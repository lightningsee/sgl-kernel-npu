# 测试文件按算子合并方案

## Story 名称

按算子整合 FusedDeepMoe 与 MegaMoe 的 A3/A5 测试，DispatchFFNCombine 保持 A3 独立测试。

## Story 描述

DeepEP 相关测试按算子组织文件：

- **FusedDeepMoe**：将 A5 测试合入 `test_fused_deep_moe.py`，通过现有 `--quant` 参数选择 A3/A5 执行路径，删除独立的 A5 入口文件。
- **MegaMoe**：继续使用 `test_mega_moe.py`，对齐 A3/A5 的 case、routing、激活参考、精度/性能 baseline 和权重准备流程。
- **DispatchFFNCombine（DFC）**：目前仅在 A3 测试，继续使用 `test_dispatch_ffn_combine.py`，不合并、不增加 A5 空入口、不修改算子测试逻辑。

FusedDeepMoe 与 MegaMoe 共用的 A5 MC2 baseline 放在 `utils.py`，依赖关系统一为“测试文件 → 工具模块”，不再相互 import。

## 整体方案

本方案以算子为文件边界、以 `utils.py` 为共享层，分别推进 FusedDeepMoe 的 A3/A5 入口合并和 MegaMoe 的 A3/A5 调用链对齐。平台差异保留在各算子内部，不按平台新建包含多个算子的大文件，也不强行统一量化、权重格式和数值参考。

| 算子 | A3 | A5 | 目标文件 | 本次处理 |
|---|---|---|---|---|
| FusedDeepMoe | 已支持 | 已支持 | `test_fused_deep_moe.py` | 合并 A3/A5 入口，保留各自执行链 |
| MegaMoe | 已支持 | 已支持 | `test_mega_moe.py` | 对齐 A3/A5 测试流程，复用 MC2 小算子链 |
| DispatchFFNCombine | 已支持 | 未适配 | `test_dispatch_ffn_combine.py` | 保持独立，不修改 |

目标结构：

```text
tests/python/deepep/
├── test_fused_deep_moe.py        # FusedDeepMoe：A3/A5 统一入口
├── test_mega_moe.py              # MegaMoe：A3/A5 对齐后的独立测试
├── test_dispatch_ffn_combine.py  # DFC：保持 A3 独立测试，不修改
└── utils.py                      # 共享 MC2 baseline、量化配置和工具函数
```

不采用按平台新建大文件的方式，避免把三个算子的权重、通信组、Buffer 和计时逻辑混入同一入口。

## 整体实施顺序

1. **FusedDeepMoe 前置精简**：分别清理 A3/A5 文件中的无效代码、重复辅助函数和限定范围内的复杂控制流。
2. **抽取共享实现**：将 A5 MC2 baseline 及必要依赖放入 `utils.py`，FusedDeepMoe 和 MegaMoe 统一从工具模块导入。
3. **合并 FusedDeepMoe 入口**：通过 `--quant` 选择平台路径，更新 CI 和文档后删除旧 A5 文件。
4. **对齐 MegaMoe 调用链**：按五个功能点统一 case、routing、激活、baseline 和权重准备流程。
5. **分层回归**：分别验证 FusedDeepMoe、MegaMoe、DFC 和公共导入接口，确认平台覆盖没有误删。

## FusedDeepMoe 设计

### 1. 合并测试入口

不传 `--quant` 时运行 A3；传入 `fp8_e4m3`、`fp8_e5m2` 或 `fp4_e2m1` 时运行 A5。不增加 `--platform`，非法量化值直接报错。

```text
main()
  → _pre_parse_quant(): parse_known_args 预解析 --quant
  → platform = "a5" if quant else "a3"
  → build_parser(platform)
  → args = parser.parse_args()
  → A3：spawn(test_loop, ...)
  → A5：_validate_a5(args, parser) → spawn(run_rank, ...)
```

预解析使用 `argparse.parse_known_args(add_help=False)`，同时支持 `--quant=value` 和 `--quant value`。完整 parser 负责 help、参数校验和平台专属参数互斥。

| 参数 | 执行路径 |
|---|---|
| 不传 `--quant` | A3 |
| `--quant fp8_e4m3`、`--quant fp8_e5m2` | A5 MXFP8 |
| `--quant fp4_e2m1` | A5 MXFP4 |
| 其他值 | 参数报错 |

A5 原 `--quant` 的默认值必须改为 `None`，否则无法进入 A3。`choices` 使用 `utils.py` 中的字符串常量 `MX_QUANT_MODE_NAMES`，不在模块导入时解析 MXFP dtype。

### 2. 保留 A3/A5 各自执行逻辑

- A3 继续执行 `test_loop → test → baseline_test / fused_deep_moe`。
- A5 继续执行 `run_rank → run_small_op_baseline / run_buffer_fused`。
- 两套权重、baseline、第二返回值、随机种子、精度阈值和性能统计口径保持不变。
- 公共参数在两个 parser 中分别注册并保留原默认值；另一平台的专属参数由 argparse 自动拒绝。
- A5 参数校验从 `main()` 抽为 `_validate_a5(args, parser)`；A3 不增加无实际作用的对应封装。

### 3. 合并前精简

- **无效代码清理**：删除未使用的 import、辅助函数和已废弃的 trace、量化日志、profile debug 参数。
- **辅助函数整合**：A5 抽取 `_quant_and_cast_weight`，为 `make_random_profile_cases` 增加 `total_case_count`；A3 抽取 `prepare_all_weights`。
- **控制流精简**：A5 合并 `validate_random_profile_results` 中重复的 cross-iteration 校验；A3 向量化 topk_idx 计数循环。

清理前检查直接调用、函数对象引用和跨文件 import。`make_random_profile_cases`、`make_fixed_profile_cases` 通过 `case_builder` 调用，不能按静态名称搜索结果直接删除。

### 4. 保留公共接口

`test_fused_deep_moe.py` 继续提供以下接口，名称、签名和默认值不变：

- `init_base_weights(...)`
- `init_fused_weights_int8(...)`

`test_normal_and_fused_deep_moe.py` 保持现有 import 和调用方式。

## 共享 A5 MC2 baseline

`utils.py` 统一接收：

- `run_small_op_baseline`
- `MX_QUANT_MODE_NAMES` 和 `_MX_QUANT_META`
- `get_mx_quant_config`
- FusedDeepMoe 专用的 `fused_situ_reference`

量化配置分两层：

1. `_MX_QUANT_META` 只保存字符串，模块导入时不访问 MXFP dtype。
2. `get_mx_quant_config(quant_name: str)` 在运行时通过 `getattr` 解析 dtype，返回 dispatch 和中间量化所需配置。

FusedDeepMoe 的 `fused_situ_reference` 保持 `exp/add/div` 顺序和 `beta=None` 语义；MegaMoe 的 `situ_reference` 使用 `torch.sigmoid` 且参数约定不同，两者不得合并成同一份数值实现。

## MegaMoe A3/A5 调用链对齐

CANN 原生 MegaMoe 与 DeepEP 封装的 wrapper-equivalence 已完成验证，后续可将 A5 精度参考切换为 MC2 小算子链。目标调用链：

```text
build_case_matrix
  → prepare_case
  → run_small_op_reference（A3/A5 各自量化实现）
  → run_fused_reference
  → 公共精度校验
  → bench(小算子链, MegaMoe)
```

### 功能点 1：统一 case 矩阵

A3/A5 都覆盖 activation、beta、token 全量/半量和 rank jitter；性能模式仍使用单长度、无 jitter 输入。

- 修改 `build_case_matrix`：A3 case 增加 `num_tokens=tokens + rank * token_jitter`，`routing_dtype` 仍只用于 A5。
- `prepare_case` 已读取 `case["num_tokens"]`，A3 已通过 all_reduce 获取最大 token 数，无需修改。
- A3 `run_baseline_reference` 已支持 padding、`x_active_mask` 和输出裁剪，无需修改。
- `run_a5_scenes` 的 capacity 已覆盖最大 rank token 数，`run_fused_reference` 和 `finalize_case` 无需修改。

### 功能点 2：对齐 routing dtype 与精度校验

A3 routing weights 已为 FP32；A5 使用 `--routing-dtype fp32` 即可对齐。A3 的 int32 与 A5 FP32 路由场景的 int64 `topk_idx` 属于接口差异，不强行统一。

若将 `routing_dtype` 扩展为 A3 测试维度：

- 修改 `build_case_matrix`，为 A3 case 增加 `routing_dtype`。
- 修改 `prepare_case`，将 softmax 和 routing weight dtype 转换整理为公共逻辑。
- 修改 `build_parser`，将 `--routing-dtype` 调整为公共参数，并分别限制平台合法值。
- 检查 `run_baseline_reference`/`run_mc2_combine` 对 BF16 routing weights 的支持；不支持时 A3 仅开放 FP32。

精度校验统一检查输出形状、有限性、`calc_diff` 和 `expert_token_nums`。`finalize_case` 复用公共校验结构，但 `DIFF_THRESHOLDS`、`rtol/atol` 按量化模式分别保留。

### 功能点 3：对齐激活参考

- 修改 `utils.py::run_small_op_baseline`：增加可选 SiTU reference 参数，默认继续使用 `fused_situ_reference`。
- 修改 `make_a5_small_op_call`：显式传入 MegaMoe 的 `situ_reference`。
- `situ_reference`、`apply_activation` 和 FusedDeepMoe 调用方无需修改。

### 功能点 4：精度与性能复用同一 MC2 小算子链

- 修改 `launch_case`：A5 精度路径由 `run_a5_reference` 改为 `make_a5_small_op_call(...)()`。
- 修改 `test_loop`：A5 MC2 通信组在精度模式也创建，不再仅限性能对比模式。
- 修改 `run_a5_scenes`：完全弃用 CANN 参考后，删除参考路径 `reference_buffer` 的创建和销毁，并将小算子支持场景校验移到公共入口。
- `run_a5_reference` 不再进入主调用链，可删除或保留为可选调试参考。
- `run_test_cases` 已在性能段使用 `make_a5_small_op_call`，无需修改；性能-only 模式继续使用 `bench_kineto`。

切换优先覆盖 MXFP8、stacked、ND、等长 token 和 SwiGLU；per-expert、NZ、FP4、A8W4 及不等长 token 在完成适配后扩展。MC2 精度对比使用独立容差，不沿用 wrapper-equivalence 阈值。

### 功能点 5：统一权重准备接口

统一由 `prepare_*_weights` 准备完整的 baseline/fused 权重，不强行统一 A3 INT8 与 A5 MXFP 的权重键名和格式。

- 修改 `prepare_a5_scene_weights`：增加 `include_baseline` 参数，同时准备 MC2 baseline 所需的权重和 scale。
- 修改 `run_a5_scenes`：传入 `include_baseline=not args.performance_only`。
- 修改 `make_a5_small_op_call`：直接消费预准备的 baseline 权重，移除函数内临时 transpose/转换逻辑。
- A3 `prepare_scene_weights` 已支持 `include_baseline`，无需修改。
- `run_baseline_reference`、`run_fused_reference` 和 `utils.py` 在此功能点无需修改。

## 文件修改清单

下表中的测试文件均位于 `tests/python/deepep/`：

| 文件 | 修改内容 |
|---|---|
| `test_fused_deep_moe.py` | 合入 A5 实现，整理 parser 和 `main()`，保留公共权重接口 |
| `test_fused_deep_moe_a5.py` | 引用迁移并验证后删除，不保留兼容 wrapper |
| `test_mega_moe.py` | 完成五项 A3/A5 调用链对齐，统一从 `utils` 导入共享 MC2 baseline |
| `utils.py` | 共享 MC2 baseline、量化配置和 SiTU reference；支持 MegaMoe 激活参考注入 |
| `test_dispatch_ffn_combine.py` | **不修改** |
| `test_normal_and_fused_deep_moe.py` | 保持原 import 和调用方式 |
| `.github/workflows/pr-test-deepep-npu.yml` | 更新 FusedDeepMoe A5 路径并显式传入 `--quant`；DFC 命令不变 |
| `scripts/generalization_test_fused_deep_moe.sh` | 保持原命令 |
| `python/deep_ep/README.md` | 更新 FusedDeepMoe A5 命令和 MegaMoe 使用说明 |
| `docs/design_a5_megamoe_param_adaptation.md` | 同步 MegaMoe baseline、参数和 SiTU 行为说明 |

FusedDeepMoe A5 命令调整为：

```bash
python3 tests/python/deepep/test_fused_deep_moe.py \
  --num-processes 4 --quant fp8_e4m3 --num-warmups 5 --num-tests 30
```

## Commit 拆分建议

1. **精简 FusedDeepMoe A3/A5**：分别完成无效代码、辅助函数和控制流清理。
2. **抽取共享 `utils.py`**：迁移 MC2 baseline、量化配置和 `fused_situ_reference`，更新两个算子的 import。
3. **合并 FusedDeepMoe 入口**：完成 parser、A5 实现搬移、旧文件删除以及 CI/文档更新。
4. **对齐 MegaMoe 输入与校验**：完成 case、routing 和公共精度校验。
5. **切换 MegaMoe MC2 精度参考**：完成激活注入、权重准备、通信组和性能复用改造。
6. **分层回归**：补充 A3/A5 覆盖验证；DFC 只执行回归，不产生代码改动。

## 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 删除 FusedDeepMoe A5 旧文件 | 回滚需要恢复文件 | 合并路径完整验证后再删除，利用 git 历史恢复 |
| `get_mx_quant_config` 签名变化 | 调用方可能不兼容 | 统一改传量化模式字符串，并检查所有调用方 |
| 模块导入时解析 MXFP dtype | A3 或缺少 MXFP 能力的环境无法导入 | 静态字符串表与运行时 dtype 解析分层，增加导入测试 |
| A5 MC2 baseline 场景受限 | 不能直接覆盖全部 MegaMoe 精度 case | 先覆盖 MXFP8/stacked/ND/等长/SwiGLU，再逐步扩展 |
| 各 rank 不等长导致 `global_bs` 和 chunk 不一致 | MC2 集合通信参数错误或执行失败 | 统一全局 token 计算和各 rank chunk 调度，必要时仅对 baseline 做 padding/mask |
| 两个 SiTU reference 误合并 | 引入数值偏差 | 通过参数注入保留各自实现，分别回归 |
| MC2 与 fused 舍入路径不同 | 原 wrapper-equivalence 阈值过严 | 为 MC2 精度对比设置独立、经数据校准的容差 |
| DFC 被误纳入平台合并 | 产生无实现的 A5 入口 | 明确 DFC 文件和 CI 均不修改 |

## 验收标准

| 验收项 | 通过标准 |
|---|---|
| FusedDeepMoe A3 | 不传 `--quant`，原有正确性和性能测试通过 |
| FusedDeepMoe A5 | 三种 MXFP 模式按原覆盖通过，profiler 统计口径不变 |
| 平台参数互斥 | 错传另一平台专属参数时 argparse 明确报错 |
| MegaMoe case 对齐 | A3/A5 均覆盖约定的 token 长度和 rank jitter，性能模式输入保持稳定 |
| MegaMoe 精度 | MC2 baseline 与 fused 的输出误差、专家计数均满足对应量化模式阈值 |
| MegaMoe 性能 | 精度与性能复用同一小算子链，性能-only 仍正确输出 MegaMoe profile 耗时 |
| DFC 不变 | `test_dispatch_ffn_combine.py` 无代码差异，原 A3 测试和 CI 通过 |
| 公共接口兼容 | `test_normal_and_fused_deep_moe.py` 无需修改即可运行 |
| `utils.py` 导入 | 缺少 MXFP dtype 时仍可导入，A3 公共权重接口可用 |
| CI 与覆盖 | FusedDeepMoe、MegaMoe、DFC 原有效测试均通过，无静默跳过 |

## 方案优点与边界

**优点**：

- 文件按算子划分，测试入口和通信资源边界清晰。
- FusedDeepMoe 的 A3/A5 公共行为可在同一文件维护。
- MegaMoe 的 A3/A5 校验流程对齐，同时保留平台量化差异。
- 共享依赖统一放入 `utils.py`，避免测试文件相互依赖。
- DFC 不制造无实现的 A5 空壳入口。
- 可以分 commit 实施和回滚。

**边界**：

- 本方案不实现 A5 DFC。
- MegaMoe 切换 MC2 精度参考属于数值验证改造，不是纯代码搬移，必须独立回归。
- 不为了形式统一而合并不同平台的量化配置、权重格式、SiTU 数值实现或精度阈值。
