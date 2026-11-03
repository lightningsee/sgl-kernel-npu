# FusedDeepMoe 测试按算子合并方案

## Story名称

合并 A3/A5 FusedDeepMoe 测试入口。

## Story描述

目前 A3 和 A5 的 FusedDeepMoe 测试放在两个文件里，入口和参数解析分别维护。这个方案把 A5 测试合入 `test_fused_deep_moe.py`，通过现有 `--quant` 参数选择执行路径，减少一个测试入口。

A3、A5 的权重、baseline、通信 Buffer 和计时实现各自保留，原有输入、参数默认值、随机种子、精度阈值和测试覆盖不变。`test_dispatch_ffn_combine.py` 和 `test_mega_moe.py` 仍作为独立测试文件维护，不合并进 FusedDeepMoe 文件；但 MegaMoe 对 `run_small_op_baseline` 的 import 来源会改为 `utils.py`（见"共享 A5 MC2 baseline"）。

## 整体方案

本方案采用“按算子组织测试、按平台保留执行链”的整合方式：FusedDeepMoe 对外只保留一个测试文件和命令行入口，A3/A5 在文件内部继续使用各自的实现，不为了形式上的复用统一权重、baseline 或数值语义。共享代码仅抽取已有明确复用关系的 A5 MC2 baseline，使测试模块统一依赖 `utils.py`，不再相互 import。

目标结构如下：

```text
tests/python/deepep/
├── test_fused_deep_moe.py        # 合并 A3/A5 FusedDeepMoe，统一 parser 和 main()
├── test_dispatch_ffn_combine.py  # 保持独立，不纳入本次合并
├── test_mega_moe.py              # 保持独立，从 utils 导入共享 MC2 baseline
└── utils.py                      # 接收共享 A5 MC2 baseline 及其必要依赖
```

实施按以下顺序推进：

1. **前置精简**：分别在原 A3、A5 文件中清理无效代码、整合辅助函数并完成限定范围的控制流精简，避免与代码搬移混在同一 diff 中。
2. **抽取共享实现**：将 `run_small_op_baseline` 及必要依赖迁入 `utils.py`，FusedDeepMoe 和 MegaMoe 同步修改 import，并验证共享路径行为不变。
3. **合并测试入口**：把 A5 剩余实现按原调用关系合入 `test_fused_deep_moe.py`，通过 `--quant` 预解析选择平台 parser 和 spawn 入口；不传 `--quant` 走 A3，三种 MXFP 模式走 A5。
4. **迁移引用并删除旧入口**：更新 CI、文档和跨文件引用，确认新路径可用后删除 `test_fused_deep_moe_a5.py`，不保留兼容 wrapper。
5. **分层回归**：依次完成语法与导入检查、A3/A5 FusedDeepMoe 设备回归、MegaMoe 共享 baseline 回归及混合测试公共接口检查。

整个过程保持原有输入生成顺序、随机种子、第二返回值、精度阈值、profiler 事件和统计口径不变；A3 对外权重接口保持名称和签名不变，MXFP dtype 仅在实际使用时解析。只修改当前工作树，不处理嵌套镜像目录。

## 功能点名称

1. 合并测试入口。
2. 保留 A3/A5 各自的执行逻辑。
3. 合并前对两个文件分三步精简：无效代码清理、前置辅助函数整合、限定范围的控制流精简。
4. 更新跨文件依赖、CI 和文档。

## 功能点描述

1. 不传 `--quant` 时运行 A3；传入 `fp8_e4m3`、`fp8_e5m2` 或 `fp4_e2m1` 时运行 A5。不增加 `--platform`，非法量化值直接报错。
2. A3 继续运行 `test_loop`，A5 继续运行 `run_rank`。两条路径使用自己的权重、baseline 和性能统计。A5 dtype 在实际使用时解析，避免影响 A3 模块导入。
3. 合并前完成三步精简，降低合并 PR 的 diff 噪音：
   - **无效代码清理**：A3 删除 `sys`/`time`/`partial`/`hash_tensor` 导入、`make_uniform_topk_idx`、`from_inclusive_prefix_sum`、`str_to_bool`；A5 删除 `format_triplet`、`run_buffer_fused_with_burn_in`、`log_quant_tensor` 系列、`trace_dir`/`kernel_trace_dir`/`dump_profile_events` 等已废弃参数及处理分支。
   - **前置辅助函数整合**：A5 抽取 `_quant_and_cast_weight` 消除 gmm1/gmm2 量化重复，给 `make_random_profile_cases` 加 `total_case_count` 参数消除 `make_fixed_profile_cases` 的 Namespace hack；A3 抽取 `prepare_all_weights` 整合 `init_base_weights → init_baseline_weights → init_fused_weights_int8` 链式调用。
   - **控制流精简（本次限定）**：A5 合并 `validate_random_profile_results` 中 `fixed_values` 的两轮 cross-iteration 校验；A3 topk_idx 计数循环向量化。`run_rank` 末尾统计打印嵌套留给合并后单独 PR。
4. 保留其他测试使用的公共函数，将共享 A5 MC2 baseline 移至 `utils.py`，使 FusedDeepMoe 和 MegaMoe 都从工具模块导入，不再相互依赖。A5 CI 命令替换文件路径并补上 `--quant`；A3 命令和 generalization 脚本保持原样。

## 功能实现思路

先用 `argparse` 的 `parse_known_args` 预解析 `--quant`（`add_help=False`，不消费其他参数），再构造对应平台的 parser。不用 `sys.argv` 手动扫描，`parse_known_args` 能正确处理 `--quant=fp8_e4m3` 和 `--quant fp8_e4m3` 两种写法。这样能保留两套默认值，也能拒绝另一平台的专属参数。

迁移 A5 实现时按原来的调用关系搬移，不同时重写权重或 baseline。A3 代码放在前面，A5 代码放在后面，文件末尾统一保留 `main()`。

清理函数前同时检查直接调用、函数对象引用和跨文件 import。`make_random_profile_cases`、`make_fixed_profile_cases` 通过 `case_builder` 调用，需要保留。

先更新依赖和执行命令，再删除旧 A5 文件。核对合并前后的默认参数、公共接口和执行路径，确认现有测试覆盖没有减少。只处理当前工作树，不修改嵌套镜像目录。

### 合并前精简策略

三步精简作为合并 PR 的前置 commit，按顺序提交便于 review 和回滚。三步只触碰各自文件，不跨文件改动；迁 `utils.py` 和合并入口留到后续 commit。若控制流精简范围在实现时发现风险（如 `validate_random_profile_results` 两轮校验合并后漏掉某个断言路径），该子项可回退，不影响前两步成果。

## 功能实现设计

### 代码设计

合并后的 A3/A5 测试实现放在 `tests/python/deepep/test_fused_deep_moe.py`，两个算子共用的 A5 MC2 baseline 放在 `utils.py`，入口如下：

```text
main()
  → _pre_parse_quant(): parse_known_args 抢取 --quant（default=None，add_help=False）
  → platform = "a5" if quant else "a3"
  → build_parser(platform): _build_a3_parser() 或 _build_a5_parser()
  → args = parser.parse_args()
  → A3：spawn(test_loop, ...)
  → A5：_validate_a5(args, parser) → spawn(run_rank, ...)
```

预解析器关闭 help，完整 parser 负责 help 和参数校验。A3 的 `--debug` 仍写入 `args.debug`；SiTU、shape 和 A5 性能参数只在 A5 路径处理。

跨平台参数互斥由 argparse 的"未注册即拒绝"天然生效：A3 parser 不注册 A5 专属参数（如 `--activation`），A5 parser 不注册 A3 专属参数（如 `--active-ranks`）。误传时 argparse 自动报 `unrecognized arguments`。A5 原有的 `--dump-profile-events --debug` 别名已废弃，A5 用户改用 `--print-per-rank-profile`。

| 参数 | 执行路径 |
|---|---|
| 不传 `--quant` | A3 |
| `--quant fp8_e4m3`、`--quant fp8_e5m2` | A5 MXFP8 |
| `--quant fp4_e2m1` | A5 MXFP4 |
| 其他值 | 参数报错 |

注意 A5 原 `--quant` 的 `default="fp8_e4m3"` 必须改为 `default=None`，否则不传 `--quant` 时无法触发 A3 路径。合并后 `--quant` 的 `choices` 使用字符串常量 `MX_QUANT_MODE_NAMES = ("fp8_e4m3", "fp8_e5m2", "fp4_e2m1")`（定义在 `utils.py`，见下文"共享 A5 MC2 baseline"），不引用含 MXFP dtype 的配置表。

### parser 构建与校验拆分

`build_parser(platform)` 内部分为 `_build_a3_parser()` 和 `_build_a5_parser()` 两个内部函数。共有参数（如 `--hidden`、`--num-processes`）在两个 parser 各注册一次，默认值各自保留——这是"保留两套默认值"的落地方式。A3 parser 注册 10 个参数（含 `--active-ranks`、`--topk-drop-*`、`--debug`），A5 parser 注册约 20 个参数（含 `--quant`、`--activation`、`--event-timing` 等）。

A5 当前的约 15 条校验逻辑（SiTU beta 正数、shape 范围、profile cases 互斥等）耦合在 `main()` 里，合并后抽成 `_validate_a5(args, parser)` 独立函数，从 `main()` 解耦。A3 当前无校验，不需要对应函数。

FusedDeepMoe 文件保留两个公共接口，名称和签名不变：`init_base_weights(...)`、`init_fused_weights_int8(...)`。`run_small_op_baseline(...)` 移至 `utils.py`，名称和签名同样保持不变。

`test_fused_deep_moe.py` 和 `utils.py` 均不在模块加载时解析 MXFP dtype。`utils.py` 的 `MX_QUANT_MODE_NAMES`/`_MX_QUANT_META` 只存字符串，dtype 通过 `get_mx_quant_config(quant_name)` 在调用时 `getattr` 解析（见下文"共享 A5 MC2 baseline"）。增加导入检查：模拟 `torch` 或 `torch_npu` 缺少 MXFP 属性，确认仍能导入 `utils` 及 A3 的两个权重接口；执行 A5 时再检查所需 dtype 是否可用。

A3 继续走 `test_loop → test → baseline_test / fused_deep_moe`，A5 继续走 `run_rank → run_small_op_baseline / run_buffer_fused`。两套第二返回值的含义保持不变。

### 共享 A5 MC2 baseline

`utils.py` 接收 `run_small_op_baseline`、`get_mx_quant_config`、量化配置解析以及 FusedDeepMoe 专用的 SiTU reference。

#### `MX_QUANT_CONFIGS` 拆层（解决延迟解析冲突）

当前 `MX_QUANT_CONFIGS`（A5:23-45）在模块加载时即解析 `torch.float8_e4m3fn`、`torch_npu.float4_e2m1fn_x2` 等 dtype，与"不在模块加载时解析 MXFP dtype"直接冲突。拆为两层：

- **层 1（静态字符串表，可安全迁 `utils.py`）**：`MX_QUANT_MODE_NAMES = ("fp8_e4m3", "fp8_e5m2", "fp4_e2m1")` 和 `_MX_QUANT_META` 字典，只存字符串（如 `"dtype_attr": "float8_e4m3fn"`、`"src": "torch"/"torch_npu"`），模块加载时无 dtype 解析。parser 的 `choices` 用 `MX_QUANT_MODE_NAMES`。
- **层 2（运行时构造 dtype）**：`get_mx_quant_config(quant_name: str) -> Dict`，用 `getattr(torch, attr)` 或 `getattr(torch_npu, attr)` 在调用时解析 dtype，返回含 `dispatch_quant_mode`/`dispatch_y_dtype`/`quant_dst_type`/`origin_dtype` 的字典。

**签名变化**：`get_mx_quant_config(args)` → `get_mx_quant_config(quant_name: str)`。调用方（`run_small_op_baseline`、`make_umdk_static_inputs`）改传 `args.quant`。MegaMoe 构造的 `small_args.quant` 是 `args.quant_mode.removeprefix("mx_")`（test_mega_moe.py:1547），仍是字符串，兼容。

#### SiTU reference 迁移与 MegaMoe 差异

FusedDeepMoe 的 `situ_reference` 迁移后命名为 `fused_situ_reference`，继续保持 `exp/add/div` 的运算顺序和 `beta=None` 语义（位置参数 `beta: Optional[float]`、`linear_beta: Optional[float]`）。MegaMoe 现有的同名函数（test_mega_moe.py:81-93）使用 `torch.sigmoid` 且参数约定不同（`beta` 为必填关键字参数），必须保留在原文件中，不与 FusedDeepMoe 版本合并。

#### 迁移清单

| 名称 | 原位置 | 迁后形态 | 决策 |
|------|------|------|------|
| `MX_QUANT_MODE_NAMES` | 新增 | 字符串常量 | 迁 `utils.py` |
| `_MX_QUANT_META` | A5:23-45 改造 | 字符串→字符串映射 | 迁 `utils.py` |
| `get_mx_quant_config` | A5:87-88 | 签名改 `(quant_name: str)`，运行时构造 dtype | 迁 `utils.py` |
| `run_small_op_baseline` | A5:736-895 | 签名不变，内部调 `get_mx_quant_config(args.quant)` | 迁 `utils.py` |
| `fused_situ_reference` | A5:719-734 改名 | 保持运算顺序和 `beta=None` 语义 | 迁 `utils.py` |
| `maybe_cast_weight_to_nz` | A5:91-143 | — | **不迁**，weight 构造路径专用 |
| `FUSED_COMPAT_QUANT_MODE` 等 A5 专有常量 | A5 顶部 | — | **不迁**，留在 FusedDeepMoe 文件 |

FusedDeepMoe 和 MegaMoe 均从 `utils` 导入 `run_small_op_baseline`，删除 MegaMoe 对 FusedDeepMoe 测试模块的延迟 import（test_mega_moe.py:1526）。此次移动只改变共享代码位置，不修改 MC2 算子调用顺序、权重转置、计数返回值或性能统计口径。

SwiGLU 沿用原 profiler 事件、预热和统计方式。A5 SiTU 普通模式完成正确性检查后返回，指定 `--event-timing` 时只测 fused 调用。本次不扩展 SiTU 的性能范围。

需要修改的文件如下，表中的测试文件均位于 `tests/python/deepep/`：

| 文件 | 修改内容 |
|---|---|
| `test_fused_deep_moe.py` | 合入 A5 实现，整理 parser 和 `main()`，从 `utils` 调用共享 A5 baseline |
| `test_fused_deep_moe_a5.py` | 引用迁移后删除，不保留兼容 wrapper |
| `test_mega_moe.py` | 从 `utils` 导入 `run_small_op_baseline`，移除对 FusedDeepMoe 测试模块的依赖 |
| `test_normal_and_fused_deep_moe.py` | 保持原 import 和调用方式 |
| `utils.py` | 接收共享 A5 MC2 baseline、量化配置解析和 FusedDeepMoe 专用 SiTU reference |
| `.github/workflows/pr-test-deepep-npu.yml` | A3 命令保留；A5 替换路径并显式传入 `--quant` |
| `scripts/generalization_test_fused_deep_moe.sh` | 保持原命令 |
| `python/deep_ep/README.md` | 更新 A5 测试命令 |
| `docs/design_a5_megamoe_param_adaptation.md` | 更新路径、示例和 SiTU 行为说明 |

A5 命令改为下面的形式，原来依赖默认 FP8 的命令也要补上 `--quant`：

```bash
python3 tests/python/deepep/test_fused_deep_moe.py \
  --num-processes 4 --quant fp8_e4m3 --num-warmups 5 --num-tests 30
```

## 风险与验收

### 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| 删除 `test_fused_deep_moe_a5.py` 是破坏性变更 | 回滚需恢复整个文件 | 合并 PR 前确保 A5 路径在合并后文件中完整可用；git 历史保留旧文件可随时恢复 |
| `MX_QUANT_CONFIGS` 拆层后 `get_mx_quant_config` 签名变更 | 调用方（`run_small_op_baseline`、`make_umdk_static_inputs`、MegaMoe 的 `small_args`）需同步改 | 签名从 `(args)` 改为 `(quant_name: str)`，调用方改传 `args.quant`；MegaMoe 的 `small_args.quant` 已是字符串，兼容 |
| 跨平台参数互斥可能误伤现有脚本 | A5 用户习惯传 `--debug`（已废弃别名）会报错 | 文档和 CI 命令同步更新；A5 用户改用 `--print-per-rank-profile` |
| MegaMoe 延迟 import 改为顶部 import | 若 `utils.py` 导入失败，MegaMoe 也无法启动 | `utils.py` 不在模块加载时解析 MXFP dtype，导入检查覆盖此场景 |

### Commit 拆分建议

合并 PR 建议拆 4 个 commit，便于 review 和回滚：

1. **精简 A3**：无效代码清理 + 前置辅助函数整合 + 控制流精简（仅 `test_fused_deep_moe.py`）
2. **精简 A5**：同上三步（仅 `test_fused_deep_moe_a5.py`）
3. **迁 `utils.py`**：`MX_QUANT_CONFIGS` 拆层 + `run_small_op_baseline`/`get_mx_quant_config`/`fused_situ_reference` 迁移 + MegaMoe 改顶部 import（`utils.py`、`test_mega_moe.py`）
4. **合并入口**：A5 实现合入 `test_fused_deep_moe.py`、预解析 parser、删除 `test_fused_deep_moe_a5.py`、更新 CI 和文档

若控制流精简（步骤 1/2 的第三步）发现风险，该子项可回退，不影响前两步成果。

### 验收标准

| 验收项 | 方式 | 通过标准 |
|------|------|------|
| A3 路径功能不变 | `python3 tests/python/deepep/test_fused_deep_moe.py --num-processes 4`（不传 `--quant`） | 正确性断言通过，性能输出与合并前一致 |
| A5 路径功能不变 | `python3 tests/python/deepep/test_fused_deep_moe.py --num-processes 4 --quant fp8_e4m3` | 正确性断言通过，profiler 统计输出与合并前一致 |
| A5 MXFP4 路径 | `--quant fp4_e2m1` | 正确性断言通过 |
| 跨平台参数互斥 | A3 传 `--activation situ`、A5 传 `--active-ranks 0,1` | argparse 报 `unrecognized arguments` |
| MegaMoe 不受影响 | 运行 `test_mega_moe.py` 的 A5 baseline 对比 | 从 `utils` 导入 `run_small_op_baseline` 成功，结果与合并前一致 |
| `utils.py` 可独立导入 | 模拟 `torch_npu` 缺少 `float4_e2m1fn_x2` | `import utils` 不报错，A3 的 `init_base_weights` 可调用 |
| CI 通过 | `pr-test-deepep-npu.yml` 全量 | A3/A5 两条命令均通过 |
| 测试覆盖不减少 | 对比合并前后的参数组合和执行路径 | 默认参数、随机种子、精度阈值、profile case 校验均保留 |



  ### 按算子方案的优缺点

  • 优点:改动聚焦 FusedDeepMoe,DFC/MegaMoe 只做 import 微调;A3/A5 同算子共处一文件,公共行为改一处;commit
  拆分粒度细,前两步可独立回滚。
  • 缺点:跨平台共享(run_small_op_baseline)要外迁 utils.py 桥接,MegaMoe 依赖延迟 import(L1526);--quant 切平台不如 --op
  显式;A5 DFC 无归宿(本方案不处理)。

  1. 改动面小、风险可控:只合并 FusedDeepMoe 的 A3/A5,删除 1 个旧文件(test_fused_deep_moe_a5.py);DFC 和 MegaMoe
  基本不动,只在 import 上微调。按平台方案要新建 2 个文件、删除 4 个旧文件、搬运 DFC/MegaMoe 全部代码。
  2. commit 拆分细、可分步回滚:前 2 个 commit 是单文件精简(仅 test_fused_deep_moe.py / test_fused_deep_moe_a5.py),第 3
  个迁 utils.py,第 4 个才合并入口。每步只触碰各自文件,任一步发现问题都能独立回退,不影响前序成果。按平台方案几乎每个
  commit 都横跨多算子搬运,review 压力集中。
  3. 同算子的 A3/A5 共处一文件:FusedDeepMoe 的公共行为(精度阈值、profile 校验、计数断言、profiler
  事件约定)改一处即可,两套路径天然对照。按平台方案下 A3/A5 同算子分居两文件,改公共行为要同步两边,容易漂移。
  4. DFC/MegaMoe
  保持原结构:这两个算子各有独立的通信组、Buffer、权重路径和性能统计口径,保持独立文件维护边界清晰。按平台方案强行把
  DFC(MegaMoe)和 FusedDeepMoe 塞进同一文件,会引入重名函数改名(test/test_loop/baseline_test)和 spawn 调用链调整。
  5. 不制造空壳入口:A5 DFC 底层未适配,按算子方案不处理它;按平台方案要在 A5
  文件里放入口并报不支持,文件结构领先于功能,后续若 A5 DFC 不落地就是死代码。
  6. 跨平台共享代码外迁 utils.py 更通用:run_small_op_baseline、get_mx_quant_config、fused_situ_reference 迁到 utils.py
  后,FusedDeepMoe 和 MegaMoe 都从工具模块导入,依赖关系是"测试 → 工具",而不是"测试 → 测试"。按平台方案把共享 baseline
  留在 A5 文件内,MegaMoe 仍依赖 FusedDeepMoe 所在文件,耦合方向不干净。
  7. --quant 复用现有参数语义:A5 本来就有 --quant,只是把 default 从 fp8_e4m3 改成 None 来触发 A3 路径。不增加 --op、--
  platform 等新维度,用户认知负担小;argparse 的"未注册即拒绝"天然实现跨平台参数互斥。
  8. MX_QUANT_CONFIGS 拆层有独立价值:层 1(静态字符串表)可安全迁 utils.py 且模块加载时不解析 MXFP dtype,层
  2(get_mx_quant_config(quant_name))运行时构造 dtype。这个拆分解决了"模块加载时解析 dtype 导致 A3
  路径无法导入"的隐患,无论是否合并入口都值得做。
  9. 与现有 CI/脚本兼容性高:A3 命令和 scripts/generalization_test_fused_deep_moe.sh 保持原样,A5 命令只需补 --quant
  和换路径。按平台方案要按 job 平台替换路径并增加 --op,改动范围更大。
  10. 适合渐进式推进:你现在的诉求是"只动 FusedDeepMoe 两个文件",按算子方案的前两步(A3/A5 精简)完全满足,后续迁 utils.py
  和合并入口可视情况再推进。按平台方案是一锤子买卖,必须一次性搬完三个算子才有意义。