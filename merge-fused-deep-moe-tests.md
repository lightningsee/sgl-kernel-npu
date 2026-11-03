# FusedDeepMoe 测试按算子合并方案

## Story名称

合并 A3/A5 FusedDeepMoe 测试入口。

## Story描述

目前 A3 和 A5 的 FusedDeepMoe 测试放在两个文件里，入口和参数解析分别维护。这个方案把 A5 测试合入 `test_fused_deep_moe.py`，通过现有 `--quant` 参数选择执行路径，减少一个测试入口。

A3、A5 的权重、baseline、通信 Buffer 和计时实现各自保留，原有输入、参数默认值、随机种子、精度阈值和测试覆盖不变。`test_dispatch_ffn_combine.py` 和 `test_mega_moe.py` 仍单独维护。

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

先用预解析器读取 `--quant`，再构造对应平台的 parser。这样能保留两套默认值，也能拒绝另一平台的专属参数。

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
  → 预解析 --quant
  → build_parser(platform)
  → 解析并校验该平台参数
  → A3：spawn(test_loop, ...)
  → A5：spawn(run_rank, ...)
```

预解析器关闭 help，完整 parser 负责 help 和参数校验。A3 的 `--debug` 仍写入 `args.debug`；SiTU、shape 和 A5 性能参数只在 A5 路径处理。

| 参数 | 执行路径 |
|---|---|
| 不传 `--quant` | A3 |
| `--quant fp8_e4m3`、`--quant fp8_e5m2` | A5 MXFP8 |
| `--quant fp4_e2m1` | A5 MXFP4 |
| 其他值 | 参数报错 |

FusedDeepMoe 文件保留两个公共接口，名称和签名不变：`init_base_weights(...)`、`init_fused_weights_int8(...)`。`run_small_op_baseline(...)` 移至 `utils.py`，名称和签名同样保持不变。

`test_fused_deep_moe.py` 和 `utils.py` 均不在模块加载时解析 MXFP dtype。增加导入检查：模拟 `torch` 或 `torch_npu` 缺少 MXFP 属性，确认仍能导入 `utils` 及 A3 的两个权重接口；执行 A5 时再检查所需 dtype 是否可用。

A3 继续走 `test_loop → test → baseline_test / fused_deep_moe`，A5 继续走 `run_rank → run_small_op_baseline / run_buffer_fused`。两套第二返回值的含义保持不变。

### 共享 A5 MC2 baseline

`utils.py` 接收 `run_small_op_baseline`、`get_mx_quant_config`、量化配置解析以及 FusedDeepMoe 专用的 SiTU reference。量化配置在实际调用时构造，parser 的 `choices` 使用独立的量化模式名称常量，不能通过实例化包含 MXFP dtype 的配置表获取。

FusedDeepMoe 的 `situ_reference` 迁移后命名为 `fused_situ_reference`，继续保持 `exp/add/div` 的运算顺序和 `beta=None` 语义。MegaMoe 现有的同名函数使用 `torch.sigmoid` 且参数约定不同，必须保留在原文件中，不与 FusedDeepMoe 版本合并。

FusedDeepMoe 和 MegaMoe 均从 `utils` 导入 `run_small_op_baseline`，删除 MegaMoe 对 FusedDeepMoe 测试模块的延迟 import。此次移动只改变共享代码位置，不修改 MC2 算子调用顺序、权重转置、计数返回值或性能统计口径。

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
