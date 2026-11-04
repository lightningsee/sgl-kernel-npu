# MegaMoe A3/A5 调用链场景分析

## 对齐目标

CANN 原生 MegaMoe 与 DeepEP 封装的 wrapper-equivalence 已完成验证，后续可将 A5 精度参考切换为 MC2 小算子链。目标调用链：

```text
build_case_matrix
  → prepare_case
  → run_small_op_reference（A3/A5 各自量化实现）
  → run_fused_reference
  → 公共精度校验
  → bench(小算子链, MegaMoe)
```

## 功能点 1：统一 case 矩阵

A3/A5 都覆盖 activation、beta、token 全量/半量和 rank jitter；性能模式仍使用单长度、无 jitter 输入。

- 修改 `build_case_matrix`：A3 case 增加 `num_tokens=tokens + rank * token_jitter`，`routing_dtype` 仍只用于 A5。
- `prepare_case` 已读取 `case["num_tokens"]`，A3 已通过 all_reduce 获取最大 token 数，无需修改。
- A3 `run_baseline_reference` 已支持 padding、`x_active_mask` 和输出裁剪，无需修改。
- `run_a5_scenes` 的 capacity 已覆盖最大 rank token 数，`run_fused_reference` 和 `finalize_case` 无需修改。

## 功能点 2：对齐 routing dtype 与精度校验

A3 routing weights 已为 FP32；A5 使用 `--routing-dtype fp32` 即可对齐。A3 的 int32 与 A5 FP32 路由场景的 int64 `topk_idx` 属于接口差异，不强行统一。

若将 `routing_dtype` 扩展为 A3 测试维度：

- 修改 `build_case_matrix`，为 A3 case 增加 `routing_dtype`。
- 修改 `prepare_case`，将 softmax 和 routing weight dtype 转换整理为公共逻辑。
- 修改 `build_parser`，将 `--routing-dtype` 调整为公共参数，并分别限制平台合法值。
- 检查 `run_baseline_reference`/`run_mc2_combine` 对 BF16 routing weights 的支持；不支持时 A3 仅开放 FP32。

精度校验统一检查输出形状、有限性、`calc_diff` 和 `expert_token_nums`。`finalize_case` 复用公共校验结构，但 `DIFF_THRESHOLDS`、`rtol/atol` 按量化模式分别保留。

## 功能点 3：对齐激活参考

- 修改 `utils.py::run_small_op_baseline`：增加可选 SiTU reference 参数，默认继续使用 `fused_situ_reference`。
- 修改 `make_a5_small_op_call`：显式传入 MegaMoe 的 `situ_reference`。
- `situ_reference`、`apply_activation` 和 FusedDeepMoe 调用方无需修改。

## 功能点 4：精度与性能复用同一 MC2 小算子链

- 修改 `launch_case`：A5 精度路径由 `run_a5_reference` 改为 `make_a5_small_op_call(...)()`。
- 修改 `test_loop`：A5 MC2 通信组在精度模式也创建，不再仅限性能对比模式。
- 修改 `run_a5_scenes`：完全弃用 CANN 参考后，删除参考路径 `reference_buffer` 的创建和销毁，并将小算子支持场景校验移到公共入口。
- `run_a5_reference` 不再进入主调用链，可删除或保留为可选调试参考。
- `run_test_cases` 已在性能段使用 `make_a5_small_op_call`，无需修改；性能-only 模式继续使用 `bench_kineto`。

切换优先覆盖 MXFP8、stacked、ND、等长 token 和 SwiGLU；per-expert、NZ、FP4、A8W4 及不等长 token 在完成适配后扩展。MC2 精度对比使用独立容差，不沿用 wrapper-equivalence 阈值。

## 功能点 5：统一权重准备接口

统一由 `prepare_*_weights` 准备完整的 baseline/fused 权重，不强行统一 A3 INT8 与 A5 MXFP 的权重键名和格式。

- 修改 `prepare_a5_scene_weights`：增加 `include_baseline` 参数，同时准备 MC2 baseline 所需的权重和 scale。
- 修改 `run_a5_scenes`：传入 `include_baseline=not args.performance_only`。
- 修改 `make_a5_small_op_call`：直接消费预准备的 baseline 权重，移除函数内临时 transpose/转换逻辑。
- A3 `prepare_scene_weights` 已支持 `include_baseline`，无需修改。
- `run_baseline_reference`、`run_fused_reference` 和 `utils.py` 在此功能点无需修改。
