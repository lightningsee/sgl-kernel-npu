# 深度模型（MoE）算子测试与覆盖矩阵说明

算子的测试部分包括：
1. 精度测试
2. 基线对齐
3. 绝对误差，相对误差，容忍率
4. 激活测试
5. 内存排布方面：per-expert, stacked

## 路由场景（topk_idx 生成）

| 场景 | topk_idx 生成方式 | 触发参数 | 代码位置 |
|---|---|---|---|
| 全 rank 随机路由 | 每个 rank 独立生成随机专家分数，并通过 TopK 选择全局专家 ID | 不需要，默认执行 | 121-132 行（生成）；1342-1344 行（调用） |

# MegaMoe 测试参数说明

`tests/python/deepep/test_mega_moe.py` 的主要命令行参数如下：

| 参数 | 类型 | 默认值 | 用途 / 意图 |
|---|---|---:|---|
| `--num-processes` | `int` | `16` | 本机启动的 worker 进程数，每个进程对应一个 rank。 |
| `--quant-mode` | `str` | `"w4a8"` | 指定权重/激活量化模式。A3：`w4a8`、`w8a8`；A5：`mx_fp8_e4m3`、`mx_fp8_e5m2`、`mx_fp4_e2m1`、`mx_fp8_e4m3_w4`。 |
| `--num-tokens` | `int` | `64` | 每个 rank 生成的输入 token 数。A5 精度测试还会覆盖约一半长度的场景；性能测试只测指定长度。 |
| `--hidden` | `int` | `7168` | 输入及输出 token 的隐藏层维度。 |
| `--moe-intermediate-size` | `int` | `3072` | 每个 MoE 专家的中间层隐藏维度。 |
| `--num-topk` | `int` | `8` | 每个 token 路由选择的专家数量。 |
| `--num-experts` | `int` | `64` | EP 集群的全局专家总数，不是单卡本地专家数。 |
| `--activation` | `str` | `"swiglu"` | 指定激活函数测试项，支持逗号分隔，例如 `swiglu`、`situ` 或 `swiglu,situ`。 |
| `--beta` | `str` | `"4.0"` | SiTU 激活的 `beta` 测试值；支持逗号分隔的浮点数列表，如 `1.0,4.0`。 |
| `--linear-beta` | `str` | `"25.0"` | SiTU 激活的线性项 `linear_beta` 测试值；支持逗号分隔，数值 `0` 会转换为 `None`。 |
| `--enable-performance` | 布尔标志 | `False` | 精度校验完成后，增加 MegaMoe 与 MC2 小算子链的性能对比；传入即为 `True`。 |
| `--performance-only` | 布尔标志 | `False` | 跳过精度 baseline，仅执行 MegaMoe 性能测试；同时自动开启 `--enable-performance`。 |
| `--performance-iters` | `int` | `30` | 性能测试的迭代次数。 |
| `--weight-layout` | `str` | `"all"` | 仅 A5：权重及 scale 的 TensorList 布局。可选 `all`、`per-expert`、`stacked`；`all` 表示两种布局都测试。 |
| `--weight-format` | `str` | `"all"` | 仅 A5：权重存储格式。可选 `all`、`ND`、`NZ`；实际允许格式由量化模式决定。 |
| `--routing-dtype` | `str` | `"all"` | 仅 A5：路由权重的数据类型。可选 `all`、`bf16`、`fp32`；`all` 表示两种类型都测试。 |

# MegaMoe 正确性校验

校验主入口为 `finalize_case`（`test_mega_moe.py` 1467-1529 行）。A3 以 `run_baseline_reference`（878-1001 行）为参考，A5 以 `run_a5_reference`（1004-1037 行）为参考，二者均与 `run_fused_reference`（1241-1271 行）的融合算子输出进行比对。

| 校验项 | 阈值 | 代码位置 | 说明 |
|---|---|---|---|
| 全卡最大 `calc_diff` | A3 上测试：`1e-4`；A5 上测试：`1e-5` | 35-44、1492-1497、1516-1519 行 | `all_reduce(MAX)` 汇总全卡最大 diff，`max_diff >=` 阈值则失败。 |
| 专家 token 计数一致性（`counts_match`） | 完全相等，无容差 | 176-187、1499-1506、1516-1519 行 | `baseline_counts` 与 `fused_counts` 统一格式后用 `torch.equal` 比较本地专家 token 数；任一 rank 不一致则失败。 |

```python
baseline_out = launched_case["baseline_out"]
fused_out = launched_case["fused_out"]
local_diff = calc_diff(baseline_out.float(), fused_out.float(),)
diff_tensor = torch.tensor(local_diff, dtype=torch.float32, device=device)
dist.all_reduce(diff_tensor, op=dist.ReduceOp.MAX)

baseline_counts = launched_case["baseline_counts"]
fused_counts = launched_case["fused_counts"]
counts_match = torch.equal(
    normalize_expert_counts(baseline_counts, num_local_experts, device),
    normalize_expert_counts(fused_counts, num_local_experts, device),
)
counts_mismatch = torch.tensor(
    0 if counts_match else 1, dtype=torch.int32, device=device
)
dist.all_reduce(counts_mismatch, op=dist.ReduceOp.MAX)
```


# MegaMoe 性能测试

| 测试对象 | 计时方式 | 代码位置 | 说明 |
|---|---|---|---|
| baseline 小算子链（性能对比模式，`small_fn`） | `bench` 函数 | 1693-1699 行 | 对完整 baseline 小算子链计时，结果记为 `small_time`。 |
| MegaMoe 融合算子（性能对比模式，`fused_fn`） | `bench` 函数 | 1693-1699 行 | 对完整 MegaMoe 融合路径计时，结果记为 `fused_time`，并与 `small_time` 计算加速比。 |
| MegaMoe 融合算子（单独性能测试） | `bench_kineto` 函数 | 1711-1728 行 | 在 `--performance-only` 下跳过 baseline 和精度校验，只执行 MegaMoe；|

说明：性能对比模式的两条路径使用同一 case 的输入和相同计时参数。调用 `bench` 时设置 `num_tests=args.performance_iters + 1`、`sync_fn=dist.barrier`，通过 NPU Event 计时并同步各 rank；首个计时样本被丢弃，对其余样本取平均值。最终输出 `small_ops_time`、`megamoe_time`，并按 `small_time / fused_time` 计算 MegaMoe 加速比。表中 `bench` 两行对应 `--enable-performance` 且非 `--performance-only` 的性能对比路径，`bench_kineto` 一行对应 `--performance-only` 的单独性能测试路径。

```python
small_fn = partial(
    run_baseline_reference, *call_args, group_ep=baseline_group_ep, barrier_group=baseline_group,
    ep_world_size=num_ranks, ep_rank_id=rank, enable_dispatch_v2=enable_dispatch_v2,)
fused_fn = partial(run_fused_reference, fused_buffer, *call_args)
small_time, fused_time = (
    bench(
        fn,
        num_tests=args.performance_iters + 1,
        sync_fn=dist.barrier,
    )[0]
    for fn in (small_fn, fused_fn)
)

fused_time = bench_kineto(
    partial(
        run_fused_reference,
        fused_buffer,
        launched_case["x"],
        launched_case["topk_idx"],
        launched_case["topk_weights"],
        launched_case["max_num_tokens"],
        args.num_experts,
        launched_case["weights"],
        case["activation"],
        case["beta"],
        case["linear_beta"],
    ),
    "MegaMoe",
    num_tests=args.performance_iters,
    barrier_comm_profiling=True,
)
```

# A5上MegaMoe覆盖的测试场景
1. 量化模式: mx_fp8_e4m3, mx_fp8_e5m2, mx_fp4_e2m1, mx_fp8_e4m3_w4
```python
A5_QUANT_MODES = (
    "mx_fp8_e4m3",
    "mx_fp8_e5m2",
    "mx_fp4_e2m1",
    "mx_fp8_e4m3_w4",
)
A5_FP4_WEIGHT_MODES = frozenset({
    "mx_fp4_e2m1",
    "mx_fp8_e4m3_w4",
})
```
2. 权重布局：per-expert: 每个专家独立Tensor; stacked: 专家堆叠Tensor. 默认all, 两种都测
3. 权重格式：`mx_fp8_e4m3` ：ND、NZ；`mx_fp8_e5m2` ：ND；`mx_fp4_e2m1` ：NZ；`mx_fp8_e4m3_w4` ：NZ
```python
formats = {
    "mx_fp8_e4m3": ("ND", "NZ"),
    "mx_fp8_e5m2": ("ND",),
    "mx_fp4_e2m1": ("NZ",),
    "mx_fp8_e4m3_w4": ("NZ",),
}[args.quant_mode]
```
4. 路由方式：全 rank 独立随机生成专家分数并执行 TopK，使用全局专家 ID，可覆盖跨 rank 路由
```python
    scores = torch.randn(
    (num_tokens, num_experts),
    dtype=torch.float32,
    device=device,
)
topk_weights, topk_ids = torch.topk(
    scores,
    num_topk,
    dim=-1,
    largest=True,
    sorted=False,
)
- `scores` 的列数是`num_experts` ，即 全局专家总数。
- `torch.topk` 返回被选专家的列号，因此`topk_idx` 是`[0, num_experts-1]` 范围内的全局专家 ID。
每个 case 调用路由生成函数
topk_idx, topk_weights = make_topk_inputs(
    num_tokens, args.num_experts, args.num_topk, device, topk_dtype
)
```
5. token长度：精度测试覆盖`num_tokens` 和`max(1, num_tokens // 2)` 两种基础长度
6. 跨rank不等长：精度测试中，各rank的token数为`基础长度 + rank × 8` ，覆盖不同 rank 输入长度不同的场景，不做 padding
```python
核心代码在build_case_matrix
token_jitter = 0 if args.enable_performance else A5_ACCURACY_TOKEN_JITTER
num_tokens = tokens + rank * token_jitter
```
7. 激活函数：默认覆盖 SwiGLU；传入`--activation situ` 可覆盖 SiTU，并可通过`--beta` 、`--linear-beta` 组合多组参数
```python
cases = []
    for activation in activations:
        if activation == "swiglu":
            cases.append(
                {
                    "activation": "swiglu",
                    "beta": 1.0,
                    "linear_beta": None,
                }
            )
            continue
        if activation != "situ":
            raise ValueError(f"Unsupported activation case: {activation}")
        for beta in betas:
            for linear_beta in linear_betas:
                cases.append(
                    {
                        "activation": "situ",
                        "beta": beta,
                        "linear_beta": linear_beta,
                    }
                )
```
8. 性能对比：覆盖 MegaMoe 与 MC2 小算子链的性能对比；小算子链支持按每 rank 512 tokens 分块，MegaMoe 单次处理完整batch
9. 纯性能模式：`--performance-only` 下跳过精度和 baseline，只测 MegaMoe，使用`bench_kineto` 计时

# A3上MegaMoe覆盖的测试场景
1. 量化模式：`w4a8` 、`w8a8` ，每次运行选择一种
```python
prepare_weights = (
    prepare_w4a8_weights if quant_mode == "w4a8" else prepare_w8a8_weights
)
```
2. 精度参考：MC2 Dispatch → GMM1 → 激活 → 动态量化 → GMM2 → Combine 小算子链
```python
run_mc2_dispatch(...)
run_grouped_matmul_quantized(...)  # GMM1
apply_activation(...)
torch_npu.npu_dynamic_quant(...)
run_grouped_matmul_quantized(...)  # GMM2
run_mc2_combine(...)
```
3. 被测路径： DeepEP`fused_deep_moe` / MegaMoe 融合路径
4. 激活： 默认 SwiGLU；可配置 SiTU，并组合多组`beta` 、`linear_beta`
5. token: 每个 case 使用`--num-tokens` 指定的长度；A3 不自动增加半长度和 rank jitter
```python
代码在build_case_matrix函数，因为不是A5,不会新增num_tokens // 2 的半长度case
不会计算tokens + rank * token_jitter, 直接使用tokens

num_tokens = case.get("num_tokens", args.num_tokens)
x = torch.randn((num_tokens, args.hidden), dtype=torch.bfloat16, device=device)
```
6. 精度校验： 融合输出与 MC2 baseline 的`calc_diff < 1e-4` ，并校验专家 token 计数一致
```python
DIFF_THRESHOLDS = {
    "w4a8": 1e-4,
    "w8a8": 1e-4,
    ...
}
local_diff = calc_diff(
    baseline_out.float(),
    fused_out.float(),
)
diff_tensor = torch.tensor(local_diff, ...)
dist.all_reduce(diff_tensor, op=dist.ReduceOp.MAX)
比较专家 token 计数:
counts_match = torch.equal(
    normalize_expert_counts(baseline_counts, ...),
    normalize_expert_counts(fused_counts, ...),
)
if counts_mismatch.item() != 0:
    failures.append("expert_token_nums mismatch across ranks")

if max_diff >= diff_threshold:
    failures.append(...)
```
7. 性能模式：可测 MC2 小算子链与 MegaMoe 的耗时和加速比，或使用`--performance-only` 只测 MegaMoe
```python
small_fn = partial(
    run_baseline_reference,
    *call_args,
    group_ep=baseline_group_ep,
    barrier_group=baseline_group,
    ep_world_size=num_ranks,
    ep_rank_id=rank,
    enable_dispatch_v2=enable_dispatch_v2,
)
fused_fn = partial(run_fused_reference, fused_buffer, *call_args)
small_time, fused_time = (
    bench(fn, ...)[0]
    for fn in (small_fn, fused_fn)
)
```

# A5上仍存在的覆盖盲点
1. 只有全 rank 随机 TopK，没有强制构造热点专家、单专家集中、指定空专家、特定源 rank→目标 rank 等场景；随机出现不等于稳定覆盖
2. FP4、A8W4、`per-expert` 、NZ 等场景没有 baseline 加速比；性能测试也只测单 token 长度且不做 rank jitter
3. A5精度参考是run_a5_reference中的CANN原生MegaMoe，校验的是DeepEP wrapper与 CANN的等价性，不能替代MC2小算子链的数值精度验证
