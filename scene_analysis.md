

# 深度模型（MoE）算子测试与覆盖矩阵说明
## A3 部分

## 一、 测试场景设计

### 1. 路由场景 (`topk_idx` 生成)

| **场景**          | **触发参数**                 | **代码位置** | **覆盖意图**                                   |
| --------------- | ------------------------ | -------- | ------------------------------------------ |
| **全 rank 随机路由** | `--active-ranks` 为空（默认）  | L293-300 | scores `abs+1` 后 `topk`，token 可能分布到任意 rank |
| **限定活跃 rank**   | `--active-ranks "0,1,3"` | L253-292 | 只从指定 rank 的 expert 池采样，测部分 rank 空载         |
全rank随机路由：每个 token 的 num_topk 个 expert 可能落在任意 rank 上。
限定活跃rank： 测部分 rank 空载的路由分布，只有指定 rank 的 expert 池会被采样，其他 rank 的 expert 不会被命中。此列为模拟负载不均衡下的场景，与全rank随机路由进行对比，此处专家的采样为   
```
topk_idx = valid_experts[
            torch.randint(0, len(valid_experts), (num_tokens, num_topk), device="npu")
```
与全rank随机路由的不同
```
    scores = torch.randn((num_tokens, num_experts)).abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
```
在前者的逻辑中是会要重复的，这是否合理？

### 2. Top-k Drop 场景 (测 `-1` padding 路径)

| **场景**        | **触发参数**                                    | **代码位置** | **覆盖意图**                                      |
| ------------- | ------------------------------------------- | -------- | --------------------------------------------- |
| **无 drop**    | `--topk-drop-prob 0 --topk-drop-col -1`（默认） | L380-382 | 正常全 expert 命中                                 |
| **随机概率 drop** | `--topk-drop-prob 0.3`                      | L341-357 | 按概率置 `-1`，保证每 token 至少 1 个有效                  |
| **固定列 drop**  | `--topk-drop-col 2`                         | L360-371 | 指定列全置 `-1`，测 `topk-1` 场景 (`test_topk_minus1`) |
 随机：topk_idx 里某些位置被置成 -1
 固定：   指定位置
 --topk-drop-prob  每个位置按概率随机置 -1
   --topk-drop-col   指定某一列全部置 -1
测的是 topk_idx 里某些位置被置成 -1（表示"无效 expert"）时，fused kernel 能否正确处理。
边界：至少保留一个有效的专家，前者随机是在某一行某一列上进行drop，后者是指定列上都drop。

## 二、 验证与性能指标

### 1. 正确性校验

| **校验项**          | **阈值**                       | **代码位置** | **说明**                                        |
| ---------------- | ---------------------------- | -------- | --------------------------------------------- |
| **输出数值**         | `avg_diff < 4e-4`            | L432-444 | fused vs baseline 逐元素绝对误差均值                   |
| **`recv_count`** | `max_recv_count_diff < 1e-4` | L480-495 | fused 的 `ep_recv_count` 与 all_gather 推导的期望值对比 |
|                  |                              |          |                                               |
主要是used_output ， baseline_output的比较，涉及到
```python
    max_diff = torch.max(torch.abs(fused_output - baseline_output)).item()
    avg_diff = torch.mean(torch.abs(fused_output - baseline_output)).item()
    baseline_output_avg = torch.mean(torch.abs(baseline_output)).item()
    fused_output_avg = torch.mean(torch.abs(fused_output)).item()
    diff2 = calc_diff(fused_output, baseline_output)

```
### 2. 性能测试

| **测量对象**           | **计时方式**                 | **代码位置** | **说明**                                       |
| ------------------ | ------------------------ | -------- | -------------------------------------------- |
| **baseline（小算子链）** | `bench_kineto` 5 个事件     | L526-536 | dispatch / GMM1 / SwiGLU / GMM2 / combine 全链 |
| **fused**          | `bench_kineto` 1 个事件     | L537-541 | `FusedDeepMoe` 单算子                           |
| **baseline 汇总**    | `sum + baseline_time[2]` | L543-548 | GMM 被计两次，额外加回                                |
最后一行逻辑问题，此处在字符匹配后得到的结果是gemm1+gemm2的时长，然而在最后又加一次，变成2*（gemm1,+gemm2）,缺点一是数值变大，缺点二是数值并没有严格的分开。
```python
    baseline_time = bench_kineto(

        lambda: baseline_test(**baseline_args),

        (

            "aclnnInplaceOne_OnesLikeAiCore_OnesLike",

            "MoeLowLatencyDispatchV2",

            "aclnnGroupedMatmulWeightNz_GroupedMatmul_GroupedMatmul",

            "DequantSwigluQuant",

            "MoeLowLatencyCombineV2",

        ),

        barrier_comm_profiling=True,

    )
```

A5部分是分开的
```python
SMALL_OP_EVENT_PATTERNS = (

    ("MoeDistributeDispatchV2", "MoeDistributeDispatchV3"),

    (

        "GroupedMatmul",

        "GroupedMatMul",

        "aclnnGroupedMatmulV4_GroupedMatmul_GroupedMatmul",

    ),

    ("Swiglu", "SwiGlu"),

    ("DynamicMxQuant", "DynamicMXQuant"),

    (

        "GroupedMatmul",

        "GroupedMatMul",

        "aclnnGroupedMatmulV4_GroupedMatmul_GroupedMatmul",

    ),

    ("MoeDistributeCombineV2", "MoeDistributeCombineV3"),

)
```
对应baseline 测试的是5个小算子的耗时，对应fused部分是测试单个
## 三、 参数

### 1. 参数


| **参数名称**                  | **类型**  | **默认值** | **范围与约束**         | **覆盖意图 / 说明**                      |
| ------------------------- | ------- | ------- | ----------------- | ---------------------------------- |
| `--num-processes`         | `int`   | `16`    | 无校验               | rank 数，决定 EP 并行度                   |
| `--num-tokens`            | `int`   | `256`   | 无校验               | 每 rank token 数，决定 dispatch 规模      |
| `--hidden`                | `int`   | `7168`  | 无校验               | 隐藏维度，影响 GMM 计算量                    |
| `--moe-intermediate-size` | `int`   | `4096`  | 无校验               | 中间维度（GMM1 输出=D，激活后=D/2）            |
| `--num-topk`              | `int`   | `8`     | 无校验               | 每 token 选 expert 数                 |
| `--num-experts`           | `int`   | `256`   | 无校验               | 全局 expert 数                        |
| `--active-ranks`          | `str`   | `""`    | 逗号分隔整数            | 限定活跃 rank（对应测试场景二）                 |
| `--topk-drop-prob`        | `float` | `0.0`   | 无校验               | 随机 drop 概率（对应测试场景 3a）              |
| `--topk-drop-col`         | `int`   | `-1`    | 小于 `num_topk` 时生效 | 固定列 drop（对应测试场景 3b）                |
| `--debug`                 | `flag`  | `False` | —                 | 打印权重 shape、drop 比例、recv diff 等调试信息 |
可能存在的问题：
  • num_processes=0 会 spawn 0 进程，无报错但什么都不跑
  • num_tokens=-1 会在 torch.randn 报错，但不是友好提示
  • num_experts % num_processes != 0 在 L241 assert num_experts % num_ranks == 0 才报错

  对比 A5 有 15 条 parser.error() 校验（如 1 <= num_tokens <= 256、512 <= hidden <= 8192），A3 的参数校验很粗糙。

  ### --moe-intermediate-size 的语义， 这个参数在 A3 和 A5 的含义不同？

## 四、 当前未覆盖的场景（潜在缺口）


1. **`return_recv_hook=True` 路径**：L333 硬编码为 `False`，async/hook 路径目前未被测到。
	    这个是干什么的，有没有必要增加
      
    
2. **`cumulative_local_expert_recv_stats` 非零**：L330 初始化为零，累积统计路径未真正验证。
		累计统计是否有效，该功能测试场景需进一步确认。
```python
          cumulative_local_expert_recv_stats = torch.zeros(
        (num_local_experts,), dtype=torch.int32, device="npu"
    )
```
    
3. **变长 token（per-rank 不同 token 数）**：目前所有 rank 均使用统一的 `num-tokens`，缺乏 jitter 抖动测试。
```python
	    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="npu")
```
		在A5中
```python
		    if args.num_token_jitter > 0:
        token_jitter_rng = random.Random(NUM_TOKEN_JITTER_SEED + rank)
        token_delta = token_jitter_rng.randint(
            -args.num_token_jitter, args.num_token_jitter
        )
    else:
        token_delta = 0
    local_num_tokens = max(1, min(256, args.num_tokens + token_delta))

```
      该jitter场景是缺失的，导致负载不均衡和buffer的padding填充未进行充分验证。
    
4. **单 active rank / 空 token rank**：A5 参数支持 `--single-active-rank`，但当前部分模块（如 A3）尚未覆盖。
	    A5中
```python
	
    if args.single_active_rank is not None:
        token_delta = 0
        local_num_tokens = 1 if rank == args.single_active_rank else 0
```
      属于边界测试场景：空 token rank 的 dispatch,以及recv_count 边界 Buffer padding  ，all_reduce/all_gather，精度校验
    
5. **精度阈值动态化**：`avg_diff < 4e-4` 为单一固定阈值，缺乏基于 quant / dtype 的差异化配置（当前 A3 仅涉及 INT8，问题相对较小）。
    
      
    
## A5部分

 # A5 测试场景覆盖分析


### 一、A5 已覆盖的场景与工程增强

**【算法与功能场景覆盖】****1. 量化精度矩阵（A5 核心新增）**

- **MX_QUANT_CONFIGS 三种模式**：支持 fp8_e4m3 / fp8_e5m2 / fp4_e2m1。
    
- 每种模式独立验证 `dispatch_quant_mode=4` + `dispatch_y_dtype` 链路。
    
- FP4 走 `torch_npu.float4_e2m1fn_x2` 打包格式，并专门校验 NZ 存储字节数（L129-143）。
    
- 通过 `--quant` 单选切换，可在 Linux 一条命令测一种。
```python
    MX_QUANT_CONFIGS = {
        "fp8_e4m3":  { "dispatch_quant_mode": 4, "dispatch_y_dtype": torch.float8_e4m3fn, ...},
        "fp8_e5m2":  { "dispatch_quant_mode": 4, "dispatch_y_dtype": torch.float8_e5m2, ...},
        "fp4_e2m1":  { "dispatch_quant_mode": 4, "dispatch_y_dtype": torch_npu.float4_e2m1fn_x2, ...},
    }
```

**2. 权重格式**

- `--weight-format NZ`：默认 NZ，走 `maybe_cast_weight_to_nz` 完整 NZ cast + 字节数校验。
    
- 非 NZ：`view(origin_dtype)` 直接透传。
    
- FP4 NZ 失败时给出明确报错（L113-120）。
    

**3. 激活函数（A5 核心新增）**


- 支持 `--activation swiglu`（默认）/ `--activation situ`。
    
- SiTU 配套 `--beta` / `--linear_beta` 参数。
    
- SiTU 因精度不稳定，引入了 `SITU_MAX_MISMATCH_RATIO=1e-4` 容忍率校验（非全等比对）。
    

**4. 变长 Token 负载（A5 新增）**

- `--num-token-jitter <max>`：每个 rank 在 `[base-max, base]` 内随机偏移，模拟真实不均匀负载。
    
- `NUM_TOKEN_JITTER_SEED=2026` 保证可复现。
    
- `make_small_op_padded_inputs` 对 small-op 路径做 padded 补零 + `x_active_mask` 对齐。
    
- jitter 时跳过 strict expected_counts 检查（L1183-1190）。
    
**测试工程与性能压测工具****5. 严格的 Profile 模式与状态隔离（A3 完全没有）**

- **random profile**：每次迭代用不同 seed 生成不同输入，确保无缓存复用；跨迭代校验 `validate_random_profile_results` 保证正式迭代结果一致。
```python
    case_seed = RANDOM_PROFILE_SEED + case_idx * world_size + rank 
```
- **fixed profile**：所有迭代用相同输入但**克隆存储隔离**，通过 `assert_unique_tensor_storage` 强校验 data_ptr 唯一，专门用于排查算子底层的**状态污染**问题。
```python
        source = make_random_profile_cases(..., total_case_count=1)[0]  # 
    for case_idx in range(total_case_count):
        fused_inputs = dict(source["fused_inputs"])
        for name in ("x", "expert_ids", "expert_scales"):
            fused_inputs[name] = source["fused_inputs"][name].clone()  # 

```
- **NaN/Inf 拦截**：在 profile 校验阶段增加非数值监控。**在A5中也可以额外增加Inf的拦截**
```python
        padded_small_nan_count = (
        torch.isnan(small_output[local_num_tokens:]).sum().item()          # L513
    )

```
如果后续考虑增加profile测试，需要增加相应的NAN拦截

**6. 事件级时序剖析与稳态控制**

- `--event-timing`：用 `profile_npu_event_sequences` 抓 NPU event 序列。
	    此处在utils.py里有，可考虑迁移至A3？
- 细粒度拆解：Fused 路径整体抓取，Small-op 路径分 6 段抓取 (dispatch, gmm1, swiglu, requant, gmm2, combine)。

| **剖析维度** | **A3 (bench_kineto)** | **A5 (profile_npu_event_sequences)** | 收益                |
| -------- | --------------------- | ------------------------------------ | ----------------- |
| **匹配方式** | 按算子名独立散列匹配            | 按有序 pattern 序列强校验匹配                  | 避免外层框架同名算子被误统计    |
| **迭代概念** | 无（全局同名事件混淆累加）         | 有（完整序列匹配闭环 = 一轮迭代）                   | 隔离异步流交叠造成的统计误差    |
| **返回粒度** | 单一算子的全局平均耗时           | 每轮总耗时 + 多轮 Min/Max/方差                | 不仅看均值             |
| **拆解能力** | 仅按底层 Op 名拆分           | 按 MoE 逻辑阶段拆分 (Dispatch/GMM等)         | 性能瓶颈直接映射到具体的算法逻辑段 |
| **稳态控制** | 无明确 warmup 隔离概念       | `dropped_warmups` 精准剔除前 N 轮          | 剥离 NPU 冷启动与频率爬坡开销 |

GroupedMatmul 在 small-op 路径里出现两次（gmm1 和 gmm2），如果用 bench_kineto
  按算子名匹配，两次会被混在一起分不开。profile_npu_event_sequences 按时序序列匹配，第一次 GroupedMatmul 归
  gmm1，第二次归 gmm2，能正确拆分。


- **Burn-in 稳态预热（性能压测手段）**：首轮 warmup 引入 `SMALL_FIRST_WARMUP_GMM_BURN_IN_REPEATS=100` 次大矩阵 Matmul，克服设备端 Cold Start 和频率爬坡，提升 Benchmark 置信度。
    
```
 def make_small_warmup_burn_in_buffers(local_num_tokens, args):
 ...
     if gmm_burn_in_repeats > 1 and warmup_burn_in_buffers is not None:
        # This burn-in only runs on the first profiler warmup iteration.
        # It does not change the routed output; it only inserts extra device
        # work before the normal small-op chain of that warmup iteration.
        burn_in_lhs, burn_in_rhs = warmup_burn_in_buffers
        for _ in range(gmm_burn_in_repeats):
            _ = torch.matmul(burn_in_lhs, burn_in_rhs)
```
收益低，不加，此处在bench_kineto处已有，如果改动前面的字符匹配的问题，那就需要考虑好适当补充bench kineto的功能
**7. 强参数校验与精度水位**

- `main()` 补充了 15+ 条 `parser.error` 规则（如 `--num-topk` 限制、`single-active-rank` 边界等）。
    
- `calc_diff` + `ACCURACY_ATOL=2.0` / `ACCURACY_RTOL=0.02`（考虑 MX 量化噪声适配了宽松水位）。 ##此处A3是否应该也考虑呢 但是A3只做了 INT8的量化模式
    
**8. 通信资源与统计**

- `cumulative_local_expert_recv_stats`：零初始化、非零校验。
    
- `ep_recv_count` 与 `small_counts` 严格相等校验。
    

### 二、A5 相对 A3 的能力对比汇总

| **维度分类** | **测试特性**                 | **A3 现状**       | **A5 现状**               | **状态定性**        |
| -------- | ------------------------ | --------------- | ----------------------- | --------------- |
| **功能场景** | 量化模式                     | INT8 单一         | FP8 e4m3/e5m2 + FP4 三选一 |                 |
|          | 激活函数                     | SwiGLU 固定       | SwiGLU + SiTU           |                 |
|          | Token 负载抖动 (Jitter)      | ❌ 未覆盖           | ✅ 随机偏移容错验证              | **新增覆盖**        |
|          | **Token 路由丢弃 (Drop=-1)** | ✅ 概率/定列 Drop 验证 | ❌ 未构造 `-1` 场景           | **回归 **         |
|          | **多级稀疏 Rank 组合**         | ✅ 列表级任意组合       | ❌ 退化为单 Rank 极值          | **能力降级**        |
| **测试工程** | Profile 内存/状态防污染         | ❌ 缺失            | ✅ Random + Fixed 双模式    | **架构级优势**       |
|          | 事件级时序拆解                  | ❌ 缺失            | ✅ 6段时序拦截                | **架构级优势**       |
|          | 稳态预热 (Burn-in)           | ❌ 缺失            | ✅ 100次大矩阵占位             | **Benchmark增强** |
|          | 异常态拦截 (NaN/Inf)          | ❌ 缺失            | ✅ 强制校验                  | **稳定性增强**       |

### 三、A5 仍存在的覆盖盲点


1. **路由丢弃 (Token Drop = -1) 场景缺失**：在真实 MoE 训练/推理中，Expert 超载导致 Token 被 Drop 是极频繁触发的核心机制。A3 支持 `--topk-drop-prob` 且对 `-1` 做了充分验证，而 A5 剥离了该逻辑。融合算子如果对 `-1` 越界读取会导致 Core Dump 或精度雪崩，这是必须补齐的安全红线。
      
2. **多 Rank 任意组合稀疏激活退化**：A3 的 `--active-ranks` 支持列表传入（如 "0,1,3" 激活），A5 目前仅保留了 `--single-active-rank` 单卡极值模式，失去了多节点间复杂空载/满载交替的通信验证能力。
    

 3. **异步通信 Hook (`return_recv_hook=True`) 未测**：这是 A3 和 A5 共同存在的历史遗留盲点，当前脚本均硬编码固定了 `return_recv_hook=False`（同步路径），尚未覆盖异步收发并行重叠场景。

4. **多 `num_topk` 组合未遍历**：默认 `num_topk=8`，未跑 `num_topk=1/2/4` 等极端值。 5. **Cumulative Stats 累加行为未充分验证**：当前仅校验了非零态，未在多次调用后严格验证数值累加的绝对正确性。 6. **FP4 NZ 失败硬件兜底**：仅测了 Happy Path，未模拟在老旧 CANN 版本不支持 FP4 转换时的环境回退行为（仅写了报错文本）。

 5. **分组与权重格式变体**：写死了 `group_list_type=1, group_type=0` 和默认 NZ，未对 ND 或其他 Group Type 组合做遍历。

