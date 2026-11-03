⏺ ## 文档主要缺点

  ### 1. "清理无用函数和 import，简化分支"过于笼统（最大问题）

  这正是我们之前三步讨论要解决的核心，但文档功能点 3 一句话带过，没有：

  • 死代码清单：具体删哪些函数/import（A3 的 make_uniform_topk_idx 等已点名，A5 的没列）
  • 提炼目标：哪些重复逻辑要抽辅助函数（如 gmm1/gmm2 量化重复、Namespace hack、权重准备链）
  • 分支精简范围：哪些 if/else 要合并或简化（A5 的 validate_random_profile_results 两轮校验、maybe_cast_weight_to_nz 的
  fp4 判断等）
  • 执行顺序：三步谁先谁后，是否在合并前做（强烈建议合并前做完，否则合并 PR 的 diff 会被噪音淹没）

  ### 2. 合并入口的技术细节不够可操作

  • 预解析器用什么 API？parse_known_args 还是手动 sys.argv 扫描？
  • --quant 缺省时判 A3，有值判 A5——这个二分逻辑写在哪？预解析器里还是 build_parser 里？
  • 两个平台 parser 参数互斥的报错机制：A3 传了 --activation situ 怎么报？A5 传了 --active-ranks 怎么报？是 parser.error
  还是忽略？
  • 没有给 build_parser(platform) 的签名和关键分支示例

  ### 3. utils.py 迁移边界不清

  run_small_op_baseline 依赖一大堆 A5
  专有物：get_mx_quant_config、maybe_cast_weight_to_nz、situ_reference、MX_QUANT_CONFIGS、quant_cfg["dispatch_quant_mode
  "] 等。迁到 utils 后：

  • 这些依赖一起迁还是作为参数传入？
  • MX_QUANT_CONFIGS 含 torch_npu.float4_e2m1fn_x2 等 dtype，与"A5 dtype 延迟解析""不在模块加载时解析
  MXFP"直接冲突——这块怎么改？改成函数返回？常量只存字符串 key？
  • 文档说迁 get_mx_quant_config 和"量化配置解析"，但没说 MX_QUANT_CONFIGS 表本身的形态变化

  ### 4. "A5 dtype 延迟解析"缺少落地路径

  • "在实际使用时解析"——是改成 get_mx_quant_config(quant_name) 返回 dtype？还是懒加载？
  • "模拟 torch/torch_npu 缺少 MXFP 属性的导入检查"——这是单元测试还是运行时 gate？放哪？

  ### 5. 风险与验收标准缺失

  • 合并是破坏性变更（删 test_fused_deep_moe_a5.py），回滚策略没写
  • "确认现有测试覆盖没有减少"——验收方式是什么？跑 CI？对比 profiler 输出？diff 命令？
  • 单 PR 还是拆 commit？这个体量建议拆 3-4 个 commit（精简 A3 → 精简 A5 → 迁 utils → 合并入口），文档没说

  ### 6. "简化分支"完全没落地

  文档提到"简化分支"但全文找不到一个分支精简的例子。要么删掉这个提法（只做清理+提炼），要么把具体例子写进去。

  --------

  ## 建议：把三步思路写进去，但分层放

  不要塞进"功能点描述"——功能点是给 PM/review 看的，保持高层。

  应该在"功能实现设计"里新增一节，比如：

    ### 代码精简策略（合并前置）

    合并前对两个文件分别完成三步精简，降低合并 PR 的 diff 噪音：

    1. 无效代码清理
       - A3: 删除 sys/time/partial/hash_tensor 导入, make_uniform_topk_idx,
         from_inclusive_prefix_sum, str_to_bool
       - A5: 删除 format_triplet、run_buffer_fused_with_burn_in、
         log_quant_tensor 系列调试辅助函数、trace_dir/kernel_trace_dir/
         dump_profile_events 等已废弃参数及其处理分支

    2. 前置辅助函数整合
       - A5: 抽取 _quant_and_cast_weight 消除 gmm1/gmm2 量化重复;
         make_random_profile_cases 加 total_case_count 参数,
         消除 make_fixed_profile_cases 的 Namespace hack
       - A3: 抽取 prepare_all_weights 整合 init_base_weights →
         init_baseline_weights → init_fused_weights_int8 链式调用

    3. 控制流精简(本次范围限定)
       - A5: 合并 validate_random_profile_results 中 fixed_values 的
         两轮 cross-iteration 校验
       - A3: topk_idx 计数循环向量化
       - 暂不处理 run_rank 末尾统计打印嵌套(留给合并后单独 PR)

  这样：

  • 功能点保持简洁（3-4 条）
  • 实现设计有可执行的具体动作
  • reviewer 能看到每步的产出和边界
  • 合并 PR 时能清晰对照"精简已完成，本 PR 只做合并"

  一个判断：如果你这次合并 PR 不打算同时做控制流精简，那第 3
  步直接不写，避免承诺过多。文档里"简化分支"这种模糊词宁可删掉，也不要留一个没计划的口子。

