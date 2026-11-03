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


  • 功能点保持简洁（3-4 条）
  • 实现设计有可执行的具体动作
  • reviewer 能看到每步的产出和边界
  • 合并 PR 时能清晰对照"精简已完成，本 PR 只做合并"

  一个判断：如果你这次合并 PR 不打算同时做控制流精简，那第 3
  步直接不写，避免承诺过多。文档里"简化分支"这种模糊词宁可删掉，也不要留一个没计划的口子。


优缺点

  按算子合并方案(merge-fused-deep-moe-tests.md):一个文件装 A3+A5 的 FusedDeepMoe,用 --quant 切平台。DFC 和
  MegaMoe 各自独立。

  按平台拆分方案(merge-fused-deep-moe-tests-by-platform.md):两个文件
  test_fused_moe_a3.py/test_fused_moe_a5.py,每个文件内用 --op 切 FusedDeepMoe/DFC/MegaMoe 三个算子。

  ## 按平台方案 
 

  ### 按算子方案的相对取舍

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