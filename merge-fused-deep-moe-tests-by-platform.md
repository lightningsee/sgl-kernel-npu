# 融合 MoE 测试按平台拆分方案

## Story名称

按 A3/A5 整理三类融合 MoE 测试。

## Story描述

把 FusedDeepMoe、DispatchFFNCombine 和 MegaMoe 的测试按设备平台归到两个文件：A3 一个文件，A5 一个文件。文件名确定平台，`--op` 选择算子，方便在同一平台内维护测试和复用相同的准备步骤。

这是[按算子合并方案](merge-fused-deep-moe-tests.md)的另一种做法，实施时选择其中一种。当前 DFC 只[注册了 ascend910_93](csrc/deepep/ops/op_host/dispatch_ffn_combine_def.cpp)，没有 ascend950 配置，A5 文件先保留入口并报不支持。底层适配完成后再启用测试，文件整理本身不计为完成 A5 DFC 支持。

## 整体方案

本方案采用“按平台组织文件、按算子选择入口、按原实现保留执行链”的整合方式：A3 和 A5 各自提供一个测试文件，文件名确定平台，`--op` 确定本次运行的算子。每次启动只执行一个 worker，不把三类算子串成统一计算流程，也不跨平台统一权重、baseline 或数值语义。

目标结构如下：

```text
tests/python/deepep/
├── test_fused_moe_a3.py  # --op fused_deep_moe / dispatch_ffn_combine / mega_moe
├── test_fused_moe_a5.py  # 三类算子入口；DFC 暂明确报不支持
└── utils.py              # 保留通用分布式与计时工具，不承载平台测试主体
```

实施按以下顺序推进：

1. **迁移平台实现**：将四个旧文件中的测试主体按平台归入两个新文件，处理重名函数并保持 worker 位于模块顶层，继续使用 `torch.multiprocessing.spawn`。
2. **建立统一入口**：分别实现 A3、A5 的 parser 和 `main()`，通过 `--op` 选择参数集合、校验逻辑和 worker；A5 DFC 仅保留能力检查，不伪装为已支持。
3. **抽取同平台复用步骤**：只复用已确认一致的权重准备和 baseline 计算，保留调用方的容量参数、计数校验、布局转换及精度参考差异；通信组和 Buffer 仍由各算子独立管理。
4. **迁移引用并删除旧入口**：更新混合测试 import、CI、generalization 脚本和文档，确认新路径回归通过后删除四个旧测试文件，不保留兼容 wrapper。
5. **分层回归**：完成语法与懒加载导入检查，再分别回归 A3/A5 的三类算子入口、混合测试、非法参数、精度、计数和性能统计。

两个平台文件不互相 import，平台专属 dtype 和依赖仅在实际使用时加载。整个过程保持原有参数默认值、输入生成顺序、随机种子、第二返回值、精度阈值和 profiler 统计口径不变；只修改当前工作树，不处理嵌套镜像目录。

## 功能点名称

1. 按平台整理文件和测试入口。
2. 保留各算子的参数和默认值。
3. 复用相同的权重准备和 baseline 步骤。
4. 分开管理各算子的依赖和通信资源。
5. 迁移 import、CI、脚本和文档。

## 功能点描述

1. 新增 `test_fused_moe_a3.py`、`test_fused_moe_a5.py`，分别接收三类测试的入口。每次启动只运行一个算子。
2. 用 `--op` 选择 parser 和 worker，不增加 `--platform`，也不根据量化模式切换平台。同名参数在不同测试中的原有含义保留。
3. A3 FusedDeepMoe 与 DFC 共用相同的权重源和 low-latency baseline 计算步骤；A5 MegaMoe 在本文件内调用 MC2 小算子函数。权重转换、精度参考和计数差异分别处理。
4. MXFP dtype 和 MegaMoe 依赖在实际使用时加载。各测试创建自己的通信组和 Buffer，保留原有隔离、复用检查和释放逻辑。
5. 更新混合测试的权重 import、CI 命令和 generalization 脚本。现有可运行路径回归通过后，删除四个旧测试文件，不保留兼容 wrapper。

## 功能实现思路

先按平台拆出测试主体，再处理重名函数。将原来的 `test`、`test_loop`、`baseline_test` 改成能区分算子的名称，worker 仍放在模块顶层，继续使用 `torch.multiprocessing.spawn`。

两个平台文件不互相 import。通用分布式和计时工具继续放在现有 `utils.py`，不再增加一个存放测试主体的公共文件。

代码复用先从已确认相同的步骤开始。A3 FusedDeepMoe 和 DFC 共用权重生成与 baseline 计算，融合权重转换仍分开；MegaMoe 的量化、布局和直接 CANN 参考按原实现保留。

迁移时对照新旧参数、用例列表、权重尺寸和调用顺序，再运行原有设备用例。检查范围包括默认值、非法参数、空 token rank、变长输入、精度、计数和性能统计。清理无用函数时保留函数对象的间接调用，只处理当前工作树，不修改嵌套镜像目录。

## 功能实现设计

### 代码设计

两个新文件都放在 `tests/python/deepep/`，原代码的去向如下：

| 原文件 | `test_fused_moe_a3.py` | `test_fused_moe_a5.py` |
|---|---|---|
| `test_fused_deep_moe.py` | 原有 A3 测试 | 无 |
| `test_fused_deep_moe_a5.py` | 无 | 原有 A5 测试 |
| `test_dispatch_ffn_combine.py` | 原有 INT8 测试 | 先放入口和能力检查 |
| `test_mega_moe.py` | W4A8/W8A8 路径 | 四种 MXFP 路径 |

入口使用 `--op fused_deep_moe|dispatch_ffn_combine|mega_moe`，默认 `fused_deep_moe`。预解析器使用 `add_help=False` 和 `parse_known_args()`，完整 parser 再注册 `--op` 和该算子的参数，支持 `--op=mega_moe`。设备检查只报告文件与设备不匹配，不切换平台。

```text
main()
  → 读取 --op，调用 build_parser(op)
  → 校验参数和设备支持情况，设置环境变量
  → spawn 对应的顶层 worker
      → 创建通信组和 Buffer
      → 构造输入、权重，执行 baseline 和 fused
      → 校验结果，按原方式计时
      → 释放资源
```

每个文件按 FusedDeepMoe、DFC、MegaMoe 分节，分别提供 `run_fused_deep_moe_test`、`run_dispatch_ffn_combine_test`、`run_mega_moe_test`。不增加 `--op all`，也不让三个算子在同一轮进程中共用通信资源。

参数默认值按下面的配置保留。A5 MegaMoe 由文件名确定平台，默认量化改为 `mx_fp8_e4m3`：

| 平台和算子 | 进程数 | tokens | hidden | moe-intermediate-size | experts | top-k | 默认量化 |
|---|---:|---:|---:|---:|---:|---:|---|
| A3 FusedDeepMoe | 16 | 256 | 7168 | 4096 | 256 | 8 | 原 INT 配置 |
| A3 DFC | 16 | 256 | 7168 | 4096 | 256 | 8 | INT8 |
| A3 MegaMoe | 16 | 64 | 7168 | 3072 | 64 | 8 | `w4a8` |
| A5 FusedDeepMoe | 8 | 32 | 7168 | 3072 | 64 | 6 | `fp8_e4m3` |
| A5 DFC | 16 | 256 | 7168 | 4096 | 256 | 8 | 当前报不支持 |
| A5 MegaMoe 精度模式 | 16 | 64 | 7168 | 3072 | 64 | 8 | `mx_fp8_e4m3` |

A5 MegaMoe 性能模式仍默认使用 256 experts、top-k 6、stacked ND 权重和 FP32 routing，显式参数优先。其他参数按算子分别注册：

- A5 FusedDeepMoe 使用 `--quant`；MegaMoe 使用 `--quant-mode`。A3 MegaMoe 接受 `w4a8|w8a8`，A5 接受 `mx_fp8_e4m3`、`mx_fp8_e5m2`、`mx_fp4_e2m1`、`mx_fp8_e4m3_w4`。
- A5 FusedDeepMoe 的 SiTU 参数是单个浮点数，`--no-linear-beta` 表示 up 直通。MegaMoe 保留参数列表，`--linear-beta 0` 表示直通。
- DFC 使用 SwiGLU、INT8 和 INT64 scale，不接收 SiTU 或 MXFP 参数。它的 `--performance-only` 跳过 baseline 和精度检查；MegaMoe 的该选项只测 fused。
- A5 FusedDeepMoe 保留 `--num-warmups`、`--num-tests`、`--event-timing`。A3 调试行为沿用原实现，已删除的 trace、量化日志和 profile debug 参数不恢复。

`--moe-intermediate-size` 的含义也按原测试保留，不在公共入口统一乘二或除二：

| 测试 | 参数为 D 时的 GMM1 输出宽度 | 激活后宽度 |
|---|---:|---:|
| A3 FusedDeepMoe、原 DFC | D | D / 2 |
| A3 MegaMoe、A5 FusedDeepMoe、A5 MegaMoe | 2D | D |

A3 文件内，`init_base_weights`、`init_baseline_weights` 各保留一份。供混合测试使用的 `init_base_weights`、`init_fused_weights_int8` 名称、签名和默认值不变。FusedDeepMoe 继续做 GMM1 和 scale permutation，DFC 继续使用普通 NZ 权重与 INT64 编码 scale。

共用 low-latency baseline 时分别返回 `packed_recv_count` 和 `handle[1]` 对应的 `ep_recv_count`，由各算子取原来使用的统计。dispatch、INT32 GMM1、dequant-SwiGLU-quant、GMM2、combine 的顺序，以及异步标志、路由传参和计时事件不变。A3 MegaMoe 的 W4 打包、补偿 bias、W8 scale 和 MC2 baseline 另行保留。

A5 文件内保留 `run_small_op_baseline`，供 FusedDeepMoe 和 MegaMoe 小算子性能比较调用。前者权重方向是 `[E,K,N]`，后者是 `[E,N,K]`，scale 也需要转换。保留原 adapter，在计时外转换；FP4 packed 数据和 NZ 格式按原路径构造。MegaMoe 性能比较仍只开放原来允许的 MXFP8、stacked ND 组合。

MegaMoe 的精度参考继续直接调用 CANN，FusedDeepMoe 仍对比小算子结果。SiTU 普通模式校验后返回，Event 模式只测 fused，不套用 SwiGLU 的 profiler 拆分。A5 DFC 底层适配完成前以非零状态报不支持，不改调其他算子或把跳过当成功。

dtype 使用时再解析，help 用字符串 choices。`cann_ops_transformer.ops` 和 JIT 依赖只在 MegaMoe 路径加载；模块导入不解析 CLI、不创建通信资源。检查缺少 FP4、E8M0 或 `cann_ops_transformer` 时，未使用这些能力的入口仍能加载，A3 的公共权重接口仍能导入。

各算子保留自己的通信组和 Buffer，baseline 与被测路径隔离。A5 MegaMoe 保留两个 SymmBuffer 的创建、缓存复用检查和销毁；HCCL 控制组与 MC2 数据组的配置、collective 顺序不变。环境变量在选定算子后设置，spawn worker 不依赖父进程临时生成的配置。

依赖修改如下，表中的测试文件均位于 `tests/python/deepep/`：

| 文件 | 修改内容 |
|---|---|
| `test_normal_and_fused_deep_moe.py` | 从 A3 新文件导入原有两个权重接口 |
| `.github/workflows/pr-test-deepep-npu.yml` | 按 job 平台替换路径并增加 `--op`，保留其余参数和环境变量 |
| `scripts/generalization_test_fused_deep_moe.sh` | 替换 SCRIPT，使用 `--op fused_deep_moe` |
| `python/deep_ep/README.md` | 更新中英文命令和支持范围 |
| `docs/design_a5_megamoe_param_adaptation.md` | 更新路径、示例和 SiTU 行为说明 |

先对照新旧 parser 的默认值、显式覆盖和非法参数，再检查 MegaMoe 用例的顺序、重复项、激活、routing dtype 和 token 数。设备回归沿用原 CI，并覆盖 DFC 容量与 performance-only、A5 单活跃和空 token rank、jitter、SiTU 饱和参数、Event 计时、SwiGLU random/fixed profile cases，以及 MegaMoe 的 E8M0 scale、SymmBuffer 复用和 MC2 分块输入。精度阈值、计数布局、预热和统计次数、rank 汇总方式保持原样。

回归通过并完成引用迁移后，删除四个旧文件，不保留兼容 wrapper。旧命令需要替换脚本路径并增加 `--op`，例如：

```bash
python3 tests/python/deepep/test_fused_moe_a3.py --op mega_moe --num-processes 8 --quant-mode w4a8
python3 tests/python/deepep/test_fused_moe_a5.py --op fused_deep_moe --num-processes 4 --quant fp4_e2m1
```
