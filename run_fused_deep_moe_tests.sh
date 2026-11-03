# FusedDeepMoe 测试命令（Linux 服务器）

## A3 路径（不传 --quant）

```bash
python3 tests/python/deepep/test_fused_deep_moe.py --num-processes 4
```

## A5 路径

### MXFP8 e4m3（默认量化）

```bash
python3 tests/python/deepep/test_fused_deep_moe_a5.py --num-processes 4 --quant fp8_e4m3
```

### MXFP8 e5m2

```bash
python3 tests/python/deepep/test_fused_deep_moe_a5.py --num-processes 4 --quant fp8_e5m2
```

### MXFP4

```bash
python3 tests/python/deepep/test_fused_deep_moe_a5.py --num-processes 4 --quant fp4_e2m1
```

### A5 完整性能对比（profiler + random profile cases）

```bash
python3 tests/python/deepep/test_fused_deep_moe_a5.py \
  --num-processes 4 --quant fp8_e4m3 \
  --num-warmups 5 --num-tests 30 \
  --random-profile-cases
```

### A5 fixed-value cross-iteration 校验

```bash
python3 tests/python/deepep/test_fused_deep_moe_a5.py \
  --num-processes 4 --quant fp8_e4m3 \
  --num-warmups 5 --num-tests 30 \
  --fixed-profile-cases
```

### A5 SiTU 激活

```bash
python3 tests/python/deepep/test_fused_deep_moe_a5.py \
  --num-processes 4 --quant fp8_e4m3 \
  --activation situ
```

### A5 NPU Event 计时（仅 fused）

```bash
python3 tests/python/deepep/test_fused_deep_moe_a5.py \
  --num-processes 4 --quant fp8_e4m3 \
  --event-timing --num-warmups 5 --num-tests 30
```
