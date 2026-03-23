# CANN 后端算子优化总览

本文档汇总 OminiX-Ascend 项目中针对华为昇腾（Ascend 910B）CANN 后端的全部算子优化工作。

---

## 1. 编译与环境准备

```bash
# 设置 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 编译（启用 CANN 后端）
cmake -B build -DGGML_CANN=ON -DCMAKE_BUILD_TYPE=Release -DSOC_TYPE=ascend910b
cmake --build build --config Release -j$(nproc)
```

---

## 2. 算子优化清单

### 2.1 GGML_OP_SET 实现

| 项目 | 说明 |
|------|------|
| **问题** | Qwen3.5 的 delta-net 架构使用 `ggml_set_inplace`，CANN 后端未实现该算子，回退到 CPU 后因解引用设备指针导致 segfault |
| **修复** | 实现 `ggml_cann_set()`，将 src1 通过自定义步长拷贝到 dst 的子区域（参考 `ggml_cann_acc`） |
| **支持类型** | F32, I32 |
| **文件** | `ggml/src/ggml-cann/aclnn_ops.cpp`, `ggml/src/ggml-cann/ggml-cann.cpp` |

### 2.2 REPEAT 算子重写（绕过 CANN SDK Bug）

| 项目 | 说明 |
|------|------|
| **问题** | CANN SDK 8.5.0 的 `aclnnRepeat` 在特定 shape（如 ne[2]=36）下触发 MTE 越界写入崩溃 |
| **修复** | 单维度 repeat 改用 `aclnnInplaceCopy` + tensor view 切片的 O(log2 R) 倍增策略；identity repeat 直接 memcpy；多维度回退 `aclnnRepeat` |
| **文件** | `ggml/src/ggml-cann/aclnn_ops.cpp` |

### 2.3 REPEAT + Binary Op 融合

| 项目 | 说明 |
|------|------|
| **问题** | REPEAT 后接 MUL/ADD/SUB/DIV 时，先展开再计算浪费显存和计算 |
| **修复** | 跳过 REPEAT，直接让 ACLNN 的 binary op 利用原生 broadcasting |
| **启用方式** | 自动生效（编译进 `execute_graph_nodes` 循环） |
| **融合 op** | REPEAT+MUL, REPEAT+ADD, REPEAT+SUB, REPEAT+DIV |
| **文件** | `ggml/src/ggml-cann/aclnn_ops.cpp` (`ggml_cann_op_repeat_binary_fused`), `ggml/src/ggml-cann/ggml-cann.cpp` |

### 2.4 RMS_NORM 手动分解

| 项目 | 说明 |
|------|------|
| **问题** | `aclnnRmsNorm` 在大 tensor（>32768 元素）上极慢 |
| **修复** | 手动拆为 5 个 ACLNN kernel：Mul → Mean → InplaceAdds → InplaceRsqrt → Mul |
| **阈值** | 元素数 > 32768 时走手动分解，小 tensor（如 QK_NORM）仍用原生 `aclnnRmsNorm` |
| **文件** | `ggml/src/ggml-cann/aclnn_ops.cpp` (`ggml_cann_rms_norm`) |

### 2.5 SSM_CONV decode 快速路径

| 项目 | 说明 |
|------|------|
| **问题** | decode 阶段 n_t=1 时 `aclnnConvolution` 开销大 |
| **修复** | n_t=1 时替换为逐元素 MUL + ReduceSum |
| **文件** | `ggml/src/ggml-cann/aclnn_ops.cpp` |

### 2.6 BF16 类型支持

| 项目 | 说明 |
|------|------|
| **问题** | BF16 模型推理时 `supports_op` 不包含 BF16，全部回退 CPU（tg 仅 1.6 t/s） |
| **修复** | 为 MUL_MAT, MUL_MAT_ID, GET_ROWS, SET_ROWS, CPY, CONT 添加 BF16 支持 |
| **额外修复** | FRACTAL_NZ 格式限制为仅 F16 权重，F32/BF16 使用 ND 格式（避免 910B2 上找不到算子内核） |
| **性能提升** | pp512: 7.8 → **1482 t/s** (190x)，tg128: 1.6 → **41 t/s** (25x) |
| **文件** | `ggml/src/ggml-cann/aclnn_ops.cpp`, `ggml/src/ggml-cann/ggml-cann.cpp` |

### 2.7 V cache SET_ROWS 优化

| 项目 | 说明 |
|------|------|
| **问题** | 转置 V cache 布局导致 ScatterUpdate 退化为逐元素操作（524,288 个索引），单核执行，占总耗时 42% |
| **修复** | 检测 V cache 转置布局（ne[0]==1 且 ne[1]>1），通过 view_src 链追溯原始形状，改用列级 `InplaceIndexCopy` 实现全核并行 |
| **性能提升** | pp512: 1135 → **5901 t/s** (5.2x) |
| **文件** | `ggml/src/ggml-cann/aclnn_ops.cpp` (`ggml_cann_set_rows_v_cache_optimized`) |

### 2.8 ACL Graph Capture 预热修复

| 项目 | 说明 |
|------|------|
| **问题** | ACL Graph capture 期间 `aclrtMalloc` 导致崩溃（capture 模式下不允许内存分配） |
| **修复** | capture 前增加一次预热执行（pre-warm），初始化所有内存池和缓存；预热后重置 `rope_cache.cached = false` 确保重新录制 |
| **文件** | `ggml/src/ggml-cann/ggml-cann.cpp` (`evaluate_and_capture_cann_graph`) |

### 2.9 ADD + RMS_NORM 算子融合

| 项目 | 说明 |
|------|------|
| **启用方式** | 设置环境变量 `GGML_CANN_OPERATOR_FUSION=on` |
| **文件** | `ggml/src/ggml-cann/ggml-cann.cpp` |

---

## 3. 运行时环境变量

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `GGML_CANN_OPERATOR_FUSION` | off | 启用算子融合（ADD+RMS_NORM 融合） |
| `GGML_CANN_WEIGHT_NZ` | **on** | 权重 NZ 格式转换（默认开启，提升 MatMul 性能） |
| `GGML_CANN_ACL_GRAPH` | off | 启用 ACL Graph 模式（需编译时 `USE_ACL_GRAPH` 宏） |
| `GGML_CANN_PREFILL_USE_GRAPH` | off | prefill 阶段也使用 ACL Graph |
| `GGML_CANN_MEM_POOL` | 默认池 | 内存池类型选择 |
| `GGML_CANN_NO_PINNED` | 未设置 | 禁用 pinned 内存 |
| `GGML_CANN_DISABLE_BUF_POOL_CLEAN` | off | 禁用缓冲池清理 |

---

## 4. 性能测试命令

### 4.1 llama-bench 基准测试

```bash
# 基础性能测试
./build/bin/llama-bench \
  -m /path/to/model.gguf \
  -p 512 -n 128 \
  -ngl 99

# 启用 Flash Attention
./build/bin/llama-bench \
  -m /path/to/model.gguf \
  -p 512 -n 128 \
  -ngl 99 -fa 1

# 启用算子融合 + Flash Attention（最佳性能）
GGML_CANN_OPERATOR_FUSION=on ./build/bin/llama-bench \
  -m /path/to/model.gguf \
  -p 512 -n 128 \
  -ngl 99 -fa 1
```

### 4.2 llama-cli 交互测试

```bash
# 交互式推理
./build/bin/llama-cli \
  -m /path/to/model.gguf \
  -ngl 99 -fa 1 \
  -p "Hello, how are you?"

# 带算子融合
GGML_CANN_OPERATOR_FUSION=on ./build/bin/llama-cli \
  -m /path/to/model.gguf \
  -ngl 99 -fa 1 \
  -p "Hello, how are you?"
```

### 4.3 调试定位

```bash
# 启用调度器调试日志
GGML_SCHED_DEBUG=2 ./build/bin/llama-bench -m model.gguf -p 32 -n 8 -ngl 99

# 逐算子同步（定位异步错误）
# 在 ggml_cann_compute_forward 中每个 op 后插入 aclrtSynchronizeStream
```

---

## 5. 性能汇总

### Qwen3-8B (BF16 优化前后)

| 阶段 | pp512 (t/s) | tg128 (t/s) |
|------|-------------|-------------|
| 优化前（CPU 回退） | 7.8 | 1.6 |
| BF16 支持后 | **1482** | **41** |
| 提升 | **190x** | **25x** |

### Qwen3-8B Q8_0 (逐步优化)

| 优化阶段 | pp512 (t/s) | tg128 (t/s) |
|----------|-------------|-------------|
| Baseline | 1135 | 27.15 |
| + V cache SET_ROWS 优化 | 5901 | 27.15 |
| + Flash Attention | 5901 | 42.82 |
| + 算子融合 (FA+Fusion) | **6220** | **42.82** |
| 总提升 | **5.48x** | **1.58x** |

### 竞品对比 (tg, tokens/s)

| 模型 | OminiX-Ascend | Torch-npu / vLLM | 加速比 |
|------|--------------|-------------------|--------|
| DeepSeek OCR 2 | **33** | 5 | 6.6x |
| Qwen3-8B | **42** | 28 | 1.5x |
| Qwen3.5-9B | **28** | 8.9 | 3.1x |
| Qwen3-32B | **10** | OOM | -- |
| GLM-4-32B | **10** | OOM | -- |
| Ministral-3-14B | **21** | 4 | 5.25x |
| Llama2-7B | **50** | 31 | 1.6x |

---

## 6. 相关文件索引

| 文件 | 说明 |
|------|------|
| `ggml/src/ggml-cann/aclnn_ops.cpp` | 算子实现（SET, REPEAT, RMS_NORM, SET_ROWS, repeat+binary fusion） |
| `ggml/src/ggml-cann/aclnn_ops.h` | 算子声明 |
| `ggml/src/ggml-cann/ggml-cann.cpp` | 后端调度、supports_op、融合检测、ACL Graph |
| `docs/backend/CANN.md` | 上游 CANN 后端文档 |
| `reports/cann_debugging_methodology.md` | 调试方法论 |
| `reports/qwen35_cann_segfault_fix.md` | Qwen3.5 崩溃修复报告 |
| `change.md` | 变更日志与性能数据 |
| `docs/benchmarks.md` | 竞品对比基准测试 |
