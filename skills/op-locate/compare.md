# 子流程：compare — 算子级对比与误差判定

> SKILL.md 阶段 4-6 的细化规范。

## 通用张量对比

```python
from lib import compare_tensors

r = compare_tensors("layer1.mlp_out", hf_tensor, vllm_tensor)
print(r.verdict())   # ✅ 一致 / ❌ 发散
# 字段：max_abs_diff, mean_abs_diff, cos, relative_diff, is_close
```

判定阈值（默认）：
- `is_close`：`torch.allclose(rtol=1e-2, atol=1e-2)`
- "一致"：`cos >= 0.999`
- "发散"：`cos < 0.999`，看 `max_abs_diff` 量级

## 逐层对比（粗粒度）

按计算顺序对比，**定位误差首次出现位置**：

```python
from lib import TensorComparator

cmp = TensorComparator()
stages = ["embedding", "layer0_in", "layer1_in", ..., "layer31_in", "final_ln", "logits"]
first_diff = None
for s in stages:
    r = cmp.compare(s, hf_inter[s], vllm_inter[s])
    print(r.verdict())
    if r.cos < 0.999 and first_diff is None:
        first_diff = s
        # 不要 break——继续看衰减曲线，但根因在 first_diff
```

示例曲线（仅示意形态，不代表任何具体模型）：
```
embedding     cos=0.999999  ✅
layer0_in     cos=0.999999  ✅
layer1_in     cos=0.940     ← 某层开始偏
layer8_in     cos=0.011     💀 发散
```
→ 找到 cos 首次掉的那一层，误差通常从那里开始逐层放大。

## topk 对比（MoE router 专用）

**必须按 id 对齐**，不能各自 sorted 逐位比（陷阱 5）：

```python
from lib import compare_topk

r = compare_topk(hf_ids, hf_weights, vllm_ids, vllm_weights)
print(r.verdict())
# ids ✅/❌ + weights ✅/❌
# 字段：ids_exact_match_rate, weights_max_abs_diff, weights_max_rel_diff
```

判定：
- `ids_match=True` + `weights_zero_error=True` → router 完全一致，根因在别处
- `ids_match=False` → **选错专家**，根因在 router 算子（如 fused_gate）
- `ids_match=True` + `weights_zero_error=False` → 选专家对但权重不同（如 lightop），精度受影响

## 层内 A-F 拆解对比

定位到某层发散后，逐点对比层内算子。

**MoE 模型**：A-F 六点法（见 `knowledge/vllm_forward_paths.md`）。
典型情形：A-D 全 cos≈1.0，**E (mlp) 唯一发散** → 矛头指向 MoE。

**Dense 模型**：
```python
ops_in_layer = ["A_input_norm", "B_attn", "C_residual", "D_post_attn_norm", "E_mlp", "F_layer_out"]
for op in ops_in_layer:
    r = compare_tensors(op, hf[op], vllm[op])
    print(r.verdict())
```
嫌疑路径：norm 发散 → RMSNorm/LayerNorm 精度；attn 发散 → flash_attn vs eager；mlp 发散 → 融合 GEMM 精度。

> 以实际对比为准，不要预判。

## 单算子对比（阶段 6）

锁定嫌疑算子后，隔离变量做对照实验：

### 通用方法

- **fused kernel vs native**：同一输入对比 vLLM 融合 kernel 与纯 PyTorch 回退路径
- **环境变量回退**：设对应开关端到端验证（如 `VLLM_ATTENTION_BACKEND=TORCH_SDPA`）
- 均用 `lib.tensor_compare` 对比输出

### MoE 专属
```python
# 同一组 router_logits + bias，分别喂两个算子
from vllm.model_executor.layers.fused_moe.cpu_fused_moe import grouped_topk as py_topk
from vllm import _custom_ops as ops
# ... 喂相同输入，compare_topk 对比输出
```

### fused vs native FFN
```python
experts.forward_cuda(...)   # fused
experts.forward_native(...) # 纯 PyTorch
# compare_tensors 对比，若一致 → fused kernel 无精度问题
```

参考脚本：`examples/compare_layers.py`（`--op router` + `--env` 做有/无 fused_gate 对照）。

## 误差判定速查

| cos | 含义 | 动作 |
|---|---|---|
| ≥ 0.999 | 一致 | 排除该算子，往下找 |
| 0.99-0.999 | 轻微偏差 | 记录，可能是数值精度，通常非根因 |
| 0.9-0.99 | 明显偏差 | 嫌疑算子，需细化 |
| < 0.9 | 严重发散 | 强嫌疑，重点排查 |
| < 0 或 ≈ 0 | 完全发散 | 选错专家/数值爆炸，根因级 |

## 环境变量绕过验证

定位到某嫌疑算子后，若有对应的"切换到参考路径"开关，可端到端验证：
```bash
# 例：怀疑 fused_gate 路径时，切到 Python 参考路径
VLLM_ENABLE_MOE_FUSED_GATE=0 python3 <quickstart 或 probe 脚本>
# 输出恢复正常 + 与参考实现一致 → 支持该算子为根因
```
注意：绕过开关是**对照验证手段**，不是通用补丁。仅当探针确认该算子为根因时才用。
