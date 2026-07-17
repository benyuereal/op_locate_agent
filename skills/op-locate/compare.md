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

AntAngelMed 实测曲线（参考基准）：
```
embedding     cos=0.999999  ✅
layer0_in     cos=0.999999  ✅  (L0 dense)
layer1_in     cos=0.940     ← L1 (第一个 MoE) 开始偏
layer8_in     cos=0.011     💀 发散
```
→ 误差从第一个 MoE 层开始，逐层放大。

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

定位到某层 mlp_out 发散后，逐点对比层内算子：

```python
# A=input_layernorm B=attention C=post_attn_res D=post_attn_norm E=mlp F=layer_out
ops_in_layer = ["A_input_ln", "B_attn", "C_post_attn_res", "D_post_attn_norm", "E_mlp", "F_layer_out"]
for op in ops_in_layer:
    r = compare_tensors(op, hf_op_out[op], vllm_op_out[op])
    print(r.verdict())
```

AntAngelMed 实测：A-D 全 cos≈1.0，**E (mlp) cos=0.63 唯一发散** → 根因在 MoE。

## 单算子对比（阶段 6）

锁定嫌疑算子后，隔离变量做单算子对比：

### fused_gate vs python grouped_topk
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

参考脚本：`precision_compare/test_lightop_vs_torch_min.py`、`test_moe_native_l1.py`。

## 误差判定速查

| cos | 含义 | 动作 |
|---|---|---|
| ≥ 0.999 | 一致 | 排除该算子，往下找 |
| 0.99-0.999 | 轻微偏差 | 记录，可能是数值精度，通常非根因 |
| 0.9-0.99 | 明显偏差 | 嫌疑算子，需细化 |
| < 0.9 | 严重发散 | 强嫌疑，重点排查 |
| < 0 或 ≈ 0 | 完全发散 | 选错专家/数值爆炸，根因级 |

## 环境变量绕过验证

定位到 fused_gate 嫌疑后，端到端验证修复：
```bash
VLLM_ENABLE_MOE_FUSED_GATE=0 python3 probe_envvar.py
# 输出正常中文 + 无 NULL → 确认根因 + 验证修复
```
