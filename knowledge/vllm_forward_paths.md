# vLLM Forward Paths — 前向核心路径速查

> 规划 hook 点时查。按模型族列出 vLLM 前向的关键节点与对应模块路径。
> 模块路径是 hook 注册用的点分隔字符串（见 `lib/hook_manager.py`）。

## 通用 Transformer 前向骨架

```
input_ids
  │
  ▼
embed_input_ids (VocabParallelEmbedding, CustomOp)   ← 用 monkey-patch 抓，hook 不可靠
  │
  ▼
for layer in model.layers:
    ├─ input_layernorm                               ← pre_hook 抓层输入
    ├─ self_attn (attention)                         ← post_hook 抓 attn_out
    ├─ residual add
    ├─ post_attention_layernorm
    ├─ mlp (dense 或 MoE)                             ← post_hook 抓 mlp_out
    └─ residual add
  │
  ▼
model.norm (final layernorm)
  │
  ▼
lm_head → logits
```

## 标准 hook 点（dense 模型）

`lib/hook_manager.transformer_hook_points` 自动生成：

| hook 名 | 模块路径 | kind | 抓什么 |
|---|---|---|---|
| `layer{i}_in` | `model.layers.{i}` | pre | 层输入（in-place 污染前的干净值） |
| `layer{i}_attn_out` | `model.layers.{i}.self_attn` | post | attention 输出 |
| `layer{i}_mlp_out` | `model.layers.{i}.mlp` | post | MLP/MoE 输出 |

> 层输出 = 下一层输入，由 `layer{i+1}_in` 覆盖，不必单独抓。

## MoE 专用 hook 点

`lib/hook_manager.moe_router_hook_points` 自动生成（`first_k_dense_replace` 之后才抓）：

| hook 名 | 模块路径 | kind | 抓什么 |
|---|---|---|---|
| `layer{i}_router_logits` | `model.layers.{i}.mlp.experts.router` | pre | router 输入（gate logits） |
| `layer{i}_topk` | `model.layers.{i}.mlp.experts.router` | post/patch | router 输出（topk_ids, topk_weights） |

> router 是 CustomOp/自定义对象，**优先用 patch 模式**抓返回值（口径可靠）。
> MoE 内部更深拆解（input_layernorm / attn / post_attn_norm / mlp / layer_out）
> 参考 `precision_compare/test_l1_ops.py` 的 A-F 六点法。

## MoE 层内拆解（定位到 MoE 后的细化）

发现 `mlp_out` 发散后，按此顺序逐点对比，定位 MoE 内具体算子：

```
层输入 (pre)
  ├─ A. input_layernorm        → model.layers.{i}.input_layernorm (post)
  ├─ B. attention              → model.layers.{i}.self_attn (post)
  ├─ C. post_attn_residual     → (A+B，抓 self_attn 后的 residual)
  ├─ D. post_attention_layernorm → model.layers.{i}.post_attention_layernorm (post)
  ├─ E. mlp (MoE)              → model.layers.{i}.mlp (post)   ← MoE 发散点通常在这
  └─ F. layer_out              → model.layers.{i+1} (pre)
```

AntAngelMed 实测：A-D 全部 cos≈1.0，**E (mlp) cos=0.63 是唯一发散点**。

## MoE 内部更进一步（router vs 专家 FFN）

定位到 E (mlp) 发散后，再拆 MoE 内部：

```
mlp.gate (router gate)         → router_logits      (抓 pre router)
  ├─ router.select_experts     → topk_ids, topk_weights  (抓 post/patch router)
  │    ├─ ops.moe_fused_gate   (use_fused_gate=True)   ← gfx936 坑点
  │    ├─ lightop.moe_fused_gate (VLLM_USE_LIGHTOP=1)  ← 权重不一致
  │    └─ grouped_topk (Python)  (正确参考)
  ├─ experts (FusedMoE)        → 专家 FFN 输出
  │    ├─ forward_cuda (fused kernel)
  │    └─ forward_native (纯 PyTorch)
  └─ shared_expert             → 共享专家输出
```

判定 router 是否为根因：对比 vLLM **实际运行时**的 topk_ids 与 HF。
（注意：用纯 Python 重算的 topk 不代表 vLLM 实际用的——这是 AntAngelMed 调查中的关键转折。）
