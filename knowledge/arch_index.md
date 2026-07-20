# Architecture Index — 架构 → vLLM 代码路径映射

> 拿到 ModelProfile 后第一查。按 model_type / architectures 定位 vLLM 模型文件。
> `lib/path_resolver.py` 已程序化实现此映射，本文件作为人工补充与已知坑标注。

## 映射表

| model_type | architectures | vLLM 文件 | MoE | 备注 |
|---|---|---|---|---|
| `bailing_moe_v2` | `BailingMoeV2ForCausalLM` | `models/bailing_moe.py` | ✅ | sigmoid+bias，走 fused_gate 路径（见 precision_known_issues.md） |
| `bailing_moe` | `BailingMoeForCausalLM` | `models/bailing_moe.py` | ✅ | 同上 |
| `deepseek_v2` | `DeepseekV2ForCausalLM` | `models/deepseek_v2.py` | ✅ | softmax 路径 |
| `deepseek_v3` | `DeepseekV3ForCausalLM` | `models/deepseek_v3.py` | ✅ | |
| `qwen2_moe` | `Qwen2MoeForCausalLM` | `models/qwen2_moe.py` | ✅ | |
| `qwen3_moe` | `Qwen3MoeForCausalLM` | `models/qwen3_moe.py` | ✅ | |
| `glm4_moe` | `GLM4MoeForCausalLM` | `models/glm4_moe.py` | ✅ | |
| `granitemoe` | `GraniteMoeForCausalLM` | `models/granitemoe.py` | ✅ | |
| `ernie45_moe` | `Ernie45MoeForCausalLM` | `models/ernie45_moe.py` | ✅ | |
| `exaone_moe` | `ExaoneMoeForCausalLM` | `models/exaone_moe.py` | ✅ | |
| `mixtral` | `MixtralForCausalLM` | `models/mixtral.py` | ✅ | softmax |
| `llama` | `LlamaForCausalLM` | `models/llama.py` | ❌ | dense; attention=LlamaAttention, norm=LlamaRMSNorm |
| `qwen2` | `Qwen2ForCausalLM` | `models/qwen2.py` | ❌ | dense; attention=Qwen2Attention, norm=Qwen2RMSNorm |
| `qwen3` | `Qwen3ForCausalLM` | `models/qwen3.py` | ❌/✅ | 部分变体 MoE |

## 通用底座路径（MoE + dense）

| 组件 | 路径 | 作用 |
|---|---|---|
| Attention | `vllm/attention/` | flash_attn / eager / sdpa 实现 |
| RMSNorm/LayerNorm | `model_executor/layers/layernorm.py` | 融合 norm kernel |
| 激活函数 | `model_executor/layers/activation.py` | SiLU / GELU 等 |

## 共用 MoE 底座路径（与 model_type 无关）

vLLM 所有 MoE 模型共用同一套 fused_moe 底座，路径稳定：

| 组件 | 路径 | 作用 |
|---|---|---|
| MoE 层 | `model_executor/layers/fused_moe/layer.py` | `FusedMoE`，`use_fused_gate` 定义在 603 行 |
| Router 目录 | `model_executor/layers/fused_moe/router/` | 各类 router |
| GroupedTopk Router | `router/grouped_topk_router.py` | sigmoid+bias 路径分发在此（343 行） |
| Python grouped_topk | `fused_moe/cpu_fused_moe.py` | 纯 PyTorch 参考实现（正确） |
| fused gate 算子 | `_custom_ops.py` | `ops.moe_fused_gate`（fused 实现；已知案例见 precision_known_issues.md） |
| aiter fused MoE | `fused_moe/rocm_aiter_fused_moe.py` | ROCm aiter 加速（本机未启用） |

## 判定规则：何时会走 fused_gate 路径

满足以下**全部**条件时，`ops.moe_fused_gate` 会被调用（这是结构性事实，不预设是否有问题）：

1. `profile.is_moe == True`
2. `profile.score_function == "sigmoid"`
3. `profile.e_score_correction_bias is not None`（有 expert bias 机制）
4. `profile.num_expert_group is not None`（grouped topk）
5. 未设 `VLLM_ENABLE_MOE_FUSED_GATE=0`

> 满足条件只意味着"走了 fused 路径"，**不等于有问题**。
> 是否有已知案例查 `precision_known_issues.md`，是否有问题用运行时探针验证。
> `lib/path_resolver.py:_infer_key_ops` 据此列出该算子入口（不带 bug 结论）。
