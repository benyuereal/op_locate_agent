# Architecture Index — 架构 → vLLM 代码路径映射

> 拿到 ModelProfile 后第一查。按 model_type / architectures 定位 vLLM 模型文件。
> `lib/path_resolver.py` 已程序化实现此映射，本文件作为人工补充与已知坑标注。

## 映射表

| model_type | architectures | vLLM 文件 | MoE | 备注 |
|---|---|---|---|---|
| `bailing_moe_v2` | `BailingMoeV2ForCausalLM` | `models/bailing_moe.py` | ✅ | sigmoid+bias → 见 fused_gate 坑 |
| `bailing_moe` | `BailingMoeForCausalLM` | `models/bailing_moe.py` | ✅ | 同上 |
| `deepseek_v2` | `DeepseekV2ForCausalLM` | `models/deepseek_v2.py` | ✅ | softmax 路径，无 fused_gate 坑 |
| `deepseek_v3` | `DeepseekV3ForCausalLM` | `models/deepseek_v3.py` | ✅ | |
| `qwen2_moe` | `Qwen2MoeForCausalLM` | `models/qwen2_moe.py` | ✅ | |
| `qwen3_moe` | `Qwen3MoeForCausalLM` | `models/qwen3_moe.py` | ✅ | |
| `glm4_moe` | `GLM4MoeForCausalLM` | `models/glm4_moe.py` | ✅ | |
| `granitemoe` | `GraniteMoeForCausalLM` | `models/granitemoe.py` | ✅ | |
| `ernie45_moe` | `Ernie45MoeForCausalLM` | `models/ernie45_moe.py` | ✅ | |
| `exaone_moe` | `ExaoneMoeForCausalLM` | `models/exaone_moe.py` | ✅ | |
| `mixtral` | `MixtralForCausalLM` | `models/mixtral.py` | ✅ | softmax |
| `llama` | `LlamaForCausalLM` | `models/llama.py` | ❌ | dense |
| `qwen2` | `Qwen2ForCausalLM` | `models/qwen2.py` | ❌ | dense |
| `qwen3` | `Qwen3ForCausalLM` | `models/qwen3.py` | ❌/✅ | 部分变体 MoE |

## 共用 MoE 底座路径（与 model_type 无关）

vLLM 所有 MoE 模型共用同一套 fused_moe 底座，路径稳定：

| 组件 | 路径 | 作用 |
|---|---|---|
| MoE 层 | `model_executor/layers/fused_moe/layer.py` | `FusedMoE`，`use_fused_gate` 定义在 603 行 |
| Router 目录 | `model_executor/layers/fused_moe/router/` | 各类 router |
| GroupedTopk Router | `router/grouped_topk_router.py` | sigmoid+bias 路径分发在此（343 行） |
| Python grouped_topk | `fused_moe/cpu_fused_moe.py` | 纯 PyTorch 参考实现（正确） |
| fused gate 算子 | `_custom_ops.py:3049` | `ops.moe_fused_gate`（gfx936 有坑） |
| aiter fused MoE | `fused_moe/rocm_aiter_fused_moe.py` | ROCm aiter 加速（本机未启用） |

## 判定规则：何时查 fused_gate 坑

满足以下**全部**条件时，`ops.moe_fused_gate` 会被调用，需查 `moe_known_issues.md`：

1. `profile.is_moe == True`
2. `profile.score_function == "sigmoid"`
3. `profile.e_score_correction_bias is not None`（有 expert bias 机制）
4. `profile.num_expert_group is not None`（grouped topk）
5. 平台 `is_cuda()=False`（gfx936/DCU）
6. 未设 `VLLM_ENABLE_MOE_FUSED_GATE=0`

`lib/path_resolver.py:_infer_key_ops` 已自动据此推断 `ops.moe_fused_gate` 入口。
