# Precision Known Issues — 精度已知坑库

> MoE + dense 模型在 vLLM 上的**已记录案例**库。按"算子"组织，每条带验证来源。
>
> **重要心态**：这里的每一条都是**某个具体模型在某个具体环境下的已确认案例**，
> 不是普适结论。新模型/新环境可能复现，也可能不复现——必须用运行时探针独立验证，
> 不要因为"长得像"就直接套用。根因以运行时探针为准，不以代码静态分析为准。

---

## 通用算子（MoE + dense）

### attention (flash_attn / eager / sdpa)

- **文件**: `vllm/attention/`；`vllm/_custom_ops.py` → flash_attn kernel
- **触发条件**: `VLLM_ATTENTION_BACKEND` 环境变量
- **通用排查要点**：vLLM 默认走 flash_attn 融合 kernel，与 HF eager/sdpa 累加顺序不同。
  若怀疑 flash_attn 精度问题，设 `VLLM_ATTENTION_BACKEND=TORCH_SDPA` 回退对比。
- **已记录案例**：（暂无；待后续定位补充）

### RMSNorm / LayerNorm

- **文件**: `vllm/model_executor/layers/layernorm.py`
- **通用排查要点**：fp16/bf16 下 RMSNorm 的 eps 加法顺序、power 计算的精度影响。
  vLLM 用 CUDA fused LayerNorm 的 bf16 中间精度可能与 HF 不同。
- **已记录案例**：（暂无；待后续定位补充）

### MLP/FFN (gate/up/down 投影 + 激活)

- **文件**: `vllm/model_executor/layers/activation.py`
- **通用排查要点**：dense 模型用分组 GEMM,MoE 模型走 fused_moe——同是 MLP 但实现路径完全不同。
  先分 dense/MoE,再算子细化。
- **已记录案例**：（暂无；待后续定位补充）

---

## MoE 专属算子

- **文件**: `vllm/_custom_ops.py` → `torch.ops._moe_C.moe_fused_gate`
- **触发条件**: `use_fused_gate=True`
  - 定义于 `fused_moe/layer.py`：
    `use_fused_gate = VLLM_ENABLE_MOE_FUSED_GATE and e_score_correction_bias is not None and num_expert_group is not None`
  - 分发于 `fused_moe/router/grouped_topk_router.py`：
    ```python
    if self.use_fused_gate:
        if envs.VLLM_USE_LIGHTOP:
            ... lightop.op.moe_fused_gate ...
        else:
            topk_weights, topk_ids = ops.moe_fused_gate(...)
    else:
        topk_weights, topk_ids = grouped_topk_impl(...)
    ```
  - `VLLM_ENABLE_MOE_FUSED_GATE` 实际默认 **True**
    （envs 里 lambda `os.getenv(..., "1")` 覆盖了声明的 False）
- **通用排查要点**：此算子是 fused 实现，与 Python `grouped_topk` 是两条不同代码路径。
  若怀疑路由错误，对比这两条路径的输出（用 `compare_topk` 按 id 对齐）。
  绕过手段：`VLLM_ENABLE_MOE_FUSED_GATE=0` 走 Python 路径（正确性参考，但更慢）。
- **已记录案例**：
  - 模型 BailingMoeV2，平台 gfx936：fused_gate 选错专家（与 HF 0/32 匹配）→
    输出全 NULL。绕过 `VLLM_ENABLE_MOE_FUSED_GATE=0` 后正常。
    验证来源：reports/antangelmed_20260717（待迁入），2026-07-17，状态 已确认。
  - **不代表**所有 sigmoid+bias 模型在所有平台上都有此问题。

### 易混淆点：ops.grouped_topk ≠ ops.moe_fused_gate

> 这两个算子容易混淆。判定到底走了哪条路径，**必须看运行时**而非静态读代码：
> - `ops.grouped_topk` 受 `current_platform.is_cuda()` 守卫，非 CUDA 平台**不进入** fused 分支。
> - `ops.moe_fused_gate` 经 `use_fused_gate=True` 触发，与 is_cuda 无关。
>
> 历史教训：曾据静态分析误判根因为 `ops.grouped_topk`，运行时探针才发现该算子
> 在非 CUDA 平台根本没被调用，真正执行的是 `ops.moe_fused_gate`。
> **根因必须用运行时探针确认。**

---

## lightop.op.moe_fused_gate（lightop 高性能算子）

- **文件**: `lightop.op.moe_fused_gate(router_logits, bias, num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor)`
- **触发条件**: `use_fused_gate=True AND VLLM_USE_LIGHTOP=1`
- **通用排查要点**：lightop 是 fused_gate 的另一实现。若启用，需验证其输出（ids 与 weights）
  是否与 Python 参考路径一致。权重对比**必须按 expert id 对齐**（见下"教训"）。
- **已记录案例**：
  - 模型 BailingMoeV2，平台 gfx936：ids 与 torch 100% 一致，但权重不同
    （max_abs≈0.29，max_rel≈60%），疑为归一化/缩放公式差异。不会 NULL，但精度不齐。
    验证来源：reports/antangelmed_20260717，状态 已确认（选专家正确，权重有真实数值差）。
  - **不代表**所有模型/平台上 lightop 都有此差异。

### 教训：topk 权重对比必须"按 id 对齐"

对比两组 topk 权重时，不能各自 sorted 后逐位比——顺序差异会产生假误差。
必须按 expert id 对齐，且处理重复 id 用 multiset + 贪心配对。
此逻辑已固化在 `lib/tensor_compare.py:compare_topk`。这是**通用规则**，不限模型。

---

## aiter fused MoE（rocm_aiter）

- **文件**: `fused_moe/router/grouped_topk_router.py`；`fused_moe/rocm_aiter_fused_moe.py`
- **触发条件**: `rocm_aiter_ops.is_fused_moe_enabled()=True`
- **通用排查要点**：启用时 router 走 aiter 路径而非 Python `grouped_topk`。
  排查前先确认 `is_fused_moe_enabled()` 的值，才知道实际走了哪条路径。
- **已记录案例**：某 gfx936 环境下 `is_fused_moe_enabled()=False`（aiter 未启用）。
  非问题，是现状。其他环境/其他 gfx 可能不同。

---

## fused vs native MoE 专家 FFN

- **文件**: `fused_moe/layer.py`（`forward_cuda`=fused kernel，`forward_native`=纯 PyTorch）
- **通用排查要点**：怀疑 fused FFN kernel 精度问题时，对比 fused vs native 输出。
  若两者一致但都与参考实现有差 → 问题不在 fused FFN 精度，而在算法/路由层。
  vLLM 启动若报 MoE 配置文件缺失（如 `E=..,N=..,device_name=xxx.json not found`），
  通常只是**性能警告**，不影响正确性——但需用 fused vs native 对比证实。
- **已记录案例**：模型 BailingMoeV2，gfx936：fused vs native 一字不差，但都与 HF 差 0.63
  → 证伪"fused FFN 精度问题"假设，根因在路由层。状态 已证伪。

---

## routed_scaling_factor 重复相乘

- **通用排查要点**：部分模型 vLLM router 内部已把 `routed_scaling_factor` 乘进 topk_weights，
  若模型 forward 外层又乘一次，会重复。检查 vLLM 模型文件与 HF modeling 的 scaling 逻辑。
- **已记录案例**：模型 BailingMoeV2：去掉外层 scaling 后 cos 仅 0.630→0.667，
  不是主因。状态 已证伪（次要）。
