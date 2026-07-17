# MoE Known Issues — MoE 已知坑库

> MoE 模型在 vLLM 上的已知算子问题。按"算子"组织，每条带验证来源。
> **重要**：根因以运行时探针为准，不以代码静态分析为准（静态分析曾导致误判）。

---

## ops.moe_fused_gate（fused gate kernel）— 已确认

- **文件**: `vllm/_custom_ops.py:3049` → `torch.ops._moe_C.moe_fused_gate`
- **触发条件**: `use_fused_gate=True`
  - 定义于 `fused_moe/layer.py:603`：
    `self.use_fused_gate = envs.VLLM_ENABLE_MOE_FUSED_GATE and e_score_correction_bias is not None and num_expert_group is not None`
  - 分发于 `fused_moe/router/grouped_topk_router.py:343-356`：
    ```python
    if self.use_fused_gate:
        if envs.VLLM_USE_LIGHTOP:
            ... lightop.op.moe_fused_gate ...
        else:
            topk_weights, topk_ids = ops.moe_fused_gate(...)  # ← bug 路径
    else:
        topk_weights, topk_ids = grouped_topk_impl(...)       # ← 正确路径
    ```
  - `VLLM_ENABLE_MOE_FUSED_GATE` 实际默认 **True**
    （`envs.py:1794` 的 lambda `os.getenv(..., "1")` 覆盖了 `envs.py:279` 声明的 `False`）
- **平台**: gfx936 (Hygon DCU, `is_cuda()=False`)
- **症状**: vLLM 输出全 NULL（token id 188 = `\x00`）。MoE 选错专家（与 HF 0/32 匹配）→ routed_sum cos=0.126 → 逐层放大 → 全 NULL。
- **根因**: `ops.moe_fused_gate` 这个 C++/HIP fused gate kernel 在 gfx936 上、针对 `n_group=8, topk_group=4, sigmoid+expert_bias` 选出的专家与正确实现不一致。
- **修复**: `VLLM_ENABLE_MOE_FUSED_GATE=0` → `use_fused_gate=False` → 走 Python `grouped_topk`。验证输出正常连贯中文，NULL 消失。
- **代价**: Python `grouped_topk` 比 fused 慢。以正确性换速度。
- **验证**: AntAngelMed (BailingMoeV2), 2026-07-17, `precision_compare/`（probe_envvar.py 端到端验证）
- **状态**: 已确认

### 易混淆点：ops.grouped_topk ≠ ops.moe_fused_gate

> **历史误判**：早期调查记录（`INVESTIGATION.md` 第 6 节）把根因记为 `ops.grouped_topk`。
> 运行时探针证伪：本机 `is_cuda()=False`，`grouped_topk` 的 fused 分支
> （`grouped_topk_router.py:82` 的 `if ... current_platform.is_cuda() ...`）**根本不进入**，
> 所以 `ops.grouped_topk` 从未被调用。真正执行的是 `ops.moe_fused_gate`（经 `use_fused_gate=True`）。
>
> **教训**：根因必须用运行时探针确认，不能只看代码静态路径。

---

## lightop.op.moe_fused_gate（lightop 高性能算子）— 不建议

- **文件**: `lightop.op.moe_fused_gate(router_logits, bias, num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor)`
- **触发条件**: `use_fused_gate=True AND VLLM_USE_LIGHTOP=1`（见上面分发代码的 344 行分支）
- **平台**: gfx936
- **症状**: **不会** NULL（选专家正确），但权重与 torch 路径不一致。
- **根因**: lightop 选出的专家 ids 与 torch 100% 一致，但分配的**路由权重不同**
  （max_abs_diff≈0.294，max_rel_diff≈60%）。疑为归一化/缩放步骤公式差异
  （bias 是否参与归一化分母、×2.5 的顺序）。
- **影响**: 权重差 60% → MoE 加权输出偏 → 与 transformers 精度对不齐。**不会崩，但精度不达标**。
- **修复**: 无。不建议启用 `VLLM_USE_LIGHTOP`。若要速度且精度达标，需查清 lightop 归一化公式差异并修复。
- **验证**: AntAngelMed, 2026-07-17, `precision_compare/test_lightop_vs_torch_min.py`
  - ids 行精确匹配 100% (64/64)
  - weights 按id对齐: max_abs=2.943e-01, max_rel=6.000e-01, cos=1.000000
- **状态**: 已确认（选专家正确，权重有真实数值差）

### 教训：topk 权重对比必须"按 id 对齐"

对比两组 topk 权重时，不能各自 sorted 后逐位比——顺序差异会产生假误差。
必须按 expert id 对齐（且处理重复 id 用 multiset + 贪心配对）。
此逻辑已固化在 `lib/tensor_compare.py:compare_topk`。

---

## aiter fused MoE（rocm_aiter）— 未启用

- **文件**: `fused_moe/router/grouped_topk_router.py:226,331`；`rocm_aiter_fused_moe.py`
- **触发条件**: `rocm_aiter_ops.is_fused_moe_enabled()=True` 时走 aiter 路径
- **平台**: gfx936
- **现状**: 本机 `is_fused_moe_enabled()=False`，故 `grouped_topk_impl = grouped_topk`（Python），不走 aiter。
- **影响**: 无（未启用）。但意味着 aiter 加速路径在本环境不可用。
- **未深挖**: 为什么 `is_fused_moe_enabled()=False`（aiter 未正确初始化？DCU 兼容？）
- **状态**: 观察（非问题，是现状）

---

## fused vs native MoE 专家 FFN — 已证伪

- **文件**: `fused_moe/layer.py`（`forward_cuda` = fused Triton/HIP kernel，`forward_native` = 纯 PyTorch）
- **触发条件**: 专家 FFN 计算路径选择
- **假设**: gfx936 MoE 配置缺失（`E=256,N=256,device_name=gfx936_64cu_nn.json` not found）导致 fused kernel 精度问题
- **验证**: fused vs native 结果一字不差（cos=1.000001, max_diff=0.000），但都和 HF 差 0.63。
- **结论**: **假设证伪**。问题不在 fused FFN kernel 的精度/配置，配置缺失只是性能警告。
- **验证**: AntAngelMed, 2026-07-17, `precision_compare/test_moe_native_l1.py`
- **状态**: 已证伪

---

## routed_scaling_factor 重复相乘 — 已证伪（次要问题）

- **文件**: vLLM `GroupedTopKRouter.select_experts` 内部已把 `routed_scaling_factor` 乘进 topk_weights；`BailingMoE.forward` 外层又乘一次
- **假设**: 2.5×2.5=6.25 倍 bug 是 0.63 误差主因
- **验证**: 去掉外层 scaling，cos 仅 0.630→0.667，NULL 仍在。
- **结论**: **假设证伪**（不是主因）。重复相乘确实是个问题，但不是 0.63 主因。
- **验证**: AntAngelMed, 2026-07-17, `precision_compare/test_fix_scaling.py`
- **状态**: 已证伪（次要）
