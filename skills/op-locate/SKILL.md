---
name: op-locate
description: 定位 vLLM 推理精度问题到具体算子。输入模型路径，逐步用 hook 抓中间态、对比 transformers，锁定误差首次出现的算子。适用于任意 HF/ModelScope 模型在 vLLM 上输出异常（乱码/NULL/精度不齐）的算子级定位。
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, WebFetch
---

# op-locate — vLLM 算子级精度定位

## 何时用

用户给出模型路径（本地目录），抱怨 vLLM 推理输出异常：
- 全 NULL / 乱码 / token id 异常（如全 188）
- 与 transformers(HF) 输出对不齐
- 逐层精度衰减

目标：把"vLLM 输出和 transformers 对不齐"定位到**具体算子**，从天级别压到小时级别。

## 输入

- **必需**：模型本地路径（含 config.json），如 `/models/AntAngelMed`
- **可选**：modelscope / huggingface URL（用 WebFetch 拉官方 model card 补充背景）
- **可选**：测试 prompt（默认用 `"你好，我是一名医生"`，4-8 token）

## 输出

`reports/<model>_<date>/` 下：
- `report.md`：人读结论（现象、根因、修复、被证伪假设）
- `verdict.json`：机读结论（bug 算子、修复方法、置信度）
- `evidence/`：中间态、对比日志

并提示人工 review 后回写 `knowledge/moe_known_issues.md`（不自动回写）。

## 工具与知识

- **工具库**：`op_locate_agent/lib/`（config_loader, path_resolver, hook_manager, tensor_compare）
- **知识库**：`op_locate_agent/knowledge/`（arch_index, vllm_forward_paths, moe_known_issues, hook_pitfalls, platforms/gfx936）

> 假设 cwd 为 `op_locate_agent/`。lib 可 `import lib`，knowledge 用 Read 读。

## 流程（7 阶段）

### 阶段 0：环境确认

1. 探测平台：
   ```python
   from lib import probe_platform, recommended_idle_gpu
   info = probe_platform()
   print(info.summary())   # 如 "Hygon DCU BW100 (gfx936), HIP 6.3.26113, 8 devices, is_cuda=False"
   gpu = recommended_idle_gpu(info)   # 避开在线服务占用卡
   ```
2. 若 `info.is_dcu`：Read `knowledge/platforms/hygon_dcu.md`。
   - 注意 gfx936 与 gfx938 是不同架构，kernel/aiter 兼容性可能不同，不要假设。
   - 记住 `is_cuda()=False` 会改变 vLLM 调度（见 hygon_dcu.md）。
3. 选定空闲卡，后续脚本统一用 `HIP_VISIBLE_DEVICES=<gpu>`。
4. **重要心态**：历史案例（AntAngelMed 的 fused_gate bug）只是**一个案例**，
   不代表当前问题也是它。每个案例独立用运行时探针定位，不要预判根因。

### 阶段 1：解析模型配置

```python
from lib import load_model_profile
profile = load_model_profile("/models/AntAngelMed")
print(profile.moe_summary())
```

记录：arch、is_moe、MoE 参数、bias 机制、custom_py_files。

### 阶段 2：解析代码路径

```python
from lib import resolve_code_paths
cp = resolve_code_paths(profile)
for k in cp.key_ops:
    print(k.name, k.trigger_condition, k.known_issue)
```

读 `knowledge/arch_index.md` 确认 vLLM 模型文件。
读 `knowledge/vllm_forward_paths.md` 规划 hook 点。

**关键判定**：若 `key_ops` 含 `ops.moe_fused_gate` 且平台是 gfx936 → **先读 `knowledge/moe_known_issues.md` 的 fused_gate 条目**，这是已知坑，可能直接命中。

### 阶段 3：获取官方背景（可选）

若用户给了 modelscope/hf URL，WebFetch 拉 model card，关注：
- 官方推荐的推理框架与版本
- 已知 issue / 限制
- MoE 实现细节提示

### 阶段 4：粗粒度逐层定位

目的：找到误差**首次出现**的层。

1. 用 `knowledge/hook_pitfalls.md` 的规范写 hook（pre_hook 优先、CustomOp 用 patch、owner 存储）。
2. 在 vLLM 和 HF 上跑**同一输入**，抓逐层中间态。
3. `lib/tensor_compare` 逐层算 cos。
4. 定位：embedding 应 cos≈1.0；找到第一个 cos 掉的层。

参考脚本：`precision_compare/test_stage_compare.py`（粗粒度曲线）、`test_l1_ops.py`（层内 A-F 拆解）。

> **必读**：`knowledge/hook_pitfalls.md` 陷阱 1-3，否则会得到假误差。

### 阶段 5：细粒度算子定位

定位到某层 `mlp_out` 发散后，按 `knowledge/vllm_forward_paths.md` 的 MoE 拆解树细化：

1. 抓 router_logits（gate 输出）→ 对比 HF。若发散，根因在 gate。
2. 抓 **vLLM 实际运行时** topk（patch `select_experts`，**不用重算**）→ 对比 HF topk_ids。
   - 用 `lib.compare_topk`（按 id 对齐，处理重复 id）。
3. 若 topk_ids 不匹配 → 根因在 router 算子。查 `key_ops` 与 `moe_known_issues.md`。
4. 若 topk_ids 匹配但 mlp_out 仍发散 → 根因在专家 FFN 或 shared expert。

> **必读**：`knowledge/hook_pitfalls.md` 陷阱 4（重算 vs 实际）、陷阱 5（topk 排序错位）。

### 阶段 6：单算子对比验证

锁定嫌疑算子后，做单算子级对比确认：

- 若嫌疑是 fused_gate：对比 `ops.moe_fused_gate` vs Python `grouped_topk` vs lightop
  （参考 `precision_compare/test_lightop_vs_torch_min.py`）。
- 若嫌疑是 fused FFN：对比 `forward_cuda` vs `forward_native`
  （参考 `precision_compare/test_moe_native_l1.py`）。
- 若需要环境变量绕过：设 `VLLM_ENABLE_MOE_FUSED_GATE=0` 端到端验证输出恢复。

### 阶段 7：报告与回写

按 `compare.md` 和 `report.md` 子流程：
1. 写 `reports/<model>_<date>/report.md`（现象、根因、修复、被证伪假设）。
2. 写 `verdict.json`（bug 算子、修复方法、置信度、证据路径）。
3. **提示人工 review** 后回写 `knowledge/moe_known_issues.md`。**不自动回写**。

## 加速原则

- **先查知识库再动手**：阶段 2 的 key_ops 若命中已知坑，可直接跳到阶段 6 验证，省掉阶段 4-5。
- **粗到细**：先逐层（阶段 4）再层内（阶段 5）再算子（阶段 6），不要一上来就钻单算子。
- **运行时探针为准**：任何"重算"结果都不能代替 vLLM 实际运行时中间态（陷阱 4）。

## 黄金回归

`tests/test_e2e_antangelmed.py` 是 AntAngelMed 的端到端黄金用例。
新模型定位前可先跑它确认环境正常；定位方法论变更后用它回归。
