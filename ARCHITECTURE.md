# OpLocate Agent — 设计文档 v0.1

> 目标：把"vLLM 输出和 transformers 对不齐，定位到具体算子"这件事，从**天级别**压到**小时级别**。
> 适用：任意 HuggingFace/ModelScope 模型在 vLLM 上推理精度异常的算子级定位。

---

## 0. 设计动机（为什么这样选）

### 0.1 能力拆解：定位精度靠什么

| 能力层 | 来源 | 是否依赖 Claude Code |
|---|---|---|
| **执行/探针** | Python 脚本（hook、抓中间态、对比算子） | ❌ 不依赖，纯代码 |
| **知识匹配** | 分层 markdown 索引（arch → vllm 路径 → 已知坑） | ❌ 不依赖，规则化检索 |
| **推理定位** | LLM 推理（"下一步 hook 哪层？这个差是数值差还是算法差？"） | ✅ 依赖**强 LLM**，但不必须是 Claude Code 这个 CLI |

**结论**：真正离不开的不是 Claude Code 这个交互式 CLI，而是"一个强推理 LLM + 它能调用的工具"。Claude Code 本质 = `强LLM + 文件/搜索/bash 工具 + 对话循环`。

### 0.2 三条路线的取舍

| 路线 | 脱离 CLI | 推理能力 | 工程量 | 风险 |
|---|---|---|---|---|
| LangGraph + 远程 LLM API | ✅ 独立服务 | 取决于接的 LLM | 大（StateGraph + 工具胶水 + 状态管理） | 编排逻辑硬编码，资产没沉淀就先修最贵的层 |
| **Claude Code headless/SDK（本方案）** | ✅ 脱离交互式 CLI（`claude -p` 可脚本化） | 满血 Claude | 中（写 skill + 抽象脚本） | 依赖 Claude API 可用性 |
| 纯 pydantic pipeline（无 LLM） | ✅ | 无推理，固定流程 | 小 | 遇到新模型新算子就卡死，做不到"一步步锁定" |

### 0.3 选定方案：Claude Code headless + Skill + 脚本工具库

- **推理引擎**：Claude，通过 Claude Code headless（`claude -p`）或 SDK 调用 → 脱离交互式 CLI，可塞进 cron/CI/被别的系统调。
- **工具层**：把现有 `precision_compare/` 的 40 个文件抽象成可复用脚本库（`lib/`），hook 方法论沉淀为模板。
- **知识层**：分层 markdown 索引（`knowledge/`），Claude 读索引定位，不把源码塞进上下文。
- **入口**：`run/` 下的标准启动指令（vLLM + HF 两套），用户给模型路径即可跑。
- **留扩展口**：以后真要做无 Claude 的独立服务，把推理层换成 LangGraph + 任意 LLM，**工具层和知识层不用动**。编排是最后一公里，不是第一公里。

**为什么不直接上 LangGraph**：现有最大资产是 hook 方法论 + 即将建的索引库，这些和编排框架无关。先把这些做扎实；先上 LangGraph 等于先修最贵、最易变的层，下面没沉淀，graph 里全是 if-else 硬编码。

---

## 1. 目录结构

```
op_locate_agent/
├── ARCHITECTURE.md              # 本文档
├── README.md                    # 快速上手（5 分钟跑通第一个模型）
│
├── skills/
│   └── op-locate/
│       ├── SKILL.md             # Claude Code skill 定义：定位流程编排
│       ├── probe.md             # 子流程：hook 注入与中间态抓取规范
│       ├── compare.md           # 子流程：算子级对比与误差判定标准
│       └── report.md            # 子流程：结论归档与索引回写
│
├── lib/                         # 可复用脚本工具库（agent 的"手"）
│   ├── __init__.py              # 统一导出
│   ├── platform_probe.py        # Hygon DCU 硬件探测（gfx 型号/型号/HIP 版本）
│   ├── config_loader.py         # 解析 config.json + auto_map → 标准化 ModelProfile
│   ├── config_patch.py          # 补 model config 缺失字段（如 BailingMoeV2 MTP）
│   ├── path_resolver.py         # ModelProfile → vLLM/transformers 实际代码路径 + key_ops
│   ├── hook_manager.py          # HookManager + 口径修正（pre_hook/patch/owner，防 in-place 污染）
│   ├── tensor_compare.py        # TensorComparator + topk 按 id 对齐对比
│   ├── probe_runner.py          # (规划中) 在 vLLM/HF 上跑指定输入，抓指定 hook 点
│   ├── op_compare.py            # (规划中) 单算子级对比（lightop vs torch vs vllm_fused）
│   └── report_writer.py         # (规划中) 生成结构化报告 + 回写 knowledge 索引
│
├── knowledge/                   # 分层 markdown 索引（agent 的"脑"）
│   ├── README.md                # 索引使用说明
│   ├── arch_index.md            # 架构 → vLLM 代码路径映射（按 model_type/architectures）
│   ├── moe_known_issues.md      # MoE 已知坑（fused_gate / grouped_topk / aiter / lightop）
│   ├── model_config_issues.md   # 模型 config 缺失字段坑（BailingMoeV2 MTP 等）
│   ├── vllm_forward_paths.md    # vLLM 前向核心路径速查（按模型族）
│   ├── hook_pitfalls.md         # hook 抓取口径陷阱（in-place 污染 / collective_rpc / CustomOp）
│   └── platforms/               # 平台相关（Hygon DCU 家族 / cuda 等）
│       └── hygon_dcu.md
│
├── examples/                    # 完整启动例子（门面）
│   └── quickstart_antangelmed.py # 端到端 vLLM+HF 启动对比，clone 后第一个跑
│
├── run/                         # 标准启动脚本（卡用 HIP_VISIBLE_DEVICES 前置指定）
│   ├── run_vllm.sh              # 标准化 vLLM 离线推理
│   └── run_hf.sh                # 标准化 transformers 推理
│
├── reports/                     # 每次定位的结构化报告（按模型+日期，.gitignore）
│   └── <model>_<date>/
│       ├── report.md            # 人读结论
│       ├── evidence/            # 抓到的中间态、对比日志
│       └── verdict.json         # 机读结论（bug 算子、修复方法、置信度）
│
├── docs/
│   └── SECURITY.md              # 凭证安全（密码/token 绝不入库）
│
└── tests/                       # lib/ 的单元测试 + 端到端回归
    ├── conftest.py
    ├── test_config_loader.py
    ├── test_path_resolver.py
    ├── test_hook_manager.py
    ├── test_tensor_compare.py
    ├── test_platform_probe.py
    └── test_e2e_antangelmed.py  # (规划中) AntAngelMed 黄金回归
```

---

## 2. 核心数据流

```
用户输入：模型本地路径 (+ 可选 modelscope/hf URL)
    │
    ▼
[1] config_loader → ModelProfile
    （config.json + auto_map 自定义 py + README → 标准化配置对象）
    │
    ▼
[2] path_resolver → CodePaths
    （ModelProfile.arch → 查 knowledge/arch_index.md → vLLM 实际代码路径）
    │
    ▼
[3] WebFetch 官方 model card（可选）→ 补充实现细节提示
    │
    ▼
[4] Claude 推理：根据 arch + CodePaths + 已知坑，规划 hook 点
    （读 knowledge/vllm_forward_paths.md + moe_known_issues.md）
    │
    ▼
[5] probe_runner：在 vLLM 和 HF 上跑同一输入，抓 hook 点中间态
    （用 hook_manager，遵循 hook_pitfalls.md 的口径规范）
    │
    ▼
[6] tensor_compare：逐点 cos，定位误差首次出现位置
    │
    ▼
[7] Claude 推理：定位到某算子后，决定下一步（更细粒度 hook / 单算子对比 / 排除假设）
    （循环 5-7，直到锁定具体算子）
    │
    ▼
[8] report_writer → reports/<model>_<date>/
    （report.md + verdict.json + 回写 knowledge 索引）
```

---

## 3. 关键抽象

### 3.1 ModelProfile（config_loader 的输出）

```python
@dataclass
class ModelProfile:
    # 基础
    local_path: str
    arch: str                    # architectures[0]，如 "BailingMoeV2ForCausalLM"
    model_type: str              # config.json model_type，如 "bailing_moe_v2"
    # 结构
    num_layers: int
    hidden_size: int
    num_heads: int
    kv_heads: int
    vocab_size: int
    # MoE（若有）
    is_moe: bool
    num_experts: Optional[int]
    num_experts_per_tok: Optional[int]      # top_k
    num_expert_group: Optional[int]         # n_group
    topk_group: Optional[int]
    routed_scaling_factor: Optional[float]
    score_function: Optional[str]           # "sigmoid" | "softmax"
    e_score_correction_bias: Optional[str]  # bias 张量名，存在则触发 fused_gate
    num_shared_experts: Optional[int]
    first_k_dense_replace: Optional[int]    # 前 N 层 dense
    # 自定义
    auto_map: Dict[str, str]     # {"AutoModelForCausalLM": "modeling_xxx.XXXForCausalLM"}
    custom_py_files: List[str]   # 本地自定义 py 文件路径
    # 平台
    torch_dtype: str
```

### 3.2 CodePaths（path_resolver 的输出）

```python
@dataclass
class CodePaths:
    vllm_model_dir: str          # 如 .../vllm/model_executor/models/bailing_moe_v2.py
    vllm_moe_layer: str          # 如 .../fused_moe/layer.py
    vllm_router: str             # 如 .../fused_moe/router/grouped_topk_router.py
    hf_modeling: str             # 本地 modeling_xxx.py
    # 关键算子入口（由知识库填充）
    key_ops: List[KeyOp]         # [{name, file, line, condition, known_issue}]
```

### 3.3 KeyOp（知识库条目）

```markdown
# knowledge/moe_known_issues.md 条目格式

## ops.moe_fused_gate
- **文件**: vllm/_custom_ops.py:3049 → torch.ops._moe_C.moe_fused_gate
- **触发条件**: use_fused_gate=True
  （VLLM_ENABLE_MOE_FUSED_GATE=1 且 e_score_correction_bias≠None 且 num_expert_group≠None）
- **平台**: gfx936 (is_cuda()=False)
- **症状**: MoE 选错专家 → 输出全 NULL(token 188)
- **修复**: VLLM_ENABLE_MOE_FUSED_GATE=0（走 Python grouped_topk）
- **验证**: AntAngelMed, 2026-07-17, reports/antangelmed_20260717/
- **状态**: 已确认
```

---

## 4. 与 Claude Code 的集成方式

### 4.1 交互式（开发/调试时）

把 `skills/op-locate/SKILL.md` 软链或复制到 `~/.claude/skills/`，在 Claude Code 里直接：
```
/op-locate /models/AntAngelMed
```

### 4.2 Headless（脱离交互式 CLI）

```bash
# 一键定位，可塞进 cron/CI
claude -p "用 op-locate skill 定位 /models/NewMoE 模型的 vLLM 精度问题" \
  --allowedTools "Bash,Read,Write,Edit,Glob,Grep,WebFetch,Skill"
```

### 4.3 SDK（被别的程序调用）

```python
# 伪代码，后续 lib/ 可封装
from claude_code_sdk import query
async for msg in query(prompt="op-locate /models/NewMoE", allowed_tools=[...]):
    ...
```

> 三种入口共用同一套 skill + lib + knowledge，只是调用方式不同。这是"脱离 CLI 但能力不打折"的关键。

---

## 5. 速度目标拆解（天 → 小时）

| 阶段 | 人工做（现状） | agent 做（目标） | 提速点 |
|---|---|---|---|
| 读 config 找结构 | 30-60 min | < 5 min | config_loader 自动解析 |
| 找 vLLM 前向路径 | 1-2 h | < 10 min | arch_index.md 直接查 |
| 写 hook 抓中间态 | 2-4 h | < 30 min | hook_manager 模板 + hook_pitfalls.md |
| 逐层对比定位 | 2-4 h | < 30 min | probe_runner + tensor_compare 自动 |
| 算子级拆解 | 半天-1 天 | < 1 h | 已知坑索引 + 单算子对比脚本 |
| **合计** | **1-2 天** | **2-3 小时** | |

提速的本质：**把"每次重新发现"变成"查索引 + 套模板"**。第一遍建索引慢，第二遍起指数级提速。

---

## 6. 里程碑

| 里程碑 | 交付物 | 验收 |
|---|---|---|
| **M1** | lib/ 核心模块（config_loader, path_resolver, hook_manager, tensor_compare）+ 单测 | 对 AntAngelMed 跑通 config 解析和 hook 抓取 |
| **M2** | knowledge/ 首版索引（arch_index, moe_known_issues, vllm_forward_paths, hook_pitfalls） | 索引能正确指向 AntAngelMed 的 bug 算子 |
| **M3** | skills/op-locate/SKILL.md + probe/compare/report 子流程 | 交互式跑通 AntAngelMed 端到端定位 |
| **M4** | run/ 标准启动 + headless 入口 + 黄金回归测试 | `claude -p` 一键复现 AntAngelMed 定位 |
| **M5** | 第二个模型验证（验证通用性，反哺索引） | 新模型定位 < 3 小时 |

---

## 7. 待定 / 开放问题

- **7.1** headless `claude -p` 在本环境是否可用、是否需要 API key 配置 → M4 前验证。
- **7.2** 知识索引的"回写"机制：每次定位完自动追加到 `moe_known_issues.md`，还是人工 review 后追加？倾向人工 review（避免错误结论污染索引，参考 INVESTIGATION.md 根因写错的前车之鉴）。
- **7.3** 单算子对比（如 lightop vs torch）是否纳入标准流程，还是作为 MoE 专属子流程？倾向后者——只有 MoE 才有 fused/native/lightop 多实现。
- **7.4** 平台维度：当前只 gfx936，要不要预留 cuda/其他 DCU 的索引结构？预留目录 `knowledge/platforms/`，先不填。

---

## 8. 与现有资产的关系

| 现有文件 | 去向 |
|---|---|
| `precision_compare/core.py` | 抽出 `UniversalHookManager`/`TensorComparator`/`ModelConfig` → `lib/hook_manager.py` + `lib/tensor_compare.py` |
| `precision_compare/test_*.py` | 方法论提炼进 `knowledge/hook_pitfalls.md` + `skills/op-locate/probe.md`；具体脚本留作参考 |
| `precision_compare/INVESTIGATION.md` | 内容回写 `knowledge/moe_known_issues.md`（**根因需更正：`ops.grouped_topk` → `ops.moe_fused_gate`**），并作为 `tests/test_e2e_antangelmed.py` 的黄金基准 |
| `precision_compare/test_lightop_vs_torch_min.py` | 沉淀为 `lib/op_compare.py` 的 MoE 单算子对比模板 |
