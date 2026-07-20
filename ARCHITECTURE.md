# OpLocate Agent — 设计文档

> 目标：把"vLLM 输出和 transformers 对不齐，定位到具体算子"从**天级别**压到**小时级别**。
> 适用任意 HF/ModelScope 模型在 vLLM 上的精度异常定位。

---

## 1. 为什么这样设计

定位精度靠三层能力，只有"推理"层依赖强 LLM，其余可脱离 Claude Code：

| 能力层 | 来源 | 依赖 Claude Code？ |
|---|---|---|
| 执行/探针 | Python 脚本（hook、抓中间态、对比算子） | ❌ |
| 知识匹配 | 分层 markdown 索引（arch → 路径 → 已知坑） | ❌ |
| 推理定位 | LLM（"下一步 hook 哪层？这是数值差还是算法差？"） | ✅ 需强 LLM，但不必是 Claude Code CLI |

**选定**：Claude Code headless（`claude -p`）+ Skill + 脚本工具库。脱离交互式 CLI（可塞 cron/CI），推理仍满血 Claude，工具与知识层独立可复用。编排是最后一公里，不是第一公里——先把 hook 方法论和索引库做扎实，将来要换 LangGraph+任意 LLM，工具层和知识层不用动。

---

## 2. 目录结构

```
op-locate-agent/
├── examples/                    # 入口脚本（门面）
│   ├── probe_model.py           # 探测模型结构属性名 → 落盘 JSON/MD
│   ├── quickstart_antangelmed.py# 端到端 vLLM+HF token 对比
│   └── compare_layers.py        # 逐层/逐算子中间值对比，定位发散点
├── lib/                         # 工具库（agent 的"手"）
│   ├── platform_probe.py        # Hygon DCU 硬件探测（gfx/型号/HIP）
│   ├── config_loader.py         # config.json + auto_map → ModelProfile
│   ├── config_patch.py          # 补 config 缺失字段（如 BailingMoeV2 MTP）
│   ├── path_resolver.py         # ModelProfile → vLLM/HF 代码路径 + key_ops
│   ├── model_probe.py           # 模型结构属性名探测（反射+源码交叉验证）
│   ├── hook_manager.py          # HookManager + 口径修正（防 in-place 污染等）
│   └── tensor_compare.py        # 张量/topk 对比器
├── knowledge/                   # 知识库（agent 的"脑"）
│   ├── arch_index.md            # 架构 → vLLM 代码路径映射
│   ├── precision_known_issues.md      # MoE 已知坑
│   ├── model_config_issues.md   # config 缺失字段坑
│   ├── vllm_forward_paths.md    # vLLM 前向路径速查
│   ├── hook_pitfalls.md         # hook 口径陷阱
│   └── platforms/hygon_dcu.md   # 平台相关
├── skills/op-locate/            # Claude Code skill（流程编排）
├── run/                         # 标准启动脚本（run_vllm.sh / run_hf.sh）
├── reports/                     # 定位报告（.gitignore）
└── tests/                       # 单测
```

---

## 3. 数据流

```
模型路径
  │
  ▼
[1] config_loader → ModelProfile        # 解析 config.json + 自定义 py
  │
  ▼
[2] model_probe → 属性名落盘             # attn/mlp/router/layer_prefix（反射+源码）
  │
  ▼
[3] quickstart → 是否对齐？              # HF vs vLLM token 对比
  │   └ 对齐 → 结束
  │   └ 不对齐 ↓
  ▼
[4] compare_layers → 首个发散层/算子      # 逐层中间值 cos
  │
  ▼
[5] Claude 推理：在该层内规划 sub-op hook # 读 knowledge 索引
  │   └ 循环 [4][5] 直到锁定具体算子
  ▼
[6] report → reports/<model>_<date>/     # report.md + verdict.json
```

---

## 4. 关键抽象

**ModelProfile**（config_loader 输出）：`arch` / `model_type` / `num_layers` / `is_moe` / `num_experts` / `first_k_dense_replace` / `custom_py_files` 等标准化字段。

**ModelProbeResult**（model_probe 输出）：`layer_prefix` / `attn_attr` / `mlp_attr` / `router_attr`，每项带 `sources`（reflection/source）与 `confidence`（high/medium/low）。落盘 JSON+MD 供 compare_layers 读取、人核。

**HookManager**（hook_manager）：`CaptureSpec(hook_points, owner, clone)`，三种口径——`pre`（防 in-place 污染）、`post`、`patch`（monkey-patch，CustomOp 可靠）。owner 存储防 collective_rpc 陷阱。

**CodePaths / KeyOp**（path_resolver）：arch → vLLM 实际代码路径 + 关键算子入口。key_ops 只标结构事实（算子+触发条件），不硬编码 bug 结论——根因以运行时探针为准。

---

## 5. 与 Claude Code 集成

三种入口共用同一套 skill + lib + knowledge：

- **交互式**：软链 `skills/op-locate` 到 `~/.claude/skills/`，`/op-locate <模型>`
- **Headless**：`claude -p "用 op-locate skill 定位 <模型>" --allowedTools "..."`
- **纯工具库**：`from lib import ...`，不要 LLM

---

## 6. 里程碑

| 里程碑 | 交付 | 状态 |
|---|---|---|
| M1 | lib/ 核心模块 + 单测 | ✅ |
| M2 | knowledge/ 首版索引 | ✅ |
| M3 | skills/op-locate 流程 | ✅ |
| M4 | examples 入口 + run 脚本 + headless | ✅ |
| M5 | 第二个模型验证通用性 | 待做 |

---

## 7. 开放问题

- 知识索引回写：倾向人工 review 后追加，防错误结论污染（曾因静态分析误判根因）。
- 单算子对比（lightop vs torch）：作为 MoE 专属子流程，不进通用主流程。
- 平台维度：当前 gfx936/gfx938，`knowledge/platforms/` 预留其他平台扩展。
