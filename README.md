# OpLocate Agent

> 把"vLLM 输出和 transformers 对不齐，定位到具体算子"从**天级别**压到**小时级别**。
> 详见 [ARCHITECTURE.md](./ARCHITECTURE.md)。

## 5 分钟上手

### 前提
- vLLM + transformers + torch 已装（本机 vLLM 0.15.1, torch 2.9.0）
- 有空闲 GPU/DCU 卡（gfx936 上避开在线服务占用的 0,1,6,7，用 2/3/7）
- 模型本地目录含 config.json

### 方式 A：Claude Code 交互式

把 skill 软链到 Claude 的 skills 目录：
```bash
mkdir -p ~/.claude/skills
ln -s $(pwd)/skills/op-locate ~/.claude/skills/op-locate
```
然后在 Claude Code 里：
```
/op-locate /models/AntAngelMed
```

### 方式 B：Claude Code headless（脱离交互式 CLI，可塞 cron/CI）

```bash
cd op_locate_agent
claude -p "用 op-locate skill 定位 /models/NewMoE 模型的 vLLM 精度问题" \
  --allowedTools "Bash,Read,Write,Edit,Glob,Grep,WebFetch,Skill"
```

### 方式 C：只用工具库（不要 LLM，自己跑）

```python
from lib import load_model_profile, resolve_code_paths

profile = load_model_profile("/models/AntAngelMed")
print(profile.moe_summary())

cp = resolve_code_paths(profile)
for k in cp.key_ops:
    print(k.name, k.known_issue)
```

## 目录

```
op_locate_agent/
├── ARCHITECTURE.md          # 设计文档（先读这个）
├── README.md                # 本文件
├── skills/op-locate/        # Claude Code skill（流程编排）
│   ├── SKILL.md  probe.md  compare.md  report.md
├── lib/                     # 工具库（agent 的"手"）
├── knowledge/               # 知识库（agent 的"脑"）
├── run/                     # 标准启动脚本
├── reports/                 # 定位报告输出
└── tests/                   # 单测 + 黄金回归
```

## 测试

```bash
cd op_locate_agent
python3 -m pytest tests/ -v          # 全部单测（32 个，含 AntAngelMed fixture）
python3 -m pytest tests/test_e2e_antangelmed.py -v   # 黄金回归
```

## 验证 lib 自检

```bash
python3 lib/config_loader.py /models/AntAngelMed    # 解析配置
python3 -m lib.path_resolver /models/AntAngelMed    # 解析代码路径
python3 lib/tensor_compare.py                       # 对比器自检
```

## 当前已沉淀的模型

| 模型 | 状态 | 根因 | 报告 |
|---|---|---|---|
| AntAngelMed (BailingMoeV2) | ✅ 已定位 | `ops.moe_fused_gate` 在 gfx936 选错专家 | `precision_compare/`（待迁入 reports/） |

修复：`VLLM_ENABLE_MOE_FUSED_GATE=0`（走 Python grouped_topk，正确但慢）。

## 设计要点（一句话版）

- **不依赖 langchain**：Claude 本身就是最好的任务编排器，工具层+知识层先沉淀，编排是最后一公里。
- **脱离 CLI 但能力不打折**：用 `claude -p` headless，推理仍满血 Claude，工具仍是本库。
- **运行时探针为准**：根因以运行时 hook 抓取为准，不以代码静态分析为准（曾导致误判）。
- **不自动回写知识库**：人工 review 是质量门，防错误结论污染。
