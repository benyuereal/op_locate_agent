# 子流程：report — 结论归档与索引回写

> SKILL.md 阶段 7 的细化规范。

## 报告目录结构

```
reports/<model>_<date>/
├── report.md          # 人读结论
├── verdict.json       # 机读结论
└── evidence/          # 证据
    ├── stage_cos.csv      # 逐层 cos 曲线
    ├── topk_compare.txt   # topk 对比明细
    ├── envvar_test.log    # 环境变量绕过验证日志
    └── ...                # 其他中间态 dump
```

`<date>` 用 YYYYMMDD（无 `Date.now()`，由调用方传入或手动填）。

## report.md 模板

```markdown
# <Model> vLLM 精度问题定位报告

## 1. 问题现象
- 模型：<arch> (<model_type>)
- 环境：vLLM <version>, <platform>, <dtype>, TP=<n>
- 现象：<输出异常描述>
- 对照：transformers <version> 输出 <正常/异常>

## 2. 模型配置
<profile.moe_summary() 输出>

## 3. 定位过程
### 3.1 粗粒度逐层
<stage_cos 曲线 + 首个发散层>

### 3.2 层内拆解
<A-F 拆解结果，定位到哪个算子>

### 3.3 算子级确认
<单算子对比结果>

## 4. 根因
<具体算子 + 为什么错，基于运行时探针>

## 5. 修复
<环境变量 / patch / 配置，附验证结果>

## 6. 被证伪的假设
| 假设 | 验证方式 | 结论 |
|---|---|---|
| ... | ... | ❌ 证伪 |

## 7. 代价与后续
<修复的代价，如速度；更好的长期修复方向>
```

## verdict.json 模板

```json
{
  "model": "<模型名>",
  "arch": "<architectures[0]>",
  "date": "<YYYYMMDD>",
  "platform": "<gfxXXX / cuda / ...>",
  "vllm_version": "<x.y.z>",
  "symptom": "<输出异常描述>",
  "bug_operator": "<定位到的算子，如 ops.xxx；未定位则 null>",
  "bug_file": "<相对 vllm 包的路径:行号>",
  "root_cause": "<基于运行时探针的根因描述>",
  "fix": "<修复方法；未修复则 null>",
  "fix_verified": <true|false>,
  "fix_tradeoff": "<修复代价，如更慢；无则空字符串>",
  "confidence": "<high|medium|low>",
  "evidence_dir": "reports/<model>_<date>/evidence",
  "falsified_hypotheses": ["<排查过并被证伪的假设>"]
}
```

`confidence` 取值：`high`（运行时探针 + 绕过验证双确认）/ `medium`（探针确认但未绕过验证）/ `low`（推测）。

## 回写知识库（人工 review）

**不自动回写**。报告完成后，提示用户：

> 定位完成。若结论已确认（confidence=high），建议人工 review 后将根因追加到
> `knowledge/moe_known_issues.md`，附"验证来源"（本报告路径 + 日期）。
> 回写时按 `knowledge/README.md` 的条目格式。

回写原因：避免错误结论污染知识库（`INVESTIGATION.md` 根因写错的前车之鉴）。
人工 review 是质量门。

## 冒烟验证

`examples/quickstart_antangelmed.py` 做端到端冒烟（HF vs vLLM）。
`tests/` 下为 lib 单测；新模型定位前跑 quickstart 确认环境。
