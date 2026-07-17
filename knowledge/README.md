# Knowledge Index — 算子定位知识库

> agent 的"脑"。分层 markdown 索引，Claude 读索引定位，**不把源码塞进上下文**。
> 每条记录都带"验证来源"——避免错误结论污染（参考 `INVESTIGATION.md` 根因写错的前车之鉴）。

## 索引文件

| 文件 | 内容 | 何时查 |
|---|---|---|
| `arch_index.md` | 架构 → vLLM 代码路径映射（按 model_type / architectures） | 拿到 ModelProfile 后第一查 |
| `vllm_forward_paths.md` | vLLM 前向核心路径速查（按模型族：MoE / Dense / Attention） | 规划 hook 点时查 |
| `moe_known_issues.md` | MoE 已知坑（fused_gate / grouped_topk / aiter / lightop） | MoE 模型必查；定位到 router 后深查 |
| `hook_pitfalls.md` | hook 抓取口径陷阱（in-place / collective_rpc / CustomOp） | 写 hook 前必查 |
| `platforms/hygon_dcu.md` | Hygon DCU 家族（gfx936/gfx938）平台特性与坑 | DCU 环境必查 |

## 条目格式约定

每条知识记录包含：

```markdown
## <算子/现象名>
- **文件**: <相对 vllm 包的路径>:<行号>
- **触发条件**: <何时被调用 / 何时复现>
- **平台**: <gfx936 / cuda / all>
- **症状**: <表现>
- **根因**: <已确认的原因，或"未深挖">
- **修复**: <已验证的修复方法>
- **验证**: <模型, 日期, 报告路径>
- **状态**: 已确认 / 已证伪 / 推测
```

## 回写规则

- **不自动回写**。每次定位完，agent 生成报告，由人工 review 后再追加到本知识库。
- 原因：`INVESTIGATION.md` 曾把根因错记为 `ops.grouped_topk`，运行时探针才证伪。
  自动回写会让错误结论扩散。人工 review 是质量门。
- 回写时必须附"验证来源"（报告路径 + 日期 + 状态）。
