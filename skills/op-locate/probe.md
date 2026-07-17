# 子流程：probe — hook 注入与中间态抓取

> SKILL.md 阶段 4-5 的细化规范。写 hook 前必读 `knowledge/hook_pitfalls.md`。

## 原则

1. **pre_hook 优先**：抓层输入/模块输入用 `kind="pre"`，防 in-place 污染（陷阱 1）。
2. **CustomOp 用 patch**：`VocabParallelEmbedding`、router 等用 `kind="patch"`，hook 不可靠（陷阱 2）。
3. **owner 存储**：`CaptureSpec(owner=model)`，防 collective_rpc 不持久化（陷阱 3）。
4. **抓实际运行时**：不用重算值代替 vLLM 实际中间态（陷阱 4）。
5. **强制 clone**：默认 `clone=True`，捕获的 tensor 不被后续 in-place 改动。

## 标准 hook 点生成

```python
from lib import HookManager, CaptureSpec
from lib.hook_manager import transformer_hook_points, moe_router_hook_points

# dense 逐层
pts = transformer_hook_points(
    num_layers=profile.num_layers,
    layer_prefix="model.layers",   # vLLM 默认；HF 也用这个
    sample_layers=[0, 1, 2, 8, 16, 24, 31],  # 抽样省显存，或 None 全层
)

# MoE router（first_k_dense_replace 之后）
if profile.is_moe:
    rpts = moe_router_hook_points(
        num_layers=profile.num_layers,
        first_moe_layer=profile.first_k_dense_replace or 0,
        sample_layers=[1, 2, 8],   # 先抓第一个 MoE 层附近
    )
```

## 抓取模板

```python
hm = HookManager()
spec = CaptureSpec(hook_points=pts, owner=model, clone=True)
with hm.capture(model, spec):
    out = model(input_ids)
intermediates = hm.get_intermediates()  # dict[name -> tensor]
```

## vLLM 特殊：EngineCore 子进程

vLLM v1 的 EngineCore 是子进程，要在 worker 上抓，需用 `collective_rpc` 把抓取函数发到 worker 执行。
**关键**：抓取结果存 `worker.model_runner.model._captured_intermediates`（owner 机制），不要用模块级变量。

简化方案（调试用）：用 `LLM` 离线推理 + `enforce_eager=True`，直接在主进程的 model 上挂 hook。

## TP 多卡口径

- TP>1 时在 **worker 0 (rank 0)** 抓，口径是 all_reduce 后的完整向量（陷阱 6）。
- 对比 HF 时注意 HF 是单卡完整口径。
- 调试优先用 TP=1（若显存允许），避免 TP 口径复杂度。

## 抓什么（按阶段）

| 阶段 | 抓什么 | hook 点 |
|---|---|---|
| 粗粒度逐层 | 每层输入 + mlp_out + attn_out | `transformer_hook_points` |
| MoE 层内 A-F | input_layernorm/attn/post_attn_norm/mlp/layer_out | 见 `vllm_forward_paths.md` MoE 层内拆解 |
| router 拆解 | router_logits(输入) + topk(输出) | `moe_router_hook_points`，topk 用 patch |

## 常见错误

- ❌ 用 post_hook 抓层输入 → in-place 污染假误差（陷阱 1）
- ❌ 用 hook 抓 CustomOp 输出 → 口径不对（陷阱 2）
- ❌ 模块级 dict 存结果 → collective_rpc 丢失（陷阱 3）
- ❌ 用 `grouped_topk` 重算代替实际 topk → 误导（陷阱 4）
- ❌ 不 clone → 捕获后被 in-place 改掉
