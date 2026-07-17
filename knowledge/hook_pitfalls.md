# Hook Pitfalls — hook 抓取口径陷阱

> 写 hook 前必查。这些陷阱都来自 AntAngelMed 实战，每一个都曾导致**假误差**或**口径错误**。
> `lib/hook_manager.py` 已对前三个做了默认防护。

---

## 陷阱 1：in-place 污染（最常见）

**现象**：post-forward hook 抓到的输入 `args[0]` 是被污染的值，cos 偏低（如 0.65），但上下文之间明明没有操作。

**根因**：vLLM 的 `BailingMoeBlock.forward`（及多数 transformer block）第一行 `residual = hidden_states`，让 `residual` 与入参 `args[0]` **共享内存**。后续 `residual.add_(...)` 是 in-place，会改掉 `args[0]`。post-forward hook 触发时，`args[0]` 已被污染。

**修复**：
- 抓层输入用 **`register_forward_pre_hook`**（forward 执行前触发，此时入参未被污染）。
- `lib/hook_manager.HookPoint(kind="pre")` 即此模式，且强制 `clone`。
- 抓输出用 post_hook 也要 `clone`（输出可能被下游 in-place 改）。

**验证**：AntAngelMed `layer0_in` 用 post_hook 抓 cos=0.646（假象），改 pre_hook 后 cos=0.999999（真实）。
`precision_compare/test_emb_l0_resolve.py`。

---

## 陷阱 2：CustomOp 的 forward_hook 不可靠

**现象**：`VocabParallelEmbedding` 等 CustomOp 注册的 forward_hook 抓到的 tensor，不是真正传给下一层的值（TP 下尤其严重）。

**根因**：CustomOp 的 forward 路径有 all_reduce / 内部派发，hook 挂载点与实际数据流不一致。

**修复**：对 CustomOp 用 **monkey-patch forward**，直接替换 forward 函数抓返回值。
- `lib/hook_manager.HookPoint(kind="patch")` 即此模式。
- 上下文退出自动恢复原 forward。

**验证**：AntAngelMed embedding，hook 抓的与 HF 对不上，monkey-patch `embed_input_ids` 后 cos=0.999999。
`precision_compare/test_emb_l0_resolve.py`。

---

## 陷阱 3：collective_rpc 持久化陷阱（vLLM v1）

**现象**：vLLM v1 的 EngineCore 是子进程，通过 `collective_rpc` 调用 worker 上的函数。每次调用用 cloudpickle 序列化，**模块级 dict / 全局变量不跨调用持久化**——抓到的张量下次取就没了。

**根因**：cloudpickle 序列化的是函数+闭包，不是 worker 进程的模块状态。模块级变量每次调用都重新初始化。

**修复**：把抓到的张量存在 **长生命周期对象**上（`worker.model_runner.model`），而非模块级变量。
- `lib/hook_manager.CaptureSpec(owner=...)`：捕获结果同时挂到 `owner._captured_intermediates`。
- 调用方传 `owner=model`（model 是长生命周期）。

**验证**：AntAngelMed 调查中 collective_rpc 抓取丢失，改存 model 对象后稳定。

---

## 陷阱 4：用"重算"代替"实际运行时"

**现象**：对比 topk 时，用纯 Python `grouped_topk` **重算**的结果和 HF 一致（100%），就以为 vLLM 选专家正确。但实际 vLLM 运行时用的是 fused kernel，选的专家和重算结果**完全不同**。

**根因**：重算用的是参考实现，不是 vLLM 实际调度的算子。两者可能走不同代码路径。

**修复**：必须抓 **vLLM 实际运行时**的中间态（monkey-patch `select_experts` 抓其真实返回值），不能用重算值代替。

**验证**：AntAngelMed 调查关键转折——重算 topk 100% 匹配（误导），抓实际 topk 0/32 匹配（真因）。
`precision_compare/test_moe_internals.py`。

---

## 陷阱 5：topk 权重对比的排序错位

**现象**：两组 topk 权重，各自 sorted 后逐位对比，出现 0.29 的差，以为是算法差异。

**根因**：两组选的专家 id 集合相同但**顺序不同**，sorted 后权重按各自 id 排序，位置错配产生假误差。

**修复**：权重对比必须**按 expert id 对齐**——对每个 ref 的 (id, weight)，在 test 里找同 id 的权重配对。处理重复 id 用 multiset + 贪心配对（取权重最接近的）。
- `lib/tensor_compare.compare_topk` 已实现。

**验证**：AntAngelMed lightop vs torch，按 id 对齐后仍 max_abs=0.294 → 证明是真实算法差，非排序假象。
`precision_compare/test_lightop_vs_torch_min.py`。

---

## 陷阱 6：TP 下 worker 0 的口径

**现象**：TP>1 时，不同 worker 看到的中间态不同。worker 0 的 embedding 输出是 all_reduce 后的完整向量，其他 worker 不是。

**修复**：抓中间态要在 **worker 0**（rank 0）上抓，或用 all_reduce 后的口径。对比 HF 时注意 HF 是单卡完整口径。

**验证**：AntAngelMed TP=4 调查中确认 worker 0 口径。
