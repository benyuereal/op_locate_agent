"""
hook_manager.py — 通用 Hook 管理器（带口径修正）

抽自 precision_compare/core.py 的 UniversalHookManager，并融入 INVESTIGATION.md
里总结的口径陷阱修正：

陷阱1 — in-place 污染：
    vLLM 的 MoE block 第一行 `residual = hidden_states` 让 residual 与入参
    共享内存，后续 in-place 相加会改掉入参。post-forward hook 拿到的 args[0]
    是被污染的值。
    修正：默认用 register_forward_pre_hook（forward 执行前触发）+ 立即 clone。
    对输出抓取，单独提供 post_hook 但强制 clone。

陷阱2 — CustomOp forward_hook 不可靠：
    TP 下 VocabParallelEmbedding 等 CustomOp 的 hook 抓到的可能不是真正传给
    下一层的 tensor。
    修正：提供 monkey-patch 接口（patch_forward），直接替换 forward 抓返回值，
    口径 100% 可靠。

陷阱3 — collective_rpc 持久化陷阱（vLLM v1）：
    EngineCore 子进程里 collective_rpc 用 cloudpickle 序列化每次调用，模块级
    dict 不跨调用持久化。必须把抓到的张量存在 worker.model_runner.model 这个
    长生命周期对象上。
    修正：本管理器把捕获结果存在传入的 owner 对象上（而非模块级变量），
    由调用方保证 owner 长生命周期。见 CaptureSpec.owner。

设计原则：
- import 本模块不触发 GPU / 不读环境变量
- pre_hook 优先；输出抓取显式 clone
- 支持 forward_hook / forward_pre_hook / monkey-patch 三种口径
- 容错：单个 hook 注册失败不影响其他
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

logger = logging.getLogger("op_locate.hook")


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class HookPoint:
    """单个 hook 点定义"""
    name: str               # 中间结果名称，如 "layer1.mlp_out"
    module_path: str        # 点分隔路径，如 "model.layers.1.mlp"
    kind: str = "pre"       # "pre" | "post" | "patch"
    output_index: int = 0   # 若模块返回 tuple，取第几个


@dataclass
class CaptureSpec:
    """一次捕获的完整规格"""
    hook_points: List[HookPoint]
    owner: Any = None       # 捕获结果挂在哪（长生命周期对象，避免 collective_rpc 陷阱）
    clone: bool = True      # 是否强制 clone（默认 True，防 in-place 污染）


# ============================================================================
# Hook 管理器
# ============================================================================

class HookManager:
    """
    通用 Hook 管理器

    用法：
        hm = HookManager()
        spec = CaptureSpec(hook_points=[...], owner=model)
        with hm.capture(model, spec):
            model(input_ids)
        intermediates = hm.get_intermediates()  # dict[name -> tensor]
    """

    def __init__(self):
        self._hooks: List[Any] = []
        self._patches: List[tuple] = []          # (module, orig_forward) 用于恢复 monkey-patch
        self._intermediates: Dict[str, torch.Tensor] = {}
        self._owner: Any = None
        self._clone: bool = True

    @contextmanager
    def capture(self, model: torch.nn.Module, spec: CaptureSpec):
        """上下文：注册 hook → yield → 清理"""
        try:
            self.register(model, spec)
            yield self
        finally:
            self.remove()

    def register(self, model: torch.nn.Module, spec: CaptureSpec) -> None:
        """注册所有 hook"""
        self.clear()
        self._owner = spec.owner if spec.owner is not None else self
        self._clone = spec.clone

        ok = 0
        for hp in spec.hook_points:
            try:
                module = self._get_module_by_path(model, hp.module_path)
                if hp.kind == "patch":
                    self._register_patch(module, hp)
                elif hp.kind == "pre":
                    h = module.register_forward_pre_hook(self._make_pre_hook(hp))
                    self._hooks.append(h)
                else:  # "post"
                    h = module.register_forward_hook(self._make_post_hook(hp))
                    self._hooks.append(h)
                ok += 1
            except Exception as e:
                logger.warning(f"注册 hook 失败 {hp.name} -> {hp.module_path}: {e}")

        logger.info(f"已注册 {ok}/{len(spec.hook_points)} 个 hook")

    def _register_patch(self, module: torch.nn.Module, hp: HookPoint) -> None:
        """monkey-patch forward，抓返回值（CustomOp 可靠口径）"""
        orig = module.forward

        def patched(*args, **kwargs):
            out = orig(*args, **kwargs)
            try:
                tensor = out[hp.output_index] if isinstance(out, tuple) else out
                self._store(hp.name, tensor)
            except Exception as e:
                logger.warning(f"patch hook {hp.name} 抓取失败: {e}")
            return out

        module.forward = patched
        self._patches.append((module, orig))

    def _make_pre_hook(self, hp: HookPoint) -> Callable:
        """pre_hook：抓输入（防 in-place 污染，forward 前触发）"""
        def hook(_module, args):
            try:
                tensor = args[0] if args else None
                if tensor is None and len(args) > 1:
                    tensor = args[hp.output_index]
                self._store(hp.name, tensor)
            except Exception as e:
                logger.warning(f"pre hook {hp.name} 抓取失败: {e}")
        return hook

    def _make_post_hook(self, hp: HookPoint) -> Callable:
        """post_hook：抓输出（强制 clone）"""
        def hook(_module, _args, output):
            try:
                tensor = output[hp.output_index] if isinstance(output, tuple) else output
                self._store(hp.name, tensor)
            except Exception as e:
                logger.warning(f"post hook {hp.name} 抓取失败: {e}")
        return hook

    def _store(self, name: str, tensor: Any) -> None:
        """存储捕获结果（强制 clone，防 in-place）"""
        if tensor is None or not torch.is_tensor(tensor):
            return
        t = tensor.detach().clone() if self._clone else tensor.detach()
        self._intermediates[name] = t
        # 同时挂到 owner（防 collective_rpc 陷阱）
        if self._owner is not None and self._owner is not self:
            store = getattr(self._owner, "_captured_intermediates", None)
            if not isinstance(store, dict):
                store = {}
                setattr(self._owner, "_captured_intermediates", store)
            store[name] = t

    @staticmethod
    def _get_module_by_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
        """点分隔路径取模块，支持数字索引（如 layers.1）"""
        module = model
        for part in path.split("."):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def remove(self) -> None:
        """移除所有 hook 和 patch"""
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
        for module, orig in self._patches:
            try:
                module.forward = orig
            except Exception:
                pass
        self._patches.clear()

    def clear(self) -> None:
        """清空捕获结果（不动 hook）"""
        self._intermediates.clear()
        if self._owner is not None and self._owner is not self:
            store = getattr(self._owner, "_captured_intermediates", None)
            if isinstance(store, dict):
                store.clear()

    def get_intermediates(self) -> Dict[str, torch.Tensor]:
        """返回捕获结果"""
        return dict(self._intermediates)


# ============================================================================
# 标准模板：常见架构的 hook 点生成
# ============================================================================

def transformer_hook_points(
    num_layers: int,
    layer_prefix: str = "model.layers",
    mlp_attr: str = "mlp",
    attn_attr: str = "self_attn",
    input_norm_attr: str = "input_layernorm",
    post_attn_norm_attr: str = "post_attention_layernorm",
    sample_layers: Optional[List[int]] = None,
) -> List[HookPoint]:
    """
    生成标准 transformer 逐层 hook 点。

    默认抓每层的：层输入(pre)、attn 输出(post)、mlp 输出(post)、层输出(pre of next)。
    sample_layers 指定时只抓指定层（省显存），None=全层。
    """
    layers = sample_layers if sample_layers is not None else list(range(num_layers))
    pts: List[HookPoint] = []
    for i in layers:
        base = f"{layer_prefix}.{i}"
        pts.append(HookPoint(f"layer{i}_in", base, kind="pre"))
        pts.append(HookPoint(f"layer{i}_attn_out", f"{base}.{attn_attr}", kind="post"))
        pts.append(HookPoint(f"layer{i}_mlp_out", f"{base}.{mlp_attr}", kind="post"))
        # norm hook 点（如 rmsnorm / layernorm 精度问题排查）
        if input_norm_attr:
            pts.append(HookPoint(f"layer{i}_input_norm_out",
                        f"{base}.{input_norm_attr}", kind="post"))
        if post_attn_norm_attr:
            pts.append(HookPoint(f"layer{i}_post_attn_norm_out",
                        f"{base}.{post_attn_norm_attr}", kind="post"))
    # 层间衔接：每层输出 = 下一层输入(pre)，由下一个 layer{i+1}_in 覆盖
    return pts


def generic_hook_point(
    layer_idx: int, attr_path: str, *, layer_prefix: str = "model.layers",
    kind: str = "post", label: Optional[str] = None,
) -> HookPoint:
    """按任意属性路径生成单个 hook 点，不限于预定义类别。

    用法：
        generic_hook_point(1, "rmsnorm", kind="post")
        generic_hook_point(1, "attention.flash_attn", kind="pre")
    """
    full_path = f"{layer_prefix}.{layer_idx}.{attr_path}"
    name = label or f"layer{layer_idx}_{attr_path.replace('.', '_')}_{kind}"
    return HookPoint(name, full_path, kind=kind)


def moe_router_hook_points(
    num_layers: int,
    first_moe_layer: int,
    layer_prefix: str = "model.layers",
    router_path_tmpl: Optional[str] = None,
    sample_layers: Optional[List[int]] = None,
) -> List[HookPoint]:
    """
    MoE router 专用 hook 点。

    router_path_tmpl 是相对模型的 router 模块路径模板，{base} 为层前缀+层号。
    不同架构 router 位置不同（mlp.gate / mlp.experts.router / ...），**应来自
    model_probe 的探测结果**而非硬编码。默认 None 时回退 mlp.experts.router
    并发警告——仅作占位，不保证对具体模型正确。

    注意：pre-hook 抓的是该模块的输入，post-hook 抓的是输出。若 router 模块
    的输出不是最终 topk（如 BailingMoeV2 的 gate 只产 logits，topk 在后续
    逻辑里），需用 patch 模式或挂到真正产 topk 的位置——见 compare_layers.py。

    first_k_dense_replace 之前的层是 dense，跳过。
    """
    if router_path_tmpl is None:
        logger.warning("moe_router_hook_points 未传 router_path_tmpl，回退 mlp.experts.router；"
                       "应传 model_probe 探测结果，否则可能挂错位置")
        router_path_tmpl = "{base}.mlp.experts.router"
    layers = sample_layers if sample_layers is not None else list(range(first_moe_layer, num_layers))
    pts: List[HookPoint] = []
    for i in layers:
        base = f"{layer_prefix}.{i}"
        rp = router_path_tmpl.format(base=base)
        pts.append(HookPoint(f"layer{i}_router_logits", rp, kind="pre"))
        pts.append(HookPoint(f"layer{i}_topk", rp, kind="post"))
    return pts
