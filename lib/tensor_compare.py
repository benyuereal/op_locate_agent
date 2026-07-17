"""
tensor_compare.py — 张量对比器

抽自 precision_compare/core.py 的 TensorComparator，扩展：
- 通用张量对比（max_abs / mean_abs / cos / allclose / rel）
- topk 对比专用：ids 精确匹配率 + 按 id 对齐的权重对比（避免排序错位假误差）
- 阈值判定标准化

设计原则：
- torch 只在调用时需要，import 本模块不触发 GPU 初始化
- 所有对比返回 CompareResult，可序列化
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import torch


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class CompareResult:
    """通用张量对比结果"""
    stage: str
    max_abs_diff: float
    mean_abs_diff: float
    cos: float
    relative_diff: float
    is_close: bool
    shape: Tuple[int, ...] = ()
    ref_mean: float = 0.0
    ref_std: float = 0.0
    test_mean: float = 0.0
    test_std: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["shape"] = list(self.shape)
        return d

    def verdict(self, atol: float = 1e-2, cos_thresh: float = 0.999) -> str:
        """生成一句话判定"""
        if self.is_close or self.cos >= cos_thresh:
            return f"✅ {self.stage}: cos={self.cos:.6f} (一致)"
        return f"❌ {self.stage}: cos={self.cos:.6f} max_abs={self.max_abs_diff:.4e} (发散)"


@dataclass
class TopkCompareResult:
    """topk 专用对比结果"""
    ids_exact_match_rate: float        # 行级（每行 id 集合相同）匹配率
    ids_elementwise_rate: float        # 逐元素一致率（sorted 后）
    weights_max_abs_diff: float        # 按 id 对齐后的权重 max abs diff
    weights_mean_abs_diff: float
    weights_max_rel_diff: float
    weights_cos: float
    ids_match: bool                    # ids 是否 100% 一致
    weights_zero_error: bool           # 权重是否零误差（按 id 对齐）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def verdict(self) -> str:
        parts = []
        parts.append(f"ids {'✅100%一致' if self.ids_match else '❌不一致'}"
                     f"({self.ids_exact_match_rate*100:.1f}%)")
        parts.append(f"weights {'✅零误差' if self.weights_zero_error else '❌有差'}"
                     f"(max_abs={self.weights_max_abs_diff:.4e})")
        return " | ".join(parts)


# ============================================================================
# 通用张量对比
# ============================================================================

class TensorComparator:
    """
    标准张量对比器

    rtol/atol 用于 allclose 判定；cos_thresh 用于"一致"的宽松判定。
    """

    def __init__(self, rtol: float = 1e-2, atol: float = 1e-2,
                 cos_thresh: float = 0.999):
        self.rtol = rtol
        self.atol = atol
        self.cos_thresh = cos_thresh

    def compare(self, name: str, ref: torch.Tensor, test: torch.Tensor) -> CompareResult:
        """对比两个张量"""
        ref_f = ref.float().reshape(-1)
        test_f = test.float().reshape(-1)

        if ref_f.shape != test_f.shape:
            # shape 不一致，裁到较短长度做数值对比，但标记 is_close=False
            n = min(ref_f.numel(), test_f.numel())
            ref_f = ref_f[:n]
            test_f = test_f[:n]
            shape_match = False
        else:
            shape_match = True

        diff = torch.abs(ref_f - test_f)
        max_diff = diff.max().item() if diff.numel() else 0.0
        mean_diff = diff.mean().item() if diff.numel() else 0.0

        denom = torch.abs(ref_f) + torch.abs(test_f) + 1e-8
        rel_diff = (diff / denom).mean().item() if diff.numel() else 0.0

        # cosine
        dot = torch.dot(ref_f, test_f).item() if diff.numel() else 0.0
        nr = torch.norm(ref_f).item()
        nt = torch.norm(test_f).item()
        cos = dot / (nr * nt + 1e-12) if (nr > 0 and nt > 0) else 0.0

        is_close = shape_match and torch.allclose(
            ref.float(), test.float(), rtol=self.rtol, atol=self.atol
        )

        return CompareResult(
            stage=name,
            max_abs_diff=max_diff,
            mean_abs_diff=mean_diff,
            cos=cos,
            relative_diff=rel_diff,
            is_close=is_close,
            shape=tuple(ref.shape),
            ref_mean=ref_f.mean().item() if diff.numel() else 0.0,
            ref_std=ref_f.std().item() if diff.numel() else 0.0,
            test_mean=test_f.mean().item() if diff.numel() else 0.0,
            test_std=test_f.std().item() if diff.numel() else 0.0,
        )


def compare_tensors(name: str, ref: torch.Tensor, test: torch.Tensor,
                    rtol: float = 1e-2, atol: float = 1e-2) -> CompareResult:
    """一次性对比的便捷函数"""
    return TensorComparator(rtol=rtol, atol=atol).compare(name, ref, test)


# ============================================================================
# topk 专用对比
# ============================================================================

def compare_topk(
    ref_ids: torch.Tensor,
    ref_weights: torch.Tensor,
    test_ids: torch.Tensor,
    test_weights: torch.Tensor,
    atol: float = 1e-6,
) -> TopkCompareResult:
    """
    对比两组 topk 结果。

    关键：权重对比必须"按 expert id 对齐"，而不是各自 sorted 后逐位比——
    否则顺序差异会产生假误差（AntAngelMed lightop vs torch 调查中的教训）。

    Args:
        ref_ids / test_ids: (n, topk) int, 专家 id
        ref_weights / test_weights: (n, topk) float, 对应权重
        atol: 权重零误差阈值

    Returns:
        TopkCompareResult
    """
    ref_ids = ref_ids.detach().cpu().long()
    test_ids = test_ids.detach().cpu().long()
    ref_w = ref_weights.detach().cpu().float()
    test_w = test_weights.detach().cpu().float()

    n = ref_ids.shape[0]
    topk = ref_ids.shape[1]

    # —— ids 行级匹配率（用 multiset/Counter，正确处理重复 id）——
    from collections import Counter
    row_match = 0
    for i in range(n):
        if Counter(ref_ids[i].tolist()) == Counter(test_ids[i].tolist()):
            row_match += 1
    ids_exact_rate = row_match / n if n else 0.0

    # —— ids 逐元素一致率（各自 sorted 后）——
    elem_match = 0
    total = 0
    for i in range(n):
        r = sorted(ref_ids[i].tolist())
        t = sorted(test_ids[i].tolist())
        if len(r) == len(t):
            total += topk
            elem_match += sum(1 for a, b in zip(r, t) if a == b)
    ids_elem_rate = elem_match / total if total else 0.0

    # —— 权重按 id 对齐（处理重复 id：贪心配对同 id 中权重最接近的）——
    # 对每行：把 test 的 (id, weight) 按 id 分组（保留 list 支持重复），
    # 对 ref 每个 (id, weight)，在同 id 的 test 候选中取权重最接近且未配对的。
    abs_diffs = []
    rel_diffs = []
    cos_num = 0.0
    cos_r2 = 0.0
    cos_t2 = 0.0
    for i in range(n):
        # test 的 id -> [权重列表]（可消费）
        tmap: Dict[int, list] = {}
        for j, tid in enumerate(test_ids[i].tolist()):
            tmap.setdefault(int(tid), []).append(float(test_w[i, j].item()))

        for j, rid in enumerate(ref_ids[i].tolist()):
            rw = float(ref_w[i, j].item())
            cands = tmap.get(int(rid))
            if not cands:
                # ref 选的专家 test 没选 → 视为最大差异（对端权重 0）
                tw = 0.0
            else:
                # 贪心：在同 id 候选中取权重最接近 rw 的，消费掉
                best_k = min(range(len(cands)), key=lambda k: abs(cands[k] - rw))
                tw = cands.pop(best_k)
            d = abs(rw - tw)
            abs_diffs.append(d)
            denom = abs(rw) + abs(tw) + 1e-8
            rel_diffs.append(d / denom)
            cos_num += rw * tw
            cos_r2 += rw * rw
            cos_t2 += tw * tw

    max_abs = max(abs_diffs) if abs_diffs else 0.0
    mean_abs = sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0
    max_rel = max(rel_diffs) if rel_diffs else 0.0
    cos = cos_num / (math.sqrt(cos_r2) * math.sqrt(cos_t2) + 1e-12) if (cos_r2 > 0 and cos_t2 > 0) else 0.0

    return TopkCompareResult(
        ids_exact_match_rate=ids_exact_rate,
        ids_elementwise_rate=ids_elem_rate,
        weights_max_abs_diff=max_abs,
        weights_mean_abs_diff=mean_abs,
        weights_max_rel_diff=max_rel,
        weights_cos=cos,
        ids_match=ids_exact_rate >= 1.0,
        weights_zero_error=max_abs <= atol,
    )


# ============================================================================
# CLI 自检
# ============================================================================

if __name__ == "__main__":
    # 零误差 case
    a = torch.randn(64, 8)
    b = a.clone()
    r1 = compare_tensors("zero", a, b)
    print(r1.verdict())

    # 有差 case
    c = a + 0.1 * torch.randn_like(a)
    r2 = compare_tensors("diff", a, c)
    print(r2.verdict())

    # topk：ids 相同但权重顺序不同 → 必须按 id 对齐才能识别为零误差
    ids = torch.randint(0, 256, (4, 8))
    w = torch.rand(4, 8)
    # 打乱每行顺序
    perm = torch.randperm(8)
    ids2 = ids[:, perm]
    w2 = w[:, perm]
    r3 = compare_topk(ids, w, ids2, w2)
    print("topk (permuted, should be zero-error):", r3.verdict())
