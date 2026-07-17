"""
test_tensor_compare.py — tensor_compare 单测

重点验证 topk 按 id 对齐逻辑（lightop vs torch 调查的教训）：
顺序不同但 id 集合相同 → 必须识别为零误差，不能报假误差。
"""

import math
import torch
import pytest

from lib import compare_tensors, compare_topk, TensorComparator


class TestTensorCompare:

    def test_identical_tensors(self):
        a = torch.randn(32, 8)
        r = compare_tensors("x", a, a)
        assert r.is_close is True
        assert r.cos == pytest.approx(1.0, abs=1e-5)
        assert r.max_abs_diff == pytest.approx(0.0, abs=1e-7)

    def test_different_tensors(self):
        a = torch.randn(32, 8)
        b = a + 0.5 * torch.randn_like(a)
        r = compare_tensors("x", a, b)
        assert r.is_close is False
        assert r.cos < 0.999
        assert r.max_abs_diff > 0.01

    def test_shape_mismatch_not_close(self):
        a = torch.randn(32, 8)
        b = torch.randn(16, 8)
        r = compare_tensors("x", a, b)
        assert r.is_close is False


class TestTopkCompare:

    def test_identical_topk_zero_error(self):
        ids = torch.randint(0, 256, (64, 8))
        w = torch.rand(64, 8)
        r = compare_topk(ids, w, ids, w)
        assert r.ids_match is True
        assert r.weights_zero_error is True
        assert r.ids_exact_match_rate == 1.0

    def test_permuted_order_zero_error(self):
        """关键测试：顺序打乱但 id 集合相同 → 必须零误差（按 id 对齐）"""
        torch.manual_seed(0)
        ids = torch.randint(0, 256, (64, 8))
        w = torch.rand(64, 8)
        # 每行用相同 permutation 打乱 ids 和 weights
        ids2 = ids.clone()
        w2 = w.clone()
        for i in range(64):
            perm = torch.randperm(8)
            ids2[i] = ids[i, perm]
            w2[i] = w[i, perm]
        r = compare_topk(ids, w, ids2, w2)
        assert r.ids_match is True
        assert r.weights_zero_error is True, \
            f"permuted should be zero-error, got max_abs={r.weights_max_abs_diff}"

    def test_different_ids_detected(self):
        ids1 = torch.zeros(4, 8, dtype=torch.long)
        ids2 = torch.ones(4, 8, dtype=torch.long)
        w = torch.rand(4, 8)
        r = compare_topk(ids1, w, ids2, w)
        assert r.ids_match is False
        assert r.ids_exact_match_rate == 0.0

    def test_same_ids_different_weights(self):
        """ids 相同但权重不同 → ids_match=True, weights_zero_error=False"""
        ids = torch.randint(0, 256, (64, 8))
        w1 = torch.rand(64, 8)
        w2 = w1 + 0.3 * torch.rand(64, 8)
        r = compare_topk(ids, w1, ids, w2)
        assert r.ids_match is True
        assert r.weights_zero_error is False
        assert r.weights_max_abs_diff > 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
