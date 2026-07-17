"""
test_hook_manager.py — hook_manager 单测

用一个迷你 torch 模型验证三种口径（pre/post/patch）+ in-place 污染防护 +
owner 存储（collective_rpc 陷阱防护）。不需要 GPU，纯 CPU torch。
"""

import os
import torch
import torch.nn as nn
import pytest

from lib import HookManager, HookPoint, CaptureSpec


class TinyBlock(nn.Module):
    """模拟 vLLM MoE block 的 in-place 污染陷阱：
    forward 第一行 residual = x（共享内存），后续 in-place 相加会改掉 x。"""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        residual = x              # 共享内存
        h = self.lin(x)
        residual.add_(h)         # in-place！污染入参 x
        return residual


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock(), TinyBlock()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TestHookManager:

    def test_pre_hook_captures_clean_input(self):
        """pre_hook 抓到的是 in-place 污染前的干净输入"""
        model = TinyModel()
        hm = HookManager()
        spec = CaptureSpec(hook_points=[
            HookPoint("layer0_in", "layers.0", kind="pre"),
        ], owner=model)
        x = torch.ones(1, 4)
        with hm.capture(model, spec):
            model(x)
        inter = hm.get_intermediates()
        assert "layer0_in" in inter
        # 干净输入应全 1（in-place 污染后会变）
        assert torch.allclose(inter["layer0_in"], torch.ones(1, 4))

    def test_post_hook_captures_output(self):
        model = TinyModel()
        hm = HookManager()
        spec = CaptureSpec(hook_points=[
            HookPoint("layer0_out", "layers.0", kind="post"),
        ], owner=model)
        x = torch.ones(1, 4)
        with hm.capture(model, spec):
            out = model(x)
        inter = hm.get_intermediates()
        assert "layer0_out" in inter
        # post 抓的应等于模型实际输出（第一层）
        assert inter["layer0_out"].shape == (1, 4)

    def test_patch_hook_reliable(self):
        """monkey-patch forward 抓返回值（CustomOp 可靠口径）"""
        model = TinyModel()
        hm = HookManager()
        spec = CaptureSpec(hook_points=[
            HookPoint("layer0_patched", "layers.0", kind="patch"),
        ], owner=model)
        x = torch.ones(1, 4)
        with hm.capture(model, spec):
            model(x)
        inter = hm.get_intermediates()
        assert "layer0_patched" in inter
        assert inter["layer0_patched"].shape == (1, 4)

    def test_owner_storage(self):
        """捕获结果必须同时挂到 owner（防 collective_rpc 模块级 dict 不持久化）"""
        model = TinyModel()
        hm = HookManager()
        spec = CaptureSpec(hook_points=[
            HookPoint("layer0_in", "layers.0", kind="pre"),
        ], owner=model)
        x = torch.ones(1, 4)
        with hm.capture(model, spec):
            model(x)
        owner_store = getattr(model, "_captured_intermediates", None)
        assert isinstance(owner_store, dict)
        assert "layer0_in" in owner_store

    def test_clone_prevents_inplace(self):
        """clone=True 时，捕获的 tensor 不被后续 in-place 改动"""
        model = TinyModel()
        hm = HookManager()
        spec = CaptureSpec(hook_points=[
            HookPoint("layer0_in", "layers.0", kind="pre"),
        ], owner=model, clone=True)
        x = torch.ones(1, 4)
        with hm.capture(model, spec):
            model(x)
        captured = hm.get_intermediates()["layer0_in"]
        # 捕获后 captured 不应被污染
        assert torch.allclose(captured, torch.ones(1, 4))

    def test_hooks_removed_after_context(self):
        model = TinyModel()
        hm = HookManager()
        spec = CaptureSpec(hook_points=[
            HookPoint("layer0_in", "layers.0", kind="pre"),
        ], owner=model)
        x = torch.ones(1, 4)
        with hm.capture(model, spec):
            model(x)
        # 退出 context 后，再跑不应再捕获
        hm.clear()
        model(x)
        assert len(hm.get_intermediates()) == 0

    def test_patch_restored_after_context(self):
        """monkey-patch 必须在退出后恢复：再跑不应再捕获"""
        model = TinyModel()
        hm = HookManager()
        spec = CaptureSpec(hook_points=[
            HookPoint("x", "layers.0", kind="patch"),
        ], owner=model)
        with hm.capture(model, spec):
            model(torch.ones(1, 4))
        assert "x" in hm.get_intermediates()
        # 退出 context 后 patch 已恢复，清空后再跑不应再捕获
        hm.clear()
        model(torch.ones(1, 4))
        assert "x" not in hm.get_intermediates()
        assert len(hm.get_intermediates()) == 0

    def test_invalid_path_warns_not_raises(self):
        """单个 hook 路径错误不应抛异常，只 warn"""
        model = TinyModel()
        hm = HookManager()
        spec = CaptureSpec(hook_points=[
            HookPoint("good", "layers.0", kind="pre"),
            HookPoint("bad", "layers.99.nonexistent", kind="pre"),
        ], owner=model)
        with hm.capture(model, spec):
            model(torch.ones(1, 4))
        inter = hm.get_intermediates()
        assert "good" in inter
        assert "bad" not in inter


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
