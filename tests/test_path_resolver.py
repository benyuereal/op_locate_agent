"""
test_path_resolver.py — path_resolver 单测

验证 vLLM/HF 代码路径定位正确性 + 关键算子推断。
"""

import os
import pytest

from lib import resolve_code_paths


class TestPathResolver:

    def test_vllm_model_file_located(self, code_paths):
        assert code_paths.vllm_model_file is not None
        assert os.path.isfile(code_paths.vllm_model_file)
        assert code_paths.vllm_model_file.endswith("bailing_moe.py")

    def test_moe_layer_located(self, code_paths):
        assert code_paths.vllm_moe_layer is not None
        assert os.path.isfile(code_paths.vllm_moe_layer)
        assert code_paths.vllm_moe_layer.endswith("fused_moe/layer.py")

    def test_router_dir_located(self, code_paths):
        assert os.path.isdir(code_paths.vllm_router_dir)

    def test_custom_ops_located(self, code_paths):
        assert os.path.isfile(code_paths.vllm_custom_ops)

    def test_hf_modeling_located(self, code_paths):
        assert code_paths.hf_modeling is not None
        assert os.path.isfile(code_paths.hf_modeling)
        assert code_paths.hf_modeling.endswith("modeling_bailing_moe_v2.py")

    def test_key_ops_includes_fused_gate(self, code_paths):
        """sigmoid + bias 必须推断出 ops.moe_fused_gate"""
        names = [k.name for k in code_paths.key_ops]
        assert "ops.moe_fused_gate" in names

    def test_fused_gate_has_known_issue_and_fix(self, code_paths):
        """已知问题与修复方法必须带上（来自 AntAngelMed 实战）"""
        op = next(k for k in code_paths.key_ops if k.name == "ops.moe_fused_gate")
        assert op.known_issue is not None
        assert "NULL" in op.known_issue or "选错专家" in op.known_issue
        assert op.fix is not None
        assert "VLLM_ENABLE_MOE_FUSED_GATE=0" in op.fix

    def test_key_ops_includes_moe_forward(self, code_paths):
        names = [k.name for k in code_paths.key_ops]
        assert any("FusedMoE.forward" in n for n in names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
