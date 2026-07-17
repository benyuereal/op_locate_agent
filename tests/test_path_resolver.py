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

    def test_fused_gate_op_is_structural(self, code_paths):
        """path_resolver 对 fused_gate 只标注结构性事实（算子+触发条件），
        不预置具体 bug 结论——避免对任意 sigmoid+bias 模型先入为主。
        具体 known_issue/fix 由知识库提供，运行时探针验证。"""
        op = next(k for k in code_paths.key_ops if k.name == "ops.moe_fused_gate")
        assert op.trigger_condition is not None
        assert "use_fused_gate" in op.trigger_condition
        # 结构性推断不应硬编码具体 bug 结论
        assert op.known_issue is None
        assert op.fix is None

    def test_key_ops_includes_moe_forward(self, code_paths):
        names = [k.name for k in code_paths.key_ops]
        assert any("FusedMoE.forward" in n for n in names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
