"""
test_config_loader.py — config_loader 单测

用 AntAngelMed 做真实 fixture，验证 MoE 参数解析正确性。
"""

import os
import pytest

from lib import load_model_profile


class TestConfigLoader:
    """config_loader 单测"""

    def test_loads_without_error(self, model_path):
        p = load_model_profile(model_path)
        assert p is not None

    def test_basic_fields(self, profile):
        assert profile.arch == "BailingMoeV2ForCausalLM"
        assert profile.model_type == "bailing_moe_v2"
        assert profile.num_layers == 32
        assert profile.hidden_size == 4096
        assert profile.num_heads == 32
        assert profile.kv_heads == 4
        assert profile.head_size == 128
        assert profile.vocab_size == 157184

    def test_moe_detected(self, profile):
        assert profile.is_moe is True
        assert profile.num_experts == 256
        assert profile.num_experts_per_tok == 8     # top_k
        assert profile.num_expert_group == 8        # n_group
        assert profile.topk_group == 4
        assert profile.routed_scaling_factor == 2.5
        assert profile.score_function == "sigmoid"
        assert profile.num_shared_experts == 1
        assert profile.first_k_dense_replace == 1   # 前 1 层 dense

    def test_bias_mechanism_detected(self, profile):
        """关键：bias 机制必须被检测到，否则 path_resolver 不会推断 fused_gate 路径"""
        assert profile.e_score_correction_bias is not None
        assert "expert_bias" in profile.e_score_correction_bias

    def test_auto_map_and_custom_py(self, profile):
        assert "AutoModelForCausalLM" in profile.auto_map
        assert any("modeling_bailing_moe_v2" in f for f in profile.custom_py_files)
        assert any("configuration_bailing_moe_v2" in f for f in profile.custom_py_files)

    def test_custom_py_files_exist(self, profile):
        for f in profile.custom_py_files:
            assert os.path.isfile(f), f"{f} not found"

    def test_readme_loaded(self, profile):
        assert profile.readme_path is not None
        assert os.path.isfile(profile.readme_path)
        assert profile.readme_summary is not None
        assert len(profile.readme_summary) > 0

    def test_moe_summary_string(self, profile):
        s = profile.moe_summary()
        assert "MoE" in s
        assert "E=256" in s
        assert "topk=8" in s

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model_profile(str(tmp_path))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
