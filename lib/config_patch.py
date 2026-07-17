"""
config_patch.py — 模型 config 缺失字段补丁

某些模型的 modeling 代码读取 config.json 未定义的字段（如 MTP/NextN 相关），
直接 from_pretrained 会 AttributeError。本模块在加载前补默认值。

当前覆盖：
- BailingMoeV2: num_nextn_predict_layers 等 MTP 字段

设计：按 model_type 分派补丁，未知 model_type 不动（不越权改 config）。
"""

from __future__ import annotations

from typing import Any, Dict


# 按 model_type 的补丁默认值
_PATCH_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "bailing_moe_v2": {
        "add_router_probs": False,
        "mtp_loss_scaling_factor": 0.0,
        "num_nextn_predict_layers": 0,
        # n_layers / n_positions 需运行时从 config 推导，见 patch_config
    },
    "bailing_moe": {
        "add_router_probs": False,
        "mtp_loss_scaling_factor": 0.0,
        "num_nextn_predict_layers": 0,
    },
}


def patch_config(config) -> list:
    """
    就地补齐 config 缺失字段。返回实际补上的字段名列表。

    Args:
        config: transformers AutoConfig 实例

    Returns:
        list[str]: 实际补上的字段名（用于日志）
    """
    model_type = getattr(config, "model_type", "")
    patched = []

    defaults = dict(_PATCH_DEFAULTS.get(model_type, {}))

    # 运行时推导的字段
    if model_type in ("bailing_moe_v2", "bailing_moe"):
        defaults["n_layers"] = getattr(config, "num_hidden_layers", 0)
        defaults["n_positions"] = getattr(config, "max_position_embeddings", 32768)

    for k, v in defaults.items():
        if not hasattr(config, k):
            try:
                setattr(config, k, v)
                patched.append(k)
            except Exception:
                pass

    return patched


def needs_patch(model_type: str) -> bool:
    """该 model_type 是否有已知 config 补丁"""
    return model_type in _PATCH_DEFAULTS


if __name__ == "__main__":
    # 演示：加载 AntAngelMed config 并打补丁
    import sys
    from transformers import AutoConfig
    path = sys.argv[1] if len(sys.argv) > 1 else "/models/AntAngelMed"
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    print(f"model_type: {cfg.model_type}")
    print(f"needs_patch: {needs_patch(cfg.model_type)}")
    patched = patch_config(cfg)
    print(f"patched fields: {patched}")
    print(f"num_nextn_predict_layers = {cfg.num_nextn_predict_layers}")
