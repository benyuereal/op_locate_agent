"""
config_loader.py — 模型配置解析

输入：模型本地目录
输出：ModelProfile（标准化配置对象）

解析顺序：
1. config.json → 基础结构 + MoE 参数
2. auto_map → 自定义 py 文件路径（HF remote code）
3. README* → 官方介绍摘要（可选，供 WebFetch 补充）

设计原则：
- 零 GPU 依赖，纯文件/JSON 解析，可在任何环境跑
- 对缺失字段优雅降级（Optional + None），不抛异常
- MoE 检测基于多字段联合判断，兼容各家命名差异
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ModelProfile:
    """标准化模型配置 — agent 全流程的起点"""

    # —— 基础 ——
    local_path: str
    arch: str                       # architectures[0]，如 "BailingMoeV2ForCausalLM"
    model_type: str                 # config.json model_type
    config_path: str                # config.json 绝对路径

    # —— 结构 ——
    num_layers: int
    hidden_size: int
    num_heads: int
    kv_heads: int
    head_size: int
    vocab_size: int
    intermediate_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None

    # —— MoE ——
    is_moe: bool = False
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None   # top_k
    num_expert_group: Optional[int] = None      # n_group
    topk_group: Optional[int] = None
    routed_scaling_factor: Optional[float] = None
    score_function: Optional[str] = None        # "sigmoid" | "softmax"
    e_score_correction_bias: Optional[str] = None  # 触发 fused_gate 的关键标志
    num_shared_experts: Optional[int] = None
    first_k_dense_replace: Optional[int] = None  # 前 N 层 dense
    moe_intermediate_size: Optional[int] = None
    moe_router_enable_expert_bias: Optional[bool] = None

    # —— 自定义 remote code ——
    auto_map: Dict[str, str] = field(default_factory=dict)
    custom_py_files: List[str] = field(default_factory=list)

    # —— dtype / 杂项 ——
    torch_dtype: str = "bfloat16"
    eos_token_id: Optional[Any] = None
    bos_token_id: Optional[Any] = None
    norm_topk_prob: Optional[bool] = None

    # —— 官方介绍（可选，由 WebFetch 填充）——
    readme_path: Optional[str] = None
    readme_summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def moe_summary(self) -> str:
        """人类可读的 MoE 配置摘要，用于 prompt 注入"""
        if not self.is_moe:
            return f"{self.arch}: dense (no MoE)"
        return (
            f"{self.arch}: MoE "
            f"E={self.num_experts} topk={self.num_experts_per_tok} "
            f"group={self.num_expert_group} topk_group={self.topk_group} "
            f"rsf={self.routed_scaling_factor} score={self.score_function} "
            f"bias={self.e_score_correction_bias is not None} "
            f"shared={self.num_shared_experts} "
            f"dense_first={self.first_k_dense_replace}"
        )


# ============================================================================
# 解析逻辑
# ============================================================================

# 不同模型库里 MoE 字段命名差异的归一化映射
_MOE_FIELD_ALIASES = {
    "num_experts": ["num_experts", "n_routed_experts", "num_local_experts"],
    "num_experts_per_tok": ["num_experts_per_tok", "top_k"],
    "num_expert_group": ["num_expert_group", "n_group", "num_expert_groups"],
    "topk_group": ["topk_group", "top_k_group", "num_expert_group_candidates"],
    "routed_scaling_factor": ["routed_scaling_factor", "router_scale"],
    "score_function": ["score_function", "scoring_func"],
    "num_shared_experts": ["num_shared_experts", "n_shared_experts"],
    # 注意：moe_layer_freq（MoE 层频率）语义不同于 first_k_dense_replace（前 N 层 dense 计数），
    # 不作为别名，避免误用。
    "first_k_dense_replace": ["first_k_dense_replace"],
    "moe_intermediate_size": ["moe_intermediate_size"],
    "moe_router_enable_expert_bias": ["moe_router_enable_expert_bias"],
}

# 触发 fused_gate 路径的关键字段名候选（bias 张量命名各家不同）
_BIAS_FIELD_CANDIDATES = [
    "e_score_correction_bias",
    "router_bias",
    "expert_bias",
]


def _get_any(cfg: Dict[str, Any], keys: List[str], default=None):
    """从 cfg 里按候选 key 列表取第一个存在的值"""
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    return default


def _resolve_bias_field(cfg: Dict[str, Any]) -> Optional[str]:
    """检测 bias 字段名。config.json 里通常只存 bool 标志（如
    moe_router_enable_expert_bias=true），真正 bias 张量在权重文件里。
    这里返回一个"是否存在 bias 机制"的标志字符串，供后续 path_resolver
    判断是否走 fused_gate 路径。"""
    # 显式字段
    for k in _BIAS_FIELD_CANDIDATES:
        if k in cfg and cfg[k] is not None:
            return k
    # 标志位：有 enable_expert_bias=True 即视为有 bias 机制
    if cfg.get("moe_router_enable_expert_bias") is True:
        return "expert_bias (via moe_router_enable_expert_bias=true)"
    return None


def _detect_is_moe(cfg: Dict[str, Any]) -> bool:
    """MoE 判定：任一 MoE 标志字段存在即视为 MoE"""
    moe_signals = [
        "num_experts", "n_routed_experts", "num_local_experts",
        "moe_intermediate_size", "num_shared_experts",
        "first_k_dense_replace", "num_expert_group", "n_group",
    ]
    return any(cfg.get(s) is not None for s in moe_signals)


def load_model_profile(model_path: str, readme_chars: int = 4000) -> ModelProfile:
    """
    从本地模型目录加载标准化 ModelProfile。

    Args:
        model_path: 模型本地目录（含 config.json）
        readme_chars: README 摘要截断长度，0 表示不读 README

    Returns:
        ModelProfile

    Raises:
        FileNotFoundError: config.json 不存在
        json.JSONDecodeError: config.json 解析失败
    """
    model_path = os.path.abspath(model_path)
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # —— 基础 ——
    architectures = cfg.get("architectures", [])
    arch = architectures[0] if architectures else cfg.get("model_type", "Unknown")
    model_type = cfg.get("model_type", "unknown")

    num_layers = cfg.get("num_hidden_layers") or cfg.get("n_layer") or 0
    hidden_size = cfg.get("hidden_size") or cfg.get("n_embd") or 0
    num_heads = cfg.get("num_attention_heads") or cfg.get("n_head") or 0
    kv_heads = cfg.get("num_key_value_heads") or num_heads
    head_size = cfg.get("head_dim") or (hidden_size // num_heads if num_heads else 0)
    vocab_size = cfg.get("vocab_size", 0)

    # —— MoE ——
    is_moe = _detect_is_moe(cfg)
    num_experts = _get_any(cfg, _MOE_FIELD_ALIASES["num_experts"])
    num_experts_per_tok = _get_any(cfg, _MOE_FIELD_ALIASES["num_experts_per_tok"])
    num_expert_group = _get_any(cfg, _MOE_FIELD_ALIASES["num_expert_group"])
    topk_group = _get_any(cfg, _MOE_FIELD_ALIASES["topk_group"])
    routed_scaling_factor = _get_any(cfg, _MOE_FIELD_ALIASES["routed_scaling_factor"])
    score_function = _get_any(cfg, _MOE_FIELD_ALIASES["score_function"])
    num_shared_experts = _get_any(cfg, _MOE_FIELD_ALIASES["num_shared_experts"])
    first_k_dense_replace = _get_any(cfg, _MOE_FIELD_ALIASES["first_k_dense_replace"])
    moe_intermediate_size = _get_any(cfg, _MOE_FIELD_ALIASES["moe_intermediate_size"])
    moe_router_enable_expert_bias = cfg.get("moe_router_enable_expert_bias")
    e_score_correction_bias = _resolve_bias_field(cfg) if is_moe else None

    # —— 自定义 remote code ——
    auto_map = cfg.get("auto_map", {}) or {}
    custom_py_files = _list_custom_py(model_path, auto_map)

    # —— README ——
    readme_path, readme_summary = _read_readme(model_path, readme_chars)

    profile = ModelProfile(
        local_path=model_path,
        arch=arch,
        model_type=model_type,
        config_path=config_path,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        kv_heads=kv_heads,
        head_size=head_size,
        vocab_size=vocab_size,
        intermediate_size=cfg.get("intermediate_size"),
        max_position_embeddings=cfg.get("max_position_embeddings"),
        is_moe=is_moe,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
        score_function=score_function,
        e_score_correction_bias=e_score_correction_bias,
        num_shared_experts=num_shared_experts,
        first_k_dense_replace=first_k_dense_replace,
        moe_intermediate_size=moe_intermediate_size,
        moe_router_enable_expert_bias=moe_router_enable_expert_bias,
        auto_map=auto_map,
        custom_py_files=custom_py_files,
        torch_dtype=cfg.get("torch_dtype", "bfloat16"),
        eos_token_id=cfg.get("eos_token_id"),
        bos_token_id=cfg.get("bos_token_id"),
        norm_topk_prob=cfg.get("norm_topk_prob"),
        readme_path=readme_path,
        readme_summary=readme_summary,
    )
    return profile


def _list_custom_py(model_path: str, auto_map: Dict[str, str]) -> List[str]:
    """列出本地自定义 py 文件（auto_map 引用的 + 顶层 modeling/configuration_*.py）"""
    files: List[str] = []
    seen = set()

    # 从 auto_map 提取文件名（取 . 前的模块名）
    for ref in auto_map.values():
        mod = ref.split(".")[0]
        if not mod:
            continue
        for ext in (".py", ""):
            cand = os.path.join(model_path, mod + ext)
            if os.path.isfile(cand) and cand not in seen:
                files.append(cand)
                seen.add(cand)

    # 兜底：顶层 modeling_*.py / configuration_*.py
    if os.path.isdir(model_path):
        for name in sorted(os.listdir(model_path)):
            if name.endswith(".py") and (
                name.startswith("modeling_") or name.startswith("configuration_")
            ):
                full = os.path.join(model_path, name)
                if full not in seen and os.path.isfile(full):
                    files.append(full)
                    seen.add(full)

    return files


def _read_readme(model_path: str, chars: int) -> tuple[Optional[str], Optional[str]]:
    """读 README（优先中文版），截取摘要"""
    if chars <= 0:
        return None, None
    for name in ("README_zh-CN.md", "README_zh.md", "README.md", "README.rst"):
        p = os.path.join(model_path, name)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                return p, None
            # 去掉 markdown 图片/链接噪音，截断
            text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
            text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
            return p, text[:chars]
    return None, None


# ============================================================================
# CLI 自检
# ============================================================================

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/path/to/model"
    p = load_model_profile(path)
    print("=== ModelProfile ===")
    print(f"arch        : {p.arch}")
    print(f"model_type  : {p.model_type}")
    print(f"is_moe      : {p.is_moe}")
    print(f"moe_summary : {p.moe_summary()}")
    print(f"auto_map    : {p.auto_map}")
    print(f"custom_py   : {p.custom_py_files}")
    print(f"bias        : {p.e_score_correction_bias}")
    print(f"readme      : {p.readme_path}")
