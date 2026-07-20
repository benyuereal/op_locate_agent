"""
path_resolver.py — 模型配置 → vLLM/HF 实际代码路径

输入：ModelProfile
输出：CodePaths（vLLM 模型文件、MoE layer、router、HF modeling、关键算子入口）

解析策略：
1. vLLM 模型文件：model_type → 文件名映射（含已知别名）+ 文件存在性校验
2. MoE 路径：固定（vLLM 的 fused_moe 结构稳定）
3. 关键算子入口：查 knowledge/precision_known_issues.md（先尝试程序化匹配，
   匹配不到返回空列表，由 agent 读 markdown 补充）
4. HF modeling：ModelProfile.custom_py_files 里的 modeling_*.py

设计原则：
- 不依赖知识库也能给出基础路径（程序化推断）
- 知识库条目作为"已知坑"叠加，缺失时不阻塞
- 所有路径返回前校验存在性，不存在的标注 None
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .config_loader import ModelProfile


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class KeyOp:
    """关键算子入口（来自知识库或程序化推断）"""
    name: str                 # 如 "ops.moe_fused_gate"
    file: str                 # 相对 vllm 包的路径
    line: Optional[int] = None
    trigger_condition: str = ""   # 何时被调用
    known_issue: Optional[str] = None  # 已知问题简述
    fix: Optional[str] = None       # 已知修复


@dataclass
class CodePaths:
    """模型相关的全部代码路径"""
    vllm_root: str                       # vllm 包根目录
    vllm_model_file: Optional[str]       # vllm/model_executor/models/xxx.py
    vllm_moe_layer: Optional[str]        # fused_moe/layer.py (MoE 模型)
    vllm_router_dir: str                 # fused_moe/router/ (MoE 模型)
    vllm_attention_dir: str              # vllm/attention/ (通用)
    vllm_norm_file: str                  # layernorm.py 路径 (通用)
    vllm_custom_ops: str                 # _custom_ops.py
    hf_modeling: Optional[str]           # 本地 modeling_*.py
    hf_config_py: Optional[str]          # 本地 configuration_*.py
    key_ops: List[KeyOp] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["key_ops"] = [asdict(k) for k in self.key_ops]
        return d


# ============================================================================
# model_type → vLLM 模型文件名映射
# ============================================================================

# 已知映射（model_type → 文件名，不含 .py）
_MODEL_FILE_MAP: Dict[str, str] = {
    # Bailing
    "bailing_moe_v2": "bailing_moe",
    "bailing_moe": "bailing_moe",
    # DeepSeek
    "deepseek_v2": "deepseek_v2",
    "deepseek_v3": "deepseek_v3",
    "deepseek_mtp": "deepseek_mtp",
    # Qwen MoE
    "qwen2_moe": "qwen2_moe",
    "qwen3_moe": "qwen3_moe",
    "qwen3": "qwen3",
    # GLM MoE
    "glm4_moe": "glm4_moe",
    "glm4_moe_lite": "glm4_moe_lite",
    # Granite
    "granitemoe": "granitemoe",
    "granitemoeshared": "granitemoeshared",
    "granitemoehybrid": "granitemoehybrid",
    # ERNIE
    "ernie45_moe": "ernie45_moe",
    # Exaone
    "exaone_moe": "exaone_moe",
    # 通用 dense
    "llama": "llama",
    "qwen2": "qwen2",
    "gemma2": "gemma2",
    "mixtral": "mixtral",
}

# architectures → 文件名 的补充映射（有些 model_type 不直观）
_ARCH_FILE_MAP: Dict[str, str] = {
    "BailingMoeV2ForCausalLM": "bailing_moe",
    "BailingMoeForCausalLM": "bailing_moe",
    "DeepseekV2ForCausalLM": "deepseek_v2",
    "DeepseekV3ForCausalLM": "deepseek_v3",
    "MixtralForCausalLM": "mixtral",
}


def _vllm_root() -> str:
    """vllm 包根目录"""
    mod = importlib.import_module("vllm")
    return os.path.dirname(mod.__file__)


def _resolve_model_file(profile: ModelProfile, vllm_root: str) -> Optional[str]:
    """定位 vLLM 模型文件，多重 fallback"""
    models_dir = os.path.join(vllm_root, "model_executor", "models")
    candidates: List[str] = []

    # 1. model_type 直接映射
    f = _MODEL_FILE_MAP.get(profile.model_type)
    if f:
        candidates.append(f + ".py")

    # 2. arch 直接映射
    f = _ARCH_FILE_MAP.get(profile.arch)
    if f:
        candidates.append(f + ".py")

    # 3. model_type 去后缀尝试（bailing_moe_v2 → bailing_moe）
    mt = profile.model_type
    for suffix in ("_v2", "_v3", "_lite", "_mtp"):
        if mt.endswith(suffix):
            base = mt[: -len(suffix)]
            if base in _MODEL_FILE_MAP:
                candidates.append(_MODEL_FILE_MAP[base] + ".py")

    # 4. model_type 直接当文件名
    candidates.append(mt + ".py")

    for cand in candidates:
        full = os.path.join(models_dir, cand)
        if os.path.isfile(full):
            return full
    return None


def _resolve_hf_files(profile: ModelProfile) -> tuple[Optional[str], Optional[str]]:
    """从 custom_py_files 找 modeling / configuration"""
    modeling = config_py = None
    for f in profile.custom_py_files:
        bn = os.path.basename(f)
        if bn.startswith("modeling_") and modeling is None:
            modeling = f
        elif bn.startswith("configuration_") and config_py is None:
            config_py = f
    return modeling, config_py


# ============================================================================
# 关键算子推断（程序化，不依赖 markdown）
# ============================================================================

def _infer_key_ops(profile: ModelProfile, vllm_root: str) -> List[KeyOp]:
    """
    根据 ModelProfile 推断可能的关键算子入口。
    知识库（precision_known_issues.md）的详细条目由 agent 在运行时读取叠加，
    这里只给出"结构性"入口——MoE 模型和 dense 模型都覆盖。
    """
    ops: List[KeyOp] = []
    custom_ops = os.path.join(vllm_root, "_custom_ops.py")
    router_dir = os.path.join(vllm_root, "model_executor", "layers",
                              "fused_moe", "router")
    moe_layer = os.path.join(vllm_root, "model_executor", "layers",
                             "fused_moe", "layer.py")

    # --- 通用算子（MoE + dense 都可能有） ---
    # Attention 实现路径
    attn_dir = os.path.join(vllm_root, "attention")
    ops.append(KeyOp(
        name="attention (flash_attn / eager / sdpa)",
        file=os.path.relpath(attn_dir, vllm_root),
        trigger_condition="根据 VLLM_ATTENTION_BACKEND 选择实现",
    ))
    # Norm 实现路径
    norm_py = os.path.join(vllm_root, "model_executor", "layers", "layernorm.py")
    ops.append(KeyOp(
        name="RMSNorm / LayerNorm",
        file=os.path.relpath(norm_py, vllm_root) if os.path.isfile(norm_py) else "model_executor/layers/layernorm.py",
        trigger_condition="所有 transformer 层通用",
    ))
    # Dense MLP / FFN
    ops.append(KeyOp(
        name="MLP/FFN (gate/up/down 投影 + 激活)",
        file=os.path.relpath(
            os.path.join(vllm_root, "model_executor", "layers", "activation.py"), vllm_root)
        if os.path.isfile(os.path.join(vllm_root, "model_executor", "layers", "activation.py"))
        else "model_executor/layers/activation.py",
        trigger_condition="所有 transformer 层通用；dense 走逐层 GEMM，MoE 走 fused_moe",
    ))

    if not profile.is_moe:
        return ops

    # --- MoE 专属算子 ---
    if profile.score_function == "sigmoid" and profile.e_score_correction_bias:
        ops.append(KeyOp(
            name="ops.moe_fused_gate",
            file=os.path.relpath(custom_ops, vllm_root),
            trigger_condition=(
                "use_fused_gate=True (VLLM_ENABLE_MOE_FUSED_GATE 默认1 且 "
                "e_score_correction_bias≠None 且 num_expert_group≠None)"
            ),
            known_issue=None,
            fix=None,
        ))
        ops.append(KeyOp(
            name="GroupedTopKRouter.select_experts / _compute_routing",
            file=os.path.relpath(
                os.path.join(router_dir, "grouped_topk_router.py"), vllm_root),
            trigger_condition="MoE 层路由分发（use_fused_gate 分支判断在此）",
        ))
    else:
        ops.append(KeyOp(
            name="grouped_topk (fused / python)",
            file=os.path.relpath(
                os.path.join(router_dir, "grouped_topk_router.py"), vllm_root),
            trigger_condition="num_expert_group≠None 时走 grouped topk",
        ))
    ops.append(KeyOp(
        name="FusedMoE.forward (fused / native)",
        file=os.path.relpath(moe_layer, vllm_root),
        trigger_condition="专家 FFN 计算；forward_cuda=fused kernel, forward_native=纯 PyTorch",
    ))

    return ops


# ============================================================================
# 主入口
# ============================================================================

def resolve_code_paths(profile: ModelProfile) -> CodePaths:
    """
    ModelProfile → CodePaths

    Returns:
        CodePaths，未定位到的字段为 None
    """
    vllm_root = _vllm_root()
    models_dir = os.path.join(vllm_root, "model_executor", "models")
    moe_layer = os.path.join(vllm_root, "model_executor", "layers",
                             "fused_moe", "layer.py")
    router_dir = os.path.join(vllm_root, "model_executor", "layers",
                              "fused_moe", "router")
    custom_ops = os.path.join(vllm_root, "_custom_ops.py")

    vllm_model_file = _resolve_model_file(profile, vllm_root)
    hf_modeling, hf_config_py = _resolve_hf_files(profile)
    key_ops = _infer_key_ops(profile, vllm_root)

    attn_dir = os.path.join(vllm_root, "attention")
    norm_file = os.path.join(vllm_root, "model_executor", "layers", "layernorm.py")

    return CodePaths(
        vllm_root=vllm_root,
        vllm_model_file=vllm_model_file,
        vllm_moe_layer=moe_layer if os.path.isfile(moe_layer) else None,
        vllm_router_dir=router_dir,
        vllm_attention_dir=attn_dir if os.path.isdir(attn_dir) else "",
        vllm_norm_file=norm_file if os.path.isfile(norm_file) else "",
        vllm_custom_ops=custom_ops,
        hf_modeling=hf_modeling,
        hf_config_py=hf_config_py,
        key_ops=key_ops,
    )


# ============================================================================
# CLI 自检
# ============================================================================

if __name__ == "__main__":
    import sys
    from .config_loader import load_model_profile
    path = sys.argv[1] if len(sys.argv) > 1 else "/path/to/model"
    profile = load_model_profile(path)
    cp = resolve_code_paths(profile)
    print("=== CodePaths ===")
    print(f"vllm_root       : {cp.vllm_root}")
    print(f"vllm_model_file : {cp.vllm_model_file}")
    print(f"vllm_moe_layer  : {cp.vllm_moe_layer}")
    print(f"vllm_router_dir : {cp.vllm_router_dir}")
    print(f"vllm_custom_ops : {cp.vllm_custom_ops}")
    print(f"hf_modeling     : {cp.hf_modeling}")
    print(f"hf_config_py    : {cp.hf_config_py}")
    print(f"--- key_ops ({len(cp.key_ops)}) ---")
    for k in cp.key_ops:
        print(f"  {k.name}")
        print(f"    file   : {k.file}")
        print(f"    trig   : {k.trigger_condition}")
        if k.known_issue:
            print(f"    issue  : {k.known_issue}")
        if k.fix:
            print(f"    fix    : {k.fix}")
