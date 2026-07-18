"""
op-locate-agent.lib — 可复用工具库

agent 的"手"：config 解析、路径解析、hook、对比、启动、报告。
所有模块设计为零副作用导入（import 不会启动 GPU / 不会读环境变量）。
"""

from .config_loader import ModelProfile, load_model_profile
from .path_resolver import CodePaths, KeyOp, resolve_code_paths
from .hook_manager import (
    HookManager, HookPoint, CaptureSpec,
    transformer_hook_points, moe_router_hook_points,
)
from .tensor_compare import (
    TensorComparator,
    CompareResult,
    compare_topk,
    compare_tensors,
)
from .platform_probe import PlatformInfo, probe_platform, recommended_idle_gpu
from .model_probe import ModelProbeResult, probe_model, save_probe_result
from . import config_patch

__all__ = [
    "ModelProfile",
    "load_model_profile",
    "CodePaths",
    "KeyOp",
    "resolve_code_paths",
    "HookManager",
    "HookPoint",
    "CaptureSpec",
    "transformer_hook_points",
    "moe_router_hook_points",
    "TensorComparator",
    "CompareResult",
    "compare_topk",
    "compare_tensors",
    "PlatformInfo",
    "probe_platform",
    "recommended_idle_gpu",
    "ModelProbeResult",
    "probe_model",
    "save_probe_result",
    "config_patch",
]
