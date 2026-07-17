"""
platform_probe.py — Hygon DCU 硬件探测

自动识别当前机器的 gfx 架构与型号，返回标准化 PlatformInfo。
不依赖 rocm-info（DCU 上常不在 PATH），优先用 /opt/dtk/bin/rocminfo。

零副作用：import 不初始化 GPU。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


# Hygon DCU 型号 → gfx 映射（来自厂商规格）
_HYGON_DCU_MAP = {
    "BW100": "gfx936",
    "BW150": "gfx936",
    "BW1000": "gfx936",
    "K100AI": "gfx938",
    "BW1100": "gfx938",
}


@dataclass
class PlatformInfo:
    """标准化平台信息"""
    is_dcu: bool                      # 是否 Hygon DCU
    gfx: Optional[str]                # 如 "gfx936"
    marketing_name: Optional[str]     # 如 "BW100"
    hip_version: Optional[str]        # 如 "6.3.26113"
    rocm_path: Optional[str]          # 如 "/opt/dtk"
    device_count: int = 0             # GPU 数量（torch 侧）
    is_cuda: Optional[bool] = None    # vLLM current_platform.is_cuda()
    is_rocm: Optional[bool] = None

    def summary(self) -> str:
        if not self.is_dcu:
            return f"non-DCU platform (is_cuda={self.is_cuda}, is_rocm={self.is_rocm})"
        return (
            f"Hygon DCU {self.marketing_name} ({self.gfx}), "
            f"HIP {self.hip_version}, {self.device_count} devices, "
            f"is_cuda={self.is_cuda} is_rocm={self.is_rocm}"
        )

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def _find_rocminfo() -> Optional[str]:
    """找 rocminfo 可执行文件"""
    for cand in ("/opt/dtk/bin/rocminfo", "/opt/rocm/bin/rocminfo"):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    p = shutil.which("rocminfo")
    return p


def _parse_rocminfo(path: str) -> tuple[Optional[str], Optional[str]]:
    """从 rocminfo 输出提取 GPU 的 gfx 与 Marketing Name"""
    try:
        out = subprocess.run([path], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception:
        return None, None
    gfx = mkt = None
    # rocminfo 每个 agent 块里有 Name: 和 Marketing Name:。
    # GPU agent 的 Name 是 gfxXXX；CPU agent 的 Name 是 CPU 型号。
    lines = out.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("Name:") and s.split(":", 1)[1].strip().startswith("gfx"):
            gfx = s.split(":", 1)[1].strip()
            # 往下找同块的 Marketing Name
            for j in range(i, min(i + 15, len(lines))):
                mj = lines[j].strip()
                if mj.startswith("Marketing Name:"):
                    mkt = mj.split(":", 1)[1].strip()
                    break
            if mkt:
                break
    return gfx, mkt


def _hipconfig_version() -> tuple[Optional[str], Optional[str]]:
    """从 hipconfig 取 HIP 版本与 ROCM_PATH"""
    hip = shutil.which("hipconfig") or "/opt/dtk/bin/hipconfig"
    if not os.path.isfile(hip):
        return None, os.environ.get("ROCM_PATH")
    try:
        out = subprocess.run([hip], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None, os.environ.get("ROCM_PATH")
    ver = rocm = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("HIP version"):
            ver = s.split(":", 1)[1].strip()
        elif s.startswith("ROCM_PATH"):
            rocm = s.split(":", 1)[1].strip()
    return ver, rocm


def _torch_device_count() -> tuple[int, list]:
    """torch 侧设备数与名称（惰性，避免无 GPU 时崩）"""
    try:
        import torch
        n = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        return n, names
    except Exception:
        return 0, []


def _vllm_platform_flags() -> tuple[Optional[bool], Optional[bool]]:
    """vLLM current_platform 的 is_cuda / is_rocm"""
    try:
        from vllm.platforms import current_platform
        return bool(current_platform.is_cuda()), bool(current_platform.is_rocm())
    except Exception:
        return None, None


def probe_platform() -> PlatformInfo:
    """探测当前平台，返回 PlatformInfo"""
    rocminfo = _find_rocminfo()
    gfx, mkt = _parse_rocminfo(rocminfo) if rocminfo else (None, None)
    hip_ver, rocm_path = _hipconfig_version()
    dev_count, dev_names = _torch_device_count()

    # 若 torch 报出已知 DCU 型号但 rocminfo 没抓到，用 torch 名字补
    if mkt is None and dev_names:
        for nm in dev_names:
            if nm in _HYGON_DCU_MAP:
                mkt = nm
                if gfx is None:
                    gfx = _HYGON_DCU_MAP[nm]
                break

    is_dcu = gfx is not None and gfx.startswith("gfx")
    is_cuda, is_rocm = _vllm_platform_flags()

    return PlatformInfo(
        is_dcu=is_dcu,
        gfx=gfx,
        marketing_name=mkt,
        hip_version=hip_ver,
        rocm_path=rocm_path,
        device_count=dev_count,
        is_cuda=is_cuda,
        is_rocm=is_rocm,
    )


# gfx → 空闲卡建议（避免抢在线服务；机器配置不同需人工确认）
def recommended_idle_gpu(info: PlatformInfo, avoid: str = "0,1,6,7") -> str:
    """建议空闲卡：避开 avoid 列表，取最靠前可用卡"""
    avoid_set = {x.strip() for x in avoid.split(",") if x.strip()}
    for i in range(info.device_count):
        if str(i) not in avoid_set:
            return str(i)
    return "0"


if __name__ == "__main__":
    info = probe_platform()
    print("=== PlatformInfo ===")
    print(info.summary())
    print(info.to_dict())
    print("recommended idle gpu:", recommended_idle_gpu(info))
