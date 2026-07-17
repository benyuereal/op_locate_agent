"""
conftest.py — pytest 共享 fixture

用 AntAngelMed 做真实 fixture（黄金回归用例）。
HIP_VISIBLE_DEVICES 默认设为 7（空闲卡），避免和在线服务抢卡。
"""

import os
import sys
import pytest

# 确保能 import lib
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

# 默认用空闲卡，可被环境变量覆盖
os.environ.setdefault("HIP_VISIBLE_DEVICES", "7")

ANTANGELMED_PATH = "/models/AntAngelMed"


@pytest.fixture(scope="session")
def model_path():
    if not os.path.isdir(ANTANGELMED_PATH):
        pytest.skip(f"{ANTANGELMED_PATH} not available")
    return ANTANGELMED_PATH


@pytest.fixture(scope="session")
def profile(model_path):
    from lib import load_model_profile
    return load_model_profile(model_path)


@pytest.fixture(scope="session")
def code_paths(profile):
    from lib import resolve_code_paths
    return resolve_code_paths(profile)
