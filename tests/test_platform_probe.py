"""
test_platform_probe.py — 平台探测单测
"""

import pytest

from lib import probe_platform, recommended_idle_gpu


class TestPlatformProbe:

    def test_probe_returns_info(self):
        info = probe_platform()
        assert info is not None
        # 本机是 Hygon DCU
        assert info.is_dcu is True
        assert info.gfx in ("gfx936", "gfx938")
        assert info.marketing_name is not None
        assert info.device_count > 0

    def test_is_cuda_false_on_dcu(self):
        """DCU 上 vLLM is_cuda() 必须是 False（关键调度因素）"""
        info = probe_platform()
        if info.is_dcu:
            assert info.is_cuda is False
            assert info.is_rocm is True

    def test_summary_string(self):
        info = probe_platform()
        s = info.summary()
        assert "DCU" in s
        assert info.gfx in s

    def test_recommended_idle_gpu(self):
        """空闲卡推荐逻辑：避开 avoid 列表，取最靠前可用卡。
        注意：测试在 HIP_VISIBLE_DEVICES 限制下 device_count 可能被裁剪，
        所以构造一个足够大的虚拟 device_count 来测逻辑。"""
        from lib.platform_probe import PlatformInfo
        # 虚构 8 卡场景
        info = PlatformInfo(
            is_dcu=True, gfx="gfx936", marketing_name="BW100",
            hip_version="6.3", rocm_path="/opt/dtk", device_count=8,
        )
        gpu = recommended_idle_gpu(info, avoid="0,1,6,7")
        assert gpu == "2"  # 避开 0,1,6,7 后最靠前是 2
        # 全部避开时回退到 0
        assert recommended_idle_gpu(info, avoid="0,1,2,3,4,5,6,7") == "0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
