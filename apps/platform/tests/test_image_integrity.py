# apps/platform/tests/test_image_integrity.py
"""image_integrity 的解码判定:纯函数级测试,不依赖数据集/权重/网络。"""
from __future__ import annotations

import pytest

from od_platform.data_validation.checks.image_integrity import decode_error

pytestmark = pytest.mark.unit


def test_valid_png_decodes(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    p = tmp_path / "ok.png"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(p)
    assert decode_error(p) is None


def test_garbage_bytes_are_rejected(tmp_path):
    pytest.importorskip("PIL.Image")
    p = tmp_path / "fake.png"
    p.write_bytes(b"not an image at all\n")
    err = decode_error(p)
    assert err is not None
    assert ":" in err                      # "UnidentifiedImageError: ..."


def test_truncated_jpeg_is_rejected(tmp_path, monkeypatch):
    """只 open 不 load 抓不到截断文件 —— 这个用例钉住"必须真解码"。"""
    Image = pytest.importorskip("PIL.Image")
    ImageFile = pytest.importorskip("PIL.ImageFile")
    monkeypatch.setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", False)

    p = tmp_path / "half.jpg"
    Image.new("RGB", (64, 64), (0, 255, 0)).save(p, format="JPEG")
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])
    assert decode_error(p) is not None


def test_missing_pillow_degrades_to_error_string(tmp_path, monkeypatch):
    """Pillow 不可用时 helper 返回原因串而不是抛异常(整轮质检不能因此崩)。"""
    import od_platform.data_validation.checks.image_integrity as mod

    monkeypatch.setattr(mod, "_load_image_module", lambda: None)
    assert decode_error(tmp_path / "whatever.png").startswith("PillowUnavailable")