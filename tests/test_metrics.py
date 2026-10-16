"""PSNR / SSIM / 灰度转换的口径测试。

论文的数字是在**特定口径**下算出来的：MATLAB 的 rgb2gray 系数、彩色逐通道算 PSNR 再平均。
口径错了不会让程序报错，只会让所有数字对不上论文 —— 所以这里是全套测试里最该钉死的一组。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import MATLAB_RGB2GRAY, psnr, rgb2gray_matlab, ssim  # noqa: E402


def test_psnr_identical_is_infinite():
    assert psnr(np.zeros((8, 8)), np.zeros((8, 8))) == float("inf")


def test_psnr_known_value():
    """常数偏移 d 时 PSNR = 20·log10(1/d)，手算校验公式本身。"""
    a = np.full((16, 16), 0.5)
    b = np.full((16, 16), 0.6)
    d = 0.1
    assert psnr(a, b) == pytest.approx(20 * np.log10(1 / d), abs=1e-9)


def test_psnr_color_is_mean_over_channels():
    """彩色按通道算 PSNR 再平均（论文 Set12 的做法），不是对整图一起算 MSE。

    这两种算法结果不同，且差得不少 —— 混用会导致和论文差零点几 dB。
    """
    rng = np.random.RandomState(0)
    a = rng.rand(16, 16, 3)
    b = a.copy()
    b[..., 0] += 0.1   # 只让 R 通道有误差

    per_channel = np.mean([psnr(a[..., c], b[..., c]) for c in range(3)])
    whole_image = 10 * np.log10(1.0 / np.mean((a - b) ** 2))
    assert psnr(a, b) == pytest.approx(per_channel)
    assert per_channel != pytest.approx(whole_image, abs=0.5)  # 确认两种口径确实不同


def test_psnr_rewards_lower_error():
    rng = np.random.RandomState(1)
    clean = rng.rand(32, 32)
    good = clean + rng.randn(32, 32) * 0.01
    bad = clean + rng.randn(32, 32) * 0.10
    assert psnr(clean, good) > psnr(clean, bad) + 10


def test_ssim_identical_is_one():
    rng = np.random.RandomState(0)
    x = rng.rand(32, 32)
    assert ssim(x, x) == pytest.approx(1.0, abs=1e-6)


def test_ssim_drops_with_noise():
    rng = np.random.RandomState(2)
    clean = rng.rand(64, 64)
    mild = clean + rng.randn(64, 64) * 0.02
    heavy = clean + rng.randn(64, 64) * 0.15
    assert ssim(clean, mild) > ssim(clean, heavy)


@pytest.mark.parametrize("channel,expected", [(0, 76), (1, 150), (2, 29)])
def test_rgb2gray_matches_matlab(channel, expected):
    """MATLAB rgb2gray([255 0 0]) = 76, [0 255 0] = 150, [0 0 255] = 29。

    这三个值是判据：换成 skimage/Rec.709 默认系数会得到 54/183/18，
    PSNR 因此偏掉约 0.1 dB —— 足以让「复现成功」的判定翻车。
    """
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    img[0, 0, channel] = 255
    assert rgb2gray_matlab(img)[0, 0] == expected


def test_rgb2gray_coefficients_sum_to_one():
    assert sum(MATLAB_RGB2GRAY) == pytest.approx(1.0, abs=1e-6)


def test_rgb2gray_white_stays_white():
    """系数和为 1，白图必须还是 255（没有整体偏暗）。"""
    white = np.full((4, 4, 3), 255, dtype=np.uint8)
    assert (rgb2gray_matlab(white) == 255).all()


def test_rgb2gray_differs_from_rec709():
    """明确锁死：这不是 Rec.709。用 Rec.709 会静默产生系统性偏差。"""
    rng = np.random.RandomState(3)
    img = rng.randint(0, 256, (8, 8, 3)).astype(np.uint8)
    ours = rgb2gray_matlab(img).astype(np.float64)
    rec709 = (img.astype(np.float64) * np.array([0.2126, 0.7152, 0.0722])).sum(-1)
    assert not np.allclose(ours, rec709, atol=0.5)