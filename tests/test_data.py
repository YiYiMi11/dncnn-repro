"""数据管线测试：噪声强度标定与可复现性。

两条不变量：
  1. 加进去的噪声，其真实标准差（换算到 0–255 尺度）必须等于请求的 sigma；
  2. 同一个 seed 必须给出逐比特相同的噪声 —— 否则「跑两遍对不上」没法调试。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 改名导入：TestSet 以 "Test" 开头，pytest 会误以为它是测试类并尝试收集
from src.data import TestSet as TestImageSet  # noqa: E402
from src.data import TrainingImages, psnr  # noqa: E402


@pytest.fixture
def fake_testsets(tmp_path):
    """造一个临时 testsets/BSD68，免去依赖真实数据集。"""
    from PIL import Image

    d = tmp_path / "testsets" / "BSD68"
    d.mkdir(parents=True)
    rng = np.random.RandomState(0)
    for i in range(3):
        Image.fromarray(rng.randint(0, 256, (48, 48, 3)).astype(np.uint8)).save(
            d / f"img{i:03d}.png"
        )
    return tmp_path


@pytest.mark.parametrize("sigma", [5, 25, 50])
def test_noise_sigma_is_calibrated(fake_testsets, sigma):
    """y = x + v，v ~ N(0, (sigma/255)²)。实测标准差必须落在 sigma 附近。

    这是最容易出错又最难发现的一步：常见的错误是忘了除 255，或者除成了 sqrt(3)（三通道），
    程序照跑、数字照出，只是全都不对。
    """
    ts = TestImageSet(str(fake_testsets), "BSD68", color=False, sigma=sigma, seed=0)
    stds = []
    for i in range(len(ts)):
        _, clean, noisy = ts[i]
        # 逐像素差值的标准差就是噪声的标准差（干净图那一项被减掉了）
        stds.append(float((noisy - clean).std() * 255.0))
    # 单张 48×48 的采样噪声波动不小，放宽到 15%
    assert np.mean(stds) == pytest.approx(sigma, rel=0.15)


def test_noise_is_reproducible_with_same_seed(fake_testsets):
    """同 seed 两次加载 → 逐比特相同。"""
    a = TestImageSet(str(fake_testsets), "BSD68", color=False, sigma=25, seed=0)
    b = TestImageSet(str(fake_testsets), "BSD68", color=False, sigma=25, seed=0)
    for i in range(len(a)):
        na, _, ya = a[i]
        nb, _, yb = b[i]
        assert na == nb
        assert np.array_equal(ya, yb)


def test_different_seed_gives_different_noise(fake_testsets):
    """不同 seed → 噪声必须不同，否则 seed 这个参数形同虚设。"""
    a = TestImageSet(str(fake_testsets), "BSD68", color=False, sigma=25, seed=0)
    b = TestImageSet(str(fake_testsets), "BSD68", color=False, sigma=25, seed=1)
    assert not np.array_equal(a[0][2], b[0][2])


def test_different_image_gets_different_noise(fake_testsets):
    """每张图的噪声也必须独立（否则三张图会被加上一模一样的噪声）。"""
    ts = TestImageSet(str(fake_testsets), "BSD68", color=False, sigma=25, seed=0)
    n0 = ts[0][2] - ts[0][1]
    n1 = ts[1][2] - ts[1][1]
    assert not np.array_equal(n0, n1)


def test_gray_mode_returns_single_channel(fake_testsets):
    ts = TestImageSet(str(fake_testsets), "BSD68", color=False, sigma=25, seed=0)
    _, clean, noisy = ts[0]
    assert clean.shape == (48, 48, 1) and noisy.shape == (48, 48, 1)


def test_color_mode_returns_three_channels(fake_testsets):
    ts = TestImageSet(str(fake_testsets), "BSD68", color=True, sigma=25, seed=0)
    _, clean, _ = ts[0]
    assert clean.shape == (48, 48, 3)


def test_noisy_psnr_matches_analytic_expectation(fake_testsets):
    """含噪图的 PSNR 应约为 20·log10(255/sigma)。

    去噪实验都拿这个数当基线；它不对的话，整张表的「提升幅度」就都是错的。
    """
    sigma = 25
    ts = TestImageSet(str(fake_testsets), "BSD68", color=False, sigma=sigma, seed=0)
    vals = [psnr(ts[i][1], ts[i][2]) for i in range(len(ts))]
    expected = 20 * np.log10(255 / sigma)
    assert np.mean(vals) == pytest.approx(expected, abs=0.6)


def test_missing_dataset_raises_helpful_error(tmp_path):
    """数据集没下载时应给出可操作的提示，而不是一个裸的 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        TestImageSet(str(tmp_path), "BSD68", color=False, sigma=25)


def test_training_holdout_excludes_images_from_sampling(tmp_path):
    """验证集必须真的不参与 patch 采样，否则训练曲线就在偷看验证集。"""
    from PIL import Image

    d = tmp_path / "BSDS500" / "train"
    d.mkdir(parents=True)
    rng = np.random.RandomState(0)
    for i in range(6):
        Image.fromarray(rng.randint(0, 256, (64, 64, 3)).astype(np.uint8)).save(
            d / f"{i:03d}.png"
        )
    ti = TrainingImages(str(tmp_path), color=False, n_holdout=2, verbose=False)
    assert len(ti) == 4
    assert len(ti.val_images) == 2
    assert not (set(ti.paths) & set(ti.val_paths))


def test_sample_batch_shape_and_range(tmp_path):
    """采样出的 batch 形状 [B,C,P,P]、dtype float、值域落在 [0,1]。"""
    from PIL import Image

    d = tmp_path / "BSDS500" / "train"
    d.mkdir(parents=True)
    rng = np.random.RandomState(0)
    for i in range(4):
        Image.fromarray(rng.randint(0, 256, (64, 64, 3)).astype(np.uint8)).save(
            d / f"{i:03d}.png"
        )
    ti = TrainingImages(str(tmp_path), color=False, verbose=False)
    batch = ti.sample_batch(8, 40, np.random.default_rng(0), "cpu")
    assert batch.shape == (8, 1, 40, 40)
    assert batch.dtype.is_floating_point
    assert 0.0 <= float(batch.min()) and float(batch.max()) <= 1.0