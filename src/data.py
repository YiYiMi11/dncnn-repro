"""数据集与 PSNR / SSIM 指标。

颜色 -> 灰度的转换用 **MATLAB rgb2gray** 的系数，而不是 skimage 默认的 Rec.709 系数。
这一点很重要：论文里 BSD68 是灰度评测，MATLAB 的 rgb2gray 用
    Y = 0.2989 R + 0.5870 G + 0.1140 B
Rec.709(0.2125/0.7154/0.0721) 会得到略微不同的灰度图，PSNR 会有 0.1dB 级别的偏差。
"""

import os
import glob

import numpy as np
import torch
from PIL import Image

# MATLAB rgb2gray 的精确系数（uint8 输入时先转 float 再加权，最后四舍五入回 uint8）
MATLAB_RGB2GRAY = (0.298936021293776, 0.587043074451121, 0.114020904255103)


def imread_rgb(path):
    """读入 RGB uint8 -> HWC uint8"""
    img = Image.open(path)
    return np.array(img.convert("RGB"), dtype=np.uint8)


def rgb2gray_matlab(img_u8):
    """RGB uint8 -> Gray uint8，与 MATLAB rgb2gray() 行为一致"""
    x = img_u8.astype(np.float64) / 255.0
    w = np.asarray(MATLAB_RGB2GRAY).reshape(1, 1, 3)
    y = (x * w).sum(axis=2)
    return np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)


def list_images(root, exts=(".png", ".jpg", ".jpeg", ".bmp")):
    files = []
    for e in exts:
        files += glob.glob(os.path.join(root, f"*{e}"))
    return sorted(files)


def _load(path, color):
    a = imread_rgb(path)
    if not color:
        a = rgb2gray_matlab(a)[..., None]          # HxWx1
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)


class TrainingImages:
    """把 400 张训练图一次性读进内存（uint8，约 190MB），之后在 CPU 上切 patch 极快。

    n_holdout > 0 时，会从末尾留出 n_holdout 张图 **不参与 patch 采样**，
    用作训练期间的验证集 —— 这样监控曲线就不会偷看测试集。
    """

    def __init__(self, root, color=True, verbose=True, n_holdout=0):
        files = list_images(os.path.join(root, "BSDS500/train")) + list_images(
            os.path.join(root, "BSDS500/val")
        )
        if not files:  # 也允许直接给一个平铺目录
            files = list_images(root)
        if not files:
            raise FileNotFoundError(
                f"{root} 下没找到训练图片。请先运行: python download_data.py"
            )
        n_holdout = min(n_holdout, max(0, len(files) - 1))
        val_files = files[-n_holdout:] if n_holdout else []
        train_files = files[: len(files) - n_holdout]

        self.color = color
        bad = []

        def load_many(paths):
            out = []
            for p in paths:
                try:
                    out.append(_load(p, color))
                except Exception:      # 下载中断会留下损坏文件，跳过并提示重下
                    bad.append(p)
            return out

        self.paths = train_files
        self.val_paths = val_files
        self.images = load_many(train_files)
        self.val_images = load_many(val_files)
        if bad:
            print(f"[data] 跳过 {len(bad)} 个损坏/未下完的文件，重跑 "
                  f"`python download_data.py` 可补齐，例如 {os.path.basename(bad[0])}")
        if not self.images:
            raise FileNotFoundError(
                f"{root} 下没有可用的训练图片，请先运行: python download_data.py")
        if verbose:
            print(f"[data] 训练 {len(self.images)} 张 / 验证 {len(self.val_images)} 张, "
                  f"{'RGB' if color else 'Gray'}")

    def __len__(self):
        return len(self.images)

    def sample_batch(self, batch_size, patch_size, rng, device, aug=False):
        """随机裁 batch_size 个 patch_size x patch_size 的干净图块，返回 float tensor [0,1]。

        aug=True 时对每个 patch 随机做水平/垂直翻转和转置（论文 DnCNN-S 未使用，
        属于常见的额外增广，能略微提升泛化，报告里需说明）。
        """
        out = torch.empty(batch_size, self.images[0].shape[0], patch_size, patch_size,
                          dtype=torch.uint8)
        idxs = rng.integers(0, len(self.images), size=batch_size)
        for i, ix in enumerate(idxs):
            img = self.images[ix]
            _, h, w = img.shape
            y = rng.integers(0, h - patch_size + 1)
            x = rng.integers(0, w - patch_size + 1)
            out[i] = img[:, y:y + patch_size, x:x + patch_size]
            if aug:
                if rng.random() < 0.5:
                    out[i] = torch.flip(out[i], dims=[2])
                if rng.random() < 0.5:
                    out[i] = torch.flip(out[i], dims=[1])
                if rng.random() < 0.5:
                    out[i] = out[i].transpose(1, 2)
        x = out.to(device, non_blocking=True).float().div_(255.0)
        return x


class TestSet:
    """测试集：Set12(彩色) / BSD68(彩色转灰度)。噪声用固定 seed 生成，保证可复现。"""

    def __init__(self, root, name, color, sigma, seed=0):
        d = os.path.join(root, "testsets", name)
        files = list_images(d)
        if not files:
            raise FileNotFoundError(f"{d} 为空，请先运行: python download_data.py")
        self.name = name
        self.color = color
        self.sigma = sigma
        self.seed = seed
        self.images = []
        for p in files:
            a = imread_rgb(p)
            if not color:
                a = rgb2gray_matlab(a)[..., None]
            self.images.append(a.astype(np.float32) / 255.0)
        self.names = [os.path.splitext(os.path.basename(p))[0] for p in files]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        clean = self.images[i]
        # 固定 seed + 固定索引 -> 每张图每个噪声等级对应一份可复现的噪声
        rng = np.random.RandomState(self.seed + i)
        noisy = clean + rng.randn(*clean.shape).astype(np.float32) * (self.sigma / 255.0)
        return self.names[i], clean, noisy


# --------------------------------------------------------------------------- #
#                                   指标                                       #
# --------------------------------------------------------------------------- #
def psnr(a, b, data_range=1.0):
    """a, b: float 数组，同形状。彩色图按通道算 PSNR 再取平均（论文 Set12 的做法）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 3 and a.shape[2] == 3:
        return float(np.mean([psnr(a[..., c], b[..., c], data_range) for c in range(3)]))
    mse = np.mean((a - b) ** 2)
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def ssim(a, b, data_range=1.0):
    """SSIM。彩色按通道算再平均。全图计算（论文 Table V 的口径）。"""
    from skimage.metrics import structural_similarity
    a = np.clip(np.asarray(a, np.float64), 0, 1)
    b = np.clip(np.asarray(b, np.float64), 0, 1)
    if a.ndim == 3 and a.shape[2] == 1:
        a, b = a[..., 0], b[..., 0]
    if a.ndim == 3 and a.shape[2] == 3:
        return float(np.mean([ssim(a[..., c], b[..., c], data_range) for c in range(3)]))
    return float(structural_similarity(a, b, data_range=data_range, channel_axis=None))