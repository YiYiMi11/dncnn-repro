"""对照实验：找出为什么比论文低 1.5 dB。

已确认的事实：
  * 网络比简单高斯模糊好 1.5 dB —— 说明确实学到了东西，但不够好
  * 训练 loss 从 epoch 3 起卡在 0.00200 不动 —— 优化卡住了
  * BN 的 gamma 训练 19200 步后几乎没变（梯度比卷积小 100 倍）
  * 论文的 29.23 dB 对应 MSE≈0.00119，我们是 0.00200（高 68%）

这个脚本对多组 (init, optimizer, lr, weight_decay) 各跑同样的步数，
打印 loss 轨迹，用来定位是哪个因素在限制。

    python tools/diag.py --iters 1500
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import TrainingImages                    # noqa: E402
from src.model import DnCNN                            # noqa: E402
from src.train import build_optimizer, get_device      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=1500)
ap.add_argument("--sigma", type=float, default=25)
ap.add_argument("--data-root", default="data")
ap.add_argument("--gray", action="store_true", default=True)
A = ap.parse_args()

dev, _ = get_device()
tr = TrainingImages(A.data_root, color=A.gray, n_holdout=8, verbose=False)
IN_CH = 3 if A.gray else 1
NOISE = A.sigma / 255.0

COMBOS = [
    ("基线(当前默认)",   dict(init="orthogonal", opt="sgd",  lr=0.01,  wd=1e-4)),
    ("lr 大 3 倍",      dict(init="orthogonal", opt="sgd",  lr=0.03,  wd=1e-4)),
    ("lr 小 1/2",       dict(init="orthogonal", opt="sgd",  lr=0.005, wd=1e-4)),
    ("lr 0.02",         dict(init="orthogonal", opt="sgd",  lr=0.02,  wd=1e-4)),
    ("无 weight decay", dict(init="orthogonal", opt="sgd",  lr=0.01,  wd=0.0)),
    ("Adam 1e-4 (KAIR)", dict(init="orthogonal", opt="adam", lr=1e-4, wd=1e-4)),
    ("He 初始化",       dict(init="he",         opt="sgd",  lr=0.01,  wd=1e-4)),
    ("He + Adam 1e-4",  dict(init="he",         opt="adam", lr=1e-4, wd=1e-4)),
]


def run(cfg, iters):
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    m = DnCNN(17, 64, IN_CH, init=cfg["init"]).to(dev)
    a = type("A", (), dict(weight_decay=cfg["wd"], optimizer=cfg["opt"],
                           lr=cfg["lr"], momentum=0.9))()
    opt = build_optimizer(m, a)
    curve, div = [], None
    for it in range(iters):
        for g in opt.param_groups:
            g["lr"] = cfg["lr"] * (1e-3 ** (it / max(1, iters - 1)))
        c = tr.sample_batch(128, 40, rng, dev, False)
        loss = torch.nn.functional.mse_loss(m(c + torch.randn_like(c) * NOISE), c)
        v = loss.item()
        if not np.isfinite(v):
            div = it + 1
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        curve.append(v)
    return curve, div


if __name__ == "__main__":
    print(f"device={dev}  iters={A.iters}  sigma={A.sigma}")
    print(f"loss 理论目标 = 0（完美预测噪声）；论文 29.23dB 对应 MSE≈0.00119")
    print(f"我们目前卡在 0.00200\n")
    print(f"{'配置':<20}{'结果':<10}{'后 1/4 平均 loss':<20}{'等效 PSNR':<12}{'耗时'}")
    print("-" * 76)
    for name, cfg in COMBOS:
        t0 = time.time()
        curve, div = run(cfg, A.iters)
        if div:
            print(f"{name:<20}{'第'+str(div)+'步发散':<10}{'-':<20}{'-':<12}"
                  f"{time.time()-t0:.0f}s", flush=True)
            continue
        tail = float(np.mean(curve[-A.iters // 4:]))
        psnr_eq = 10 * np.log10(1.0 / tail)
        print(f"{name:<20}{'收敛':<10}{tail:<20.5f}{psnr_eq:<12.2f}"
              f"{time.time()-t0:.0f}s", flush=True)