"""扫 (init, optimizer, lr) 组合，实测谁能收敛、收敛得怎么样。

论文给的是 MatConvNet 上的超参（SGD + lr 1e-1，He 初始化），直接照搬到 PyTorch
会第一个 iteration 就发散。这个脚本把结论跑出来，不靠猜。

    python tools/lr_sweep.py                         # 默认 data/、sigma=25、灰度
    python tools/lr_sweep.py --data-root data --sigma 50 --color
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import TrainingImages, psnr          # noqa: E402
from src.model import DnCNN                        # noqa: E402
from src.train import get_device                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="data")
ap.add_argument("--sigma", type=float, default=25)
ap.add_argument("--iters", type=int, default=300)
ap.add_argument("--color", action="store_true")
ap.add_argument("--lr-min-factor", type=float, default=1e-3,
                help="训练结束时 lr / 起始 lr，模拟指数衰减")
A = ap.parse_args()

dev, _ = get_device()
tr = TrainingImages(A.data_root, color=A.color, n_holdout=8)
IN_CH = 3 if A.color else 1
NOISE = A.sigma / 255.0


@torch.no_grad()
def val_psnr(m):
    tot = 0.0
    for t in tr.val_images:
        c = t.float().to(dev) / 255.0
        if c.dim() == 3:                      # 灰度：HxWx1 -> 1xHxW
            c = c.unsqueeze(0)
        g = torch.Generator(device=dev).manual_seed(0)
        n = c + torch.randn(c.shape, generator=g, device=dev) * NOISE
        o = m(n).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        tot += psnr(o, c[0].permute(1, 2, 0).cpu().numpy())
    return tot / len(tr.val_images)


COMBOS = [("orthogonal", "sgd", 0.1), ("orthogonal", "sgd", 0.02),
          ("orthogonal", "sgd", 0.01), ("orthogonal", "sgd", 0.005),
          ("orthogonal", "adam", 1e-3), ("orthogonal", "adam", 1e-4),
          ("he", "sgd", 0.01), ("he", "adam", 1e-3)]

print(f"sigma={A.sigma}  iters={A.iters}  device={dev}  "
      f"loss 下界(噪声能量)={NOISE ** 2:.6f}", flush=True)
print("init        opt   lr        iters  loss        valPSNR", flush=True)
print("-" * 54, flush=True)
for init, optn, lr in COMBOS:
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    m = DnCNN(17, 64, IN_CH, init=init).to(dev)
    if optn == "sgd":
        opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        opt = torch.optim.Adam(m.parameters(), lr=lr)
    losses = []
    for it in range(A.iters):
        f = it / max(1, A.iters - 1)
        for g in opt.param_groups:
            g["lr"] = lr * (A.lr_min_factor ** f)
        c = tr.sample_batch(128, 40, rng, dev, False)
        loss = torch.nn.functional.mse_loss(m(c + torch.randn_like(c) * NOISE), c)
        if not np.isfinite(loss.item()):
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    ok = bool(losses) and bool(np.isfinite(losses[-1]))
    if ok:
        print(f"{init:<12}{optn:<6}{lr:<9}{len(losses):<7}"
              f"{np.mean(losses[-50:]):<12.6f}{val_psnr(m):.3f}", flush=True)
    else:
        print(f"{init:<12}{optn:<6}{lr:<9}{'发散':<7}{'-':<12}-", flush=True)