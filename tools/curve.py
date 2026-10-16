"""评估一个实验目录下所有 checkpoint，产出「BSD68 PSNR vs 训练量」曲线。

用途：回答「到底要训多久才够」。训练还在跑的时候就能反复执行，
已经存下来的 checkpoint 会被立刻评估并追加到 CSV 里，不用等训练结束。

    # 训练还在跑，每隔一会儿执行一次，曲线会越画越长
    python -m src.train --gray --sigma 25 --preset fast --ckpt-every 1000 &
    python tools/curve.py experiments/probe_gray25 --sigma 25

    # 训练结束后
    python tools/curve.py experiments/gray_sigma25 --sigma 25 --plot

输出：
    <实验目录>/curve.csv    step, psnr, psnr_noisy, elapsed_min
    <实验目录>/curve.png    曲线图（带论文基线横线）
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import TestSet, psnr                      # noqa: E402
from src.model import DnCNN                              # noqa: E402
from src.test import denoise                             # noqa: E402
from src.train import get_device                         # noqa: E402

# 论文 Table II（BSD68 灰度）
PAPER_BSD68 = {15: 31.73, 25: 29.23, 50: 26.23}


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    color = a.get("color", False)
    m = DnCNN(depth=a.get("depth", 17), n_channels=a.get("width", 64),
              image_channels=3 if color else 1,
              init=a.get("init", "orthogonal"), init_gain=a.get("init_gain", 0.2))
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), color, ck.get("step")


def _total_iters(exp_dir, ckpts):
    """推断这次训练一共跑了多少 iteration —— 用作最终模型的 x 坐标。

    优先级：
      1. status.json 里的 total（训练脚本自己写的，最准）
      2. args 里的 epochs × 每 epoch 的 iteration 数
      3. 最大 ckpt step（保底）
    """
    import json
    sp = os.path.join(exp_dir, "status.json")
    if os.path.exists(sp):
        try:
            with open(sp) as f:
                t = json.load(f).get("total")
            if t:
                return int(t)
        except Exception:
            pass

    try:
        ck = torch.load(ckpts[0], map_location="cpu", weights_only=False)
        a = ck.get("args", {})
        if a.get("epochs") and a.get("patches_per_epoch") and a.get("batch_size"):
            return int(a["epochs"] * a["patches_per_epoch"] / a["batch_size"])
    except Exception:
        pass

    steps = []
    for c in ckpts:
        try:
            ck = torch.load(c, map_location="cpu", weights_only=False)
            if ck.get("step"):
                steps.append(int(ck["step"]))
        except Exception:
            pass
    return max(steps) if steps else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--dataset", default="BSD68")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--self-ensemble", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--recompute", action="store_true", help="忽略已有 CSV，全部重算")
    args = ap.parse_args()

    dev, _ = get_device()
    ckpts = sorted(glob.glob(os.path.join(args.exp_dir, "ckpt_*.pth")))
    final = os.path.join(args.exp_dir, "model_final.pth")
    if os.path.exists(final):
        ckpts.append(final)
    if not ckpts:
        sys.exit(f"{args.exp_dir} 下没有 ckpt_*.pth —— 训练时加 --ckpt-every 1000")

    csvp = os.path.join(args.exp_dir, "curve.csv")
    done = {}
    if os.path.exists(csvp) and not args.recompute:
        with open(csvp) as f:
            for r in csv.DictReader(f):
                done[int(r["step"])] = float(r["psnr"])
        print(f"已有 {len(done)} 个结果，跳过")

    ds = TestSet(args.data_root, args.dataset, color=False, sigma=args.sigma)
    total_iters = _total_iters(args.exp_dir, ckpts)
    todo = []
    for c in ckpts:
        m, color, step = load_model(c, dev)
        if color:
            print(f"  跳过 {os.path.basename(c)}（彩色模型，与 {args.dataset} 灰度不匹配）")
            continue
        # 最终模型的 step 是 None。用训练总步数作为它的真实 x 坐标，
        # 而不是拿一个哨兵大数 —— 否则 x 轴会被拉到 1e9，
        # 所有真实数据被压成左边一条竖线，图完全没法看。
        if step is None:
            step = total_iters
        key = int(step)
        if key in done:
            continue
        todo.append((key, c, m))
    todo.sort()

    if not todo:
        print("没有新的 checkpoint 可评估")
    else:
        print(f"评估 {len(todo)} 个 checkpoint（{args.dataset}, σ={args.sigma:.0f}）...")
        for key, c, m in todo:
            tot, base = 0.0, 0.0
            for i in range(len(ds)):
                _, clean, noisy = ds[i]
                out = denoise(m, noisy, dev, args.self_ensemble)
                tot += psnr(out, clean)
                base += psnr(noisy, clean)
            n = len(ds)
            done[key] = tot / n
            label = f"{key}"
            print(f"  {label:>8s} iter   PSNR {tot/n:6.3f} dB"
                  f"   (含噪 {base/n:.2f} dB, 提升 {tot/n - base/n:+.2f})", flush=True)

        with open(csvp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "psnr", "psnr_noisy"])
            for k in sorted(done):
                w.writerow([k, f"{done[k]:.4f}", ""])
        print(f"\n-> {csvp}")

    if not done:
        return
    ks = sorted(done)
    print(f"\n{'iter':>9s}  {'PSNR':>8s}  {'vs 论文':>9s}")
    paper = PAPER_BSD68.get(int(args.sigma))
    for k in ks:
        d = f"{done[k] - paper:+.3f}" if paper else "-"
        print(f"{k:>9d}  {done[k]:8.3f}  {d:>9s}")
    if paper:
        best = max(done.values())
        print(f"\n论文 DnCNN-S = {paper:.2f} dB, 本实验最高 {best:.3f} dB "
              f"(差 {best - paper:+.3f})")
        near = [k for k in ks if done[k] >= paper - 0.15]
        if near:
            print(f"→ 达到「相差 ≤0.15 dB」最早在 {min(near)} iter")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # ★ 必须显式指定中文字体，否则图上所有汉字都是 □□□
        #   （macOS 默认 DejaVu Sans 不含 CJK 字形）
        plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC",
                                            "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ys = [done[k] for k in ks]
        ax.plot(ks, ys, "o-", lw=1.6, ms=4, color="#1565C0",
                label=f"本实现（BSD68 灰度）")
        if paper:
            ax.axhline(paper, ls="--", color="crimson",
                       label=f"论文 DnCNN-S = {paper:.2f} dB")
            ax.axhline(paper - 0.15, ls=":", color="gray", lw=1.2,
                       label="论文 −0.15 dB（复现合格线）")
            ax.fill_between(ks, paper - 0.15, paper, color="crimson", alpha=.08)

        # 标注最终模型
        ax.annotate(f"最终模型\n{ks[-1]} iter\n{ys[-1]:.2f} dB",
                    (ks[-1], ys[-1]), textcoords="offset points",
                    xytext=(-12, -46), fontsize=9, color="#1565C0",
                    ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#E3F2FD", ec="#1565C0"))

        ax.set_xlabel("训练 iteration 数")
        ax.set_ylabel(f"{args.dataset} 平均 PSNR (dB)")
        ax.set_title(f"σ={args.sigma:.0f} 收敛曲线：训多久才够？", fontsize=12.5)
        ax.set_xlim(0, max(ks) * 1.04)
        lo, hi = min(ys), max(ys)
        pad = max(0.15, (hi - lo) * 0.35)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=.3)
        ax.legend(fontsize=9, loc="lower right")
        fig.tight_layout()
        p = os.path.join(args.exp_dir, "curve.png")
        fig.savefig(p, dpi=150)
        print(f"-> {p}")


if __name__ == "__main__":
    main()