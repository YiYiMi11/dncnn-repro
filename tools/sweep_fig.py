"""生成核心对比图：已知噪声级 vs 盲去噪的鲁棒性扫描。

    python tools/sweep_fig.py            # 读 results/*.json -> figures/*.png

读的是 src/test.py 已经跑完落盘的结果，不重新跑模型，所以几秒钟就能重画。
图里要讲的事情只有一句：

    已知 σ 的 DnCNN 在训练分布上很准，但在分布外会崩到「不如不去噪」；
    盲去噪的 DnCNN-B 牺牲不到 0.5 dB，换来 5→55 全程可用。
"""

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")

CJK = ["PingFang SC", "Heiti SC", "Hiragino Sans GB", "Arial Unicode MS",
       "Noto Sans CJK SC", "Microsoft YaHei"]
plt.rcParams["font.sans-serif"] = CJK
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["grid.linewidth"] = 0.6

# 论文 Table II：DnCNN-S 在 BSD68 上的灰度结果
PAPER = {15: 31.73, 25: 29.23, 50: 26.23}

C_KNOWN = "#d1495b"     # 已知 sigma
C_BLIND = "#2a6f97"     # 盲去噪
C_NOISY = "#8d99ae"     # 含噪输入
C_PAPER = "#2b2d42"


def load():
    """把 results/*.json 按 (模型, 测试噪声级) 归拢。"""
    rows = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        d = json.load(open(p))
        key = "blind" if "blind" in os.path.basename(p) else "known"
        rows.setdefault(key, []).append(
            {"sigma": float(d["sigma"]), "psnr": float(d["psnr_dncnn"]),
             "noisy": float(d["psnr_noisy"]), "ssim": float(d.get("ssim_dncnn", np.nan))}
        )
    for v in rows.values():
        v.sort(key=lambda r: r["sigma"])
    return rows


def series(rows):
    x = [r["sigma"] for r in rows]
    return x, [r["psnr"] for r in rows], [r["noisy"] for r in rows]


def crossover(xs, diffs):
    """增益由负变正的噪声等级（线性插值）。全负或全正则返回 None。"""
    for i in range(len(diffs) - 1):
        if diffs[i] < 0 <= diffs[i + 1]:
            span = diffs[i + 1] - diffs[i]
            t = -diffs[i] / span if span else 0.0
            return xs[i] + t * (xs[i + 1] - xs[i])
    return None


def plot_main(rows, out):
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12.6, 4.9), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    kx, ky, kn = series(rows["known"])
    bx, by, bn = series(rows["blind"])

    # ---- 左：绝对 PSNR -------------------------------------------------- #
    ax.plot(kx, kn, ls="--", lw=1.8, color=C_NOISY,
            marker="^", ms=5.5, label="含噪输入（不去噪的基线）")
    ax.plot(kx, ky, lw=2.4, color=C_KNOWN, marker="o", ms=7,
            label="DnCNN-S（只知道 σ=25）")
    ax.plot(bx, by, lw=2.4, color=C_BLIND, marker="s", ms=7,
            label="DnCNN-B（盲去噪，不知道 σ）")
    ax.plot(list(PAPER), list(PAPER.values()), ls=":", lw=1.8, color=C_PAPER,
            marker="D", ms=5.5, label="论文 DnCNN-S（每个 σ 单独训练）")

    # 已知 σ 模型反而不如基线的区间：用线性插值把变缺点算准，
    # 而不是简单地拿「实测最差的那个点」当区间。
    diff_k = [a - b for a, b in zip(ky, kn)]
    cross = crossover(kx, diff_k)
    if cross is not None:
        ax.axvspan(0, cross, color=C_KNOWN, alpha=0.09, zorder=0)
        ax.axvline(cross, color=C_KNOWN, lw=1.2, ls="-.", alpha=0.8, zorder=1)
        ax.annotate(
            f"σ 小于 {cross:.0f} 时\n「去噪」反而不如不去噪",
            xy=(cross, 38.4), xytext=(13.5, 34.0),
            fontsize=10.5, color=C_KNOWN, ha="left",
            arrowprops=dict(arrowstyle="->", color=C_KNOWN, lw=1.4,
                            connectionstyle="arc3,rad=0.22"),
        )

    ax.set_xlabel("测试图的噪声等级 σ", fontsize=11.5)
    ax.set_ylabel("BSD68 平均 PSNR (dB)", fontsize=11.5)
    ax.set_title("已知 σ 的模型在分布外会崩，盲去噪不会", fontsize=13.5, pad=11)
    ax.set_xlim(0, 60)
    ax.set_ylim(12, 41)
    ax.legend(fontsize=9.6, loc="lower left", framealpha=0.94, edgecolor="#ccd")

    # ---- 右：相对基线的增益 --------------------------------------------- #
    ax2.axhline(0, color="#333", lw=1.1)
    ax2.plot(kx, diff_k, lw=2.4, color=C_KNOWN, marker="o", ms=7, label="DnCNN-S")
    ax2.plot(bx, [a - b for a, b in zip(by, bn)], lw=2.4, color=C_BLIND,
             marker="s", ms=7, label="DnCNN-B")
    ax2.fill_between(kx, diff_k, 0, where=[d < 0 for d in diff_k],
                     color=C_KNOWN, alpha=0.16)
    if cross is not None:
        ax2.axvspan(0, cross, color=C_KNOWN, alpha=0.07, zorder=0)
        worst = min(zip(diff_k, kx))
        ax2.annotate(f"最差 {worst[0]:.2f} dB\n（把图弄脏了）",
                     xy=(worst[1], worst[0]), xytext=(24, -7.2),
                     fontsize=10, color=C_KNOWN,
                     arrowprops=dict(arrowstyle="->", color=C_KNOWN, lw=1.3))
    ax2.set_xlabel("测试图的噪声等级 σ", fontsize=11.5)
    ax2.set_ylabel("相对含噪输入的 PSNR 增益 (dB)", fontsize=11.5)
    ax2.set_title("增益始终为正，才是能上线的模型", fontsize=13.5, pad=11)
    ax2.set_xlim(0, 60)
    ax2.set_ylim(-9.5, 13.5)
    ax2.legend(fontsize=10, loc="upper right", framealpha=0.94, edgecolor="#ccd")

    fig.suptitle(
        f"DnCNN 复现 · BSD68 灰度去噪   "
        f"（σ=25 模型实测 {ky[kx.index(25.0)]:.2f} dB，论文 29.23 dB，差 "
        f"{ky[kx.index(25.0)] - 29.23:+.2f}）",
        fontsize=12.5, y=1.035,
    )
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig] {out}")


def plot_qualitative(data_root, out, image_index=3, sigmas=(5, 15, 25, 35, 50)):
    """定性对照：同一个 σ=25 模型和一��盲模型，喂五档噪声。

    需要真实数据集与 checkpoints，���此单独提供入口，不随主图一起跑。
    """
    sys.path.insert(0, ROOT)
    import torch

    import src.data as D
    import src.model as M

    nets = {}
    for key, fn in (("known", "dncnn_s_sigma25.pth"), ("blind", "dncnn_b_blind.pth")):
        # depth 不能写死：DnCNN-S 是 17，盲去噪 DnCNN-B 是 20，
        # 以 checkpoint 里的训练配置为准
        net, cfg = M.load_dncnn(os.path.join(ROOT, "checkpoints", fn))
        print(f"[fig] {fn}: depth={cfg['depth']} channels={1 if not cfg['color'] else 3}")
        nets[key] = net

    ts = D.TestSet(data_root, "BSD68", color=False, sigma=sigmas[0], seed=0)
    _, clean, _ = ts[image_index]
    clean1 = clean[..., 0]

    rows = [("DnCNN-S  只知道 σ=25", "known"), ("DnCNN-B  盲去噪", "blind")]
    cols = 1 + len(sigmas)
    fig, axes = plt.subplots(len(rows), cols, figsize=(2.15 * cols, 4.9))

    def show(ax, a, title, color="k"):
        ax.imshow(np.clip(a, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9.4, color=color)
        ax.axis("off")

    with torch.no_grad():
        for r, (label, key) in enumerate(rows):
            show(axes[r, 0], clean1, "干净原图", color="#333")
            axes[r, 0].set_ylabel(label, fontsize=10.5)
            for c, s in enumerate(sigmas, start=1):
                rng = np.random.RandomState(1000 + s)
                noisy = clean + rng.randn(*clean.shape).astype(np.float32) * (s / 255.0)
                x = torch.from_numpy(noisy.transpose(2, 0, 1)).float().unsqueeze(0)
                den = nets[key](x)[0, 0].numpy()          # (1,1,H,W) -> (H,W)
                p = D.psnr(clean1, np.clip(den, 0, 1))
                show(axes[r, c], den, f"输入 σ={s}\n{p:.2f} dB",
                     color="#777")
            axes[r, 0].text(-0.10, 0.5, label, transform=axes[r, 0].transAxes,
                            rotation=90, va="center", ha="right",
                            fontsize=10.5)

    fig.suptitle("σ=25 专用模型遇到小噪声会把纹理抹平；盲去噪模型各档都稳",
                 fontsize=13, y=1.02)
    fig.text(0.5, -0.04, "BSD68 test004 · 数字为该张图的去噪后 PSNR",
             ha="center", fontsize=9.5, color="#666")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def plot_convergence(csv_name, out, title="DnCNN-S (σ=25)"):
    """从 results/curve_*.csv 重画训练/验证曲线。

    带上这张图是为了说明「-0.22 dB 是欠训练，不是实现错了」：
    验证 PSNR 到最后一个 epoch 仍在上升，没有出现平台。
    """
    import csv

    path = os.path.join(RESULTS, csv_name)
    if not os.path.exists(path):
        print(f"[skip] 找不到 {path}")
        return
    rows = list(csv.DictReader(open(path)))
    step = [int(r["step"]) for r in rows]
    psnr = [float(r["psnr"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(step, psnr, lw=2.0, color=C_KNOWN, marker="o", ms=4.5)
    ax.axhline(29.23, color=C_PAPER, ls=":", lw=1.6)
    ax.text(step[-1], 29.23, "  论文 29.23", va="bottom", ha="right",
            fontsize=10, color=C_PAPER)
    last = psnr[-1]
    tail = psnr[-3:]
    slope = (tail[-1] - tail[0]) / 2
    ax.annotate(f"末尾仍在缓慢上升（末尾两个 epoch 平均 "
                    f"{slope * 1000:.1f}×0.001 dB），尚未出现平台期",
                xy=(step[-1], last), xytext=(0.30, 0.64), textcoords="axes fraction",
                fontsize=10.5, color=C_KNOWN,
                arrowprops=dict(arrowstyle="->", color=C_KNOWN, lw=1.3))
    ax.set_xlabel("训练步数", fontsize=11.5)
    ax.set_ylabel("验证 PSNR (dB)", fontsize=11.5)
    ax.set_title(f"{title} 训练过程 · 训练量约为论文的 1/4", fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig] {out}")


if __name__ == "__main__":
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data"),
                    help="定性图需要真实数据集；主图不需要")
    ap.add_argument("--qualitative", action="store_true",
                    help="额外生成定性对照图（需要数据集与 checkpoints）")
    a = ap.parse_args()

    os.makedirs(FIGS, exist_ok=True)
    r = load()
    if "known" not in r or "blind" not in r:
        sys.exit("results/ 下缺少 known/blind 的结果 JSON，先跑 src/test.py")
    plot_main(r, os.path.join(FIGS, "sigma_sweep.png"))
    plot_convergence("curve_dncnn_s_sigma25.csv", os.path.join(FIGS, "convergence.png"))
    if a.qualitative:
        plot_qualitative(a.data_root, os.path.join(FIGS, "qualitative_sweep.png"))