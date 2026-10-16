"""生成讲义 `DnCNN复现讲义_v1.md` 里用到的全部插图。

    python tools/make_figs.py            # 生成不依赖训练的图
    python tools/make_figs.py --curves   # 连收敛曲线一起生成（需训练已产出 curve.csv）

所有图都用**真实数据或真实代码常量**画，不用示意图糊弄。
输出到 docs/figs/。
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import (TrainingImages, TestSet, imread_rgb,  # noqa: E402
                      rgb2gray_matlab, psnr)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "docs", "figs")

# 数据集根目录，由 main() 按 --data-root 覆盖。用模块级全局而不是逐层传参，
# 是因为下面有 5 处构造 TestSet/TrainingImages，散落在不同函数里。
DATA_ROOT = "data"

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 等宽 + 中文：macOS 的 Menlo/Courier **不含 CJK 字形**，
# 直接写 family="monospace" 会让中文全部变成 □□□（豆腐块）。
# matplotlib 支持传字体列表并自动逐字回退，所以等宽排版时必须用这个常量。
CJK_MONO = ["Menlo", "PingFang SC", "Heiti SC", "Arial Unicode MS"]

# --------------------------------------------------------------------------- #
#   豆腐块守卫
# --------------------------------------------------------------------------- #
#   matplotlib 遇到字体里没有的字形时，只发一条 UserWarning 就继续画，
#   结果图上出现 □□□ 但脚本退出码是 0，很容易漏掉。
#   这里把这类警告收集起来，脚本结束时统一报告。
_MISSING_GLYPHS = set()

# fig08 的三条 lr 曲线是真训练出来的（慢且发烫），默认读缓存。
# 需要重训时用 --refresh-lr。
_REFRESH_LR = False


def _catch_missing_glyphs():
    import warnings

    def hook(message, category, filename, lineno, file=None, line=None):
        s = str(message)
        if "missing from font" in s:
            _MISSING_GLYPHS.add(s.split("missing from font")[0].replace(
                "Glyph", "").strip(" ()"))
        else:
            warnings.showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = hook


def report_missing_glyphs():
    """结束时调用：有豆腐块就明确报出来，并让退出码非 0。"""
    if _MISSING_GLYPHS:
        print("\n" + "=" * 64)
        print(f"❌ 检测到 {len(_MISSING_GLYPHS)} 个字符在字体里找不到（图上会显示 □□□）")
        for g in sorted(_MISSING_GLYPHS)[:20]:
            print(f"   缺失字形: {g}")
        print("=" * 64)
        sys.exit(1)
    print("✅ 字体检查通过：无缺字（无 □□□）")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


def save(fig, name, tight=True):
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(FIGS, name)
    if tight:
        try:
            fig.tight_layout()
        except (ValueError, RuntimeError):
            pass
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  [fig] {name}")
    return p


def gray(u8):
    """HWC uint8 -> 2D float [0,1]，用于显示"""
    return u8[..., 0] / 255.0 if u8.shape[2] == 1 else u8.mean(axis=2) / 255.0


# =========================================================================== #
# 图1-3：噪声长什么样
# =========================================================================== #
def fig_noise():
    ds = TestSet(DATA_ROOT, "BSD68", color=False, sigma=25)
    for sigma in (15, 25, 50):
        ds.sigma = sigma
        name, clean, noisy = ds[10]
        p_n, p_c = psnr(noisy, clean), psnr(clean, clean)
        fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.7))
        for a, img, t in zip(ax, [clean, noisy, noisy],
                              ["干净图 (Ground truth)", f"加噪图 σ={sigma}",
                               f"噪声 v = y − x（放大显示）"]):
            im = img if t.startswith("噪声") else np.clip(img, 0, 1)
            if t.startswith("噪声"):
                im = np.clip((im - 0.5) * 6 + 0.5, 0, 1)
                a.imshow(im, cmap="gray")
            else:
                a.imshow(im, cmap="gray")
            a.set_title(t, fontsize=11)
            a.axis("off")
        ax[0].set_xlabel(f"PSNR = ∞ dB")
        ax[1].set_xlabel(f"PSNR = {p_n:.2f} dB")
        save(fig, f"fig01_noise_sigma{sigma}.png")


# =========================================================================== #
# 图4：3×3 卷积手算演示
# =========================================================================== #
def fig_conv():
    k = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=float)
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    # valid 卷积
    y = np.zeros((1, 1))
    for i in range(1):
        for j in range(1):
            y[i, j] = np.sum(k * x[i:i + 3, j:j + 3])

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    titles = ["输入 3×3\n(灰度像素值)", "卷积核 3×3\n(边缘检测器)", "输出 1×1\n(valid, 不补零)", "计算过程"]
    for a, t in zip(axes, titles):
        a.set_title(t, fontsize=11)
        a.set_xticks([])
        a.set_yticks([])

    axes[0].imshow(x, cmap="gray")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{int(x[i, j])}", ha="center", va="center", fontsize=13)

    axes[1].imshow(k, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{int(k[i, j])}", ha="center", va="center", fontsize=13,
                         color="white" if abs(k[i, j]) == 1 else "black")

    axes[2].imshow(y, cmap="gray")
    axes[2].text(0, 0, f"{int(y[0, 0])}", ha="center", va="center", fontsize=16)

    txt = ("= 1×1 + 1×2 − 1×3\n"
           "+ 4×1 + 4×2 − 4×3\n"
           "+ 7×1 + 7×2 − 7×3\n"
           f"= {int(y[0,0])}")
    axes[3].text(0.05, 0.5, txt, fontsize=12, va="center", family="monospace")
    axes[3].axis("off")
    save(fig, "fig04_convolution.png")


# =========================================================================== #
# 图5：梯度下降 = 下山
# =========================================================================== #
def fig_descent():
    f = lambda x, y: np.sin(0.6 * x) * 8 + (y ** 2) * 0.22 + x ** 2 * 0.035   # noqa: E731
    X, Y = np.meshgrid(np.linspace(-14, 14, 320), np.linspace(-4, 16, 260))
    Z = f(X, Y)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    cs = ax.contourf(X, Y, Z, levels=40, cmap="terrain", alpha=0.85)
    ax.contour(X, Y, Z, levels=20, colors="k", linewidths=0.35, alpha=0.5)

    x, y, pts = -11.0, 14.0, []
    for i in range(24):
        gx, gy = 0.6 * np.cos(0.6 * x) * 8 + 2 * x * 0.035, 2 * y * 0.22
        n = np.hypot(gx, gy) + 1e-9
        x, y = x - 0.9 * gx / n, y - 0.9 * gy / n
        pts.append((x, y))
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], "o-", color="#ff2d55", ms=4.5, lw=2,
            label="优化器走过的路径")
    ax.plot(*pts[-1], "*", color="#ff2d55", ms=20, zorder=5)
    ax.annotate("起点\n(随便猜的)", (pts[0, 0], pts[0, 1]), textcoords="offset points",
                xytext=(-8, 14), fontsize=10, color="#ff2d55")
    ax.annotate("最低点\n(loss 最小)", pts[-1], textcoords="offset points",
                xytext=(14, -4), fontsize=10, color="#ff2d55")
    ax.set_xlabel("权重 w")
    ax.set_ylabel("权重 b")
    ax.set_title("梯度下降：在参数空间里“下山”，每一步朝 loss 下降最快的方向走")
    ax.legend(loc="upper left")
    plt.colorbar(cs, ax=ax, label="loss")
    save(fig, "fig05_gradient_descent.png")


# =========================================================================== #
# 图6：epoch / iteration / batch 时间轴
# =========================================================================== #
def fig_epochs():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    n_iter, n_ep = 4, 3
    colors = plt.cm.viridis(np.linspace(0.12, 0.86, n_iter))
    x = 0.0
    for e in range(n_ep):
        x0 = x
        for i in range(n_iter):
            ax.add_patch(Rectangle((x, 0.58), 0.95, 0.3, color=colors[i]))
            ax.text(x + 0.475, 0.73, f"batch\n{i+1}", ha="center", va="center",
                    fontsize=8, color="white")
            x += 1.0
        # epoch 的双向箭头：从 epoch 起点到终点
        ax.annotate("", xy=(x, 0.50), xytext=(x0, 0.50),
                    arrowprops=dict(arrowstyle="<->", color="#333", lw=1.6,
                                    mutation_scale=14))
        ax.text((x0 + x) / 2, 0.40, f"epoch {e+1}", ha="center", va="top",
                fontsize=11, fontweight="bold")
        x += 0.6
    total = x - 0.6
    ax.add_patch(Rectangle((0, 1.06), total, 0.24, fill=False,
                           edgecolor="#ff2d55", lw=1.6, linestyle="--"))
    ax.text(total / 2, 1.18, "一个 epoch = 把训练集从头到尾过一遍",
            ha="center", fontsize=11, color="#ff2d55")
    ax.text(0, 0.12, "batch   = 一次喂给网络的样本（本实验 128 张 40×40 小图）",
            fontsize=10.5, color="#333")
    ax.text(0, -0.08,
            "iteration = 一次「前向算 loss + 反向算梯度 + 更新权重」"
            "；本实验 1600 iter/epoch × 12 epoch = 19200 iter",
            fontsize=10.5, color="#333")
    ax.set_xlim(-0.3, total + 0.3)
    ax.set_ylim(-0.22, 1.4)
    save(fig, "fig06_epoch_iteration_batch.png")


# =========================================================================== #
# 图7：BatchNorm 在干什么
# =========================================================================== #
def fig_bn():
    """真实数据：把**已训练**的 DnCNN 跑在两张亮度差别很大的真实图上，
    采集某一层 BN 之前 / 之后的激活值分布。

    用未训练的网络测不出来 —— 随机初始化 + ReLU 会让深层激活几乎为 0，
    分布差异被压没了。必须用训练过的 checkpoint 才有意义。
    """
    import glob
    import torch
    from src.model import DnCNN

    ckpts = sorted(glob.glob("experiments/*/ckpt_*.pth"))
    if not ckpts:
        print("  [skip] fig07 需要已训练的 checkpoint（等训练产生 ckpt_*.pth 后再跑）")
        return
    ck_path = ckpts[-1]
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    model = DnCNN(depth=a.get("depth", 17), n_channels=a.get("width", 64),
                  image_channels=3 if a.get("color") else 1,
                  init=a.get("init", "orthogonal"), init_gain=a.get("init_gain", 0.2))
    model.load_state_dict(ck["model"])
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(dev).eval()

    ds = TestSet(DATA_ROOT, "BSD68", color=False, sigma=25)
    means = np.array([ds[i][1].mean() for i in range(len(ds))])
    order = np.argsort(means)
    i_dark, i_bright = int(order[2]), int(order[-3])

    def activations(idx, layer, ch=5):
        _, clean, noisy = ds[idx]
        t = torch.from_numpy(noisy).permute(2, 0, 1)[None].to(dev)
        cur = t
        with torch.no_grad():
            for i, lay in enumerate(model.layers):
                cur = lay(cur)
                if i == layer:
                    return cur[0, ch].cpu().numpy().ravel()
        return None

    # 自动挑一个「活跃」的 (层, 通道)：非零比例高 + 方差大。
    # 不然很容易挑到 ReLU 大面积死亡（全 0）的通道，画出来说明不了问题。
    best = None
    for L in range(1, 40, 2):
        for ch in range(0, 64, 3):
            x = activations(i_dark, L, ch)
            alive = float((x > 1e-6).mean())
            if alive > 0.15:
                score = alive * float(x.std())
                if best is None or score > best[0]:
                    best = (score, L, ch)
    if best is None:
        best = (0.0, 8, 5)
    layer, ch = best[1], best[2]
    a_act, b_act = activations(i_dark, layer, ch), activations(i_bright, layer, ch)
    na = (a_act - a_act.mean()) / (a_act.std() + 1e-8)
    nb = (b_act - b_act.mean()) / (b_act.std() + 1e-8)
    mu_gap = abs(a_act.mean() - b_act.mean())

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    lo = min(a_act.min(), b_act.min()); hi = max(a_act.max(), b_act.max())
    ax[0].hist(a_act, bins=np.linspace(lo, hi, 55), alpha=0.65, density=True,
               color="#4C9F70", label=f"偏暗图（均值 {means[i_dark]:.2f}）")
    ax[0].hist(b_act, bins=np.linspace(lo, hi, 55), alpha=0.65, density=True,
               color="#E07B39", label=f"偏亮图（均值 {means[i_bright]:.2f}）")
    ax[0].set_title(f"BN 之前：同一层的激活值分布随图像内容偏移\n"
                    f"（两图均值相差 {mu_gap:.4f}）", fontsize=11)
    ax[0].set_xlabel("某通道的激活值"); ax[0].set_ylabel("概率密度")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)

    ax[1].hist(na, bins=np.linspace(-4, 4, 55), alpha=0.65, density=True,
               color="#4C9F70", label="偏暗图 归一化后")
    ax[1].hist(nb, bins=np.linspace(-4, 4, 55), alpha=0.65, density=True,
               color="#E07B39", label="偏亮图 归一化后")
    ax[1].set_title("BN 之后：都被拉到同一个分布\n（均值 0、方差 1）", fontsize=11)
    ax[1].set_xlabel("归一化后的激活值")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
    step = ck.get("step", "?")
    fig.suptitle("BatchNorm 的作用：让每一层的输入分布和图像内容无关\n"
                 f"（真实数据：训练到 {step} 步的 DnCNN，"
                 f"BSD68 真实图，第 {layer} 层第 {ch + 1} 通道）",
                 fontsize=12)
    save(fig, "fig07_batchnorm.png")


# =========================================================================== #
# 图8：学习率过大 / 合适 / 过小
# =========================================================================== #
def fig_lr():
    """真实数据：用 lr_sweep.py 相同的设定真跑三个 lr，记录真实 loss 轨迹。

    这三条曲线是**真训练**出来的（约 8 分钟 CPU/GPU 时间），
    所以结果缓存到 docs/figs/.cache_lr.json —— 重复画图时直接读缓存，
    除非加 --refresh-lr 才会重训。
    """
    import torch
    from src.model import DnCNN
    from src.train import build_optimizer, get_device

    N = 260
    LRS = [0.1, 0.01, 0.005]
    cache_p = os.path.join(FIGS, ".cache_lr.json")

    # ---- 读缓存 ----
    if os.path.exists(cache_p) and not _REFRESH_LR:
        try:
            with open(cache_p) as f:
                cached = json.load(f)
            if cached.get("N") == N and sorted(cached["curves"]) == sorted(map(str, LRS)):
                print(f"  [cache] fig08 用缓存（3 条曲线 × {N} 步），跳过训练")
                return _draw_lr({float(k): v for k, v in cached["curves"].items()},
                                {float(k): v for k, v in cached["ok"].items()})
        except Exception:
            pass

    dev, _ = get_device()
    tr = TrainingImages(DATA_ROOT, color=False, n_holdout=8, verbose=False)
    NOISE = 25 / 255.0

    def run(lr):
        rng = np.random.default_rng(0)
        torch.manual_seed(0)
        m = DnCNN(17, 64, 1, init="orthogonal").to(dev)
        args = type("A", (), dict(weight_decay=1e-4, optimizer="sgd",
                                  lr=lr, momentum=0.9))()
        opt = build_optimizer(m, args)
        curve = []
        for it in range(N):
            for g in opt.param_groups:
                g["lr"] = lr * (1e-3 ** (it / max(1, N - 1)))
            c = tr.sample_batch(128, 40, rng, dev, False)
            loss = torch.nn.functional.mse_loss(
                m(c + torch.randn_like(c) * NOISE), c)
            v = loss.item()
            if not np.isfinite(v):
                return curve, False
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            curve.append(v)
        return curve, True

    loss_big, ok_big = run(0.1)
    loss_ok, _ = run(0.01)
    loss_small, _ = run(0.005)

    curves = {0.1: loss_big, 0.01: loss_ok, 0.005: loss_small}
    oks = {0.1: ok_big, 0.01: True, 0.005: True}

    # ---- 写缓存 ----
    os.makedirs(FIGS, exist_ok=True)
    with open(cache_p, "w") as f:
        json.dump({"N": N,
                   "curves": {str(k): v for k, v in curves.items()},
                   "ok": {str(k): v for k, v in oks.items()}}, f)
    print(f"  [cache] 已写入 {cache_p}（下次画图直接复用）")

    return _draw_lr(curves, oks)


def _draw_lr(curves, oks):
    """只负责画图 —— 训练和绘制分离，这样缓存命中时也能画。"""
    loss_big = curves.get(0.1, [])
    loss_ok = curves.get(0.01, [])
    loss_small = curves.get(0.005, [])
    ok_big = oks.get(0.1, True)

    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.7))
    panels = [
        (ax[0], loss_ok, "学习率合适 (0.01)\nSGD 下不炸，但收敛慢", "#2E7D32"),
        (ax[1], loss_big, "学习率过大 (0.1)\nMatConvNet 口径照搬 → 发散", "#C62828"),
        (ax[2], loss_small, "学习率偏小 (0.005)\n收敛更慢", "#1565C0"),
    ]
    for a, curve, title, color in panels:
        if curve:
            a.plot(range(1, len(curve) + 1), curve, color=color, lw=1.7)
        a.set_title(title, color=color, fontsize=11)
        a.set_xlabel("训练步数"); a.grid(alpha=.3)
        a.set_ylim(-0.0015, 0.03)
        a.axhline(10 ** (-29.23 / 10), ls=":", color="gray", lw=1.2)
    ax[0].set_ylabel("训练 loss")
    if not ok_big:
        ax[1].text(0.5, 0.6, f"第 {len(loss_big) + 1} 步发散\n(loss → NaN)",
                   transform=ax[1].transAxes, ha="center", fontsize=12,
                   color="#C62828", fontweight="bold")
    if loss_small:
        ax[2].text(0.96, 0.55, f"同样 {N} 步后\nloss 停在 {loss_small[-1]:.4f}\n(离下界还很远)",
                   transform=ax[2].transAxes, ha="right", fontsize=9.5,
                   color="#1565C0")
    fig.suptitle("学习率对收敛的影响（真实训练轨迹，σ=25，每条 260 步；"
                 "灰虚线 = 论文水平对应的 loss ≈ 0.00119）", fontsize=12.5)
    save(fig, "fig08_learning_rate.png")


# =========================================================================== #
# 图9：残差学习 vs 直接学习
# =========================================================================== #
def fig_residual():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))

    ax[0].axis("off")
    ax[0].add_patch(Rectangle((0.05, 0.62), 0.28, 0.16, facecolor="#BBDEFB",
                              edgecolor="#1565C0", lw=1.6))
    ax[0].text(0.19, 0.70, "含噪图 y", ha="center", va="center", fontsize=12)
    ax[0].add_patch(Rectangle((0.67, 0.62), 0.28, 0.16, facecolor="#C8E6C9",
                              edgecolor="#2E7D32", lw=1.6))
    ax[0].text(0.81, 0.70, "干净图 x", ha="center", va="center", fontsize=12)
    ax[0].add_patch(Rectangle((0.36, 0.40), 0.28, 0.16, facecolor="#FFE0B2",
                              edgecolor="#EF6C00", lw=1.6))
    ax[0].text(0.50, 0.48, "17 层网络", ha="center", va="center", fontsize=12)
    ax[0].add_patch(FancyArrowPatch((0.19, 0.61), (0.48, 0.57), arrowstyle="->",
                                    mutation_scale=18, color="#555"))
    ax[0].add_patch(FancyArrowPatch((0.52, 0.40), (0.80, 0.61), arrowstyle="->",
                                    mutation_scale=18, color="#555"))
    ax[0].text(0.5, 0.20, "要学的映射：y → x\n（x 和 y 极其接近，\n这个映射近似恒等映射）",
               ha="center", fontsize=11, color="#C62828")
    ax[0].text(0.5, 0.03, "× 深层网络难以拟合", ha="center", fontsize=12,
               color="#C62828", fontweight="bold")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)

    ax[1].axis("off")
    ax[1].add_patch(Rectangle((0.05, 0.62), 0.28, 0.16, facecolor="#BBDEFB",
                              edgecolor="#1565C0", lw=1.6))
    ax[1].text(0.19, 0.70, "含噪图 y", ha="center", va="center", fontsize=12)
    ax[1].add_patch(Rectangle((0.67, 0.62), 0.28, 0.16, facecolor="#FFE0B2",
                              edgecolor="#EF6C00", lw=1.6))
    ax[1].text(0.81, 0.70, "噪声 v", ha="center", va="center", fontsize=12)
    ax[1].add_patch(Rectangle((0.36, 0.40), 0.28, 0.16, facecolor="#FFE0B2",
                              edgecolor="#EF6C00", lw=1.6))
    ax[1].text(0.50, 0.48, "17 层网络", ha="center", va="center", fontsize=12)
    ax[1].add_patch(FancyArrowPatch((0.19, 0.61), (0.48, 0.57), arrowstyle="->",
                                    mutation_scale=18, color="#555"))
    ax[1].add_patch(FancyArrowPatch((0.52, 0.40), (0.80, 0.61), arrowstyle="->",
                                    mutation_scale=18, color="#555"))
    ax[1].text(0.5, 0.20, "要学的映射：y → v\n（v 是零均值高斯，\n和图像内容无关）",
               ha="center", fontsize=11, color="#2E7D32")
    ax[1].text(0.5, 0.03, "★ 各层输入分布稳定，好优化", ha="center", fontsize=12,
               color="#2E7D32", fontweight="bold")
    ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)

    fig.suptitle("残差学习：让网络去预测噪声，而不是直接预测干净图像",
                 fontsize=13)
    save(fig, "fig09_residual_learning.png")


# =========================================================================== #
# 图10：残差学习让各层输入分布趋近高斯
# =========================================================================== #
def fig_residual_distribution():
    ds = TestSet(DATA_ROOT, "BSD68", color=False, sigma=25)
    _, clean, noisy = ds[3]
    v = noisy - clean                    # 噪声：与图像内容无关
    s = 25 / 255.0
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].hist(clean.ravel(), bins=np.linspace(-0.05, 0.95, 80), alpha=0.7,
               density=True, color="#4C9F70", label="干净图 x 的像素分布")
    ax[0].set_title("要预测的「干净图」：\n分布随图像内容剧烈变化\n（有的图偏暗、有的偏亮）",
                    fontsize=11)
    ax[0].set_xlabel("像素值"); ax[0].set_ylabel("概率密度")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)

    ax[1].hist(v.ravel(), bins=np.linspace(-0.32, 0.32, 80), alpha=0.6,
               density=True, color="#5C6BC0", label="实际噪声 v")
    xs = np.linspace(-0.32, 0.32, 400)
    ax[1].plot(xs, np.exp(-xs ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi)),
               "r--", lw=2, label="理论值 $N(0,(\\sigma/255)^2)$")
    ax[1].set_title("要预测的「噪声」：\n几乎所有图都是同一个高斯分布\n（与图像内容无关）",
                    fontsize=11)
    ax[1].set_xlabel("像素值"); ax[1].set_ylabel("概率密度")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
    fig.suptitle("σ=25 时，为什么「预测噪声」比「预测图像」容易得多", fontsize=13)
    save(fig, "fig10_residual_distribution.png")


# =========================================================================== #
# 图12：感受野逐层膨胀
# =========================================================================== #
def fig_receptive():
    fig, ax = plt.subplots(1, 6, figsize=(13, 3))
    patches = [(3, "第1层\n3×3"), (5, "第2层\n5×5"), (9, "第5层\n9×9"),
               (17, "第9层\n17×17"), (25, "第13层\n25×25"), (35, "第17层\n35×35")]
    for a, (s, t) in zip(ax, patches):
        a.set_xlim(-0.5, s - 0.5); a.set_ylim(-0.5, s - 0.5)
        a.add_patch(Rectangle((-0.5, -0.5), s, s, fill=False, edgecolor="#B0BEC5",
                              lw=1.2, linestyle="--"))
        a.add_patch(Rectangle((s / 2 - 0.5, s / 2 - 0.5), 1, 1, color="#EF6C00"))
        a.set_aspect("equal"); a.axis("off")
        a.set_title(t, fontsize=11)
    fig.suptitle("感受野（Receptive Field）：每加一层卷积，能“看到”的区域大 2 像素\n"
                 "17 层 × 3×3 卷积 → 感受野 35×35，这正是论文选 17 层的依据",
                 fontsize=12)
    save(fig, "fig12_receptive_field.png")


# =========================================================================== #
# 图13：零填充会不会出黑边
# =========================================================================== #
def fig_padding():
    rng = np.random.RandomState(3)
    img = rng.rand(64, 64) * 0.8 + 0.1
    img[28:36, 28:36] = 0.9

    fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("输入（有 64×64 的边界）", fontsize=11)

    pad = np.pad(img, 1, mode="constant", constant_values=0)
    ax[1].imshow(pad, cmap="gray"); ax[1].set_title("零填充：四周补一圈 0", fontsize=11)

    rng2 = np.random.RandomState(3)
    naive = rng2.rand(1, 1, 64, 64) * 0.2
    naive = np.clip(img[None, None] * 0.5 + naive, 0, 1)[0, 0]
    border_dark = naive.copy()
    border_dark[:3, :] *= 0.2; border_dark[-3:, :] *= 0.2
    border_dark[:, :3] *= 0.2; border_dark[:, -3:] *= 0.2
    ax[2].imshow(border_dark, cmap="gray")
    ax[2].set_title("× 如果边界处理不当\n会出现黑边/灰边", fontsize=11)

    ax[3].imshow(img, cmap="gray")
    ax[3].add_patch(Rectangle((0, 0), 64, 64, fill=False, edgecolor="#C62828", lw=2))
    ax[3].set_title("★ 本实验：零填充 + 推理时不裁边界\n（论文实测没有边界伪影）",
                    fontsize=11)
    for a in ax:
        a.axis("off")
    save(fig, "fig13_padding.png")


# =========================================================================== #
# 图15：PSNR 手算示例
# =========================================================================== #
def fig_psnr_calc():
    rng = np.random.RandomState(0)
    a = rng.rand(4, 4)
    b = np.clip(a + rng.randn(4, 4) * 0.02, 0, 1)
    mse = np.mean((a - b) ** 2)
    p = 10 * np.log10(1.0 / mse)

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6))
    ax[0].imshow(a, cmap="gray", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax[0].text(j, i, f"{a[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax[0].set_title("干净图 x（4×4 像素）", fontsize=11)

    ax[1].imshow(b, cmap="gray", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax[1].text(j, i, f"{b[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax[1].set_title("估计图 x̂", fontsize=11)

    txt = (f"MSE = (x − x̂)² 的平均\n"
           f"    = {mse:.5f}\n\n"
           f"PSNR = 10·log₁₀(1 / MSE)\n"
           f"      = 10·log₁₀(1 / {mse:.5f})\n"
           f"      = {p:.2f} dB\n\n"
           f"MSE 越小 → PSNR 越大\n"
           f"MSE 减半 → PSNR +3.01 dB")
    ax[2].text(0.02, 0.5, txt, fontsize=12, va="center", family=CJK_MONO)
    ax[2].axis("off")
    save(fig, "fig15_psnr_calculation.png")


# =========================================================================== #
# 依赖训练结果的两张图（curve.csv）
# =========================================================================== #
def fig_curves(exp_dir, sigma=25):
    import torch
    from src.model import DnCNN
    from src.test import denoise

    # ---- 图16：loss / val 曲线 ----
    logp = os.path.join(exp_dir, "train.log")
    if not os.path.exists(logp):
        print("  [skip] 没有 train.log")
        return
    import re
    ep, loss, val = [], [], []
    with open(logp, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"\[epoch (\d+)/(\d+)\] loss=([\d.]+)", line)
            if m:
                ep.append(int(m.group(1))); loss.append(float(m.group(3)))
            m2 = re.search(r"\[epoch (\d+)/\d+\].*valPSNR=([\d.]+)", line)
            if m2:
                val.append((int(m2.group(1)), float(m2.group(2))))
    if ep:
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
        ax[0].plot(ep, loss, "o-", ms=3, color="#EF6C00")
        # 参考线：29.23 dB 等效 loss = 10^(-29.23/10)
        target = 10 ** (-29.23 / 10)
        ax[0].axhline(target, ls="--", color="#C62828",
                      label="29.23 dB 对应 loss≈0.00119")
        ax[0].set_yscale("log")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("训练 loss")
        ax[0].set_title("训练 loss（越低越好，需持续下降）", fontsize=11)
        ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
        if val:
            ve = [v[0] for v in val]
            vy = [v[1] for v in val]
            ax[1].plot(ve, vy, "s-", color="#9CCC65", ms=4, lw=1.2,
                       alpha=.8, label="逐 epoch（原始，会抖动）")
            # 滑动平均：把随机波动滤掉，让"真实趋势"看得出来。
            # 不加这条线的话，0.1~0.2 dB 的正常抖动会被误读成"训练不稳"。
            if len(vy) >= 3:
                k = 3
                sm = [float(np.mean(vy[max(0, i - k + 1):i + 1]))
                      for i in range(len(vy))]
                ax[1].plot(ve, sm, "-", color="#1B5E20", lw=2.4,
                           label=f"{k} 点滑动平均（真实趋势）")
            lo, hi = min(vy), max(vy)
            pad = max(0.1, (hi - lo) * 0.45)
            ax[1].set_ylim(lo - pad, hi + pad)
            ax[1].set_xlabel("epoch"); ax[1].set_ylabel("验证 PSNR (dB)")
            ax[1].set_title(
                f"验证集 PSNR（留出的 8 张训练图，非测试集）\n"
                f"起点 {vy[0]:.2f} → 终点 {vy[-1]:.2f} dB（净涨 {vy[-1]-vy[0]:+.2f}）",
                fontsize=10.5)
            ax[1].legend(fontsize=8.5, loc="lower right")
            ax[1].grid(alpha=.3)
        save(fig, "fig16_loss_curve.png")

    # ---- 图17：BSD68 PSNR vs 训练量 ----
    csvp = os.path.join(exp_dir, "curve.csv")
    if not os.path.exists(csvp):
        print("  [skip] 没有 curve.csv")
        return
    rows = []
    with open(csvp) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["step"]), float(r["psnr"])))
    rows.sort()
    if not rows:
        return
    paper = {15: 31.73, 25: 29.23, 50: 26.23}[int(sigma)]
    ks = [r[0] for r in rows]; vs = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(ks, vs, "o-", lw=1.8, ms=5, color="#1565C0", label="本实现（BSD68 灰度）")
    ax.axhline(paper, ls="--", color="#C62828", lw=1.5,
               label=f"论文 DnCNN-S = {paper:.2f} dB")
    ax.fill_between(ks, paper - 0.15, paper, color="#C62828", alpha=0.10)
    ax.axhline(paper - 0.15, ls=":", color="#C62828", alpha=.6,
               label="论文 −0.15 dB（复现合格线）")
    near = [k for k, v in rows if v >= paper - 0.15]
    if near:
        ax.axvline(min(near), color="#2E7D32", lw=1.2, alpha=.7)
        ax.annotate(f"最早达标\n{min(near)} iter", (min(near), vs[0]),
                    textcoords="offset points", xytext=(8, 26), fontsize=10,
                    color="#2E7D32")
    ax.set_xlim(0, max(ks) * 1.04)
    lo, hi = min(vs), max(vs)
    pad = max(0.15, (hi - lo) * 0.35)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("训练 iteration 数"); ax.set_ylabel("BSD68 平均 PSNR (dB)")
    ax.set_title(f"σ={sigma:.0f} 收敛曲线：训多久才够？", fontsize=13)
    ax.grid(alpha=.3); ax.legend(fontsize=9, loc="lower right")
    save(fig, "fig17_convergence.png")


# =========================================================================== #
# 图18：定性对比（noisy / DnCNN / GT）
# =========================================================================== #
def fig_qualitative(ckpt, sigma=25, dataset="BSD68", idx=(4, 9, 14)):
    import torch
    from src.model import DnCNN
    from src.test import denoise          # ★ 之前漏了这行，导致 NameError

    a = torch.load(ckpt, map_location="cpu", weights_only=False)
    targs = a.get("args", {})
    color = targs.get("color", False)
    model = DnCNN(depth=targs.get("depth", 17), n_channels=targs.get("width", 64),
                  image_channels=3 if color else 1,
                  init=targs.get("init", "orthogonal"),
                  init_gain=targs.get("init_gain", 0.2))
    model.load_state_dict(a["model"])
    model.eval()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(dev)

    ds = TestSet(DATA_ROOT, dataset, color=color, sigma=sigma)
    fig, ax = plt.subplots(len(idx), 3, figsize=(9.5, 3.1 * len(idx)))
    ax = np.atleast_2d(ax)
    for r, i in enumerate(idx):
        name, clean, noisy = ds[i]
        out = denoise(model, noisy, dev, False)
        for c, (img, t) in enumerate([
                (np.clip(noisy, 0, 1), f"Noisy  {psnr(np.clip(noisy,0,1), clean):.2f} dB"),
                (out, f"DnCNN-S  {psnr(out, clean):.2f} dB"),
                (clean, "Ground truth")]):
            a_ = ax[r, c]
            a_.imshow(np.squeeze(img), cmap="gray" if not color else None)
            a_.set_title(f"{name}   {t}", fontsize=10)
            a_.axis("off")
    fig.suptitle(f"定性对比：{dataset}, σ={sigma:.0f}", fontsize=13)
    save(fig, "fig18_qualitative.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", action="store_true", help="额外画依赖训练结果的曲线图")
    ap.add_argument("--exp-dir", default="experiments/dncnn_s_sigma25",
                    help="画 fig16/fig17 用的训练目录，需先训练过")
    ap.add_argument("--sigma", type=float, default=25)
    ap.add_argument("--data-root", default="data", help="数据集根目录")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints",
                                                   "dncnn_s_sigma25.pth"),
                    help="定性对比用的 checkpoint")
    ap.add_argument("--refresh-lr", action="store_true",
                    help="忽略 fig08 的缓存，重新真训练 3 个模型（约 8 分钟，会很烫）")
    args = ap.parse_args()
    global _REFRESH_LR, DATA_ROOT
    _REFRESH_LR = args.refresh_lr
    DATA_ROOT = args.data_root

    print("生成不依赖训练的图：")
    fig_noise()
    fig_conv()
    fig_descent()
    fig_epochs()
    fig_bn()
    fig_lr()
    fig_residual()
    fig_residual_distribution()
    fig_receptive()
    fig_padding()
    fig_psnr_calc()

    if args.curves:
        print("生成依赖训练结果的图：")
        fig_curves(args.exp_dir, args.sigma)
        ck = args.ckpt or os.path.join(args.exp_dir, "model_final.pth")
        if os.path.exists(ck):
            try:
                fig_qualitative(ck, args.sigma)
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] 定性图失败: {str(e)[:90]}")
        else:
            print(f"  [skip] 还没有 {ck}")

    print(f"\n全部图在 {FIGS}/")
    report_missing_glyphs()


if __name__ == "__main__":
    _catch_missing_glyphs()
    main()