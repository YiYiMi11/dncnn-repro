"""评测 DnCNN：PSNR / SSIM + 定性对比图 + 结果 JSON。

对应论文 Table II（BSD68 灰度）与 Table III（Set12 彩色）。

    # BSD68 灰度
    python -m src.test --ckpt experiments/gray_sigma25/model_final.pth --dataset BSD68 --sigma 25
    # Set12 彩色
    python -m src.test --ckpt experiments/color_sigma25/model_final.pth --dataset Set12 --sigma 25
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from .data import TestSet, psnr, ssim
from .model import DnCNN, load_dncnn
from .train import get_device

# 自适应集成（self-ensembling）：8 种几何变换求平均，8 尺度无关，只在推理时用
def _transform(x, k):
    if k & 4:
        x = torch.flip(x, dims=[-1])
    r = k & 3
    if r:
        x = torch.rot90(x, r, dims=[-2, -1])
    return x


def _inverse(x, k):
    r = k & 3
    if r:
        x = torch.rot90(x, -r, dims=[-2, -1])
    if k & 4:
        x = torch.flip(x, dims=[-1])
    return x


@torch.no_grad()
def denoise(model, noisy, device, self_ensemble=False):
    """noisy: HWC float array -> 去噪后 HWC float（已 clip 到 [0,1]）"""
    t = torch.from_numpy(noisy).permute(2, 0, 1)[None].to(device)
    if not self_ensemble:
        out = model(t)
    else:
        acc = torch.zeros_like(t)
        for k in range(8):
            acc += _inverse(model(_transform(t, k)), k)
        out = acc / 8.0
    out = out.float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()[0]
    return out


def parse_args():
    p = argparse.ArgumentParser("DnCNN 评测")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", default="Set12", choices=["Set12", "BSD68"])
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--self-ensemble", action="store_true", help="8 变换自适应集成")
    p.add_argument("--seed", type=int, default=0, help="测试噪声随机种子")
    p.add_argument("--save-images", action="store_true", help="保存每张去噪结果 png")
    p.add_argument("--merge-bn", action="store_true", help="测试前把 BN 合并进 Conv")
    p.add_argument("--fig", action="store_true", help="生成定性对比图（patch 裁剪）")
    p.add_argument("--fig-patches", type=str, default="",
                   help='定性图指定区域，格式 "name:cx,cy:w" 或 "name:y:x:h:w"，可逗号分隔多个')
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default=None, help="结果 JSON 的标签，默认由 dataset/sigma/ensemble 推导")
    return p.parse_args()


def make_figure(model, ds, args, device, path, kind="DnCNN-S"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = [(2, 0.5, 0.5, 0.30), (4, 0.5, 0.5, 0.30), (6, 0.5, 0.5, 0.30),
           (9, 0.5, 0.5, 0.30)]
    sel = [s for s in sel if s[0] < len(ds)][:4]
    fig, axes = plt.subplots(len(sel), 3, figsize=(9, 3 * len(sel)))
    axes = np.atleast_2d(axes)
    for r, (i, cx, cy, sz) in enumerate(sel):
        name, clean, noisy = ds[i]
        out = denoise(model, noisy, device, args.self_ensemble)
        H, W = clean.shape[:2]
        h, w = int(H * sz), int(W * sz)
        y0 = max(0, min(H - h, int(cy * H - h / 2)))
        x0 = max(0, min(W - w, int(cx * W - w / 2)))
        box = (slice(y0, y0 + h), slice(x0, x0 + w))
        panels = [(noisy[box], f"Noisy / {psnr(noisy[box], clean[box]):.2f} dB"),
                  (out[box], f"DnCNN / {psnr(out[box], clean[box]):.2f} dB"),
                  (clean[box], "Ground truth")]
        for c, (img, ttl) in enumerate(panels):
            ax = axes[r, c]
            if img.shape[2] == 1:
                img = img[..., 0]
            ax.imshow(np.clip(img, 0, 1), cmap="gray" if img.ndim == 2 else None)
            ax.set_title(f"{name}  {ttl}", fontsize=9)
            ax.axis("off")
    fig.suptitle(f"{kind} vs noisy, {ds.name}, sigma={args.sigma:.0f}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path}")


def main():
    args = parse_args()
    device, dev = get_device()
    # 结构参数一律从 checkpoint 里的训练配置读，不在调用处硬编码 depth
    model, targs = load_dncnn(args.ckpt)
    color = bool(targs.get("color", args.dataset == "Set12"))
    model.to(device).eval()
    if args.merge_bn:
        model.merge_bnorm()

    ds = TestSet(args.data_root, args.dataset, color=color, sigma=args.sigma, seed=args.seed)
    out_dir = os.path.join(args.out, f"{args.dataset}_sigma{int(args.sigma)}"
                                   f"{'_se' if args.self_ensemble else ''}")
    os.makedirs(out_dir, exist_ok=True)

    rows, psnrs, ssims, psnr_noisy = [], [], [], []
    t0 = time.time()
    print(f"[test] {args.dataset} ({len(ds)} 张) sigma={args.sigma:.0f} "
          f"color={color} self_ensemble={args.self_ensemble} device={dev}")
    for i in range(len(ds)):
        name, clean, noisy = ds[i]
        out = denoise(model, noisy, device, args.self_ensemble)
        p_n, p_d = psnr(noisy, clean), psnr(out, clean)
        s_d = ssim(out, clean)
        psnrs.append(p_d); psnr_noisy.append(p_n); ssims.append(s_d)
        rows.append({"name": name, "psnr_noisy": p_n, "psnr_dncnn": p_d, "ssim_dncnn": s_d})
        print(f"  {i+1:3d}/{len(ds)}  {name:<10s} noisy {p_n:6.2f}  "
              f"DnCNN {p_d:6.3f}  SSIM {s_d:.4f}")
        if args.save_images:
            from PIL import Image
            Image.fromarray((out * 255).round().astype(np.uint8)).save(
                os.path.join(out_dir, f"{name}_sigma{int(args.sigma)}.png"))
    dt = time.time() - t0

    summary = {
        "dataset": args.dataset, "sigma": args.sigma, "color": color,
        "self_ensemble": args.self_ensemble, "ckpt": args.ckpt,
        "device": dev, "merge_bn": args.merge_bn, "seed": args.seed,
        "n_images": len(ds),
        "psnr_noisy": float(np.mean(psnr_noisy)),
        "psnr_dncnn": float(np.mean(psnrs)),
        "ssim_dncnn": float(np.mean(ssims)),
        "time_total_s": dt,
        "time_per_image_ms": dt / len(ds) * 1000,
        "per_image": rows,
        "train_args": targs,
    }
    # ★ 默认文件名要带上**实验名**，否则不同模型的结果会互相覆盖。
    #   例如 DnCNN-S(sigma=25) 和 DnCNN-B(blind) 都想写 gray_sigma25.json，
    #   后跑的会静默覆盖先跑的 —— 报告里就会混进别的模型的数字。
    #
    # 优先用 checkpoint 的文件名做模型标识（dncnn_s_sigma25.pth / dncnn_b_blind.pth，
    # 各自唯一）；只有在文件名是 model_final.pth 这类通用名时，才退回用父目录名。
    # 早先固定用父目录名，若把两个 checkpoint 放在同一个目录里就会撞名。
    ck_stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    generic = {"model_final", "model_last", "model", "checkpoint", "best"}
    exp_name = (ck_stem if ck_stem not in generic else
                os.path.basename(os.path.dirname(os.path.abspath(args.ckpt))))
    tag = args.tag or (f"{exp_name}__{args.dataset}_sigma{int(args.sigma)}"
                        f"{'_se' if args.self_ensemble else ''}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[result] {args.dataset} sigma={args.sigma:.0f}")
    print(f"   noisy   PSNR = {np.mean(psnr_noisy):.3f} dB")
    print(f"   DnCNN   PSNR = {np.mean(psnrs):.3f} dB   SSIM = {np.mean(ssims):.4f}")
    print(f"   耗时 {dt/len(ds)*1000:.1f} ms/张  ({dev})")
    print(f"   -> {args.out}/{tag}.json")

    if args.fig:
        # 图和 json 同名，避免不同模型（如 DnCNN-S vs DnCNN-B）互相覆盖
        kind = "DnCNN-B_blind" if targs.get("blind") else "DnCNN-S"
        make_figure(model, ds, args, device,
                    os.path.join(out_dir, f"figure_{ds.name}_sigma{int(args.sigma)}"
                                           f"_{kind}.png"),
                    kind="DnCNN-B (blind)" if targs.get("blind") else "DnCNN-S")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())