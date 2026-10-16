"""训练 DnCNN（复现论文 Sec.IV-A2 的参数设置）。

论文设置（MatConvNet 版，对应表 II / 表 III 的 DnCNN-S）：
    优化器   SGD + momentum 0.9, weight decay 1e-4, mini-batch 128
    学习率   指数衰减，从 1e-1 衰减到 1e-4（共 50 个 epoch）
    训练     50 epochs，每 epoch 采 128 x 1600 = 204800 个 40x40 patch
             -> 每 epoch 1600 个 iteration
    初始化   He initialization (He et al. [27])
    损失     残差上的 L2  (式 1)
    结构     depth 17, 64 通道

示例：
    # Set12 彩色（对应论文 Table III）
    python -m src.train --color --sigma 25
    # BSD68 灰度（对应论文 Table II）
    python -m src.train --gray  --sigma 25
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from .data import TrainingImages, psnr
from .model import DnCNN, count_params

# --------------------------------------------------------------------------- #
#   终端进度条
# --------------------------------------------------------------------------- #
_BOLD, _DIM, _RESET = "\033[1m", "\033[2m", "\033[0m"
_CYAN, _GREEN = "\033[36m", "\033[32m"


def _fmt_t(sec):
    sec = int(sec)
    if sec <= 0:
        return "--:--"
    h, m = divmod(sec // 60, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{sec % 60:02d}s"


class Progress:
    """训练进度：终端进度条 + 供 tools/watch.py 读取的 status.json。

    status.json 每秒原子写一次，里面是机器可读的完整状态
    （进度 / 已用时间 / 每步耗时 / 剩余时间 / loss / 验证 PSNR），
    这样后台训练时 `python tools/watch.py <目录>` 能拿到**实时**进度，
    而不是像读日志那样要等下一次 flush。
    """

    def __init__(self, total, status_path=None, width=34, meta=None):
        self.total = max(1, total)
        self.enabled = sys.stderr.isatty()
        self.width = width
        self.t0 = time.time()
        self.n = 0
        self.status_path = status_path
        self.meta = meta or {}
        self.last_write = 0.0
        self.best_val = None

    # ---------------- 内部 ----------------
    def _stats(self):
        el = time.time() - self.t0
        spd = el / self.n if self.n > 0 else 0.0
        return {
            "step": self.n,
            "total": self.total,
            "pct": round(100.0 * self.n / self.total, 2),
            "elapsed_s": round(el, 1),
            "s_per_it": round(spd, 4),
            "remain_s": round(max(0.0, (self.total - self.n) * spd), 1),
            "eta_unix": round(time.time() + max(0.0, (self.total - self.n) * spd), 1),
            "updated": round(time.time(), 1),
            **self.meta,
        }

    def _write_status(self, extra=None):
        if not self.status_path:
            return
        d = self._stats()
        if extra:
            d.update(extra)
        tmp = self.status_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(d, f)
            os.replace(tmp, self.status_path)     # 原子替换，读的人不会读到半个文件
            self.last_write = time.time()
        except OSError:
            pass

    # ---------------- 对外 ----------------
    def update(self, n, loss=None, val=None, force=False):
        self.n = n
        if val is not None and val == val:
            self.best_val = val if self.best_val is None else max(self.best_val, val)
        now = time.time()
        if force or now - self.last_write >= 1.0:
            self._write_status({
                "loss": loss, "val": val, "best_val": self.best_val,
                "ep": self.meta.get("ep"), "epochs": self.meta.get("epochs"),
            })
        if not self.enabled:
            return
        pct = min(100.0, n / self.total * 100)
        el = now - self.t0
        spd = el / max(1, n)
        rem = max(0.0, (self.total - n) * spd)
        done = int(pct / 100 * self.width)
        eta = time.strftime("%H:%M", time.localtime(now + rem))
        msg = (f"\r{_CYAN}{'█' * done}{_DIM}{'░' * (self.width - done)}{_RESET} "
               f"{pct:5.1f}%  {n}/{self.total}  "
               f"{_DIM}已用{_fmt_t(el)} 剩余{_fmt_t(rem)} ETA {eta}{_RESET}")
        if loss is not None:
            msg += f"  loss {loss:.5f}"
        if val is not None and val == val:
            msg += f"  {_GREEN}val {val:.2f}dB{_RESET}"
        sys.stderr.write(msg + "   ")
        sys.stderr.flush()

    def newline(self):
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def finish(self, **extra):
        self._write_status({"done": True, **extra})


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def parse_args():
    p = argparse.ArgumentParser("DnCNN 训练")
    # 任务
    g = p.add_mutually_exclusive_group()
    g.add_argument("--color", action="store_true", help="RGB 模型 -> 评测 Set12 (Table III)")
    g.add_argument("--gray", action="store_true", help="灰度模型 -> 评测 BSD68 (Table II)")
    p.add_argument("--sigma", type=float, help="噪声标准差 (0-255 尺度)，如 15/25/50")
    p.add_argument("--blind", action="store_true",
                   help="盲去噪 DnCNN-B：训练时 σ 在 [0, --sigma-max] 内随机采样，"
                        "一个模型通吃所有噪声等级（论文 Table II 的 DnCNN-B 列）")
    p.add_argument("--sigma-max", type=float, default=55.0,
                   help="盲去噪的 σ 采样上界，论文用 55")
    # 数据
    p.add_argument("--data-root", default="data")
    p.add_argument("--patch-size", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--patches-per-epoch", type=int, default=204800,
                   help="每 epoch 采样 patch 总数；论文为 128x1600=204800 "
                        f"(= {204800 // 128} 个 iteration)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--aug", action="store_true", help="随机翻转/转置增广（论文未用）")
    p.add_argument("--seed", type=int, default=0)
    # 优化
    p.add_argument("--depth", type=int, default=17, help="论文 DnCNN-S 用 17；DnCNN-B/DnCNN-3 用 20")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--init", default="orthogonal", choices=["orthogonal", "he"],
                   help="orthogonal(gain0.2)=MatConvNet/KAIR 默认，配 lr=0.1 才稳；"
                        "he=论文正文提到的 He 初始化，权重大 7 倍，必须配更小的 lr")
    p.add_argument("--init-gain", type=float, default=0.2)
    p.add_argument("--optimizer", default="adam", choices=["sgd", "adam"],
                   help="adam=默认。实测 SGD 在 PyTorch 下收敛极慢：同预算下 loss "
                        "是 Adam 的 4 倍、BSD68 差 1.5dB；作者 KAIR 用的也是 adam")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="起点学习率。adam 用 1e-4（与 KAIR 一致）；"
                        "若 --optimizer sgd，用 0.01（论文 1e-1 的 PyTorch 换算值）")
    p.add_argument("--lr-min", type=float, default=1e-6,
                   help="衰减终点。adam 1e-6；sgd 对应 1e-5（见 README 5.4）")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss", default="l2", choices=["l1", "l2"])
    p.add_argument("--clip-grad", type=float, default=0.0, help=">0 时启用梯度裁剪")
    p.add_argument("--preset", default="paper",
                   choices=["paper", "fast", "turbo"],
                   help="paper=完全按论文 50ep x 1600it=80000 iter (~9h)；"
                        "fast=20000 iter (~2.2h)；turbo=12000 iter (~1.3h)")
    # 速度
    p.add_argument("--amp", action="store_true",
                   help="bf16 混合精度。注意：M4/MPS 上实测反而更慢，别开")
    p.add_argument("--val-every", type=int, default=5, help="每多少个 epoch 在验证集上测一次 PSNR")
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="每多少个 iteration 存一个带编号的 checkpoint（0=不存）。"
                        "用来实测「PSNR 随训练量怎么变」，据此决定最少要训多久")
    p.add_argument("--val-holdout", type=int, default=8,
                   help="从 400 张训练图末尾留出多少张做验证集（不参与训练）")
    p.add_argument("--out", default=None, help="输出目录，默认 experiments/<color|gray>_sigma<sigma>")
    # 快速调试
    p.add_argument("--quick", action="store_true", help="冒烟测试：2 epoch、每 epoch 20 个 iter")
    return p.parse_args()


def lr_at_step(args, step, total_iters):
    """按全局 iteration 推进的指数衰减 lr（等价于按 epoch 衰减，但更平滑）"""
    if args.lr_min <= 0 or args.lr == args.lr_min:
        return args.lr
    f = min(max(step / max(1, total_iters), 0.0), 1.0)
    return args.lr * (args.lr_min / args.lr) ** f


def matlab_lr_to_pytorch(lr_mc, momentum):
    """把 MatConvNet 的学习率换算成 PyTorch 等效值。

    两者的动量实现不同：
        MatConvNet:  v <- m*v + (1-m)*g,  x <- x - lr_mc * v     (稳态 v -> g)
        PyTorch    :  v <- m*v + g,         x <- x - lr_pt * v     (稳态 v -> g/(1-m))
    要让实际步长相同，需 lr_pt * g/(1-m) = lr_mc * g，即 lr_pt = lr_mc * (1-m)。
    论文的 1e-1 在 PyTorch 里对应 1e-2。
    """
    return lr_mc * (1.0 - momentum)


def build_optimizer(model, args):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 标准做法：只对卷积核（ndim>1）做 weight decay；BN 的 gamma/beta 和所有偏置不做。
        # 论文是对全部参数 decay 1e-4，这里拆开更稳，对 PSNR 的影响 < 0.05dB。
        (no_decay if p.ndim <= 1 else decay).append(p)
    groups = [{"params": decay, "weight_decay": args.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    if args.optimizer == "adam":
        return torch.optim.Adam(groups, lr=args.lr)
    return torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum)


def main():
    args = parse_args()
    if args.blind and args.sigma is not None:
        raise SystemExit("错误：--blind 和 --sigma 互斥，盲去噪自己采样 σ")
    if not args.blind and args.sigma is None:
        raise SystemExit("错误：需要 --sigma <值>，或者用 --blind 训练盲去噪模型")
    if args.blind:
        # DnCNN-B：论文用 depth 20、patch 50、每 batch 3000 个 patch
        if args.depth == 17 and "--depth" not in sys.argv:
            args.depth = 20
    if not args.color and not args.gray:
        args.color = True
    color = args.color
    tag = ("blind_" + ("color" if color else "gray")) if args.blind \
        else f"{'color' if color else 'gray'}_sigma{int(args.sigma)}"

    PRESETS = {          # 目标 iteration 数（M4 上 0.40 s/iter）
        "paper": 80000,    # 论文原始配方，~9.0 h
        "fast": 20000,     # ~2.2 h
        "turbo": 12000,    # ~1.3 h
    }

    if args.quick:
        args.epochs, args.patches_per_epoch = 2, args.batch_size * 20
    elif args.preset != "paper":
        # 保持 epoch 大小不变（1600 iter/epoch），只压缩 epoch 数 —— 这样
        # 学习率的指数衰减曲线形状不变，只是走完得更快。
        ipe = args.patches_per_epoch // args.batch_size
        args.epochs = max(1, PRESETS[args.preset] // ipe)

    out_dir = args.out or os.path.join("experiments", tag)
    os.makedirs(out_dir, exist_ok=True)
    device, dev_name = get_device()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train = TrainingImages(args.data_root, color=color, n_holdout=args.val_holdout)
    iters_per_epoch = max(1, args.patches_per_epoch // args.batch_size)
    total_iters = iters_per_epoch * args.epochs

    in_ch = 3 if color else 1
    model = DnCNN(depth=args.depth, n_channels=args.width, image_channels=in_ch,
                  init=args.init, init_gain=args.init_gain).to(device)
    opt = build_optimizer(model, args)
    lossfn = nn.L1Loss() if args.loss == "l1" else nn.MSELoss()

    amp_dtype = torch.bfloat16
    use_amp = args.amp and dev_name in ("mps", "cuda")

    log_path = os.path.join(out_dir, "train.log")
    logf = open(log_path, "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log("=" * 70)
    log(f"[cfg] {vars(args)}")
    log(f"[dev] {dev_name} | amp={use_amp} | params={count_params(model):,}")
    log(f"[it]  {iters_per_epoch} iter/epoch x {args.epochs} epoch = {total_iters} iter")
    log(f"[lr]  {args.lr:g} -> {args.lr_min:g}（MatConvNet 口径为 "
        f"{args.lr / max(1e-12, 1 - args.momentum):.3g} -> "
        f"{args.lr_min / max(1e-12, 1 - args.momentum):.3g}）")
    log(f"[init] {args.init} (gain={args.init_gain})"
        + (f"  | 盲去噪: σ ~ U(0, {args.sigma_max})"
           if args.blind else f"  | σ={args.sigma:.0f}"))

    def lr_at_iter(step):
        """按全局 iteration 推进的指数衰减学习率（PyTorch 口径，见 5.4）"""
        return lr_at_step(args, step, total_iters)

    @torch.no_grad()
    def quick_val():
        """在**留出的验证图**上加噪测 PSNR —— 不看测试集，避免用测试集调参。"""
        if len(train.val_images) == 0:
            return float("nan")
        model.eval()
        tot, cnt = 0.0, 0
        for t_img in train.val_images:
            clean = t_img.float().to(device) / 255.0
            if clean.dim() == 3:                 # 灰度：HxWx1 -> 1xHxW
                clean = clean.unsqueeze(0)
            g = torch.Generator(device=device).manual_seed(args.seed + cnt)
            if args.blind:
                # 盲去噪：每次随机采一个 sigma 做验证，看整体水平
                s = float(torch.rand(1, generator=g, device=device)) * args.sigma_max
                ref_sigma = f"σ={s:.1f}"
            else:
                s = args.sigma
                ref_sigma = f"σ={args.sigma:.0f}"
            noisy = clean + torch.randn(clean.shape, generator=g, device=device) * (s / 255.0)
            out = model(noisy).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            tot += psnr(out, clean[0].permute(1, 2, 0).cpu().numpy())
            cnt += 1
            del ref_sigma
        model.train()
        return tot / cnt

    model.train()
    step = 0
    t0 = time.time()
    ema_loss = None
    prog = Progress(total_iters,
                    status_path=os.path.join(out_dir, "status.json"),
                    meta={"epochs": args.epochs, "sigma": args.sigma,
                          "color": color, "exp": tag, "dev": dev_name})
    prog.update(0)
    print(f"训练共 {total_iters} 个 iteration，按当前机器速度预计 "
          f"{_fmt_t(total_iters * 0.42)}\n", flush=True)

    for epoch in range(args.epochs):
        te = time.time()
        run = 0.0
        prog.meta["ep"] = epoch + 1
        for it in range(iters_per_epoch):
            lr = lr_at_iter(step)
            for g in opt.param_groups:
                g["lr"] = lr

            clean = train.sample_batch(args.batch_size, args.patch_size, rng, device, args.aug)
            # 在 GPU 上加高斯噪声：y = x + v, v ~ N(0, sigma^2/255^2)
            # 注意不对 y 做 clip —— 这是 DnCNN 系工作的通行做法，训练/测试一致
            if args.blind:
                # DnCNN-B：每个样本独立采一个 sigma ~ U(0, sigma_max)
                # 形状 [B,1,1,1] 便于广播到整张特征图
                s_pix = (torch.rand(clean.shape[0], 1, 1, 1, device=device)
                         * args.sigma_max)
                noisy = clean + torch.randn_like(clean) * (s_pix / 255.0)
            else:
                noisy = clean + torch.randn_like(clean) * (args.sigma / 255.0)

            with torch.autocast(device_type=dev_name, dtype=amp_dtype, enabled=use_amp):
                out = model(noisy)
            out = out.float()
            # 残差上的损失：|| (y - x_hat) - v ||^2，其中 v = y - x
            loss = lossfn(out, clean)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()

            l = loss.item()
            if not np.isfinite(l):
                log(f"[ERROR] loss 变成 {l}，训练发散（发生在第 {step+1} 个 iteration）。")
                log("        论文的 lr=1e-1 是 MatConvNet 口径，PyTorch 里要小 10 倍；")
                log(f"        当前用的是 lr={lr:.3e}。可以试：")
                log(f"          --lr {lr/4:.1e}   （再降一档）")
                log("          --optimizer adam --lr 1e-4   （作者 KAIR 仓库用的配方）")
                log("        也可先跑 python tools/lr_sweep.py 看看哪个组合收敛。")
                logf.close()
                return 1
            run += l
            ema_loss = l if ema_loss is None else 0.98 * ema_loss + 0.02 * l
            step += 1

            if (it + 1) % 200 == 0:
                log(f"  ep{epoch:3d} it{it+1:5d}/{iters_per_epoch}  lr={lr:.3e}  "
                    f"loss={ema_loss:.5f}  ({(time.time()-te)/(it+1):.3f}s/it)")
            if (it + 1) % 10 == 0 or it + 1 == iters_per_epoch:
                prog.update(step + 1, ema_loss)

            if args.ckpt_every and (step % args.ckpt_every == 0):
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "step": step, "ema_loss": ema_loss},
                           os.path.join(out_dir, f"ckpt_{step:06d}.pth"))

        msg = f"[epoch {epoch+1}/{args.epochs}] loss={run/iters_per_epoch:.5f} " \
              f"time={time.time()-te:.0f}s total={(time.time()-t0)/60:.1f}min"
        prog.newline()
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            v = quick_val()
            msg += f"  valPSNR={v:.3f}dB"
            prog.update(step, ema_loss, v)
        log(msg)
        torch.save({"model": model.state_dict(), "args": vars(args)},
                   os.path.join(out_dir, "model_last.pth"))

    torch.save({"model": model.state_dict(), "args": vars(args)},
               os.path.join(out_dir, "model_final.pth"))
    prog.finish(loss=ema_loss, val=prog.best_val)
    log(f"[done] 总耗时 {(time.time()-t0)/60:.1f} min, 模型: {out_dir}/model_final.pth")
    log(f"[next] python -m src.test --ckpt {out_dir}/model_final.pth "
        f"--dataset {'Set12' if color else 'BSD68'} --sigma {args.sigma}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())