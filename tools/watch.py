"""实时监控后台训练：进度条 + 已用/剩余时间 + ETA。

训练进程每秒会把完整状态原子写入 `<实验目录>/status.json`
（step / total / 已用时间 / 每步耗时 / 剩余时间 / loss / 验证 PSNR），
这个脚本直接读它，所以是**真正实时**的 —— 不像 `tail -f` 那样要等下一次日志 flush。

用法（另开一个终端窗口）：

    python tools/watch.py experiments/gray_sigma25
    python tools/watch.py experiments/            # 列出所有实验并选最新的
    python tools/watch.py experiments/gray_sigma25 --test   # 顺便在 BSD68 上试测

Ctrl-C 只退出监控，不影响训练。
"""

import argparse
import glob
import json
import os
import sys
import time

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"


def fmt_t(sec):
    sec = int(sec or 0)
    if sec <= 0:
        return "--:--"
    h, m = divmod(sec // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec % 60:02d}s"


def pick_dir(arg):
    if not arg:
        arg = "experiments"
    if os.path.isfile(os.path.join(arg, "status.json")):
        return arg
    cands = [os.path.dirname(p) for p in glob.glob(os.path.join(arg, "*", "status.json"))]
    cands += [os.path.dirname(p) for p in
              glob.glob(os.path.join(arg, "status.json"))]
    if not cands:
        sys.exit(f"{arg} 下没有 status.json —— 训练是否已启动？")
    # 取最近更新的
    return max(set(cands), key=lambda d: os.path.getmtime(os.path.join(d, "status.json")))


def read_status(d):
    p = os.path.join(d, "status.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None      # 正在被替换的那一瞬间，下一轮再读


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", nargs="?", default="experiments")
    ap.add_argument("--width", type=int, default=30)
    ap.add_argument("--once", action="store_true", help="只打印一次当前状态")
    ap.add_argument("--watch-seconds", type=float, default=0.0,
                    help="监控多少秒后自动退出（0=一直看）")
    args = ap.parse_args()

    d = pick_dir(args.exp_dir)
    sp = os.path.join(d, "status.json")
    print(f"{BOLD}监控{RESET} {d}   ({YELLOW}Ctrl-C{RESET} 退出监控，不影响训练)")
    if not os.path.exists(sp):
        print(f"{YELLOW}还没找到 status.json，等训练进程写第一帧...{RESET}")
    t0 = time.time()

    while True:
        s = read_status(d)
        now = time.time()
        if s is None:
            print("\033[2J\033[H" + f"{YELLOW}等待 status.json ...{RESET}", flush=True)
        else:
            age = now - s.get("updated", now)
            total = s.get("total", 0)
            step = s.get("step", 0)
            pct = s.get("pct", 0.0)
            w = args.width
            done = int(pct / 100 * w)
            bar = ("█" * done) + ("░" * (w - done))

            kind = "彩色 color" if s.get("color") else "灰度 gray"
            head = (f"{BOLD}{s.get('exp', '?')}{RESET}  {kind}  σ={s.get('sigma','?')}"
                    f"  {DIM}{s.get('dev','')}{RESET}")
            line1 = f"{CYAN}{bar}{RESET} {pct:5.1f}%  {step}/{total}"
            if s.get("ep") and s.get("epochs"):
                line1 += f"  {BOLD}ep {s['ep']}/{s['epochs']}{RESET}"
            line1 += f"  {DIM}{s.get('s_per_it', 0):.3f}s/it{RESET}"

            loss = s.get("loss")
            if loss is not None:
                line1 += f"  loss {loss:.5f}"
            val = s.get("val")
            if val:
                line1 += f"  {GREEN}val {val:.2f}dB{RESET}"
            best = s.get("best_val")
            if best and best != val:
                line1 += f"  {DIM}(best {best:.2f}){RESET}"

            eta = time.strftime("%H:%M", time.localtime(s.get("eta_unix", now)))
            line2 = (f"  {DIM}已用 {fmt_t(s.get('elapsed_s'))}"
                     f"   剩余 {fmt_t(s.get('remain_s'))}"
                     f"   预计 {eta} 完成{RESET}")
            if s.get("done"):
                line2 = f"  {GREEN}训练已完成{RESET}"
            if age > 15:
                line2 += f"  {RED}(状态已 {int(age)}s 未更新，进程可能已退出){RESET}"

            print("\033[2J\033[H" + head + "\n" + line1 + "\n" + line2, flush=True)

        if args.once:
            return
        if args.watch_seconds and time.time() - t0 > args.watch_seconds:
            return
        time.sleep(1.0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出监控。")