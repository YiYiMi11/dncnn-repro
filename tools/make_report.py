"""把 results/*.json 汇总成一张可放进 README 的对照表。

    python tools/make_report.py       # -> results/tables.md

为什么重写：旧版只按 (数据集, 噪声级) 匹配结果，不区分是哪个模型。
一旦同时存在 DnCNN-S 和 DnCNN-B 两套结果，同一张表里的不同行就会
来自**不同的模型**（σ15/50 取到盲模型、σ25 取到已知-σ 模型），
而表头看不出来 —— 这种错误比数字算错更危险，因为它是静默的。
所以这里强制按「模型」分组，模型名直接进表头。
"""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# 论文 Table II（BSD68 灰度，68 张平均 PSNR/dB）
PAPER = {15: 31.73, 25: 29.23, 50: 26.23}
PAPER_BASELINES = {15: (31.07, 31.42), 25: (28.57, 28.92), 50: (25.62, 25.97)}
PAPER_SIGMAS = (15, 25, 50)

# checkpoint 文件名 -> (显示名, 说明, 该模型训练时用的 σ)
# trained_sigma 用来区分「这个模型是为这一档噪声专门训的」与「只是恰好在这一档上测过」。
# 只有前者才有资格和论文 Table II 对照 —— 拿 σ=25 的模型去测 σ=50 得到 17.13 dB，
# 放进对照表里会让人误以为是 σ=50 模型的复现结果，实际上它离 26.23 差了 9 dB。
MODELS = {
    "dncnn_s_sigma25": ("DnCNN-S (σ=25)", "已知噪声级，只训了 σ=25 这一档", 25),
    "dncnn_b_blind": ("DnCNN-B (盲去噪)", "单一模型，训练时 σ 从 0 随机取到 55", None),
}


def load():
    """读全部结果，按模型分组：{model_key: {sigma: record}}。"""
    out = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        stem = os.path.basename(p)[: -len(".json")]
        key = next((k for k in MODELS if stem.startswith(k)), None)
        if key is None:
            print(f"[warn] 跳过无法归类的结果文件 {os.path.basename(p)}")
            continue
        d = json.load(open(p))
        if d.get("dataset") != "BSD68":
            continue
        out.setdefault(key, {})[int(d["sigma"])] = d
    return out


def fmt(v, w=6):
    return f"{v:.2f}".rjust(w) if v is not None else "—".rjust(w)


def main():
    data = load()
    if not data:
        sys.exit("results/ 下没有可用的 *.json，先跑 python -m src.test")

    L = ["# 实验结果汇总", ""]
    L += ["由 `python tools/make_report.py` 从 `results/*.json` 自动生成，"
          "数字来自 BSD68（68 张灰度测试图）。", ""]

    # ---------- 1. 与论文的对照 -------------------------------------------- #
    L += ["## 1. 复现对照：BSD68 灰度（平均 PSNR, dB）", "",
          "| 模型 | σ=15 | σ=25 | σ=50 | 说明 |", "|:--|--:|--:|--:|:--|"]

    def row(name, values, note, bold=False):
        cells = [f"**{v}**" if (bold and v != "—") else v for v in values]
        return f"| {name} | " + " | ".join(cells) + f" | {note} |"

    L += [row("BM3D（论文）", [f"{PAPER_BASELINES[s][0]:.2f}" for s in PAPER_SIGMAS],
              "传统方法，对照组"),
          row("TNRD（论文）", [f"{PAPER_BASELINES[s][1]:.2f}" for s in PAPER_SIGMAS],
              "传统方法，对照组"),
          row("**DnCNN-S（论文）**", [f"{PAPER[s]:.2f}" for s in PAPER_SIGMAS],
              "每个 σ 单独训练，完整训练量", bold=True)]

    for key, (name, note, trained) in MODELS.items():
        if key not in data:
            continue
        g = data[key]
        # 只在该模型「训练时对应的 σ」上填数，其余留空 —— 否则会把
        # 「用错模型的测试结果」摆成「复现结果」
        cells = []
        for s in PAPER_SIGMAS:
            r = g.get(s)
            ok = (trained is None) or (trained == s)
            cells.append(f"{r['psnr_dncnn']:.2f}" if (r and ok) else "—")
        L += [row(f"**{name}**", cells, note)]

    L += ["", "> 论文的 DnCNN-S 为每个 σ 单独训练的模型；本实现的 DnCNN-S 只训了 σ=25，"
          "所以只有中间一格有数。偏差来源见 README「复现到 29.01，差 0.22 dB 去哪了」。", ""]

    # ---------- 2. 完整噪声扫描 -------------------------------------------- #
    L += ["## 2. 完整噪声扫描：一个模型横扫 5–55", "",
          "同一个 checkpoint 在不同噪声等级上的表现。**没有为每一档单独训练模型。**", "",
          "| 输入 σ | 含噪基线 | DnCNN-S (σ=25) | 增益 | DnCNN-B (盲) | 增益 |",
          "|--:|--:|--:|--:|--:|--:|"]
    known = data.get("dncnn_s_sigma25", {})
    blind = data.get("dncnn_b_blind", {})
    for s in sorted(set(known) | set(blind)):
        k, b = known.get(s), blind.get(s)
        noisy = (k or b)["psnr_noisy"]
        L += [
            f"| {s} | {noisy:.2f} "
            + (f"| **{k['psnr_dncnn']:.2f}** | {k['psnr_dncnn'] - noisy:+.2f} "
               if k else "| — | — ")
            + (f"| **{b['psnr_dncnn']:.2f}** | {b['psnr_dncnn'] - noisy:+.2f} "
               if b else "| — | — ")
            + "|",
        ]
    L += ["", "最后一列是相对含噪输入的增益。**σ=5 时 DnCNN-S 的增益是负的 —— "
          "对一个本来就很干净的图做「去噪」，结果比不去更差。**", ""]

    # ---------- 3. 推理开销 ------------------------------------------------ #
    L += ["## 3. 推理开销（Apple M 系列，Metal Performance Shaders）", "",
          "| 模型 | 设备 | ms/张 | SSIM @σ=25 |", "|:--|:--|--:|--:|"]
    for key, (name, _, _trained) in MODELS.items():
        if key not in data:
            continue
        g = data[key]
        any_rec = next(iter(g.values()))
        s25 = g.get(25)
        L += [f"| {name} | {any_rec.get('device', '?')} | "
              f"{any_rec['time_per_image_ms']:.1f} | "
              f"{s25['ssim_dncnn']:.4f} |" if s25 else
              f"| {name} | {any_rec.get('device', '?')} | "
              f"{any_rec['time_per_image_ms']:.1f} | — |"]

    out = os.path.join(RESULTS, "tables.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"表格 -> {out}")


if __name__ == "__main__":
    main()