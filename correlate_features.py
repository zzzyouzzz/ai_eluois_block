"""相关性分析：权重 w2（行变化）为什么涨/跌，是不是"消行"喂出来的。

输入是训练追踪 CSV —— 它已经逐行记录了特征、奖励分、TD 和每一步的参数变化，
所以不需要改追踪模块，直接算就行。

七个面板：
  1. 消行步 / 普通步 的特征均值之比      → 哪类特征"沾了消行的光"
  2. 一步消掉几行 vs 行变化特征均值       → 消行时棋盘是不是更乱
  3. 每步对权重的推力 td×φ 的均值         → 推力来自哪类步
  4. td 均值随特征分位的变化             → 模型在压低 / 抬高一类什么局面
  5. 各特征的梯度质量里消行步占几成       → 消行到底贡献了多少
  6. 每局 Δw2 vs 每局步数                → 是不是"更新次数"在推
  7. 局末 w2 与 每局平均 φ2 的时间序列    → 两者谁跟着谁走

用法：
    python correlate_features.py training_trace.csv --output correlate.png
"""

import argparse
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FEATURES = ["高度", "消行", "行变化", "列变化", "洞数", "井深"]
COLORS = ["#1565c0", "#2e7d32", "#c0392b", "#6a1b9a", "#ef6c00", "#00838f"]
LINES_TO_SCORE = {1: 100, 2: 300, 3: 500, 4: 800}


def mean(values):
    return sum(values) / len(values) if values else 0.0


def corr(xs, ys):
    if len(xs) < 3:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def residual(xs, ys):
    """用 xs 线性解释 ys，返回残差：用来做偏相关。"""
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return [0.0] * len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]


def load(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    steps = []
    for row in rows:
        if not row.get("td_error"):
            continue            # 回放区还没预热时只有数据行、没有更新
        steps.append(
            {
                "episode": int(row["episode"]),
                "score": float(row["score"]),
                "td": float(row["td_error"]),
                "epsilon": float(row["epsilon"]),
                "feature": [float(row["feature%d" % i]) for i in range(6)],
                "weight_delta": [float(row["weight_delta%d" % i]) for i in range(6)],
                "weight_after": [float(row["weight_after%d" % i]) for i in range(6)],
            }
        )
    return steps


def td_by_percentile(steps, index, bins=10):
    """把样本按某个特征排序后等分成 bins 档，返回每档的 (分位中点, td 均值)。"""
    ordered = sorted(steps, key=lambda s: s["feature"][index])
    chunk = max(1, len(ordered) // bins)
    xs, ys = [], []
    for b in range(bins):
        part = ordered[b * chunk:(b + 1) * chunk]
        if not part:
            continue
        xs.append(b * 10 + 5)
        ys.append(mean([s["td"] for s in part]))
    return xs, ys


def main():
    parser = argparse.ArgumentParser(description="特征 / 奖励 / 梯度 的相关性分析")
    parser.add_argument("trace", help="训练追踪 CSV")
    parser.add_argument("--output", default="correlate_features.png")
    args = parser.parse_args()

    steps = load(args.trace)
    if not steps:
        raise SystemExit("追踪文件里没有可用的更新记录")

    clear = [s for s in steps if s["score"] > 0]
    plain = [s for s in steps if s["score"] == 0]
    clear_pct = 100.0 * len(clear) / len(steps)
    print("样本：%d 步（消行 %d 步，占 %.2f%%）" % (len(steps), len(clear), clear_pct))

    # 1) 消行步 vs 普通步 的特征均值
    ratio = []
    print("特征均值：")
    for i, name in enumerate(FEATURES):
        mn = mean([s["feature"][i] for s in plain])
        mc = mean([s["feature"][i] for s in clear])
        ratio.append(mc / mn if mn > 0.1 else float("nan"))
        print("  %-4s 普通 %8.3f  消行 %8.3f  %s" %
              (name, mn, mc, ("变大 %.2f×" % (mc / mn)) if mc > mn and mn > 0 else "变小"))

    # 2) 消几行 vs 行变化（这一档没有样本就不画点）
    bucket_phi2 = [(0, mean([s["feature"][2] for s in plain]))]
    for lines, score in LINES_TO_SCORE.items():
        part = [s["feature"][2] for s in steps if s["score"] == score]
        if part:
            bucket_phi2.append((lines, mean(part)))

    # 3) 每步推力 td*φ
    push_clear = [mean([s["td"] * s["feature"][i] for s in clear]) for i in range(6)]
    push_plain = [mean([s["td"] * s["feature"][i] for s in plain]) for i in range(6)]

    # 4) 梯度总质量里消行步占的比例
    share = []
    for i in range(6):
        mc = sum(s["td"] * s["feature"][i] for s in clear)
        mp = sum(s["td"] * s["feature"][i] for s in plain)
        share.append(100.0 * mc / (mc + mp) if abs(mc + mp) > 1e-12 else float("nan"))
    print("各特征梯度质量中消行步的占比：")
    for i, name in enumerate(FEATURES):
        print("  %-4s %+7.2f%%" % (name, share[i]))

    # 5) 每局聚合
    episodes = {}
    for s in steps:
        ep = episodes.setdefault(
            s["episode"],
            {"n": 0, "clears": 0, "dw": 0.0, "phi2": 0.0, "w2": 0.0},
        )
        ep["n"] += 1
        ep["phi2"] += s["feature"][2]
        ep["w2"] = s["weight_after"][2]
        ep["dw"] += s["weight_delta"][2]
        if s["score"] > 0:
            ep["clears"] += 1
    keys = sorted(episodes)
    steps_per_ep = [episodes[k]["n"] for k in keys]
    dw2 = [episodes[k]["dw"] for k in keys]
    clears_per_ep = [episodes[k]["clears"] for k in keys]
    phi2_per_ep = [episodes[k]["phi2"] / episodes[k]["n"] for k in keys]
    w2_series = [episodes[k]["w2"] for k in keys]

    c_steps = corr(steps_per_ep, dw2)
    c_clears = corr(clears_per_ep, dw2)
    p_clears = corr(residual(steps_per_ep, dw2), residual(steps_per_ep, clears_per_ep))
    p_steps = corr(residual(clears_per_ep, dw2), residual(clears_per_ep, steps_per_ep))
    print("每局 Δw2 与 每局步数：     corr = %+.3f" % c_steps)
    print("每局 Δw2 与 每局消行次数： corr = %+.3f" % c_clears)
    print("偏相关 消行 | 步数：       %+.3f" % p_clears)
    print("偏相关 步数 | 消行：       %+.3f" % p_steps)
    print("td 均值：普通步 %+.1f  消行步 %+.1f" %
          (mean([s["td"] for s in plain]), mean([s["td"] for s in clear])))

    # ---------- 画图 ----------
    fig, axes = plt.subplots(2, 4, figsize=(21, 9))
    idx = list(range(6))

    ax = axes[0, 0]
    ax.bar(FEATURES, [r if r == r else 0.0 for r in ratio], color=COLORS)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("消行步 / 普通步 的特征均值之比")
    ax.set_ylabel("倍数（1 = 无差别）")
    ax.tick_params(axis="x", labelsize=8)

    ax = axes[0, 1]
    ax.plot([b[0] for b in bucket_phi2], [b[1] for b in bucket_phi2], "o-", color="#c0392b")
    ax.set_title("一步消掉几行 vs 行变化特征均值")
    ax.set_xlabel("这一步消掉的行数")
    ax.set_ylabel("行变化 φ2 均值")

    ax = axes[0, 2]
    width = 0.38
    ax.bar([i - width / 2 for i in idx], push_plain, width, label="普通步", color="#90a4ae")
    ax.bar([i + width / 2 for i in idx], push_clear, width, label="消行步", color="#c0392b")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(idx)
    ax.set_xticklabels(FEATURES, fontsize=8)
    ax.set_title("每步对权重的推力 td×φ 的均值")
    ax.set_ylabel("td×φ")
    ax.legend(fontsize=8)

    ax = axes[0, 3]
    for i, name in enumerate(FEATURES):
        xs, ys = td_by_percentile(steps, i)
        ax.plot(xs, ys, "o-", color=COLORS[i], label=name, markersize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("td 均值随特征分位的变化")
    ax.set_xlabel("特征分位（0 = 最小的 10%，100 = 最大的 10%）")
    ax.set_ylabel("td 均值")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.bar(FEATURES, share, color=COLORS)
    ax.axhline(clear_pct, color="black", linestyle="--", linewidth=0.9,
               label="消行步只占 %.2f%% 的步数" % clear_pct)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(idx)
    ax.set_xticklabels(FEATURES, fontsize=8)
    ax.set_title("各特征的梯度质量里，消行步贡献的百分比")
    ax.set_ylabel("%")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.scatter(steps_per_ep, dw2, s=6, alpha=0.45, color="#1565c0")
    ax.set_title("每局 Δw2 vs 每局步数  corr=%+.2f" % c_steps)
    ax.set_xlabel("这一局走了多少步（= 更新了多少次）")
    ax.set_ylabel("这一局 Δw2")

    ax = axes[1, 2]
    ax.plot(keys, w2_series, color="#c0392b", linewidth=1.2, label="w2（局末）")
    ax.set_xlabel("训练局数")
    ax.set_ylabel("w2", color="#c0392b")
    ax2 = ax.twinx()
    ax2.plot(keys, phi2_per_ep, color="#1565c0", linewidth=0.9, alpha=0.7)
    ax2.set_ylabel("每局平均 φ2", color="#1565c0")
    ax.set_title("w2 与 行变化特征的时间关系")
    ax.grid(alpha=0.25)

    # 7) td 随特征分位的变化：模型在压低哪一类局面
    ax = axes[1, 3]
    for i, name in enumerate(FEATURES):
        ordered = sorted(steps, key=lambda s: s["feature"][i])
        chunk = max(1, len(ordered) // 10)
        xs, ys = [], []
        for b in range(10):
            part = ordered[b * chunk:(b + 1) * chunk]
            if not part:
                continue
            xs.append(b * 10 + 5)
            ys.append(mean([s["td"] for s in part]))
        ax.plot(xs, ys, "o-", color=COLORS[i], label=name, markersize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("td 均值随特征分位的变化")
    ax.set_xlabel("特征分位（0 = 最小的 10%%，100 = 最大的 10%%）")
    ax.set_ylabel("td 均值")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    ax.grid(alpha=0.25)

    ax = axes[1, 3]
    ax.set_axis_off()
    ax.text(0.02, 0.95,
            "偏相关\n\n"
            "Δw2 ~ 步数            %+.3f\n"
            "Δw2 ~ 消行次数        %+.3f\n"
            "消行 | 步数           %+.3f\n"
            "步数 | 消行           %+.3f\n\n"
            "td 均值\n\n"
            "普通步                %+.1f\n"
            "消行步                %+.1f\n\n"
            "样本\n\n"
            "总步数                %d\n"
            "消行步                %d（%.2f%%）"
            % (c_steps, c_clears, p_clears, p_steps,
               mean([s["td"] for s in plain]), mean([s["td"] for s in clear]),
               len(steps), len(clear), clear_pct),
            va="top", fontsize=11, linespacing=1.6)

    fig.suptitle("训练追踪相关性分析：%s（%d 步 / %d 局）" % (args.trace, len(steps), len(keys)))
    fig.tight_layout()
    fig.savefig(args.output, dpi=130)
    print("图已保存：%s" % args.output)


if __name__ == "__main__":
    main()
