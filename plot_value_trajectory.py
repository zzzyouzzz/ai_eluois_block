"""把一局之内 V 的变化画出来：棋盘从空到快死，估值到底动没动。

用法：
    python plot_value_trajectory.py training_trace.csv --output value_trajectory.png
"""

import argparse
import csv
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BINS = 10


def load(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("td_error")]
    episodes = defaultdict(list)
    for r in rows:
        episodes[int(r["episode"])].append(
            {
                "turn": float(r["turn"]),
                "v": float(r["current_value"]),
                "phi2": float(r["feature2"]),
                "phi4": float(r["feature4"]),
            }
        )
    for key in episodes:
        episodes[key].sort(key=lambda x: x["turn"])
    return episodes


def binned(episodes, field):
    """按局内进度分成 BINS 档，返回每档均值与标准差。"""
    buckets = [[] for _ in range(BINS)]
    for steps in episodes:
        n = len(steps)
        if n < 2:
            continue
        for i, s in enumerate(steps):
            b = min(BINS - 1, int(i * BINS / n))
            buckets[b].append(s[field])
    xs = [100.0 * (b + 0.5) / BINS for b in range(BINS) if buckets[b]]
    mean = [st.mean(buckets[b]) for b in range(BINS) if buckets[b]]
    sd = [st.pstdev(buckets[b]) for b in range(BINS) if buckets[b]]
    return xs, mean, sd


def main():
    parser = argparse.ArgumentParser(description="局内 V 的轨迹")
    parser.add_argument("trace", help="训练追踪 CSV")
    parser.add_argument("--output", default="value_trajectory.png")
    parser.add_argument("--recent", type=int, default=2000, help="统计用最近多少局")
    args = parser.parse_args()

    episodes = load(args.trace)
    keys = sorted(episodes)
    recent = [episodes[k] for k in keys[-args.recent:]]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    # 1) V 与 行变化 沿局内进度的曲线
    ax = axes[0]
    xs, mv, sv = binned(recent, "v")
    ax.fill_between(xs, [m - s for m, s in zip(mv, sv)], [m + s for m, s in zip(mv, sv)],
                    color="#1565c0", alpha=0.15)
    ax.plot(xs, mv, "o-", color="#1565c0", label="V 均值")
    ax.axhline(st.mean(mv), color="#1565c0", linestyle=":", linewidth=1, label="整局平均 V")
    ax.set_title("一局之内 V 的变化（最近 %d 局）" % len(recent))
    ax.set_xlabel("局内进度（0 = 第一步，100 = 顶出前一步）")
    ax.set_ylabel("模型给出的 V")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    xs2, mp, sp = binned(recent, "phi2")
    ax2.plot(xs2, mp, "s--", color="#c0392b", linewidth=1.2, label="行变化 φ2 均值")
    ax2.set_ylabel("行变化 φ2", color="#c0392b")
    ax2.legend(fontsize=8, loc="lower right")

    # 2) V 与 φ2 的相对变化幅度
    ax = axes[1]
    span_v = max(mv) - min(mv)
    span_p = max(mp) - min(mp)
    ax.bar(["V", "行变化 φ2", "洞数 φ4"],
           [span_v, span_p, max(binned(recent, "phi4")[1]) - min(binned(recent, "phi4")[1])],
           color=["#1565c0", "#c0392b", "#ef6c00"])
    ax.set_title("整局内的变化幅度（绝对值）")
    ax.set_ylabel("max − min")
    for i, val in enumerate([span_v, span_p,
                             max(binned(recent, "phi4")[1]) - min(binned(recent, "phi4")[1])]):
        ax.text(i, val, "%.1f" % val, ha="center", va="bottom", fontsize=9)

    # 3) 几局样本 V(turn) 原貌
    ax = axes[2]
    picks = [keys[-1], keys[-2], keys[-3], keys[len(keys) // 2], keys[len(keys) // 4]]
    for k in picks:
        steps = episodes[k]
        ax.plot([s["turn"] for s in steps], [s["v"] for s in steps],
                linewidth=1.3, label="第 %d 局（%d 步）" % (k, len(steps)))
    ax.set_title("几局样本的 V(turn)")
    ax.set_xlabel("这一局的第几步")
    ax.set_ylabel("V")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(args.output, dpi=130)
    print("V 全程 %.1f → %.1f（变化 %.1f）；行变化 %.1f → %.1f（变化 %.1f）" %
          (mv[0], mv[-1], mv[-1] - mv[0], mp[0], mp[-1], mp[-1] - mp[0]))
    print("图已保存：%s" % args.output)


if __name__ == "__main__":
    main()
