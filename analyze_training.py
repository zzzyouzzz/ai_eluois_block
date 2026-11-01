"""训练追踪 CSV 的可视化分析。

数据从哪来（见 eluose.c 开头的「参数配置区」）：
   把 TRAINING_TRACE_PATH 改成 "training_trace.csv" 再重新编译，
   然后照常 `eluose.exe --train -i 1.txt`，得到逐动作的训练追踪。

用法：
    python analyze_training.py training_trace.csv
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


FEATURE_NAMES = ["高度", "消行", "行变化", "列变化", "洞数", "井深"]
WEIGHT_NAMES = [f"w{i}（{name}）" for i, name in enumerate(FEATURE_NAMES)]
LINE_CLEAR_COUNTS = {0: 0, 100: 1, 300: 2, 500: 3, 800: 4}


def parse_number(row, key, row_number):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(
            f"CSV 第 {row_number} 行的 {key} 不是有效数字：{error}"
        ) from error
    if not math.isfinite(value):
        raise SystemExit(f"CSV 第 {row_number} 行的 {key} 不是有限数")
    return value


def rolling_mean(values, window):
    result = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        result.append(running_sum / min(index + 1, window))
    return result


def prepare_matplotlib():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def feature_scale(index, value, board_cells):
    if index == 0:
        return value / 20.0
    if index == 1:
        return value / 4.0
    return value / max(board_cells, 4.0)


def load_training_trace(path):
    required = {
        "episode",
        "turn",
        "epsilon",
        "score",
        "terminal",
        "td_error",
        "board_cells",
    }
    required.update(f"feature{i}" for i in range(6))
    required.update(f"weight_after{i}" for i in range(6))
    required.update(f"weight_before{i}" for i in range(6))
    episodes = defaultdict(
        lambda: {
            "score": 0.0,
            "steps": 0,
            "epsilon_sum": 0.0,
            "td_abs_sum": 0.0,
            "top_outs": 0,
            "line_clears": 0,
            "features": [0.0] * 6,
            "contributions": [0.0] * 6,
            "weights": [None] * 6,
        }
    )
    rows = 0

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing = required - columns
        if missing:
            raise SystemExit(f"训练追踪缺少列：{', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            try:
                episode = int(parse_number(row, "episode", row_number))
                parse_number(row, "turn", row_number)
                score = parse_number(row, "score", row_number)
                epsilon = parse_number(row, "epsilon", row_number)
                features = [
                    parse_number(row, f"feature{i}", row_number)
                    for i in range(6)
                ]
                before = [
                    parse_number(row, f"weight_before{i}", row_number)
                    for i in range(6)
                ]
                weights = [
                    parse_number(row, f"weight_after{i}", row_number)
                    for i in range(6)
                ]
                terminal = int(parse_number(row, "terminal", row_number))
                td_error = parse_number(row, "td_error", row_number)
                board_cells = parse_number(row, "board_cells", row_number)
            except ValueError as error:
                raise SystemExit(
                    f"CSV 第 {row_number} 行包含无效局数或步数：{error}"
                ) from error

            data = episodes[episode]
            data["steps"] += 1
            data["epsilon_sum"] += epsilon
            data["score"] += score
            data["top_outs"] += terminal
            lines_cleared = LINE_CLEAR_COUNTS.get(int(score), 0)
            data["line_clears"] += lines_cleared

            for index in range(6):
                normalized = feature_scale(index, features[index], board_cells)
                data["features"][index] += normalized
                data["contributions"][index] += before[index] * normalized
                data["weights"][index] = weights[index]
            data["td_abs_sum"] += abs(td_error)
            rows += 1

    if not episodes:
        raise SystemExit(f"训练追踪中没有数据：{path}")

    result = {}
    for episode, data in episodes.items():
        steps = data["steps"]
        result[episode] = {
            "score": data["score"],
            "steps": steps,
            "epsilon": data["epsilon_sum"] / steps,
            "td_abs": data["td_abs_sum"] / steps,
            "top_out_rate": data["top_outs"] / steps,
            "line_clears_per_move": data["line_clears"] / steps,
            "features": [value / steps for value in data["features"]],
            "contributions": [
                value / steps for value in data["contributions"]
            ],
            "weights": data["weights"],
        }
    return result, rows


def plot_training(trace_path, episodes, rows, window, output_path):
    episode_ids = sorted(episodes)
    scores = [episodes[e]["score"] for e in episode_ids]
    steps = [episodes[e]["steps"] for e in episode_ids]
    epsilons = [episodes[e]["epsilon"] for e in episode_ids]
    weights = [
        [episodes[e]["weights"][i] for e in episode_ids] for i in range(6)
    ]
    features = [
        rolling_mean(
            [episodes[e]["features"][i] for e in episode_ids], window
        )
        for i in range(6)
    ]
    contributions = [
        rolling_mean(
            [episodes[e]["contributions"][i] for e in episode_ids], window
        )
        for i in range(6)
    ]

    figure, axes = plt.subplots(3, 2, figsize=(17, 14), constrained_layout=True)
    figure.suptitle(
        f"训练过程分析：{trace_path.name}（{len(episode_ids)} 局，{rows:,} 步）"
    )

    ax = axes[0, 0]
    ax.plot(episode_ids, scores, color="#9bb7d4", alpha=0.4, linewidth=0.7,
            label="每局得分")
    ax.plot(
        episode_ids,
        rolling_mean(scores, window),
        color="#1769aa",
        linewidth=2,
        label=f"得分滚动均值（{window} 局）",
    )
    step_ax = ax.twinx()
    step_ax.plot(
        episode_ids,
        rolling_mean(steps, window),
        color="#e67e22",
        linewidth=1.7,
        label=f"存活步数滚动均值（{window} 局）",
    )
    ax.set_title("训练得分与存活步数")
    ax.set_xlabel("训练局数")
    ax.set_ylabel("得分")
    step_ax.set_ylabel("存活步数")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    other_handles, other_labels = step_ax.get_legend_handles_labels()
    ax.legend(handles + other_handles, labels + other_labels, fontsize=8)

    axes[0, 1].plot(episode_ids, epsilons, color="#7b1fa2", linewidth=1.5)
    axes[0, 1].set_title("探索率 epsilon")
    axes[0, 1].set_xlabel("训练局数")
    axes[0, 1].set_ylabel("epsilon")
    axes[0, 1].grid(alpha=0.25)

    for index, name in enumerate(WEIGHT_NAMES):
        axes[1, 0].plot(
            episode_ids, weights[index], label=name, linewidth=1.2
        )
    axes[1, 0].set_title("主干 6 个权重变化")
    axes[1, 0].set_xlabel("训练局数")
    axes[1, 0].set_ylabel("权重")
    axes[1, 0].set_yscale("symlog", linthresh=1)
    axes[1, 0].legend(ncol=2, fontsize=8)
    axes[1, 0].grid(alpha=0.25)

    for index, name in enumerate(FEATURE_NAMES):
        axes[1, 1].plot(
            episode_ids, features[index], label=name, linewidth=1.2
        )
    axes[1, 1].set_title("归一化特征均值（滚动）")
    axes[1, 1].set_xlabel("训练局数")
    axes[1, 1].set_ylabel("归一化特征值")
    axes[1, 1].legend(ncol=2, fontsize=8)
    axes[1, 1].grid(alpha=0.25)

    for index, name in enumerate(FEATURE_NAMES):
        axes[2, 0].plot(
            episode_ids,
            contributions[index],
            label=name,
            linewidth=1.2,
        )
    axes[2, 0].axhline(0, color="black", linewidth=0.7, alpha=0.6)
    axes[2, 0].set_title("权重 × 归一化特征（对价值的平均贡献）")
    axes[2, 0].set_xlabel("训练局数")
    axes[2, 0].set_ylabel("平均价值贡献")
    axes[2, 0].legend(ncol=2, fontsize=8)
    axes[2, 0].grid(alpha=0.25)

    top_out_rates = [
        episodes[e]["top_out_rate"] for e in episode_ids
    ]
    clear_rates = [
        episodes[e]["line_clears_per_move"] for e in episode_ids
    ]
    axes[2, 1].plot(
        episode_ids,
        rolling_mean(clear_rates, window),
        color="#2e7d32",
        label="每步消行数",
    )
    axes[2, 1].plot(
        episode_ids,
        rolling_mean(top_out_rates, window),
        color="#c62828",
        label="顶出率",
    )
    axes[2, 1].plot(
        episode_ids,
        rolling_mean(
            [episodes[e]["td_abs"] for e in episode_ids], window
        ),
        color="#455a64",
        label="平均绝对 TD 误差",
    )
    axes[2, 1].set_title("训练稳定性与消行效率")
    axes[2, 1].set_xlabel("训练局数")
    axes[2, 1].set_ylabel("滚动均值 / 比率")
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "用中文图表分析俄罗斯方块 DQN 的训练追踪。"
            "追踪文件由 eluose.c 开头的「参数配置区」"
            "（TRAINING_TRACE_PATH）控制。"
        )
    )
    parser.add_argument("trace", nargs="?", default="training_trace.csv")
    parser.add_argument(
        "--output",
        default="training_analysis.png",
        help="训练图输出文件。",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=25,
        help="滚动平均窗口大小，单位为训练局。",
    )
    args = parser.parse_args()

    if args.window < 1:
        parser.error("--window 必须大于零")
    trace_path = Path(args.trace)
    if not trace_path.is_file():
        raise SystemExit(
            f"找不到训练追踪文件：{trace_path}\n"
            "生成方法：把 eluose.c 配置区里的 TRAINING_TRACE_PATH 改成 "
            f'"{trace_path}"，重新编译后跑一次 `eluose.exe --train -i 1.txt`。'
        )

    prepare_matplotlib()
    episodes, rows = load_training_trace(trace_path)
    output_path = Path(args.output)
    plot_training(
        trace_path, episodes, rows, args.window, output_path
    )
    episode_ids = sorted(episodes)
    print(f"训练图：{output_path}")
    print(f"训练局数：{len(episode_ids)}；动作数：{rows:,}")
    print(
        f"平均每局得分："
        f"{sum(episodes[e]['score'] for e in episode_ids) / len(episode_ids):.3f}"
    )
    print(
        f"平均存活步数："
        f"{sum(episodes[e]['steps'] for e in episode_ids) / len(episode_ids):.3f}"
    )


if __name__ == "__main__":
    main()
