import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


FEATURE_NAMES = ["height", "line_clear", "drow", "dcol", "holes", "well"]


def main():
    parser = argparse.ArgumentParser(
        description="Visualize episode returns and feature/weight changes from a training trace."
    )
    parser.add_argument("trace", nargs="?", default="training_trace_1000.csv")
    parser.add_argument(
        "--output", default="training_analysis_1000.png", help="Output image path."
    )
    args = parser.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.is_file():
        raise SystemExit(f"Trace file not found: {trace_path}")

    required = {"episode", "turn", "score", "td_error"}
    for i in range(6):
        required.update(
            {
                f"feature{i}",
                f"weight_after{i}",
                f"weight_delta{i}",
            }
        )

    episodes = defaultdict(
        lambda: {
            "score": 0.0,
            "turns": 0,
            "td_abs_sum": 0.0,
            "weights": [None] * 6,
            "feature_abs_sum": [0.0] * 6,
            "delta_abs_sum": [0.0] * 6,
        }
    )
    updates = 0

    with trace_path.open("r", newline="", encoding="utf-8-sig") as trace_file:
        reader = csv.DictReader(trace_file)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Trace is missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            try:
                episode = int(row["episode"])
                score = float(row["score"])
                td_error = float(row["td_error"])
                features = [float(row[f"feature{i}"]) for i in range(6)]
                weights = [float(row[f"weight_after{i}"]) for i in range(6)]
                deltas = [float(row[f"weight_delta{i}"]) for i in range(6)]
            except (TypeError, ValueError) as error:
                raise SystemExit(f"Invalid numeric value on CSV row {updates + 2}: {error}")

            values = [score, td_error, *features, *weights, *deltas]
            if not all(math.isfinite(value) for value in values):
                raise SystemExit(
                    f"Non-finite value found on CSV row {updates + 2}; "
                    "the trace cannot be analyzed safely."
                )

            data = episodes[episode]
            data["score"] += score
            data["turns"] += 1
            data["td_abs_sum"] += abs(td_error)
            data["weights"] = weights
            for i in range(6):
                data["feature_abs_sum"][i] += abs(features[i])
                data["delta_abs_sum"][i] += abs(deltas[i])
            updates += 1

    if not episodes:
        raise SystemExit(f"No training updates found in {trace_path}")

    episode_ids = sorted(episodes)
    scores = [episodes[episode]["score"] for episode in episode_ids]
    weight_series = [
        [episodes[episode]["weights"][i] for episode in episode_ids]
        for i in range(6)
    ]
    mean_abs_features = [
        [
            episodes[episode]["feature_abs_sum"][i] / episodes[episode]["turns"]
            for episode in episode_ids
        ]
        for i in range(6)
    ]
    mean_abs_deltas = [
        [
            episodes[episode]["delta_abs_sum"][i] / episodes[episode]["turns"]
            for episode in episode_ids
        ]
        for i in range(6)
    ]
    mean_abs_td = [
        episodes[episode]["td_abs_sum"] / episodes[episode]["turns"]
        for episode in episode_ids
    ]

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    figure.suptitle(f"Q-learning training trace: {trace_path.name} ({updates:,} updates)")

    axes[0, 0].plot(episode_ids, scores, linewidth=0.8, alpha=0.55, label="Episode score")
    if len(scores) >= 10:
        window = max(1, len(scores) // 50)
        rolling = [
            sum(scores[max(0, i - window + 1) : i + 1])
            / len(scores[max(0, i - window + 1) : i + 1])
            for i in range(len(scores))
        ]
        axes[0, 0].plot(episode_ids, rolling, linewidth=2, label=f"Rolling mean ({window})")
    axes[0, 0].set_title("Episode score")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    for i, name in enumerate(FEATURE_NAMES):
        axes[0, 1].plot(episode_ids, weight_series[i], label=name, linewidth=1)
    axes[0, 1].set_title("Weights after each episode")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Weight (symlog scale)")
    axes[0, 1].set_yscale("symlog", linthresh=1)
    axes[0, 1].legend(ncol=2, fontsize=8)
    axes[0, 1].grid(alpha=0.25)

    for i, name in enumerate(FEATURE_NAMES):
        axes[1, 0].plot(episode_ids, mean_abs_deltas[i], label=name, linewidth=1)
    axes[1, 0].set_title("Mean absolute weight change per move")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Mean |weight delta|")
    axes[1, 0].set_yscale("symlog", linthresh=1e-4)
    axes[1, 0].legend(ncol=2, fontsize=8)
    axes[1, 0].grid(alpha=0.25)

    for i, name in enumerate(FEATURE_NAMES):
        axes[1, 1].plot(episode_ids, mean_abs_features[i], label=name, linewidth=1)
    axes[1, 1].set_title("Mean absolute feature value per move")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Mean |feature|")
    axes[1, 1].legend(ncol=2, fontsize=8)
    axes[1, 1].grid(alpha=0.25)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    print(f"Trace: {trace_path}")
    print(f"Episodes: {len(episode_ids)}; updates: {updates}")
    print(f"Mean episode score: {sum(scores) / len(scores):.3f}")
    print(f"Mean absolute TD error: {sum(mean_abs_td) / len(mean_abs_td):.3f}")
    print("Feature summary:")
    summary = []
    for i, name in enumerate(FEATURE_NAMES):
        mean_feature = sum(mean_abs_features[i]) / len(episode_ids)
        mean_delta = sum(mean_abs_deltas[i]) / len(episode_ids)
        final_weight = weight_series[i][-1]
        summary.append((mean_delta, name, mean_feature, final_weight))
    for mean_delta, name, mean_feature, final_weight in sorted(summary, reverse=True):
        print(
            f"  {name:10s} mean|feature|={mean_feature:.6g} "
            f"mean|delta_w|={mean_delta:.6g} final_weight={final_weight:.6g}"
        )
    print(f"Plot: {output_path}")


if __name__ == "__main__":
    main()
