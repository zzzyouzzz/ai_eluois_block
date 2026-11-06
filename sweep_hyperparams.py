#!/usr/bin/env python3
"""超参数扫描：一组参数一个配置，各自从全零权重训练，出分析图 + 测试分数。

用法（零参数即可跑默认网格）：
    python sweep_hyperparams.py
    python sweep_hyperparams.py --only DQN_BATCH_SIZE --seeds 11
    python sweep_hyperparams.py --episodes 2000 --workers 6

每个配置做的事：
    1. gcc 用 -D 覆盖参数，编译出一个只属于这个配置的 eluose.exe；
    2. 从全零权重开始（TRAIN_FRESH=1）训练 --episodes 局，写 trace.csv；
    3. 把训练好的模型拿去跑测试集（贪心，不探索），读纯棋盘得分；
    4. 调 analyze_training.py 出这个配置的训练分析图。

为什么固定种子：train_value_model 里 srand(TRAIN_SEED) 为 0 时取当前时间，
同参数两次训练的轨迹不同。这里给同一批对比实验传同一个种子（seed 相同），
参数才是唯一变量；再换几个种子重复，用中位数报结果。
"""
import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "eluose.c"
ANALYZER = ROOT / "analyze_training.py"
OUT_DIR = ROOT / "sweep"

TRAIN_SEQ = ROOT / "1.txt"          # 训练集
TEST_SEQ = ROOT / "2.txt"           # 测试集（训练时从没读过）
EPISODES = 10000                     # 每个配置训练多少局
SEEDS = [482913,
736105,
918274,
305761,
659428,
127583]                # 重复次数：同一配置换种子再跑
ANALYZE_WINDOW = 25                 # analyze_training.py 的滚动窗口（局）
PRUNE_TRACE = True                 # 出完图删掉 trace.csv（每份三十多 MB）
FIXES = {}                          # --fix 钉死的参数：写进目录名和编译宏，但不在本批扫描

# 每个参数名就是 -D 的宏名，默认值必须和 eluose.c 里的默认值一致，
# 否则网格里的"默认"那一档并不是真正的默认配置。
DEFAULTS = {
    "DQN_BATCH_SIZE": 24,
    "DQN_REPLAY_WARMUP": 5000,
    "DQN_TARGET_UPDATE_INTERVAL": 500,
    "DQN_LEARNING_RATE": 0.0001,
    "DQN_RHO_RATE": 0.01,
    "REWARD_SCALE": 0.01,
    "PENALTY_TOP_OUT": 5.0,
    "PENALTY_HOLE": 0.0,
    "PENALTY_HEIGHT": 0.0,
}

# 枚举范围参考常见 DQN 配置（Atari / 表格型小型任务的常用区间），
# 并按本项目的量级收窄：回放区 5 万、批更新很轻，太大或太小的值在这个
# 步数规模下本来就没机会体现出差别。
# 每个参数的列表会自动补上默认值，所以"默认"那一档一定在曲线里。
GRID = {
    # 批大小：16 太抖、256 在 2000 局里更新次数太少
    "DQN_BATCH_SIZE": [8, 12, 16, 24, 32, 64, 128, 256],
    # 回放预热：太小等于拿没装满的回放区当分布，太大等于前几十局不学习
    "DQN_REPLAY_WARMUP": [500, 1000, 2000, 3000, 5000, 7000, 10000, 15000],
    # 目标网络同步间隔（按"批更新次数"计）
    "DQN_TARGET_UPDATE_INTERVAL": [100, 200, 400, 600, 800, 1200, 1600, 2400],
    # Adam 学习率
    "DQN_LEARNING_RATE": [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    # ρ（差分基线）的学习率
    "DQN_RHO_RATE": [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3],
    # 奖励缩放
    "REWARD_SCALE": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 1],
    # 奖励整形里的三个惩罚系数
    "PENALTY_TOP_OUT": [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0],
    "PENALTY_HOLE": [0.0, 0.00005, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
    "PENALTY_HEIGHT": [0.0, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 1],
}

PARAM_LABELS = {
    "DQN_BATCH_SIZE": "批大小",
    "DQN_REPLAY_WARMUP": "回放预热",
    "DQN_TARGET_UPDATE_INTERVAL": "目标网络同步间隔",
    "DQN_LEARNING_RATE": "学习率",
    "DQN_RHO_RATE": "ρ 学习率",
    "REWARD_SCALE": "奖励缩放",
    "PENALTY_TOP_OUT": "顶出惩罚",
    "PENALTY_HOLE": "洞惩罚",
    "PENALTY_HEIGHT": "高度惩罚",
}


# ---------------------------------------------------------------- 工具函数

def c_value(value):
    """把 Python 数值写成能进 C 宏的形式（浮点保留小数点，避免 5 变成 5.0 的坑）。"""
    if isinstance(value, int):
        return str(value)
    text = repr(float(value))
    if "e" in text or "E" in text:
        return f"{value:.10f}".rstrip("0").rstrip(".")
    return text


def slug(param, value, seed):
    parts = [f"{param}={c_value(value)}"]
    for name in sorted(FIXES):
        parts.append(f"{name}={c_value(FIXES[name])}")
    name = "_".join(parts).replace(".", "p")
    return f"{name}_s{seed}"


def parse_value_overrides(items):
    """--values NAME=v1,v2 → {NAME: [v1, v2]}；能当整数就当整数，否则按浮点。"""
    overrides = {}
    for item in items:
        name, _, raw = item.partition("=")
        name = name.strip()
        if not name or not raw.strip():
            raise SystemExit(f"--values 要写成 NAME=v1,v2：{item}")
        values = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(int(token))
                continue
            except ValueError:
                pass
            try:
                values.append(float(token))
            except ValueError:
                raise SystemExit(f"--values 里的值不是数字：{token}") from None
        if not values:
            raise SystemExit(f"--values 没有给出任何值：{item}")
        overrides[name] = values
    return overrides


def run(cmd, cwd, timeout=1800):
    """跑一个子进程，返回 (returncode, stdout+stderr)。"""
    proc = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


# ---------------------------------------------------------------- 单个配置

def compile_config(build_dir, defines, exe_name="run.exe"):
    exe = build_dir / exe_name
    cmd = ["gcc", "-O2", "-Wall", "-o", str(exe), str(SOURCE)]
    cmd += [f"-D{key}={value}" for key, value in defines.items()]
    code, output = run(cmd, build_dir)
    if code != 0:
        raise RuntimeError(f"编译失败：\n{output}")
    # 源码里漏写 #ifndef 时 -D 会被悄悄覆盖回去，扫出来的网格全是一模一样的
    # 结果还不报错；这里直接把 redefined 当成致命错误
    if "redefined" in output:
        raise RuntimeError(f"宏被源码覆盖（缺少 #ifndef？）：\n{output}")
    return exe


def parse_test_output(text):
    """从 -i 模式的输出里取终局前最后一个棋盘得分。

    每步打印三行：「rotation x」「分数」「…」，终局打印 You lose! + 棋盘。
    分数是纯棋盘分（environment_step 的 board_score），不含奖励惩罚项。
    """
    lines = text.splitlines()
    lose_at = None
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip() == "You lose!":
            lose_at = index
            break
    end = lose_at if lose_at is not None else len(lines)
    score = None
    steps = 0
    index = 0
    while index < end:
        if re.fullmatch(r"\d+ \d+", lines[index].strip()) and index + 1 < end:
            value = lines[index + 1].strip()
            if re.fullmatch(r"-?\d+", value):
                score = int(value)
                steps += 1
                index += 2
                continue
        index += 1
    return score, steps


TRACE_COLUMNS = ("episode", "score", "bias_after", "current_value", "td_error")


def trace_stats(path):
    """从追踪 CSV 里算几个训练期的汇总指标（只读需要的列）。"""
    totals = {name: 0.0 for name in ("score", "steps", "td_abs", "value_abs")}
    last_bias = None
    episodes = 0
    previous_episode = None
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return None
        missing = [name for name in TRACE_COLUMNS if name not in header]
        if missing:
            raise RuntimeError(f"{path.name} 缺少列：{', '.join(missing)}")
        index = {name: header.index(name) for name in TRACE_COLUMNS}
        for row in reader:
            if not row:
                continue
            episode = row[index["episode"]]
            if episode != previous_episode:
                episodes += 1
                previous_episode = episode
            totals["score"] += float(row[index["score"]])
            totals["steps"] += 1
            totals["td_abs"] += abs(float(row[index["td_error"]]))
            totals["value_abs"] += abs(float(row[index["current_value"]]))
            bias = row[index["bias_after"]]
            last_bias = float(bias) if bias else last_bias
    if not episodes:
        return None
    return {
        "episodes": episodes,
        "mean_score": totals["score"] / episodes,
        "mean_steps": totals["steps"] / episodes,
        "final_bias": last_bias,
        "rel_td": totals["td_abs"] / max(totals["value_abs"], 1e-9),
    }


def run_config(param, value, seed, episodes, force=False):
    work_dir = OUT_DIR / slug(param, value, seed)
    result_path = work_dir / "result.json"
    fixes = " ".join(f"{name}={c_value(FIXES[name])}" for name in sorted(FIXES))
    if result_path.exists() and not force:
        with open(result_path, encoding="utf-8") as handle:
            cached = json.load(handle)
        # 局数或固定参数变了就不能复用：200 局和 2000 局、固定 lr 和不固定，
        # 训出来的根本不是同一个东西
        if cached.get("episodes") == episodes and cached.get("fixes", "") == fixes:
            print(f"  [跳过] {work_dir.name}（已有结果）")
            return cached

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    defines = {
        "TRAIN_EPISODES": episodes,
        "TRAIN_FRESH": 1,
        "TRAIN_SEED": seed,
        "TRAINING_TRACE_PATH": '"trace.csv"',
        "VALUE_MODEL_PATH": '"q_weights.dat"',
    }
    for name, default in DEFAULTS.items():
        if name in FIXES:
            defines[name] = c_value(FIXES[name])
        else:
            defines[name] = c_value(value if name == param else default)

    started = time.time()
    exe = compile_config(work_dir, defines)

    code, output = run([str(exe), "--train", "-i", str(TRAIN_SEQ)], work_dir)
    if code != 0:
        raise RuntimeError(f"训练失败（{param}={c_value(value)} seed={seed}）：\n{output}")
    wraps = [line for line in output.splitlines() if "回绕" in line]

    # 测试：用固定不读初始模型的那份二进制，cwd 指向本配置目录，
    # 于是它读到的是刚训出来的 q_weights.dat
    code, output = run([str(PLAY_EXE), "-i", str(TEST_SEQ)], work_dir)
    if code != 0:
        raise RuntimeError(f"测试失败（{param}={c_value(value)} seed={seed}）：\n{output}")
    test_score, test_steps = parse_test_output(output)
    if test_score is None:
        raise RuntimeError(f"没在测试输出里找到分数（{param}={c_value(value)} seed={seed}）")

    stats = trace_stats(work_dir / "trace.csv") or {}
    code, output = run(
        [sys.executable, str(ANALYZER), "trace.csv",
         "--output", "training.png", "--window", str(ANALYZE_WINDOW)],
        work_dir,
    )
    figure = work_dir / "training.png"
    if code != 0 or not figure.exists():
        raise RuntimeError(f"分析画图失败（{param}={c_value(value)} seed={seed}）：\n{output}")
    # 每份 trace 三十多 MB，一轮扫描就是几个 GB；图已经出完了，想省空间就删掉
    if PRUNE_TRACE:
        (work_dir / "trace.csv").unlink(missing_ok=True)

    result = {
        "param": param,
        "value": value,
        "seed": seed,
        "episodes": episodes,
        "test_score": test_score,
        "test_steps": test_steps,
        "train_mean_score": round(stats.get("mean_score", 0.0), 3),
        "train_mean_steps": round(stats.get("mean_steps", 0.0), 3),
        "final_bias": None if stats.get("final_bias") is None else round(stats["final_bias"], 4),
        "rel_td": None if stats.get("rel_td") is None else round(stats["rel_td"], 5),
        "wraps": "; ".join(wraps),
        "seconds": round(time.time() - started, 1),
        "fixes": fixes,
        "dir": work_dir.name,
    }
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def make_play_exe():
    """测试用的二进制：TRAIN_FRESH=0，也就是 -i 模式会读 q_weights.dat。"""
    global PLAY_EXE
    OUT_DIR.mkdir(exist_ok=True)
    PLAY_EXE = compile_config(OUT_DIR, {"TRAIN_FRESH": 0}, "play.exe")
    return PLAY_EXE


# ---------------------------------------------------------------- 汇总

def summarise(results):
    table = {}
    for row in results:
        # 固定了别的参数的那批单独成组，别和同参数的自由扫描混成一个中位数
        key = row["param"]
        if row.get("fixes"):
            key = f"{key} @ {row['fixes']}"
        table.setdefault(key, {}).setdefault(row["value"], []).append(row)
    summary = {}
    for param, by_value in table.items():
        summary[param] = {}
        for value, rows in sorted(by_value.items()):
            scores = [row["test_score"] for row in rows]
            summary[param][value] = {
                "median": statistics.median(scores),
                "min": min(scores),
                "max": max(scores),
                "scores": scores,
                "seeds": len(scores),
                "mean_train_steps": statistics.median(
                    [row["train_mean_steps"] for row in rows]
                ),
                "rel_td": statistics.median(
                    [row["rel_td"] for row in rows if row["rel_td"] is not None]
                ) if any(row["rel_td"] is not None for row in rows) else None,
            }
    return summary


def print_table(summary):
    print()
    for param, by_value in summary.items():
        label = PARAM_LABELS.get(param, param)
        print(f"{label}（{param}）")
        default = DEFAULTS.get(param)
        base = by_value.get(default) if default is not None else None
        for value, data in sorted(by_value.items()):
            mark = "  ← 默认" if value == default else ""
            ratio = ""
            if base and base["median"]:
                ratio = f"  {data['median'] / base['median']:.2f}x"
            rel = "" if data["rel_td"] is None else f"  rel_td={data['rel_td']:.3f}"
            print(
                f"  {c_value(value):>10}  测试分中位数 {data['median']:>9.0f}"
                f"  [{data['min']:.0f}~{data['max']:.0f}]"
                f"  训练均步数 {data['mean_train_steps']:>6.1f}{rel}{ratio}{mark}"
            )
        print()


def plot_summary(summary, path, seed_count=None, episodes=None):
    """每个参数一格：横轴取值、纵轴测试分中位数，竖条是各种子的极差。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 和 analyze_training.py 同一套字体，否则中文标题全是方框
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # 按 GRID 声明的顺序排面板：results 的插入顺序跟着并发完成顺序走，
    # 直接用它会让每次跑出来的小图顺序都不一样
    params = [name for name in GRID if name in summary]
    params += [name for name in summary if name not in GRID and " @ " not in name]
    # 钉在其他基准上扫的那批排在后面单独成格，否则默认基准的结论会被它盖掉
    params += sorted(name for name in summary if " @ " in name)
    if not params:
        # 全是 --fix 钉死的配置，没有可画的小格；重算汇总时别在这儿炸掉
        print("这批只有固定参数的配置，跳过汇总图")
        return
    columns = 3
    rows = (len(params) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 3.4 * rows))
    axes = axes.reshape(-1)
    for axis, param in zip(axes, params):
        by_value = summary[param]
        values = sorted(by_value)
        medians = [by_value[value]["median"] for value in values]
        low = [by_value[value]["median"] - by_value[value]["min"] for value in values]
        high = [by_value[value]["max"] - by_value[value]["median"] for value in values]
        positions = list(range(len(values)))
        # 单局贪心分是重尾的：同配置换种子能差几十上百倍（README L11），
        # 所以既画中位数+极差，也把每个种子单独点出来，免得"看着很稳"其实是巧合
        for index, value in enumerate(values):
            seeds = by_value[value].get("scores") or []
            jitter = [index + (i - (len(seeds) - 1) / 2) * 0.06 for i in range(len(seeds))]
            axis.scatter(jitter, seeds, s=18, color="#78909c", alpha=0.85, zorder=3)
        axis.errorbar(
            positions, medians, yerr=[low, high], marker="o",
            color="#1565c0", ecolor="#90a4ae", capsize=4, linewidth=1.6,
        )
        axis.set_xticks(positions)
        labels = [c_value(value) for value in values]
        # 细网格（学习率能到 0.00003/0.00005）横轴标签会挤成一团，斜着放
        rotate = len(values) > 6 or max(len(text) for text in labels) > 6
        axis.set_xticklabels(labels, fontsize=9,
                             rotation=30 if rotate else 0,
                             ha="right" if rotate else "center")
        base_param, _, fixed_text = param.partition(" @ ")
        title = PARAM_LABELS.get(base_param, base_param)
        if fixed_text:
            # 基准参数可能钉了六七个，整串塞进标题会直接糊出画布：
            # 去掉 DQN_/PENALTY_ 前缀、只留名字=值，再按整项折行（不能从中间劈开）
            items = [
                item.replace("DQN_", "").replace("PENALTY_", "")
                for item in fixed_text.split()
            ]
            lines, current = [], ""
            for item in items:
                candidate = item if not current else current + ", " + item
                if len(candidate) > 46 and current:
                    lines.append(current)
                    current = item
                else:
                    current = candidate
            if current:
                lines.append(current)
            title += "\n" + "\n".join(lines)
        axis.set_title(title, fontsize=10 if not fixed_text else 8)
        axis.set_xlabel(base_param, fontsize=8)
        axis.set_ylabel("测试分（纯棋盘分）", fontsize=8)
        # 分数跨好几个数量级时线性轴会把小值压成一条线，跨度超过 10 倍就换对数轴
        positive = [value for value in medians if value > 0]
        if len(positive) > 1 and max(positive) > 10.0 * min(positive):
            axis.set_yscale("log")
        default = DEFAULTS.get(base_param)
        if default in by_value:
            index = values.index(default)
            axis.axvline(index, color="#c62828", linestyle="--", linewidth=1.0, alpha=0.7)
            axis.annotate("默认", (index, max(medians)), fontsize=8,
                          color="#c62828", textcoords="offset points", xytext=(4, 6))
        axis.grid(alpha=0.25)
    for axis in axes[len(params):]:
        axis.axis("off")
    figure.suptitle(
        f"超参数扫描：{episodes or EPISODES} 局训练后测试集贪心得分"
        "（蓝点=各种子中位数，灰点=单次，误差条=极差）",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def finish(results, failures, episodes, seeds):
    """汇总表 / results.csv / summary.json / 汇总图，训练完和 --rebuild 共用。"""
    summary = summarise(results)
    print_table(summary)
    if failures:
        print(f"{len(failures)} 个配置失败：")
        for param, value, seed, error in failures:
            print(f"  {param}={c_value(value)} seed={seed}: {error.splitlines()[0]}")

    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({"episodes": episodes, "seeds": seeds, "params": {
            name: {c_value(value): data for value, data in by_value.items()}
            for name, by_value in summary.items()
        }}, handle, ensure_ascii=False, indent=2)

    rows_path = OUT_DIR / "results.csv"
    fields = ["param", "value", "seed", "test_score", "test_steps",
              "train_mean_score", "train_mean_steps", "final_bias",
              "rel_td", "seconds", "dir"]
    with open(rows_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(results, key=lambda item: (item["param"], item["value"], item["seed"])):
            writer.writerow({key: row.get(key) for key in fields})

    figure_path = OUT_DIR / "sweep_summary.png"
    if summary:
        plot_summary(summary, figure_path, len(seeds), episodes)
        print(f"汇总图：{figure_path}")
    print(f"明细：{rows_path}")
    print(f"成绩：{summary_path}")
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser(description="超参数扫描 + 测试集评估")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--only", type=str, default="", help="只跑某个参数（宏名）")
    parser.add_argument("--values", action="append", default=[],
                        metavar="NAME=v1,v2,...",
                        help="临时替换某个参数的枚举值（可重复），例如 "
                             "--values DQN_LEARNING_RATE=3e-5,5e-5,1e-4")
    parser.add_argument("--fix", action="append", default=[],
                        metavar="NAME=VALUE",
                        help="把某个参数钉死在给定值、不参与扫描（可重复），"
                             "用来在更好的基准上比其它参数")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--force", action="store_true", help="忽略已有结果重跑")
    parser.add_argument("--rebuild", action="store_true",
                        help="不训练，只用已有 result.json 重算汇总表和汇总图")
    parser.add_argument("--prune-trace", action="store_true",
                        help="出完训练图就删掉 trace.csv（一轮扫描能省几个 GB）")
    args = parser.parse_args()
    global PRUNE_TRACE
    PRUNE_TRACE = args.prune_trace
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    if args.rebuild:
        results = []
        for result_path in sorted(OUT_DIR.glob("*/result.json")):
            with open(result_path, encoding="utf-8") as handle:
                row = json.load(handle)
            if row.get("episodes") == args.episodes:
                results.append(row)
        if not results:
            raise SystemExit(f"{OUT_DIR} 下没有可用的 result.json，先正常跑一轮")
        print(f"只重算汇总：{len(results)} 个已有配置")
        return finish(results, [], args.episodes, seeds)

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    overrides = parse_value_overrides(args.values)
    unknown = [name for name in overrides if name not in GRID]
    if unknown:
        raise SystemExit(
            f"不认识的参数：{', '.join(unknown)}；可用：{', '.join(GRID)}"
        )
    for item in args.fix:
        parsed = parse_value_overrides([item])
        FIXES.update({name: values[0] for name, values in parsed.items()})
    unknown = [name for name in FIXES if name not in GRID]
    if unknown:
        raise SystemExit(
            f"不认识的参数：{', '.join(unknown)}；可用：{', '.join(GRID)}"
        )
    params = {}
    for name, values in GRID.items():
        if args.only and name != args.only:
            continue
        if name in FIXES:      # 钉死的参数不再扫
            continue
        params[name] = overrides.get(name, list(values))
    if not params:
        raise SystemExit(f"没有匹配的参数：{args.only}")

    make_play_exe()
    jobs = []
    for param, values in params.items():
        candidates = list(values)
        if DEFAULTS[param] not in candidates:
            candidates.append(DEFAULTS[param])
        for value in candidates:
            for seed in seeds:
                jobs.append((param, value, seed))

    print(f"共 {len(jobs)} 个训练配置（{len(params)} 个参数 × 平均 "
          f"{len(jobs) / len(params):.1f} 档 × {len(seeds)} 个种子），"
          f"{args.episodes} 局/配置，{args.workers} 并发")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = []
    failures = []
    done = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_config, param, value, seed, args.episodes, args.force):
                (param, value, seed)
            for param, value, seed in jobs
        }
        for future in as_completed(futures):
            param, value, seed = futures[future]
            done += 1
            try:
                results.append(future.result())
            except Exception as error:  # 单点失败不该毁掉整轮扫描
                failures.append((param, value, seed, str(error)))
                print(f"  [失败] {param}={c_value(value)} seed={seed}: {error}")
            elapsed = time.time() - started
            print(f"  ({done}/{len(jobs)}) 已完成，用时 {elapsed:.0f}s")

    summary = summarise(results)
    if failures:
        for param, value, seed, error in failures:
            print(f"  [失败] {param}={c_value(value)} seed={seed}: {error.splitlines()[0]}")
    return finish(results, failures, args.episodes, seeds)


if __name__ == "__main__":
    raise SystemExit(main())
