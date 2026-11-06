# 俄罗斯方块 DQN：线性价值 + 差分目标（源码导读）

单文件 C 实现的俄罗斯方块智能体：6 个棋盘特征线性逼近动作价值，外加一个常数项 `b`
（在当前实验分支里同时充当差分目标（differential target）的基线 ρ），用经验回放 +
目标网络做 Q-learning，Adam 更新。训练时每一步的中间量写进 CSV，配三个 Python 脚本
做复核与可视化。

> **当前状态**：`eluose.c` 共 1020 行；模型 7 个参数（6 权重 + b）；无隐藏层、
> 无评估模块；命令行只有 `--train` / `-i` 两个参数，其余全在文件开头的配置区。

## 目录

- [1. 快速开始](#1-快速开始)
- [2. 参数配置区](#2-参数配置区)
- [3. 算法](#3-算法)
- [4. 特征与偏置](#4-特征与偏置)
- [5. 代码结构](#5-代码结构)
- [6. 训练一局的完整时序](#6-训练一局的完整时序)
- [7. 追踪 CSV](#7-追踪-csv)
- [8. 工具与脚本](#8-工具与脚本)
- [9. 验证体系](#9-验证体系)
- [10. 经验与坑](#10-经验与坑)
- [11. 已知问题](#11-已知问题)
- [附录 A 源码索引](#附录-a-源码索引)
- [附录 B 从 Bellman 方程到 DQN](#附录-b-从-bellman-方程到-dqn)

## 1. 快速开始

```bash
# 编译（Windows 下 gcc 需要 -lm）
gcc -std=c99 -Wall -Wextra -O2 eluose.c -o eluose.exe -lm

# 训练：默认 20000 局，结束后把模型写回 q_weights.dat
./eluose.exe --train -i 1.txt

# 只下棋不训练：读 q_weights.dat，按贪心落子
./eluose.exe -i 2.txt
```

几个约定：

- **方块输入**：每行一个字母，`I O T J L S Z` 是七种方块；`X` 表示输入结束，
  `E` 表示提前终止。`1.txt` / `2.txt` / `3.txt` 各 100 万个方块。
- **坐标系**：`x` 左→右 0–9，`y` 下→上 0–19（打印时从第 19 行倒着打）。
- **模型文件**：`q_weights.dat`，7 个数（6 权重 + `b`）。也能读旧的 6 数文件（`b=0`）
  和早先的 `mlp 16` 格式（只取 6 个主干权重）。
- **训练只取决于配置区**：局数、是否从初始值开始、追踪文件名都写在源码开头；
  想临时改就用编译宏覆盖（见 §2）。
- ⚠️ **训练会把模型写进当前工作目录**的 `q_weights.dat`。做实验先切到临时目录，
  否则会覆盖仓库里的模型。

## 2. 参数配置区

文件开头 `eluose.c:8–95`，改完重新编译生效。带 `#ifndef` 的宏可以用 `-D` 覆盖：

```bash
gcc -DTRAIN_EPISODES=200 -DREWARD_SCALE=1.0 -DTRAINING_TRACE_PATH='"batch.csv"' \
    eluose.c -o eluose.exe -lm
```

| 组 | 宏 | 默认值 | 作用 |
| --- | --- | --- | --- |
| 运行 | `TRAIN_EPISODES` | 20000 | 训练局数 |
| 运行 | `TRAIN_FRESH` | 0 | 1 = 忽略模型文件，从源码初始权重开始（`-i` 模式同样生效） |
| 运行 | `TRAIN_SEED` | 0 | 0 = 用当前时间；非 0 = 固定种子，同参数可复现 |
| 运行 | `VALUE_MODEL_PATH` | `"q_weights.dat"` | 模型读写路径 |
| 运行 | `TRAINING_TRACE_PATH` | `"training_trace.csv"` | 追踪 CSV；`""` 关闭 |
| DQN | `DQN_REPLAY_CAPACITY` | 50000 | 回放区容量（注意内存，见 §10） |
| DQN | `DQN_BATCH_SIZE` | 32 | 每次更新抽多少条 |
| DQN | `DQN_REPLAY_WARMUP` | 5000 | 攒够多少条才开始更新 |
| DQN | `DQN_TARGET_UPDATE_INTERVAL` | 800 | 每多少次更新同步目标网络 |
| DQN | `DQN_GAMMA` | 0.99 | 折扣因子 |
| DQN | `DQN_LEARNING_RATE` | 0.001 | Adam 的步长（与梯度量级无关） |
| DQN | `REWARD_SCALE` | 0.01 | 消行分 0/100/300/500/800 乘以它才是训练奖励 |
| 奖励 | `PENALTY_TOP_OUT` | 10.0 | 顶出一次扣多少奖励 |
| 奖励 | `PENALTY_HOLE` | 0.005 | 每个洞每步扣多少奖励（`feature[4]`） |
| 奖励 | `PENALTY_HEIGHT` | 0.001 | 每格堆高每步扣多少奖励（`feature[0]`） |
| Adam | `ADAM_BETA1` / `ADAM_BETA2` / `ADAM_EPSILON` | 0.9 / 0.999 / 1e-8 | Adam 超参 |
| Adam | `DQN_LR_DECAY_STEPS` | 0 | 0 = 不衰减；N = 每 N 步学习率减半 |
| 差分 | `DQN_RHO_RATE` | 0.01 | 基线 ρ 的学习率：`ρ += rate · 批内平均 TD` |
| 探索 | `TRAIN_EPSILON_START` / `TRAIN_EPSILON_END` | 1 / 0.05 | 每局 ε 从 START 线性到 END |

`START == END` 即固定 ε（如 0.05/0.05，便于可复现实验）；源码默认 1/0.05，即退火。

**所有 DQN/运行宏都带 `#ifndef`**。漏写会怎样：`-D` 被源码里的 `#define` 覆盖回去，
编译器只丢一句 redefined 警告，扫出来的一整张超参数表全是同一个配置的复制品。
`sweep_hyperparams.py` 因此把 redefined 当致命错误。

## 3. 算法

### 3.1 动作 = afterstate

不对"动作编号"建模，而是先落子、再评价落子后的棋盘：

```
Q(s, a) ≈ V(s') = b + Σ_i w_i · φ_i(s')
```

`s'` 是方块落到 `(x, rotation)` 并消完行之后、下一块还没出现时的局面。这样 40 个候选
共用一个打分函数，线性模型才有可解释性；追踪里的 `current_value` 就是这里的 `V(s')`。

### 3.2 一步更新（差分目标）

```c
current_value = score_features(value_weights, value_bias, action.feature);
next_value    = max over candidates of score_features(target_weights, target_bias, ·);  // 跳过 top_out
target        = reward + gamma * next_value;     // 终局时 next_value = 0
td_error      = target - current_value;

// 批次末：ρ 用平均 TD 做一步 Robbins-Monro
value_bias   += DQN_RHO_RATE * mean(td_error);
```

几种等价写法，代数上要看清 `b` 到底进了哪几处：

```
current = V + b            (score_features 直接返回 b + Σw·φ)
next    = V' + b_target     (目标网络那份 b)
td      = reward + γ·next − current
        = [reward + γV' − V] − (1−γ)·b − γ·(b − b_target)
```

- **奖励永远不扣 ρ**：`target = reward + γ(V' + b)`。把 ρ 从 `reward` 里减掉是另一套写法
  （`target = (reward − ρ) + γV'`），在 γ<1 下等价于给每一步加成本，**不是**这里的做法。
- 在 `b = b_target` 的平稳情况下，只剩 `−(1−γ)·b = −0.01·b` 这一项：`b` 是一个**价值水平**
  常数（相当于把 `V` 整体托到 `r̄/(1−γ)` 量级），不是平均奖励估计器。
- 所以 **`b` 不能重复加减**：`score_features` 已经把它算进去了，再在 `current`/`next` 上
  补一次就是双重计数（这正是曾出现过的 bug，见 §10 L2）。
- 终局（顶出、或下一方块没有任何合法落点）不 bootstrap，target 就是这一手的奖励。

### 3.3 Adam 更新

```c
gradient      = Σ_batch td_error · φ_i / batch_size;
m = β1·m + (1-β1)·gradient;     v = β2·v + (1-β2)·gradient²;
w_i += alpha · (m/c1) / (sqrt(v/c2) + eps);   // c1/c2 为偏差修正
```

- 不再除 `1+Σφ²`：Adam 自己按参数量纲缩放，步长与梯度大小无关。
- `DQN_LR_DECAY_STEPS > 0` 时每 N 步学习率减半，用来抑制长时间训练后的漂移。
- 所有更新量在写回前做 `isfinite` 校验，出现 NaN/Inf 直接报错退出。

### 3.4 目标网络与回放

- 目标网络每 `DQN_TARGET_UPDATE_INTERVAL` 次批量更新同步一次（6 权重 + 基线整份 `memcpy`）。
- 回放区攒够 `DQN_REPLAY_WARMUP` 条才开始更新；之后每个训练步抽 32 条。
- 一条记录存的是 **整个后继候选数组**（`next[40]`），更新时才用目标网络取 max——
  这样"目标网络同步后老转移的 bootstrap 也跟着变"。

### 3.5 探索

第 `episode` 局的 ε 从 `START` 线性插值到 `END`，分母是 `episodes-1`，
最后一局正好落在 `END` 上（多批次实验里可以首尾相接）。

### 3.6 奖励

主体是消行分乘 `REWARD_SCALE`，再减掉三个惩罚项（都是线性的、每步生效）：

```c
reward = score * REWARD_SCALE
       - (top_out ? PENALTY_TOP_OUT : 0)     /* 顶出：终局那一步 */
       - PENALTY_HOLE   * feature[4]         /* 洞数 */
       - PENALTY_HEIGHT * feature[0]         /* 落点高度 */;
```

| 一次消行数 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 棋盘分 | 100 | 300 | 500 | 800 |
| 训练奖励（scale=0.01） | 1 | 3 | 5 | 8 |

三个惩罚系数默认很小（20 个洞才抵一行），只是给"堆得高、挖得多"一点梯度信号，
不改变"活着不值钱"这个基调；系数设 0 就回到纯消行分。

⚠️ **奖励 ≠ 棋盘分**：奖励里有惩罚项，所以**测试分数打的是纯棋盘分**（`-i` 模式打印的
`分数` 是 `place_block` 返回的 0/100/300/500/800 累加），追踪 CSV 的 `score` 列同理。
拿 `reward/REWARD_SCALE` 反推分数会被惩罚项污染成负数。

## 4. 特征与偏置

| # | 名称 | 定义 | 直觉 |
| --- | --- | --- | --- |
| 0 | 落点高度 | `y + block_height - 1` | 落得越低越好 |
| 1 | 消行特征 | `消行数 × 方块落在满行里的格子数` | 一次消得越多越好 |
| 2 | 行变化 | 行内相邻格不同的对数 | 行越整齐越好 |
| 3 | 列变化 | 列内上下相邻格不同的对数 | 列越平整越好 |
| 4 | 洞数 | 有方块压在上面的空格 | 洞越少越好 |
| 5 | 井深 | 被两侧夹住的竖直空槽深度和 | 槽越浅越好 |

`b` 是第 7 个参数，追踪里对应 `bias_before` / `bias_after`。

⚠️ **这 6 个特征高度共线**（行变化-列变化 0.80、列变化-洞数 0.89、行变化-高度 0.72），
所以「哪个特征学得好」这种问法本身不可靠：它们在数学上可以互相抵消，见 §10 L5。

## 5. 代码结构

| 区域 | 行号 | 内容 |
| --- | --- | --- |
| 配置区 | 8–95 | §2 的全部宏 |
| 方块与棋盘表 | 96–190 | `block[7][4][4][2]`、`board[25][15]`、初始权重、Adam 状态、`block_r/w/h` |
| 棋盘操作 | 191–306 | `place_block` / `clear_lines` / `remove_block` / `can_place` / `simulate_fall` / `evaluate_board` |
| 候选结构 | 307–317 | `state_and_action`（feature + 落法 + value） |
| 打分与特征 | 319–369 | `score_features` / `extract_action_features` / `get_block` |
| 动作与环境 | 370–444 | `generate_action` / `environment_step`（枚举只在这里做） |
| 模型文件 | 445–505 | `load_value_model` / `save_value_model` |
| 回放区 | 507–525 | `replay_transition`、`replay_buffer`、`random_below` |
| 学习 | 527–603 | `update_dqn_batch`（Adam + ρ）、`replay_reserve` |
| 追踪 | 606–758 | `trace_*` 一组（整段可删） |
| 训练 | 775–911 | `train_on_transition` / `train_value_episode` / `train_value_model`（方块流在这里 open 一次） |
| 对局 | 913–961 | `run_game` |
| 入口 | 962–1068 | `main` |

## 6. 训练一局的完整时序

```c
memset(board, 0, sizeof(board));
count = environment_step(NULL, input, candidates, &reward);   // 只读第一个方块，枚举候选
if (count <= 0) return;                                       // 输入为空

for (turn = 1; ; turn++) {
    action = generate_action(candidates, count, epsilon,      // 现算 Q + ε 采样
                             value_weights, value_bias);
    transition = replay_reserve();                            // 先占槽位
    transition->action = action;
    count = environment_step(&action, input, transition->next, &reward);  // 落子 + 读下一块 + 枚举
    if (action.top_out) count = 0;                            // 这一步顶出 = 终局
    transition->next_count = count;
    candidates = transition->next;                            // 下一轮直接在槽里挑

    trace_begin_step(...); train_on_transition(); trace_end_step();
    if (count == 0) break;
}
```

四个设计要点：

1. **枚举只发生在 `environment_step` 里**——`can_place` → `simulate_fall` →
   `extract_action_features` → 填 `r/x/y/top_out`。`generate_action` 只做"算 value + 采样"。
2. **候选直接落进回放槽**：`environment_step` 往 `transition->next` 里写，全程没有
   3.2 KB 的整块拷贝。
3. **每轮重新算 value**：`action.value` 必须用当前权重现算，否则会比权重滞后一步。
4. **方块流是跨局的**：`train_value_model` 只 `fopen` 一次，句柄跨局持有——
   上一局停在哪、下一局就从哪接着读，读穿整份文件才 `fseek(fp,0,SEEK_SET)` 回绕。
   `get_block` / `environment_step` 的接口没变，仍然只收一个 `FILE *`。

## 7. 追踪 CSV

`TRAINING_TRACE_PATH` 非空时每个训练步写一行，共 **49 列**：

| 列 | 含义 |
| --- | --- |
| `episode, turn` | 第几局、第几步 |
| `shape, rotation, x, y` | 这一步的方块与落点 |
| `epsilon` | 这一步用的 ε |
| `score` | 这一手消行得到的棋盘分（0/100/300/500/800，不含惩罚项） |
| `reward` | 训练奖励（`score × REWARD_SCALE` − 惩罚项） |
| `terminal` | 是否终局 |
| `current_value` | `V(s') = b + Σw·φ`（更新前） |
| `next_value` | `max Q_target(s'', ·)` |
| `target` | `reward + γ·next_value` |
| `td_error` | `target − current_value` |
| `feature_norm` | `1 + Σφ²`（Adam 下不再参与更新，只作参考） |
| `board_cells` | 落子前棋盘上的方块数 |
| `feature0..5` | 6 个特征（消行**之后**的局面） |
| `weight_before/contribution/weight_delta/weight_after` ×6 | 每个权重在本次更新前后的值与贡献 |
| `bias_before, bias_after` | 基线 `b` 在本次更新前后的值 |
| `rel_td_error` | 相对 TD 误差 `|TD| / max(|V|, 1e-3)`；判断收不收敛看这一列，别只看 `td_error` |

`contribution_i = weight_before_i × φ_i`，可以直接看出哪个特征在主导价值。

> **看收敛看 `rel_td_error`，不要看 `td_error` 的绝对值。** `|TD|` 会随价值尺度 `V`
> 等比放大（`V` 涨 100 倍，`|TD|` 也涨 100 倍），曲线上像爆炸，实际相对误差可能在降。
> 分母只在下限处钳位（`max(|V|, 1e-3)`），不是每步加常数——加法地板会把早期的小 `V`
> 稀释成 1，比值被压小、趋势会反过来。
> `analyze_training.py` 画图时按局聚合用 `Σ|TD| / Σ|V|`（比值的和，不是每步比值的均值），
> 跨度超过一个数量级才切对数轴。
> `b` 是纯平移量，在 argmax 里被减掉、不影响策略，只作"价值尺度/是否漂移"的参考。

> ⚠️ **追踪与更新必须同口径**。曾在 `update_dqn_batch` 里漏写 `−ρ` 而 `trace_begin_step`
> 里有，结果 ρ 变成只进不出的积分器，一路涨到 12 万——见 §10 L2。

## 8. 工具与脚本

| 文件 | 作用 |
| --- | --- |
| `analyze_training.py` | 读追踪 CSV：训练得分/存活步数、ε、6 权重、特征均值、贡献、相对 TD 误差、偏置，4×2 面板 |
| `correlate_features.py` | 专门看"消行 vs 普通步"：特征比、td×φ 推力、梯度质量占比、Δw2 归因、td 随分位变化 |
| `plot_value_trajectory.py` | 局内 V 的轨迹（驼峰诊断）：V 沿进度、变化幅度、几局的 V(turn) |
| `run_experiments.ps1` | 多批次实验：编译 → 训练 → 存模型快照 → 跑测试 → 记分 → 下一批 |
| `sweep_hyperparams.py` | 超参数扫描：一组参数一个配置，各自从全零权重训练 → 跑测试集 → 出分析图，最后汇一张"参数 vs 测试分"总图 |

```powershell
# 单次训练出图
python .\analyze_training.py .\training_trace.csv --output .\training_analysis.png

# 消行相关性诊断
python .\correlate_features.py .\training_trace.csv --output .\correlate.png

# 局内价值轨迹
python .\plot_value_trajectory.py .\training_trace.csv --output .\value_trajectory.png

# 多批次实验（20 × 200 局）
.\run_experiments.ps1 -TrainSet .\1.txt -TestSet .\2.txt -EpisodesPerBatch 200 -BatchCount 20
python .\analyze_training.py .\runs\trace_batch*.csv --test-log .\runs\test_scores.csv

# 超参数扫描（无参数即跑默认网格；产物在 .\sweep）
python .\sweep_hyperparams.py --episodes 2000 --seeds 11,22,33
```

`sweep_hyperparams.py` 的产物（默认 `.\sweep`）：每个配置一个目录
（`参数=取值_s种子`，里面是 `trace.csv` / `training.png` / `q_weights.dat` / `result.json`），
外加 `results.csv`（逐配置明细）、`summary.json`（按参数取值聚合的中位数）和
`sweep_summary.png`（每个参数一格，纵轴是测试集纯棋盘分，蓝点=中位数、灰点=单次、
误差条=极差）。
同一个配置换三个种子是为了压随机噪声——`TRAIN_SEED=0` 时种子取当前时间，
单次训练之间本来就不可比（见 §10 L10）。
`--rebuild` 不训练、只用已有 `result.json` 重算表和总图（改画图样式后很有用）；
`--prune-trace` 出完图删掉 `trace.csv`（每份三十多 MB，一轮扫描 3 GB 以上）。
想临时换网格不用改代码：`--values DQN_LEARNING_RATE=3e-5,5e-5,1e-4` 替换某个参数的枚举值，
`--fix DQN_LEARNING_RATE=1e-4 --only DQN_BATCH_SIZE` 把别的参数钉死再扫这一个
（钉死的配置单独成组、不混进自由扫描的中位数，也不进汇总图）。

`run_experiments.ps1` 的产物（默认 `.\runs`）：`trace_batchNNN.csv`（每批追踪）、
`model_batchNNN.dat`（测试前快照）、`test_scores.csv`
（`batch,episodes,cum_episodes,epsilon_start,epsilon_end,test_score,test_pieces,test_ms`）。
重跑时旧的 CSV 与快照会先归档到 `archive_<时间戳>\`。

## 9. 验证体系

| 工具 | 作用 |
| --- | --- |
| `tests/test_training_update.c` | 3 个用例：终局只用奖励、bootstrap 生效、Adam 步长与 TD 量级无关 |
| `tests/test_extract_action_features.c` | 10 个特征提取用例 |
| `tests/test_game.c` | 30 个对局/消行用例 |

```powershell
gcc -std=c99 -Wall -Wextra -O2 tests\test_training_update.c -o tests\test_training_update.exe -lm
gcc -std=c99 -Wall -Wextra -O2 tests\test_extract_action_features.c -o tests\test_extract_action_features.exe -lm
gcc -std=c99 -Wall -Wextra -O2 tests\test_game.c -o tests\test_game.exe -lm
.\tests\test_training_update.exe; .\tests\test_extract_action_features.exe; .\tests\test_game.exe
```

**缺口**：模型文件还没有"保存 → 读取"的往返测试，而这正是 §10 L1 那个 bug 逃过所有
测试的原因。建议补一个：写一个含长小数的模型 → 读回 → 逐位比较。

## 10. 经验与坑

这一节是踩过的坑，按"症状 → 根因 → 处置"记，重读时先看这里。

### L1 存得下、读不回来：`%15s` 截断

**症状**：训练日志均分 8 万，`-i 1.txt` 却 13 步 / 0 分。

**根因**：`load_value_model` 用 `char tag[16]; fscanf(fp,"%15s",tag)` 读第一个数，
而 `save_value_model` 用 `%.12g` 写出的 `-0.0152520950172` 有 **16 个字符**。
截掉的尾巴 `2` 被下一个 `%lf` 当成 `w1`，整套权重错位一格，`b` 被丢：

```
读成   w=[-0.0153, 2.0, 0.4555, -0.3375, 0.1306, -1.9203]  b=-0.1768
应为   w=[-0.0153, 0.4555, -0.3375, 0.1306, -1.9203, -0.1768]  b=56.4494
```

**处置**：`char tag[64]` + `%63s`。修前 13 步 / 0 分，修后 **16,867 步 / 765,600 分**。

**教训**：任何持久化格式都要有"写→读"往返测试；`%.12g` 的最长输出约 24 字符。

### L2 追踪与更新口径不一致 → ρ 变成积分器

**症状**：`b` 从 0 一路涨到 12 万，`|TD|` 同步涨到 12 万。

**根因**：更新路径和追踪路径对 `b` 的处理不一致——`score_features` 已经返回 `w·φ + b`，
一条路径又在外面补了一次 `±b`。于是 TD 里 b 的系数变了，b 却仍按 `b += rate·mean(td)` 走：
推高 b 并不能让 TD 归零，它就变成只进不出的积分器。

**处置**：两条路径写成一个口径（b 只由 `score_features` 负责）。**教训**：诊断量（trace）
和被诊断的量必须用同一段代码算，否则你测的是自己的显示器。

### L3 Adam 对奖励的整体缩放不变

把 `REWARD_SCALE` 从 0.01 改成 1，结果**一字不差**：奖励 ×100 ⇒ td ×100 ⇒ m ×100、
v ×10⁴ ⇒ `m̂/√v̂` 不变。所以"改奖励尺度"这个实验什么都证明不了。
想真正改变行为，得改**奖励的相对结构**（比如加生存项）或改优化器。

### L4 价值的"水平"几乎不可辨识

给所有 V 平移 c，TD 只变 `(γ−1)c = −0.01c`：平移 2,000 才让 TD 动 20。所以
"V 学到了几千""V 是负的"这类说法本身没有信息量——要看**局内的差值**和**相对排序**。

### L5 特征共线 → 近零空间 → 灾难性抵消

6 个特征两两相关 0.7–0.9，存在"改权重几乎不改 V"的方向。Adam 的 `m/√v` 会把这些方向上
的微小梯度放大成满步长，权重于是沿着零空间漂移、大数值互相抵消（`w2=+34` 的行变化项
+2,099 被 `w3=-47` 的洞数项 −1,653 抵掉）。**结论：不要解读单个权重的符号和大小**，
要看 `contribution` 或者 `V` 本身。

### L6 「从奖励里扣 ρ」和「把 b 加进两侧」不是一回事

两种写法一字之差，行为完全不同：

```
写法 A（错）  target = (reward − ρ) + γ·V'        ← 等于给每一步加"生命税"
写法 B（对）  target = reward + γ·(V' + b) − (V + b)
```

- **写法 A** 在 γ<1 下要求"每一步都值 ρ"，权重被压到 0、策略趋向短局（实测权重→0、贪心
  12 步）。真正中性的平均奖励框架要求 γ=1。
- **写法 B** 里 ρ 只以 `−(1−γ)·b` 的形式出现一次，相当于给 `V` 加了一个水平锚，
  **奖励一分不扣**。当前代码用的是 B（`score_features` 已经含 b，`target = reward + γ·next`）。
  平稳时平均 TD 归零 ⇒ **`(1−γ)·b = mean(reward + γV′ − V)`**（`V` 不含 b）。
  实测固定 ε 训练到 2,000 局时：mean r = +0.410、mean(γV′ − V) = −0.417 ⇒ 右边 +0.59，
  于是 `b* = 59`，与观测到的 b ≈ 59.8 吻合。
  注意它**不是** `r̄/(1−γ)`：因为 `γV′ − V` 通常是负的（一局越走越差），b 停在哪里取决于
  V 的漂移量而不只是奖励大小。

这条坑的价值在于：**"差分目标"这个词同时指两套数学**，改代码前必须先在纸上把
`td = r + γ(V'+b) − (V+b)` 展开，看 ρ 到底落在移动的哪一侧。

### L7 偏置对策略没有影响

`b` 对同一局面的 40 个候选是同一个常数，加多少都不改 argmax——三种归一化方式下
贪心对局逐位相同（393,100 分 / 8,475 步）就是证据。它只影响价值的**水平**。

### L8 内存带宽是真正的瓶颈

一条回放记录含 40 个候选 ≈ 3.2 KB，×50,000 = **160 MB**；每次更新要冷读 32 条 ≈ 100 KB。
把"候选只算一次"（`extract_action_features` 调用减少 47%）并不会让训练变快，因为时间
花在搬内存上；把容量降到 5,000 反而快 22%。

### L9 训练写当前目录

`--train` 会把模型写到**当前工作目录**的 `q_weights.dat`。在仓库根目录跑实验会直接覆盖
你正在用的模型。要么 `cd` 到临时目录，要么改 `VALUE_MODEL_PATH`。

### L10 训练不可复现，测试可复现

`train_value_model` 里 `srand(time(NULL))`：同配置重跑轨迹不同。贪心 + 固定序列的
`-i` 才是确定的（同一模型、同一输入 ⇒ 同一局）。所以模型之间只能比"多次训练的中位数"，
不能比单次。

### L11 单局贪心得分对权重极敏感

一次 5 局的训练就能让测试分从 100 万跳到 500 万（argmax 是分段常数，跨过一个边界就完全
换一条轨迹）。要评价模型，用多条序列取中位，别用单局。

### L12 模型 0 分时的排查顺序

1. 先跑 `tests/`（3 个都过说明代码没坏）；
2. 用 `-i` 跑一局看步数——1x 步就死，先怀疑**模型加载**（见 L1），而不是策略；
3. 打印加载后的权重与 `b`，和文件里逐位对比；
4. 再怀疑价值函数本身（`correlate_features.py` 看 td 的符号）。

### L13 想让 TD 残差降下去，先把 ε 固定住

退火（0.35→0.05）会让**行为策略逐局变化**：回放区 5 万条 ≈ 200 局，这 200 局里混着十几个
不同 ε 的策略产生的数据，TD 的目标一直在动，残差自然降不下来——多出来的那份误差最终被
`b` 吸收，看起来就是"b 一路漂"。

固定 ε=0.05 之后（2,000 局、510,771 步）：

| 指标 | 末 500 局 | 末 200 局 | 末 100 局 |
| --- | --- | --- | --- |
| `b` | 59.96 | 59.83 | 59.57 |
| 平均 \|TD\| | 1.294 | 1.317 | 1.330 |
| 每局步数 | 252.7 | 245.1 | 227.5 |

TD 的**带符号均值**是 −0.008（≈0）、40.5% 的步 TD>0——这是真正的不动点，不是"涨到某处被压住"。

但**收敛 ≠ 学对**：末 200 局 49,019 个状态上，V 均值 **45.48**、手算折扣回报 G 均值 **28.68**
（差 +16.8），`corr(V,G) = 0.397`。价值仍然偏乐观，只是不再发散。

**结论**：任何 A/B 对照都应在固定 ε 下做；退火适合"对外"出成绩，不适合拿来做归因。

## 11. 已知问题

- **max 自举的高估**：target 用 `max_a' Q_target`，会系统性偏高；实测收敛后 V 比真实折扣
  回报高约 17（V 45.5 vs G 28.7），`corr(V,G)` 只有 0.40。
- **`b` 的停点由数据分布决定**：它停在"平均 TD 归零"处，即 `(1−γ)b = mean(r + γV′ − V)`；
  退火把分布搅动时它就会一路漂（见 L13）。
- **没有生存压力**：奖励只剩消行分，长局里死亡不产生额外代价。
- **特征表达能力有限**：看不到"行内空洞分布""逐列高度差"，井深/洞数只能线性加权。
- **读穿文件会截断一局**：方块流跨局顺序读（§6 第 4 点），文件读完时若正处在某一局中间，
  这一局会提前结束——`get_block` 读到 EOF 返回 `-666`，上层按"候选为 0"收尾。
  每读穿一遍最多损失一局，之后才回绕到文件头继续。
- **README 的行号会过期**：改动代码后请以源码为准，或顺手更新附录 A。

## 附录 A 源码索引

| 行号 | 内容 |
| --- | --- |
| 8–95 | 参数配置区（§2） |
| 96–148 | `block[7][4][4][2]` 方块形状表 |
| 149 | `board[25][15]` |
| 150–153 | `FEATURE_COUNT`、初始权重 `value_weights`、基线 `value_bias` |
| 156–157 | 目标网络的 `target_weights` / `target_bias` |
| 159–170 | Adam 状态与 `reset_adam_state` |
| 172–190 | `block_r` / `block_w` / `block_h`（每个方块的旋转数与尺寸） |
| 191 | `can_place` 前置声明 |
| 193 | `reset_value_model`（回初始权重 + 清 Adam） |
| 201–306 | `place_block`(201) / `clear_lines`(236) / `remove_block`(257) / `can_place`(264) / `simulate_fall`(275) / `evaluate_board`(284) |
| 307–317 | `state_and_action` |
| 319 | `score_features` |
| 325 | `extract_action_features` |
| 352 | `get_block` |
| 370 | `MAX_ACTIONS` |
| 377 | `generate_action` |
| 409 | `environment_step`（唯一的枚举点） |
| 445 | `load_value_model` |
| 490 | `save_value_model` |
| 507–525 | `replay_transition` / `replay_buffer` / `random_below` |
| 527 | `update_dqn_batch`（Adam + ρ） |
| 598 | `replay_reserve` |
| 606–758 | 追踪模块（`trace_attach` / header / `begin_step` / `end_step`） |
| 775 | `train_on_transition` |
| 792 | `train_value_episode` |
| 840 | `train_value_model`（唯一打开训练输入的地方） |
| 913 | `run_game` |
| 962–1068 | `main` |

## 附录 B 从 Bellman 方程到 DQN

这是我（作者）对强化学习的一条理解主线：**价值函数的三种算法 → 同/异策略的 TD → Q-learning
与 DQN**，正好解释了 §3 里为什么要目标网络和经验回放。

### B.1 状态价值函数的三种算法

一切从 Bellman 期望方程出发：

```
v_π(s) = Σ_a π(a|s) · Σ_{s',r} p(s',r|s,a) · [ r + γ·v_π(s') ]
       = E_π[ r + γ·v_π(S') | S_t = s ]
```

三种算法的区别，只在于"这个期望怎么算出来"。

**1. DP（动态规划）**——**知道模型，直接求期望**。

在知道环境模型 `p(s',r|s,a)` 时，可以枚举下一时刻所有可能的 `s'`，再用转移概率加权求和算
出期望，一次算出当前的 `v`。代价是：必须完整知道模型，而且每次迭代都要扫所有状态。

**2. MC（蒙特卡洛）**——**用频率替代转移概率，放弃模型**。

按 Bellman 方程，`v` 是回报 `G` 的期望，那就直接从环境采样出 `G` 来求平均即可。本质是
**用采样频率替代转移概率**，于是不再需要模型。可以做增量更新：

```
V_n = (G_1 + G_2 + ... + G_n) / n
    = ((n-1)/n)·V_{n-1} + (1/n)·G_n
    = V_{n-1} + (1/n)·(G_n − V_{n-1})
```

此处下标 `n` 指"第 n 次更新后的 V"：`n` 越小，`V` 越老、越不可靠。因为 `G` 是整局的回报，
MC 必须等一局结束才能更新。

**3. TD（时序差分）**——**DP 与 MC 的结合**。

把 Bellman 方程展开：当前的 `V` 等于"奖励 + 下一状态 `V` 对环境的期望"。由于模型未知，
就用下一状态在环境里**采样到的真实值** —— 它本身就是下一个状态价值的一次采样 —— 去代替那个
期望，然后反复在环境里迭代：每到达这个状态一次，就对下一个状态采样一次，再求平均即可。

增量更新时用固定 `α` 而不用 `1/n`，是为了**一边估计一边优化**：

- 如果策略始终不变，`1/n` 是最优的（样本越攒越多，估计越稳）；
- 但一旦策略不停变化，旧样本就失去效力；`1/n` 会因为 `n` 太大让更新几乎停滞，而新样本
  恰恰是更重要的，所以固定 `α` 更好。

一句话对照：

| | 要不要模型 | 一次更新用多少数据 | 偏差/方差 | 何时能更新 |
| --- | --- | --- | --- | --- |
| DP | 要（显式 `p`） | 全部下一状态 | 无偏、无采样方差 | 每轮扫描 |
| MC | 不要 | 一整局回报 `G` | 无偏、方差大 | 一局结束 |
| TD | 不要 | 一步 `r + γV(S')` | 有偏（自举）、方差小 | 每一步 |

### B.2 策略评估：同策略与异策略的 TD

`TD` 估计的是哪个策略的价值，决定了算法的身份：

- **同策略 Sarsa**：用 TD 估计**行动策略**本身的价值，采样和估计是同一个 `π`。
- **异策略 Sarsa**：用 TD 估计**目标策略** `π` 的价值，但采回来的数据服从**行为策略** `b`。
  采样的是行为策略的分布，要估计的却是目标策略的期望，两者得靠重要性采样接上：

```
E_π[X] = Σ_a π(a|s)·X
       = Σ_a b(a|s)·(π(a|s)/b(a|s))·X
       = E_b[ ρ·X ],        ρ = π/b
```

`ρ` 就是重要性采样比（实现在按步连乘后通常还要截断，否则方差会爆）。

注意二者都是传统 TD，**`next_state` 都来自环境的真实采样**，区别只在"估计谁"。

### B.3 从 TD 到 Q-learning 再到 DQN

同样的 TD 也可以用来估计动作价值。把 `Q` 写成对下一状态 `s'` 的期望：

```
Q(s,a) = Σ_{s'} p(s'|s,a) · [ r + γ·V(s') ],     V(s') = max_a' Q(s',a')
```

- **外层的 `p` 用采样代替**（TD 的思路），再做增量更新；
- **内层的 `V` 用 `max_a' Q` 代替**（直接对目标策略取最优）。

这就是 **Q-learning**。因为是 `max` 直接在目标策略上做，行为策略只负责采数据，所以它天然
是异策略的，不需要 `ρ`。

**DQN** 就是在这个基础上，把 `Q` 函数从表格换成神经网络。区别在于稳定性：

- 表格法一次只更新一个格子，其它格子的目标值都不动，**目标相对稳定**；
- 网络参数一更新，**所有状态的 Q 同时改变**，自举的目标跟着一起动，学习因此非常容易发散。

由此引出两个设计：

- **在线网络 + 目标网络**：用一份滞后同步的参数算自举目标，先给学习一个相对固定的靶子；
- **经验池**：样本来自更早的策略，随机抽取可以避免短期内一直学同样的误差，也打断了相邻
  样本之间的强相关。

### B.4 与本仓库实现的对应

| 笔记里的概念 | 代码里的落点 |
| --- | --- |
| TD 增量更新 `td = r + γV' − V` | `update_dqn_batch` 里的 `td_error`（§3.2） |
| 自举目标用"采样值"代替期望 | `target = reward + γ·next_value`，`next_value` 是下一局面枚举出来的 `max` |
| 异策略 | 回放区里是历史（不同 ε）策略产生的数据，更新朝当前贪心目标走 |
| 目标网络 | `target_weights` / `target_bias`，每 `DQN_TARGET_UPDATE_INTERVAL` 次同步（§3.4） |
| 经验池 | `DQN_REPLAY_CAPACITY = 50000`，每次抽 `DQN_BATCH_SIZE = 32` 条 |
| 固定 `α` 而非 `1/n` | `DQN_LEARNING_RATE` + Adam；配 `DQN_LR_DECAY_STEPS` 可退火 |
| `V` 的水平不可辨识 | `b` 只做水平锚，不改 argmax（§10 L4、L7） |
