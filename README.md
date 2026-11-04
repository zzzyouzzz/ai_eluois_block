# 俄罗斯方块 DQN（线性价值 + 常数偏置）源码导读

一个单文件 C 实现的俄罗斯方块智能体：用 6 个棋盘特征线性逼近动作价值，配一个常数偏置项，
通过经验回放 + 目标网络做 Q-learning。训练过程把每一步的中间量写进 CSV，另有 Python 脚本
复核与画图。

> 当前版本：`eluose.c` 共 957 行，线性模型（6 个权重 + 1 个偏置），无隐藏层、无评估模块，
> 命令行只剩 `--train` / `-i` 两个参数。

## 目录

- [1. 快速开始](#1-快速开始)
- [2. 参数配置区](#2-参数配置区)
- [3. 算法](#3-算法)
- [4. 特征与偏置](#4-特征与偏置)
- [5. 代码结构](#5-代码结构)
- [6. 训练一局的完整时序](#6-训练一局的完整时序)
- [7. 追踪 CSV](#7-追踪-csv)
- [8. 多批次实验脚本](#8-多批次实验脚本)
- [9. 验证体系](#9-验证体系)
- [10. 已知问题与设计取舍](#10-已知问题与设计取舍)
- [附录 A 源码索引](#附录-a-源码索引)

## 1. 快速开始

```bash
# 编译（Windows 下 gcc 需要 -lm）
gcc -std=c99 -Wall -Wextra -O2 eluose.c -o eluose.exe -lm

# 训练：默认 2000 局，训练完把模型写回 q_weights.dat
./eluose.exe --train -i 1.txt

# 只玩游戏、不训练：读 q_weights.dat，按贪心策略落子
./eluose.exe -i 2.txt
```

几个约定：

- **方块输入文件**：每行一个字母，`I O T J L S Z` 是七种方块，`X` 表示输入结束，`E` 表示提前终止。
  `1.txt` / `2.txt` / `3.txt` 都是 100 万个方块的长序列。
- **棋盘坐标系**：`x` 从左到右 0–9，`y` 从下到上 0–19（打印时从第 19 行倒着打）。
- **模型文件**：`q_weights.dat`，7 个数（6 个权重 + 偏置）。也能读旧的 6 数文件和早先的
  `mlp 16` 格式（只取 6 个主干权重，偏置按 0）。
- **训练结果只取决于配置区**：局数、是否从初始值开始、追踪文件名都写在源码开头，命令行不再接受。
  想临时改，用编译宏覆盖（见 §2）。
- **注意**：训练会把模型写进**当前工作目录**的 `q_weights.dat`。做实验请先切到临时目录，
  免得覆盖仓库里的模型。

## 2. 参数配置区

文件开头 `eluose.c:8–67`，分四组，改完重新编译生效。带 `#ifndef` 的宏可以用 `-D` 覆盖：

```bash
gcc -DTRAIN_EPISODES=200 -DTRAINING_TRACE_PATH='"batch.csv"' eluose.c -o eluose.exe -lm
```

| 宏 | 默认值 | 作用 |
| --- | --- | --- |
| `TRAIN_EPISODES` | 2000 | 训练局数 |
| `TRAIN_FRESH` | 0 | 1 = 忽略模型文件，从源码里的初始权重开始 |
| `VALUE_MODEL_PATH` | `"q_weights.dat"` | 模型读写路径 |
| `TRAINING_TRACE_PATH` | `"training_trace.csv"` | 追踪 CSV；空串表示不追踪 |
| `DQN_REPLAY_CAPACITY` | 50000 | 回放区容量 |
| `DQN_BATCH_SIZE` | 32 | 每次更新抽多少条 |
| `DQN_REPLAY_WARMUP` | 64 | 攒够多少条才开始更新 |
| `DQN_TARGET_UPDATE_INTERVAL` | 500 | 每多少次更新同步目标网络 |
| `DQN_HUBER_DELTA` | 100.0 | TD 超过它按 ±100 截断 |
| `DQN_GAMMA` | 0.99 | 折扣因子 |
| `DQN_LEARNING_RATE` | 0.0001 | 学习率 |
| `DQN_BIAS_NORM_MODE` | 1 | 偏置 b 的梯度归一化：0 = 与权重共用 `1+Σφ²`，1 = 用 `sqrt(1+Σφ²)`，2 = 不归一化 |
| `TRAIN_EPSILON_START` | 0.1 | 第 0 局的 epsilon |
| `TRAIN_EPSILON_END` | 0.1 | 最后一局的 epsilon，同时是下限 |

`START == END` 就是固定 epsilon（当前默认），做可复现实验用；写成 0.35 / 0.05 就是退火。

## 3. 算法

### 3.1 价值的定义

不做"动作编号"建模，而是先落子、再评价落子后的棋盘（afterstate）：

```
Q(s, a) ≈ V(s') = b + Σ_i w_i · φ_i(s')
```

`s'` 是方块落到 `(x, rotation)` 并消完行之后、下一块还没出现时的棋盘。`φ` 是 §4 的 6 个特征，
`b` 是常数偏置。这样 40 个候选动作可以共用同一个打分函数，线性模型也才有可解释性。

### 3.2 一步更新

```c
current_value = score_features(value_weights, value_bias,  transition->action.features);
next_value    = transition->terminal ? 0.0
              : score_features(target_weights, target_bias, transition->next_features);
target        = transition->reward + (transition->terminal ? 0.0 : gamma * next_value);
td_error      = target - current_value;
```

- `next_features` 存的是 `s''` 的特征（收集时用**目标网络**挑出 `a*`，再取那个局面的特征），
  更新时用**当时的目标权重**重新打分——所以目标网络同步后，老转移的 bootstrap 也跟着变。
- 终局（顶出）不 bootstrap，target 就是这一手的奖励。

梯度：

```c
norm           = 1.0 + Σ φ_j²;                  // 防止大特征把步长带飞
clipped_td     = clamp(td_error, -100, +100);   // Huber 只做裁剪，没有二次段
gradients[i]  += clipped_td * features[i] / norm;
gradient_bias += clipped_td / bias_norm;    // bias_norm 由 DQN_BIAS_NORM_MODE 决定
after[i]       = value_weights[i] + alpha * gradients[i] / batch_size;
```

### 3.3 探索与目标网络

- ε-greedy：第 `episode` 局的 epsilon 从 `START` 线性插值到 `END`（分母是 `episodes-1`，
  最后一局正好落在 `END` 上）。
- 目标网络每 `DQN_TARGET_UPDATE_INTERVAL` 次批量更新同步一次（6 个权重 + 偏置整份 `memcpy`）。
- 回放区攒够 `DQN_REPLAY_WARMUP` 条才开始更新，之后每个训练步抽 32 条做一次更新。

### 3.4 奖励

**只有消行分**，没有存活奖励、也没有顶出惩罚：

| 一次消行数 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 奖励 | 100 | 300 | 500 | 800 |

所以"活着"本身不值钱，价值完全来自能消多少行。

## 4. 特征与偏置

| # | 名称 | 定义 | 直觉 |
| --- | --- | --- | --- |
| 0 | 落点高度 | `y + block_height - 1` | 落得越低越好 |
| 1 | 消行特征 | `消行数 × 方块落在满行里的格子数` | 一次消得越多越好 |
| 2 | 行变化 | 行内相邻格不同的对数 | 行越整齐越好 |
| 3 | 列变化 | 列内上下相邻格不同的对数 | 列越平整越好 |
| 4 | 洞数 | 有方块压在上面的空格 | 洞越少越好 |
| 5 | 井深 | 被两侧夹住的竖直空槽深度和 | 槽越浅越好 |

`b`（偏置）是第 7 个参数，让整体偏移不必挤进 6 个特征里。追踪 CSV 里它对应
`bias_before` / `bias_after` 两列。

## 5. 代码结构

| 区域 | 行号 | 内容 |
| --- | --- | --- |
| 配置区 | 8–67 | §2 的 13 个宏 |
| 方块与棋盘表 | 69–159 | `block[7][4][4][2]`、`board[25][15]`、`block_r/w/h`、初始权重 |
| 棋盘操作 | 160–242 | `place_block` / `clear_lines` / `remove_block` / `can_place` / `simulate_fall` |
| 评估与特征 | 243–326 | `evaluate_board` / `score_features` / `extract_action_features` / `get_block` |
| 动作与环境 | 327–400 | `evaluate_action` / `generate_action` / `environment_step` / `find_best_position` |
| 模型文件 | 401–469 | `load_value_model` / `save_value_model` |
| 学习 | 470–552 | 回放区、`update_dqn_batch`、`add_replay_transition` |
| 追踪 | 553–684 | `trace_*` 一组函数（整段可删） |
| 训练 | 685–810 | `train_on_transition` / `train_value_episode` / `train_value_model` |
| 对局 | 811–850 | `run_game` |
| 入口 | 851–957 | `main` |

## 6. 训练一局的完整时序

```c
// 进循环前
shape = get_block(input);
evaluate_action(shape, value_weights, value_bias, actions);      // 40 个候选各打一次分

for (turn = 1; ; turn++) {
    action = generate_action(actions, count, epsilon);           // 按 ε 选一个
    reward = environment_step(shape, &action, &line_score);      // 落子 + 消行 + 给奖励
    next_shape = get_block(input);

    if (invalid(next_shape) || action.top_out) {                 // 终局
        transition = {action, reward, terminal = 1};
    } else {
        evaluate_action(next_shape, target_weights, target_bias, next_actions);  // 目标网络
        next_action = generate_action(next_actions, next_count, 0.0);            // 贪心
        transition = {action, next_action.features, reward, terminal = 0};
    }
    if (!train_on_transition(&transition)) return 0;              // 入回放 + 可能更新
    if (terminal) break;

    shape = next_shape;
    evaluate_action(shape, value_weights, value_bias, actions);  // 下一轮重新打分（权重已变）
}
```

两个细节值得注意：

1. **每轮都重新 `evaluate_action`**。动作数组里的 `value` 必须用当前权重算，否则会比权重滞后一步。
2. **bootstrap 用目标网络，挑下一手动作用在线网络**。目标网络算出来的 `next_action` 只用它的特征，
   不用它的 value 来指导下一回合。

## 7. 追踪 CSV

`TRAINING_TRACE_PATH` 非空时，每个训练步写一行，共 **48 列**：

| 列 | 含义 |
| --- | --- |
| `episode, turn` | 第几局、第几步 |
| `shape, rotation, x, y` | 这一步选的方块与落子位置 |
| `epsilon` | 这一步用的 epsilon |
| `score` | 这一手消除得到的分数（0/100/300/500/800） |
| `reward` | 实际奖励（当前版本等于 `score`） |
| `terminal` | 是否顶出 |
| `current_value` | `Q(s,a)` 的预测值 |
| `next_value` | `max Q_target(s',·)` |
| `target` | `reward + γ·next_value` |
| `td_error` | `target - current_value` |
| `feature_norm` | `1 + Σφ²` |
| `board_cells` | 这一步棋盘上的方块数 |
| `feature0..5` | 6 个特征（落子后、消行前的局面） |
| `weight_before/contribution/weight_delta/weight_after` ×6 | 每个权重在本次更新前后的值和贡献 |
| `bias_before, bias_after` | 偏置在本次更新前后的值 |

`contribution_i = weight_before_i × 归一化后的 φ_i`，可以直接看出哪个特征在主导价值。

## 8. 多批次实验脚本

`run_experiments.ps1`：编译 → 训练若干局 → 保存模型快照 → 跑一次测试 → 记分 → 重复。

```powershell
.\run_experiments.ps1 -TrainSet .\1.txt -TestSet .\2.txt -EpisodesPerBatch 200 -BatchCount 20
python .\analyze_training.py .\runs\trace_batch*.csv --test-log .\runs\test_scores.csv
```

产物都在 `-WorkDir`（默认 `.\runs`）下：

- `trace_batchNNN.csv`：每批的追踪
- `model_batchNNN.dat`：每批测试前的模型快照
- `test_scores.csv`：每批的测试得分（`batch,episodes,cum_episodes,epsilon_start,epsilon_end,test_score,test_pieces,test_ms`）
- 每次重新运行时，旧的 CSV 和快照会先被归档到 `archive_<时间戳>\`

## 9. 验证体系

| 工具 | 作用 |
| --- | --- |
| `tests/test_training_update.c` | 3 个用例：终局只用奖励、bootstrap 生效、Huber 梯度被截断 |
| `tests/test_extract_action_features.c` | 10 个特征提取用例 |
| `tests/test_game.c` | 30 个对局/消行用例 |
| `tests/dump_actions.c` | 打印某一轮 40 个候选动作的全部字段 |
| `verify_actions.py` | 用纯 Python 重写一遍棋盘逻辑，逐项比对 40 个候选的打分 |
| `verify_trace.py` | 只用 CSV 列互证：`td == target-current`、`target == reward+γ·next_value`、`feature_norm == 1+Σφ²`、相邻行权重链 |
| `verify_all.ps1` | 依次跑：编译 → 单元测试 → 三序列候选复算 → 追踪自洽 |

```powershell
.\verify_all.ps1 -Trace .\training_trace.csv
```

## 10. 已知问题与设计取舍

- **max 自举的高估**：target 用 `max_a' Q_target`，会系统性高估，训练久了价值可能整体偏移。
- **线性模型的天花板**：6 个特征看不到"行内空洞分布""逐列高度差"这类信息，井深和洞数只能线性加权。
- **顶出不再扣分**：奖励只剩消行分，理论上"死得快"与"死得慢"只差在已经拿到的消行分上，长局里死亡不产生额外代价。
- **norm 归一化**：除以 `1+Σφ²` 让不同局面的步长接近，但也让大棋子局面学得更慢。
- **偏置对策略没有影响**：`b` 对同一局面的 40 个候选是同一个常数，加多少都不改变 argmax。它的作用只是把 TD 的量级拉回可用范围；三种归一化方式下贪心对局的得分逐位相同（393,100 分 / 8,475 步）就是证据。
- **训练不可复现**：`srand(time(NULL))`，同一配置重跑结果不同；测试（贪心、固定序列）才是确定的。

## 附录 A 源码索引

| 行号 | 内容 |
| --- | --- |
| 8–67 | 参数配置区 |
| 69–121 | `block[7][4][4][2]` 方块形状表 |
| 122 | `board[25][15]` |
| 123–124 | `FEATURE_COUNT`、`value_weights[6]` 初始权重 |
| 129–130 | `target_weights` / `target_bias` |
| 132–159 | `block_r` / `block_w` / `block_h`、`can_place` 前置声明、`reset_value_model` |
| 160–194 | `place_block` |
| 195–215 | `clear_lines` |
| 216–222 | `remove_block` |
| 223–233 | `can_place` |
| 234–242 | `simulate_fall` |
| 243–271 | `evaluate_board` |
| 272–277 | `score_features` |
| 278–304 | `extract_action_features` |
| 305–326 | `get_block` |
| 327–354 | `evaluate_action` |
| 355–374 | `generate_action` |
| 375–385 | `environment_step` |
| 386–400 | `find_best_position` |
| 401–445 | `load_value_model` |
| 446–469 | `save_value_model` |
| 470–480 | 回放区全局状态、`random_below` |
| 481–535 | `update_dqn_batch` |
| 536–552 | `add_replay_transition` |
| 553–684 | 追踪模块（`trace_attach` / header / `begin_step` / `end_step`） |
| 685–702 | `train_on_transition` |
| 703–770 | `train_value_episode` |
| 771–810 | `train_value_model` |
| 811–850 | `run_game` |
| 851–957 | `main` |
