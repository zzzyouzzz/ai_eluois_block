# 俄罗斯方块线性 DQN 训练说明

本文对应源文件 [`eluose.c`](./eluose.c)：用 C 实现俄罗斯方块的落子与消行、6 维棋盘特征提取，以及一个线性函数近似器；训练采用一步 DQN + 经验回放。

> 先给结论：代码里的 `value_weights` 名字像"棋盘价值"，实际训练的对象是 `Q(s,a)`——奖励折在回归目标里，不是漏加。为什么这样是对的、为什么不建议"再加一个 R"，见 §6。

## 目录

- 快速开始
- 1. 背景：强化学习与本程序
  - 1.1 概念对照
  - 1.2 一步 DQN 与经验回放
  - 1.3 线性函数近似与 6 个特征
  - 1.4 探索与利用
- 2. 代码解析
  - 2.1 全局状态与初始权重
  - 2.2 棋盘操作
  - 2.3 棋盘评估 `evaluate_board`
  - 2.4 动作特征 `extract_action_features`
  - 2.5 打分与动作生成
  - 2.6 模型读写
  - 2.7 回放区与批量更新
  - 2.8 单步训练与单局训练
  - 2.9 训练主循环 `train_value_model`
  - 2.10 评估
  - 2.11 对局与命令行
- 3. 追踪 CSV 格式
- 4. 编译、运行与可视化
- 5. 常见问题与注意事项
- 6. 概念辨析：这个模型到底在学 V 还是 Q
- 7. 函数速查：输入、输出、功能
- 附录 A 源码索引

## 快速开始

```powershell
# 编译
gcc -std=c11 -O2 .\eluose.c -lm -o .\eluose.exe

# 从头训练 1000 轮，用 2.txt 在训练前后评估
.\eluose.exe --train 1000 --fresh -i .\1.txt --test .\2.txt

# 单独评估当前模型
.\eluose.exe --test .\2.txt
```

细节见 §4。

## 1. 背景：强化学习与本程序

### 1.1 概念对照

强化学习让智能体通过与环境交互来学习决策。俄罗斯方块中，可以作如下对应：

| 强化学习概念 | 本程序中的含义 |
| --- | --- |
| 智能体（Agent） | 选择旋转角度和横向位置的落子策略 |
| 环境（Environment） | 棋盘、方块下落规则、消行和游戏结束规则 |
| 状态（State） | 当前棋盘与当前待放置方块 |
| 动作（Action） | 当前方块的旋转角度、横坐标和最终落点 |
| 奖励（Reward） | 消行得分，以及顶出棋盘时的惩罚 |
| 策略（Policy） | 根据动作价值选择落点的方法 |
| 回合（Episode） | 从空棋盘开始，直到游戏结束或输入序列耗尽的一局训练 |

智能体执行动作后，棋盘发生变化，得到下一个状态。训练的目标不是只挑眼前看起来最好的落点，而是让当前选择也考虑后续可能得到的回报。

### 1.2 一步 DQN 与经验回放

`Q(s, a)` 表示在状态 `s` 下执行动作 `a` 后，未来累计回报的估计。未来回报通常按折扣率 `gamma` 衰减：

```text
G = r₀ + gamma * r₁ + gamma² * r₂ + ...
```

其中 `r` 是奖励，`gamma` 越大，算法越重视较远的未来。程序使用时序差分（Temporal-Difference，TD）目标更新当前估值：

```text
target = reward + gamma * max_a' Q(next_state, a')
delta  = target - Q(state, action)
```

`delta` 称为 TD 误差，表示当前估值与新目标之间的差距。若游戏已经结束，则没有后继价值，目标只包含即时奖励。

程序使用一步 DQN 目标，并把转移放入有限大小的经验回放区，随机抽取小批次训练。目标权重每隔一段更新步数才从在线权重复制一次，以降低目标快速漂移。批量更新使用 Huber TD 梯度和特征范数归一化；终局转移不 bootstrap。超参数在源码的 `DQN_*` 常量中设置（见 §2.7 与附录 A）。

### 1.3 线性函数近似与 6 个特征

棋盘状态组合很多，程序没有为每个状态单独保存一项 Q 值，而是提取 6 个特征，用可训练权重计算动作价值：

```text
Q(s, a) ≈ w₀φ₀ + w₁φ₁ + ... + w₅φ₅
```

代码中的 `features` 是特征向量 `φ`，`value_weights` 是权重 `w`。当前仍是 6 个输入、单个线性输出的动作价值模型；DQN 的经验回放和目标权重机制不改变 `q_weights.dat` 的六个参数格式。

特征描述的是"把当前方块放到某个位置后"的棋盘：

| 下标 | 特征 | 代码含义 |
| --- | --- | --- |
| `0` | 落点高度 | `y + block_height - 1` |
| `1` | 消行特征 | 完成行数乘以当前方块落在这些完成行中的格子数 |
| `2` | 行方向变化 `drow` | 清除满行后的棋盘上，同一行内相邻列的占用状态差异计数 |
| `3` | 列方向变化 `dcol` | 清除满行后的棋盘上，同一列内相邻行的占用状态差异计数 |
| `4` | 洞数 `holes` | 清除满行后的棋盘上，某格为空但其上方已出现方块时的计数 |
| `5` | 井深 `well` | 清除满行后的棋盘上，左右两侧被占据而本格为空时累计的井深 |

初始权重为 `{-45, 34, -32, -93, -79, -34}`，是原先启发式评估系数的起点。训练后权重会变化；每个特征对局面好坏的影响由训练结果决定。

### 1.4 探索与利用

训练时需要平衡两种行为：

- **利用（exploitation）**：选择当前估值最高的动作。
- **探索（exploration）**：随机选一个合法动作，尝试当前估值较低但可能有潜力的局面。

程序使用 epsilon-greedy 策略：以概率 `epsilon` 随机探索，否则选择估值最高的动作。训练过程逐渐降低 `epsilon`，让前期多尝试，后期更多利用已学到的权重。

## 2. 代码解析

下面按调用顺序解析训练依赖的代码。**正文只用函数名定位，具体行号统一见附录 A**——源码改动后只需更新那张表，不会出现正文行号大面积过期。

### 2.1 全局状态与初始权重

```c
int board[25][15]={0};
enum { FEATURE_COUNT = 6 };
double value_weights[FEATURE_COUNT] = {-45.0, 34.0, -32.0, -93.0, -79.0, -34.0};
```

- `board` 声明为 25×15，实际只用到 `[0..19][0..9]`；`y=0` 是棋盘底部，`y=19` 是顶部，`x=0..9` 从左到右。
- `FEATURE_COUNT` 固定特征数量为 6，特征抽取、价值计算、模型读写和参数更新都使用这个数量。
- `value_weights` 保存线性模型参数。程序首次运行、且当前目录没有 `q_weights.dat` 时使用这里的初始权重；训练中原地更新，结束后写入模型文件。
- `reset_value_model` 又用一份局部数组写了一遍同样的 6 个初值，`--fresh` 走的就是它——**改初值时两处要一起改**。

`block_r` / `block_w` / `block_h` 是按方块和旋转角预置的旋转数、宽度与高度。注意 `block_w` / `block_h` 记的是**跨度**（最大 `dx` / `dy`），不是格子数，所以"落点高度"特征用 `y + block_h - 1` 时会比真实最高行小 1；同一决策内这是常数偏移，不影响 `argmax`。

### 2.2 棋盘操作

```c
int can_place(int shape,int rotation,int x,int y);
int place_block(int shape,int rotation,int x,int y,int fill[4],
                int full_rows[4],int *fill_line);
int simulate_fall(int shape,int rotation,int x);
void clear_lines(const int full_rows[4],int fill_line);
```

- `can_place`：越界（`x` 出 `[0,10)`、`y` 出 `[0,20)`）或与已有方块重叠时返回 `0`。
- `simulate_fall`：从 `y=16` 起逐格下探，返回最后能放的位置加一。**若 `y=16` 就放不下，它返回 `17` 这个非法值**，调用方必须自己检查——`generate_action` 正是先单独调一次 `can_place(...,16)` 把这种情况挡掉的。
- `place_block`：把方块写入棋盘，统计受影响各行的占用格数 `fill[]`、满行列表 `full_rows[]` 和满行数 `*fill_line`，**只算分不消行**。返回 `100 / 300 / 500 / 800`（对应消 1/2/3/4 行）、`0`（没消行），放不下返回 `-1`。
- `clear_lines`：双指针把非满行依次下移压实，高位补 `0`。
- `remove_block`：把方块占的格子清零，不校正悬空——只在"确实只是临时放了一下"的场合可用；特征提取那条路径中途会消行，所以它改用了整盘快照。

### 2.3 棋盘评估 `evaluate_board`

`evaluate_board(&drow,&dcol,&holes,&well)` 扫描当前棋盘，通过指针**累加**四个结构指标（调用前必须清零）：

- `drow`：同一行内相邻两格的占用状态不同则加一（横向起伏）。
- `dcol`：同一列内相邻两行的占用状态不同则加一（纵向起伏）。
- `holes`：在某列上，某格为空、且该列上方已经出现过方块，则记一个洞。
- `well`：某格为空、其左右两侧（含越界）都被占据时累计井深。

有一处边界处理不对称，读代码时容易忽略：`j==0` 时拿 `board[i][0]` 与自己比较，结果恒为 0，等价于**左边界不算墙**；`i==0` 同理不把底边算墙。而 `well` 的判定把越界位置当作已占据（**左右边界算墙**）。两种约定混用，靠边的空列会既被算作井、又不在 `drow` 里产生边界起伏。

### 2.4 动作特征 `extract_action_features`

```c
static int extract_action_features(int shape,int rotation,int x,int y,int block_height,
                                   double features[FEATURE_COUNT]){
```

函数输入方块形状、旋转角、横坐标、落点纵坐标和方块高度，输出落子后的特征。返回 `1` 表示成功，返回 `0` 表示不能放置。

```c
    for(int i=0;i<FEATURE_COUNT;i++)features[i]=0.0;
    int saved_board[25][15];
    memcpy(saved_board,board,sizeof(board));
```

先把输出特征清零，避免无效或未赋值的特征残留旧数据；再把整个棋盘（25×15）快照到 `saved_board`，因为函数中途会真的消行，必须整体回滚。

```c
    int fill[4],full_rows[4],fill_line;
    if(place_block(shape,rotation,x,y,fill,full_rows,&fill_line)<0)return 0;
```

- `fill` 接收方块涉及的各局部行的占用格数。
- `full_rows` 接收本次放置后完成的棋盘行号。
- `fill_line` 接收完成行数。
- `place_block` 临时把方块放到全局棋盘并计算消行结果。放置失败则直接返回 `0`。
- 这里**会真的消行**：先临时落子并记下"哪些行满了、方块在其中占几格"，再调用 `clear_lines`，最后用快照还原棋盘。所以特征 2–5 描述的是消行后的局面。
- `place_block` 的返回值是消行得分（0/100/300/500/800），在这里被丢弃——这也是特征里拿不到原始奖励、只能用 `fill_line × 格子数` 近似的原因（见 §6.6）。

```c
    int drow=0,dcol=0,holes=0,well=0;
    evaluate_board(&drow,&dcol,&holes,&well);
```

在临时落子并**消去满行之后**的棋盘上计算四项结构特征。局部变量先清零，因为 `evaluate_board` 是通过指针累加结果。

```c
    features[0]=y+block_height-1;
```

保存方块放置后的最高行坐标作为高度特征。

```c
    int block_cells_in_full_rows=0;
    for(int i=0;i<4;i++){
        if(fill[i]==10){
            for(int cell=0;cell<4;cell++)
                block_cells_in_full_rows+=
                    block[shape][rotation][cell][0]==i;
        }
    }
```

- `fill[i] == 10` 表示本次放置后局部第 `i` 行已满。
- 内层循环检查当前方块的 4 个格子各自位于哪一局部行。
- 每当方块的某个格子位于完成行，就把计数加一。
- 因此该特征不是简单的消行行数，而是"消行行数 × 当前方块在完成行中的格子数"的等价计数方式。
- 这个统计必须在 `clear_lines` 之前做，消行之后行号关系就乱了。

```c
    features[1]=fill_line*block_cells_in_full_rows;
    features[2]=drow;
    features[3]=dcol;
    features[4]=holes;
    features[5]=well;
```

将消行特征和四个棋盘结构指标放入向量中。特征下标与权重下标一一对应。

```c
    memcpy(board,saved_board,sizeof(board));  /* 用快照整体恢复棋盘 */
    return 1;
}
```

- 把快照整体写回。因为中途真的消过行，只调 `remove_block` 不足以还原棋盘；恢复后各候选落点之间互不污染。
- 返回 `1` 表示特征成功提取。

### 2.5 打分与动作生成

`training_action` 保存动作生成函数选出的动作及其评估结果：

```c
typedef struct{
    int valid,rotation,x,y,top_out;
    double features[FEATURE_COUNT];
    double value;
}training_action;
```

- `valid`：是否成功生成可放置的动作。
- `rotation`：旋转角编号。
- `x`、`y`：横坐标和落点纵坐标。
- `top_out`：该落点是否达到程序定义的顶出边界。
- `features`：该动作对应的棋盘特征。
- `value`：根据当前权重计算的动作价值，用于 epsilon-greedy 选择。

`score_features(value_weights,features)` 是 6 项点积 `wᵀφ`：`static` 表示只在本文件内使用，`score` 从 `0.0` 起、用 `double` 保留小数精度，权重由调用方传入所以函数不依赖全局模型状态。分数越高，实际价值估计越大，策略越倾向选择该落点。

```c
static training_action generate_action(int shape,double epsilon,
                                       const double value_weights[FEATURE_COUNT]){
    training_action actions[40];
    int count=0;
```

函数最多枚举 40 个候选动作：旋转数最多为 4，每个旋转枚举 10 个横向位置。权重和探索概率由调用方传入；无可用动作时返回 `valid` 为 0 的结果。

```c
    for(int rotation=0;rotation<block_r[shape];rotation++){
        for(int x=0;x<10;x++){
```

按照当前方块实际支持的旋转数，枚举旋转角与棋盘横坐标。

```c
            if(!can_place(shape,rotation,x,16))continue;
```

先检查方块能否从高处 `y=16` 放入棋盘。不能放入的动作跳过（这一步也顺带挡掉了 `simulate_fall` 的非法返回值 17）。

```c
            int y=simulate_fall(shape,rotation,x);
            if(y<0||y>=20)continue;
```

- `simulate_fall` 根据当前棋盘计算该横坐标下的最终落点。
- 落点不在棋盘合法纵坐标范围时跳过。

```c
            training_action *action=&actions[count++];
            action->rotation=rotation;
            action->x=x;
            action->y=y;
```

取出数组下一个位置并记录该候选动作的位置参数，同时递增候选数量。

```c
            action->top_out=y+block_h[shape][rotation]>14;
```

如果放置后高度超过 `14`，标记为顶出。贪心选择会避开这类动作；若探索选中或所有候选都会顶出，则执行后终止本局。

```c
            extract_action_features(shape,rotation,x,y,block_h[shape][rotation],
                                    action->features);
```

提取候选落点特征。这里候选已由合法性检查和模拟下落确定为可放置。

```c
            action->value=score_features(value_weights,action->features);
```

- `value` 始终是使用传入权重计算出的实际价值。
- 贪心选择优先比较非顶出动作；若所有候选都会顶出，则仍返回一个顶出动作供训练结束当前局。

```c
    if(count==0)return invalid;
    int selected=-1;
    if((double)rand()/((double)RAND_MAX+1.0)<epsilon){
        selected=rand()%count;
    }else{
        for(int i=0;i<count;i++)
            if(!actions[i].top_out&&
               (selected<0||actions[i].value>actions[selected].value))
                selected=i;
        if(selected<0)selected=0;
    }
    return actions[selected];
}
```

epsilon-greedy 选择在探索时从所有候选动作中随机选取，在利用时选择价值最高的安全动作。正常模式保留原有 `find_best_position` 接口，由它以 `epsilon=0` 和当前模型权重调用该动作生成函数。

### 2.6 模型读写

`load_value_model(path)`：

```c
    FILE *fp=fopen(path,"r");
    if(!fp){
        if(errno==ENOENT)return 0;
        fprintf(stderr,"无法读取价值模型 %s\n",path);
        return -1;
    }
```

- 文件不存在时返回 `0`，调用者可以继续使用代码里的初始权重。
- 其他打开错误会输出错误信息并返回 `-1`，不会静默地把损坏或不可读模型当成成功加载。

```c
    double loaded[FEATURE_COUNT];
    for(int i=0;i<FEATURE_COUNT;i++){
        if(fscanf(fp,"%lf",&loaded[i])!=1||!isfinite(loaded[i])){
            fprintf(stderr,"价值模型格式错误：%s\n",path);
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);
    for(int i=0;i<FEATURE_COUNT;i++)value_weights[i]=loaded[i];
    return 1;
```

- 临时读入 6 个权重；`fscanf` 失败或读到非有限数（无穷大、NaN）时报格式错误。
- 先读到临时数组，确认全部有效后才覆盖正在使用的权重。
- 返回 `1` 表示成功加载。

`save_value_model(path)`：以写入模式创建或覆盖文件，每行写一个权重（`%.12g`，足够保留训练得到的小数），最后检查 `fclose` 的返回值。文件打不开或关闭失败都输出错误并返回 `0`。

模型文件默认名为 `q_weights.dat`，路径相对于程序运行时的当前工作目录。

### 2.7 回放区与批量更新

超参数集中在 `DQN_*` 与 `EVALUATION_*` 宏里：

| 宏 | 值 | 含义 |
| --- | --- | --- |
| `DQN_REPLAY_CAPACITY` | 50000 | 回放区容量 |
| `DQN_BATCH_SIZE` | 32 | 每次批量更新的样本数 |
| `DQN_REPLAY_WARMUP` | 64 | 攒够多少条才开始更新 |
| `DQN_TARGET_UPDATE_INTERVAL` | 500 | 每多少次批量更新同步一次目标权重 |
| `DQN_HUBER_DELTA` | 100.0 | TD 梯度截断阈值 |
| `DQN_EARLY_END_PENALTY` | 300.0 | 提前结束的惩罚幅度 |
| `DQN_EARLY_END_DECAY` | 60.0 | 惩罚随步数衰减的时间常数 |
| `DQN_GAMMA` | 0.99 | 折扣率 |
| `DQN_LEARNING_RATE` | 0.0001 | 学习率 |
| `EVALUATION_EPISODES` | 100 | 训练前/后评估使用的片段数 |
| `EVALUATION_SAMPLE_COUNT` | 20 | 训练中途每个检查点采样的片段数 |
| `EVALUATION_MAX_STEPS` | 500 | 每个评估片段的最大步数 |

`replay_transition` 存一次环境转移需要的全部信息：

```c
typedef struct{
    training_action action;              // 状态-动作：含 features 与 value
    double next_features[FEATURE_COUNT]; // 贪心后继动作的特征
    double reward;
    int terminal;
}replay_transition;
```

`replay_buffer[50000]` 是环形缓冲（`replay_next` 回绕），`target_weights` 是目标网络参数，`dqn_update_count` 累计批量更新次数。`random_below` 用两次 `rand()` 拼出一个比 `RAND_MAX` 宽得多的无符号数再取模，避开单个 `rand()` 范围过窄带来的重复模式。

`update_dqn_batch` 的更新式为：

```text
clipped_td = clamp(target - current_value, -100, 100)
norm       = 1 + Σᵢ φᵢ²
Δwᵢ       = alpha × Σ_batch (clipped_td × φᵢ / norm) / 32
```

- 除以 `norm` 是按特征范数归一化，避免 6 维特征同时很大时梯度爆炸。
- 所谓 Huber 在这里实际只体现为**对 TD 误差做 ±100 的裁剪**，并没有二次段。
- 梯度除以批量大小，`alpha` 是学习率。

### 2.8 单步训练与单局训练

`train_on_transition` 处理一条转移：

```text
current_value = transition->action->value           // 用在线权重算出的 Q
next_value    = score(target_weights, next_features) // 目标网络 + 贪心后继动作
target        = terminal ? reward : reward + 0.99 * next_value
```

终局时目标只含即时奖励，不 bootstrap。注意 `current_value` 直接取自回放里存下的 `action->value`——**这个字段同时被"选择动作"和"算 TD 误差"两处使用**，改它的语义会同时影响两边（§6.5 会用到这个事实）。

`train_value_episode` 跑一局：

- 每步按 epsilon-greedy 选动作（`generate_action(shape,epsilon,value_weights)`），落子后拿到 `score`。
- 奖励为 `score + 1.0`：`+1` 是每存活一步的少量正向奖励，鼓励活得更久。
- 顶出时再扣 `300 * exp(-(turn-1)/60)`，越早结束罚得越重。
- 计算后继特征：若终局则无；否则读下一块，用 `target_weights` 做贪心动作取 `next_features`。读不到下一块（返回 `<0` 或 `==7`）也按终局处理——**输入耗尽在这里等价于"游戏结束"**，这一点在 §6.6 有专门讨论。
- 把转移写入回放区，攒够 warmup 后每步抽 32 条更新，并每 500 次更新同步一次 `target_weights`。

### 2.9 训练主循环 `train_value_model`

训练开始时将当前 6 个权重复制到目标权重，并清空回放区。之后的流程就是循环调用 `train_value_episode`：

- 每步按 epsilon-greedy 选择动作，执行落子和消行后保存 `features`、奖励、终局标志及下一方块的贪心动作特征。
- 回放区达到 64 条经验后，每步随机抽取 32 条进行更新；每 500 次批量更新同步一次目标权重，学习率为 `0.0001`。
- epsilon 随局数线性下降：`epsilon = 0.35 - 0.30 * episode / episodes`，下限 `0.05`。

训练局数区间的平均得分与平均存活步数每 100 局输出一次。

### 2.10 评估

`--test` 会在训练前后把测试文件按字节切成 100 个不重叠片段（`fseek` 到各段起点），对每段跑贪心策略，每段最多 500 步，最后报告平均得分、存活步数和顶出率。切分不看方块边界，所以片段边界上可能落在一个不完整的方块记录上。

`run_evaluation_episode` 负责跑单个片段，`evaluate_value_model` 负责按片段循环并汇总。训练中途的周期评估用 20 个片段，训练前/后用 100 个，差别由 `EVALUATION_SAMPLE_COUNT` / `EVALUATION_EPISODES` 控制。

### 2.11 对局与命令行

`run_game` 是交互/正常模式：预读当前块与下一块，调用 `find_best_position` 取落点，每步输出

```text
rotation x
score
```

当 `y + block_h[current_block][rotation] > 14` 时判定失败并打印棋盘（空格用 `" O"[board[row][column]]`，从 `row=19` 向下打印）。

`main` 统一解析运行模式和输入源：

- `--train [局数]` 启用训练；局数可省略（默认 2000），范围为 1 到 1,000,000。
- `-i 方块输入文件` 在训练和正常运行模式中指定方块序列；正常模式省略时使用 `stdin`，`-i -` 也表示 `stdin`。
- `--test 测试文件` 训练时比较训练前后的贪心表现；不带 `--train` 时单独评估当前模型。建议测试集与 `-i` 训练集分离。
- `--eval-trace CSV文件` 将训练前、训练中周期评估及训练后的测试结果写入独立 CSV；需要同时指定 `--test`。
- `--eval-interval 局数` 设置周期测试频率，默认每 100 局；每次中途采样 20 个测试片段，训练前/后使用 100 个片段。
- `--fresh` 训练时忽略已有 `q_weights.dat`，从源码中的初始 6 个权重开始。
- `--trace CSV文件` 可选，仅用于训练模式，记录每步训练数据。
- 参数顺序不限，例如 `eluose.exe -i .\1.txt --train 1000` 与 `eluose.exe --train 1000 -i .\1.txt` 等价。
- 追踪文件不能与 `q_weights.dat` 同名，程序会拒绝。

多轮训练必须用 `-i` 指定可重复打开的文件；不指定文件或使用 `-i -` 时只允许训练一轮。每轮训练前棋盘清空，重新打开输入文件并从头读取。输入 EOF、`E` 或 `X` 会结束当前轮，之后下一轮仍从文件开头重新读取。训练会加载已有 `q_weights.dat`（不存在则使用源码默认权重），并保存更新后的权重。

## 3. 追踪 CSV 格式

`--trace` 为每个环境动作记录一行。前 16 列为：

```text
episode,turn,shape,rotation,x,y,epsilon,score,reward,terminal,current_value,next_value,target,td_error,feature_norm,board_cells
```

之后每个特征 `i` 都会追加 `feature_i,weight_before_i,contribution_i,weight_delta_i,weight_after_i` 五列：

| 列 | 含义 |
| --- | --- |
| `feature0` | 本次动作的特征值 `φ₀` |
| `weight_before0` | 更新前权重 `w₀` |
| `contribution0` | 该特征对当前估值的贡献 `w₀ × φ₀` |
| `weight_delta0` | 本次批量更新的权重变化 `Δw₀` |
| `weight_after0` | 更新后权重 |

特征 1 到 5 同样记录。因为训练采用经验回放，一行表示一次实际环境动作；权重变化列记录此动作加入回放后执行的批量更新，**不代表单独由该行特征产生的更新**。`td_error` 是动作写入回放时计算的误差，后续重放会再次更新同一经验，因此 CSV 并非完整的逐次采样批次日志。

`shape` 是方块编号：0=I、1=T、2=O、3=J、4=L、5=S、6=Z。

`--eval-trace` 写的是另一套表头（`write_evaluation_trace_header`），按检查点记录测试集汇总指标，与上面的逐动作格式不同。

## 4. 编译、运行与可视化

### 4.1 编译

在 Windows PowerShell 中，从仓库目录执行：

```powershell
gcc -std=c11 -O2 .\eluose.c -lm -o .\eluose.exe
```

`.vscode/tasks.json` 里另有一条 `gcc -g` 的单文件调试构建。

### 4.2 训练

训练一轮时可以从标准输入读取：

```powershell
Get-Content .\1.txt | .\eluose.exe --train 1
```

指定训练局数，例如从同一个文件重复训练 10000 轮：

```powershell
.\eluose.exe --train 10000 -i .\1.txt
```

用 `1.txt` 训练，并以独立的 `2.txt` 在训练前后评估 100 个不重叠片段：

```powershell
.\eluose.exe --train 1000 --fresh -i .\1.txt --test .\2.txt
```

保存训练期间的独立测试曲线（每 100 局评估一次）：

```powershell
.\eluose.exe --train 1000 --fresh -i .\1.txt --test .\2.txt `
  --trace .\training_trace.csv --eval-trace .\evaluation_trace.csv
```

周期评估间隔可通过 `--eval-interval 50` 等方式调整。

每轮训练都会重开 `-i` 指定的文件并从头读取，棋盘顶出或文件结束时本轮结束；训练轮数由 `--train` 指定。动作探索仍使用 epsilon-greedy 随机策略。多轮训练不能使用 `stdin`，因为输入流不能重新打开；单轮训练可省略 `-i` 或使用 `-i -`。使用 `--test` 时，测试策略始终为贪心，不进行探索。

### 4.3 评估

只评估当前模型：

```powershell
.\eluose.exe --test .\2.txt
```

`--eval-trace` 会记录训练前、每隔指定局数（每个检查点 20 个片段）以及训练后的测试结果，供脚本绘制独立测试曲线。

### 4.4 追踪与出图

记录每一步特征与权重变化（例如最多训练 1000 局）：

```powershell
.\eluose.exe --train 1000 --trace .\training_trace.csv -i .\1.txt
```

CSV 可用 Excel 导入，也可用 Python、R 等工具统计。重点比较 `featureN`、`weight_beforeN`、`contributionN`、`weight_deltaN` 和 `weight_afterN`，同时观察 `td_error` 与 `feature_norm`。例如，若某维 `weight_deltaN` 长期同号且绝对值持续增大，说明该维在持续推动权重漂移；若 `td_error` 很大而多维特征同时非零，需结合每维的 `weight_deltaN` 判断具体贡献。

仓库中的 [`analyze_training.py`](./analyze_training.py) 会绘制中文训练图（滚动平均得分/存活步数、epsilon、六个权重、归一化特征、加权特征贡献、消行率和 TD 误差）；提供测试追踪时还会生成测试集得分/存活、权重、特征及加权贡献图。高度按棋盘高度归一化，消行特征按 4 格归一化，行变化、列变化、洞数和井深按落子后棋盘方块总数归一化，减少不同堆叠规模造成的量级偏差。加权贡献为 `weight_before × 归一化特征`，可观察权重变化是否伴随该特征对应贡献或特征值下降。脚本要求使用当前 DQN 版本生成的追踪 CSV；追踪格式不匹配时会报告缺失字段。旧版追踪文件需重新训练后再分析。需要 Python 与 `matplotlib`：

```powershell
python -m pip install matplotlib
python .\analyze_training.py .\training_trace.csv `
  --eval-trace .\evaluation_trace.csv --output .\training_analysis.png
```

脚本会额外生成 `training_analysis_测试集.png`；可用 `--window 50` 调整滚动均值窗口，也可用 `--eval-output` 指定测试图文件名。

### 4.5 正常游戏模式

正常模式默认从标准输入读取方块序列，也可通过 `-i` 指定输入文件：

```powershell
Get-Content .\3.txt | .\eluose.exe
.\eluose.exe -i .\3.txt
```

模型文件每行一个浮点权重，共 6 行。源文件中的 `*.txt` 忽略规则不会忽略 `q_weights.dat`。

## 5. 常见问题与注意事项

1. **每轮从文件头重读**：`-i` 指定的文件每轮重新打开；EOF、`E` 或 `X` 结束当前轮。多轮训练必须用文件输入，`stdin` 仅支持一轮。
2. **模型文件不带训练元信息**：`q_weights.dat` 是纯权重文件，不含随机种子、训练局数或超参数。程序加载后会继续训练；想比较不同实验，需自行备份。
3. **训练会覆盖模型**：默认在当前工作目录读写 `q_weights.dat`，实验前先备份。若想从源码默认权重开始（例如从旧版流程切到 DQN），加 `--fresh`；想接着 DQN 训练就不要加。
4. **模型里出现 `nan` 会被拒绝加载**：先备份并移走文件再训练，例如：
   ```powershell
   Move-Item .\q_weights.dat .\q_weights.nan.bak
   .\eluose.exe --train 1000 -i .\1.txt --trace .\training_trace.csv
   ```
5. **追踪文件名**：`--trace` / `--eval-trace` 不要指向 `q_weights.dat`。
6. **训练变久不等于变强**：可见指标会波动。固定测试序列、比较训练前后的总分/存活步数/顶出率，比看单局得分可靠。
7. **探索仍在发生**：训练按输入序列读取方块，但 epsilon-greedy 仍会随机探索落点；追踪文件记录的是每一步实际训练的方块编号。
8. **顶出边界是硬编码的 14**：`y + block_h > 14`，训练、评估、对局共用这一条规则；改棋盘尺寸或成功判据时，特征提取、候选生成、奖励和游戏逻辑要一起检查。
9. **保存的模型无法向后兼容**：`q_weights.dat` 只有 6 个数，特征是 6 维；一旦改特征维度（见 §6.6 的 one-hot 建议），旧权重文件就不能再用，需要重训。
10. **几个不影响结果但值得知道的小问题**：`board` 声明 25×15 但只用 20×10；`block_w`/`block_h` 是跨度而非格子数，`features[0]` 因此比真实最高行小 1；`get_block` 用 `char` 接收 `fgetc` 的返回值，`0xFF` 字节会被误判成 EOF（输入是字母，实际不触发）。

## 6. 概念辨析：这个模型到底在学 V 还是 Q

这一节回答一个反复出现的疑问：TD 更新里用的是 `R(s,a)`，看起来像"把 Q 该用的奖励用在了 V 上"；那是不是该改成 `Q = R + V(s')`，或者给特征加一个动作维，让它成为真正的 Q？结论是：**当前实现已经是正确的 Q-learning，不需要为此改代码**；但有三处真正的不一致值得单独讨论（§6.6）。

### 6.1 五个容易混淆的对象

| 记号 | 定义 | 本程序中的形态 |
| --- | --- | --- |
| `V_after(s')` | 落子并消行**之后**棋盘的价值，不含这一手的奖励 | 没有单独实现 |
| `Q(s,a)` | `R(s,a) + γ·V_after(s')` | 就是 `w·φ`，**被训练的对象** |
| `V(s)` | 决策前的价值（棋盘 + 当前方块） | 不存在；`generate_action` 里现算 `max_a Q` |
| `V^π(s)` | `Σ_a π(a\|s)·Q(s,a)` | 从未计算（代码用 `max` 而非期望） |
| `V*(s)` | `max_a Q(s,a)` | 隐式地由 argmax 体现 |

关键一句：变量名 `value_weights` 和"棋盘价值函数"的说法听起来像 V，**实际训练的是 Q**。

### 6.2 奖励项出现在 afterstate 的更新里，是对的

设 `A_t` 为第 `t` 次落子后的棋盘（afterstate），第 `t+1` 手的动作为 `a`，`A_{t+1} = P(A_t, a)`，奖励为 `r_{t+1}`。记 `F(A) = w·φ(A)`，代码做的是：

```text
F(A_{t+1})  ←  r_{t+1} + γ · max_{a'} F(A_{t+2}^{a'})
```

注意 `r_{t+1}` 是"**进入** `A_{t+1}` 的那条边"的奖励（也就是刚落下这一手拿到的分），不是"从 `A_{t+1}` 出发"的奖励。而 Q 的定义是：

```text
Q(A_t, a) = r + γ·V(A_{t+1}),        V(A) = max_{a'} Q(A, a')
⟹  Q(A_t, a) ← r + γ·max_{a'} Q(A_{t+1}, a')
```

两者完全一致：**`F` 拟合的就是 `Q(A_t, a)`**。因为转移是确定的，Q 可以"折叠"到它的结果棋盘上，奖励项就是这个折叠里的常数项。代码里的对应关系：

| 代码位置 | 数学含义 |
| --- | --- |
| `generate_action` 里的 `action->value = score_features(...)` | `F(A_{t+1}) ≈ Q(A_t, a)` |
| `train_on_transition` 的 `target = reward + γ·score(target_weights, next_features)` | `r + γ·max_{a'} Q_target(A_{t+1}, a')` |
| `update_dqn_batch` 的 `td_error = target - current_value` | Q 的 TD 误差 |
| `generate_action` 的贪心分支 | `argmax_a Q(A_t, a)` |

所以"奖励被用在了 V 上"只是命名造成的错觉：被更新的对象不是 V，而是 Q。

### 6.3 两步数值例子

取 `γ = 0.99`：

| 步骤 | 动作 | 奖励 | 结果 |
| --- | --- | --- | --- |
| A₀ 空盘 | a | 0 | → A₁ |
| A₁ | c | 100（终局） | → A₂ |

- 第 2 手：`F(A₂) ← 100 + 0 = 100`（终局不 bootstrap）
- 第 1 手（target 权重追上后）：`F(A₁) ← 0 + 0.99 × 100 = 99`
- 校验：`Q(A₀,a) = 0 + 0.99 · V(A₁) = 0.99 × 100 = 99` ✅
- 选动作：`argmax_c F(A₂^c) = 100 = Q(A₁,c)` ✅

两步例子说明：奖励挂在"进入当前状态的那条边"上，与"更新当前状态的价值"是自洽的，不存在错位。

### 6.4 为什么会"看起来像 V"

1. **命名**：`value_weights`、README 里的"棋盘价值函数"。
2. **特征里其实含动作信息**：`feature0 = y + block_h - 1`（落点高度）与 `feature1 = fill_line × 格子数`（这一手消了几行）都是 `(s,a)` 的函数，而不是棋盘 `s` 单独的函数。严格写应是 `φ(A_t, a)`。
3. **afterstate 的折叠**：把 Q 存到"结果棋盘"上，看起来像在评价棋盘，其实在评价"走到这个棋盘的这一手"。

### 6.5 两个"看起来该改"的方向，及其实后果

| 方案 | 想法 | 若按字面改会发生什么 |
| --- | --- | --- |
| 一 | 令 `Q = R(s,a) + V(s')`，用它做 argmax，用环境的真 R 代替估计 | **奖励被算两遍**：`F` 已≈ Q，再叠一个 `R` 等于 `argmax(Q + R)`，策略会系统性偏向立刻消行 |
| 二 | 给特征加一维动作编号（0..40），让 V 变成 Q 再训练 | 对"变成 Q"**多余**（已经是 Q）；对"加容量"**低效**（见下） |

方案一的额外坑：`action->value` 一处两用——既是 `generate_action` 的选择依据，又是 `train_on_transition` 里的 `current_value`。在选择处加 `r` 会同时抬高 TD 的"当前值"，把训练目标一起带歪。

如果确实想要"pure V + 显式 r"的另一套参数化（数学等价，学的是 `V(A)` 而不是 `Q`），需要三处同改：`extract_action_features` 把 `place_block` 的返回值带出来；选择改成 `argmax_a [r + γV]`；训练目标改成一步前瞻：

```text
V(A_{t+1})  ←  max_{a'} [ r(A_{t+1}, a') + γ·V(A_{t+2}^{a'}) ]
```

代价：目标变成"一步前瞻 + max"，方差更大、更容易偏保守。当前写法（奖励折在回归目标里）方差更低，是更实用的那种。

方案二的三个变体：

- 加"动作编号"这一维：线性模型得到"随编号线性变化"的项，而动作编号跨棋盘没有相同语义（T 的 0 号旋转和 I 的 0 号旋转毫无关系）。
- 加 one-hot：得到 `Q(s,a) = w·φ(s) + u_a`，只是"每个动作一个与棋局无关的常数偏移"。
- 加 `φ(s) ⊗ one-hot(a)`：40 套独立权重，参数从 6 涨到 240，样本效率崩。

根因：转移确定，**afterstate `s'` 已经是动作的充分统计量**，动作本身不携带额外信息。要加容量应该加棋盘特征（逐列高度、行内空洞分布、逐列高度差），不是动作维。

### 6.6 真正对不上的三处

1. **`feature1` 让 `r` 不可表示**——这才是"模型表示不了 Q"的真位置。`fill_line × 格子数` 把不同奖励折叠成同一个数：

   | 情形 | `fill_line` | 格子数 | `feature1` | 实际奖励 |
   | --- | --- | --- | --- | --- |
   | 消 1 行、占 1 格 | 1 | 1 | 1 | 101 |
   | 消 1 行、占 4 格 | 1 | 4 | 4 | 101 |
   | 消 2 行、各占 2 格 | 2 | 2 | 4 | 301 |
   | 消 4 行、各占 1 格 | 4 | 1 | 4 | 801 |

   `feature1 = 4` 对应 101 / 301 / 801 三种奖励，线性模型只能学平均。对症改法：把 `fill_line` 换成 0..4 的 one-hot，或至少单独加一维 `fill_line`。

2. **EOF 被当成终局**：`train_value_episode` 里读不到下一块就 `terminal = 1`、不 bootstrap，等于把"还能继续玩"的价值强行记成 0，序列尾部估值系统性偏低。这是截断（truncation）偏差，不是 MDP 的终止。

3. **target 网络的 max 双采样**：`argmax` 和取值都用 `target_weights`，是标准的 DQN 高估偏差（6 参数下影响小，但存在）。

### 6.7 复习用的一句话

> `Q = R + γV_after` 在确定性转移下是精确等式；代码把 Q 存在 afterstate 上，把 `R` 放进回归目标里，所以 `w·φ` **就是 Q**。
>
> 因此：不要给选择式再加 `R`（双重计奖）；不要加动作编号特征（afterstate 已是充分统计量）；真正可改的是 `feature1` 的编码与 EOF 的截断处理。

## 7. 函数速查：输入、输出、功能

`eluose.c` 共 28 个函数（含 `main`）。下面逐个给出签名、输入、输出和容易误用的地方；行号见附录 A。

### 7.1 棋盘与方块操作

#### `void reset_value_model(void)`

- **功能**：把 `value_weights` 恢复成源码里的 6 个初始权重。
- **输入**：无。
- **输出**：无（就地改全局）。
- **注意**：初值在 61–62 行与 85–86 行各写了一遍，改初值要两处同改。

#### `int can_place(int shape,int rotation,int x,int y)`

- **功能**：判断该方块以 `(x,y)` 为锚点能否放入棋盘（锚点是方块 4 个格子坐标的原点，不一定是左上角）。
- **输入**：方块编号 `0..6`、旋转角 `0..3`、棋盘横坐标、锚点纵坐标。
- **输出**：`1` 可放；`0` 越界（`x` 不在 `[0,10)`、`y` 不在 `[0,20)`、右侧溢出 `dx+x>=10`、顶部溢出 `dy+y>=20`）或与已有方块重叠。
- **注意**：函数内直接用 `block[shape][rotation]` 取格坐标，**不校验 `shape`/`rotation` 范围**，传越界值会读越界内存；`place_block` 因此在使用前自己先查了一遍。

#### `int place_block(int shape,int rotation,int x,int y,int fill[4],int full_rows[4],int *fill_line)`

- **功能**：把方块写入 `board`，统计受影响各行的占用格数与满行，并返回消行得分。
- **输入**：位置参数；三个输出参数 —— `fill[4]` 各局部行占用格数、`full_rows[4]` 满行行号（**升序**）、`*fill_line` 满行数。
- **输出**：`100/300/500/800`（消 1/2/3/4 行）、`0`（未消行）；参数非法或放不下返回 `-1`。
- **注意**：**不消行**，只算分；返回 `-1` 时三个输出参数已经被清零（清零发生在合法性检查之前），调用方不必再初始化。`clear_lines` 依赖 `full_rows` 升序。

#### `void clear_lines(const int full_rows[4],int fill_line)`

- **功能**：把非满行依次下移压实，空出的高位补 0。
- **输入**：满行行号数组（必须升序）、满行数。
- **输出**：无（就地改 `board`）。
- **注意**：单趟双指针算法，只读一遍 20 行；如果 `full_rows` 乱序会把行搬错位置。

#### `void remove_block(int shape,int rotation,int x,int y)`

- **功能**：把方块占的 4 个格子清零。
- **输入**：位置参数。
- **输出**：无。
- **注意**：不校正悬空、不消行。只适用于"确实只是临时放了一下"的场合——特征提取那条路径中途会消行，所以它改用整盘快照回滚，而不是调这个函数。

#### `int simulate_fall(int shape,int rotation,int x)`

- **功能**：模拟方块从高处自由下落到落点。
- **输入**：方块、旋转角、横坐标。
- **输出**：落点纵坐标（最后可放置的 `y`）。
- **注意**：固定从 `y=16` 起探；**若 `y=16` 就放不下，返回 `17`** 这个超范围值（原注释里保留了一行被注释掉的 `return -1`）。调用方必须自己检查，`generate_action` 是先调 `can_place(...,16)` 挡掉的，`run_game` 则由 `find_best_position` 的失败路径挡。

#### `void evaluate_board(int* drow,int* dcol,int* holes,int* well)`

- **功能**：扫描 20×10 棋盘，统计四个结构指标。
- **输入**：四个 `int*`，**累加语义**。
- **输出**：通过指针累加 —— `drow` 同一行内相邻列占用不同则加一；`dcol` 同一列内相邻行占用不同则加一；`holes` 某格为空且该列上方已有方块则加一；`well` 某格为空且左右（含越界）都占据时按连续深度累计。
- **注意**：调用前必须清零。边界处理不对称：`j==0` 自比恒为 0（左边界不算墙），`i==0` 同理（底边不算墙），但井深判定把越界当已占据（左右边界算墙）。

### 7.2 特征与打分

#### `double score_features(const double value_weights[6],const double features[6])`

- **功能**：6 维点积 `wᵀφ`，即线性价值估计。
- **输入**：权重数组、特征数组。
- **输出**：`double` 估值，可为负。
- **注意**：无状态，权重由调用方给：传 `value_weights` 就是在线网络，传 `target_weights` 就是目标网络。

#### `int extract_action_features(int shape,int rotation,int x,int y,int block_height,double features[6])`

- **功能**：算出"这一手落下去"对应的 6 维特征。
- **输入**：位置参数、方块高度（调用方传 `block_h[shape][rotation]`）、输出数组。
- **输出**：`1` 成功；`0` 不能放置（此时 `features` 已被清零）。
- **注意**：内部**真的落子并消行**（调 `place_block` → 统计 → `clear_lines`），最后靠 `memcpy` 快照整体回滚；`place_block` 返回的消行得分被丢弃，所以特征里没有原始奖励，只有 `fill_line × 格子数` 这个近似。

#### `double normalize_feature(int feature_index,double value,int board_cells)`

- **功能**：把特征缩放到可比量级，**仅用于评估统计**。
- **输入**：特征下标、原始值、当前棋盘方块总数。
- **输出**：下标 0 除以 20（棋盘高），下标 1 除以 4（最多 4 格），其余除以 `max(board_cells,4)`。
- **注意**：不参与训练，只影响测试报告里的"平均特征"和出图。

### 7.3 动作生成与对局

#### `training_action generate_action(int shape,double epsilon,const double value_weights[6])`

- **功能**：枚举所有候选落点、算特征与价值，再按 epsilon-greedy 选一个。
- **输入**：方块编号、探索概率、用于打分的权重。
- **输出**：`training_action`（`valid/rotation/x/y/top_out/features/value`）；无候选或 `shape` 非法时返回全 0 的结构（`valid=0`）。
- **注意**：候选上限 40（旋转 ≤4 × 横坐标 10）；特征提取失败的候选会被 `count--` 丢弃；贪心只在 `!top_out` 中取最大，全是顶出动作时退化为第 0 个。

#### `void find_best_position(int shape,int* rotation,int* x,int* y)`

- **功能**：以 `epsilon=0` 调 `generate_action` 取贪心落点，供正常对局使用。
- **输入**：方块编号；三个输出指针。
- **输出**：无返回值，通过指针写回落点。
- **注意**：生成失败时写的是 `rotation=0, x=0, y=-1` 这个哨兵组合，而不是报告错误；`run_game` 随后调 `place_block` 拿到 `-1` 才发现。

#### `int get_block(FILE *fp)`

- **功能**：从输入流读一个方块标记，跳过其它字符（空格、换行、数字）。
- **输入**：文件流，或 `stdin`。
- **输出**：`I/T/O/J/L/S/Z` → `0..6`；`E` → `7`；`X` → `-1`；**EOF → `-666`**。
- **注意**：返回值语义不统一，调用方普遍用 `shape>=0 && shape<7` 判断可用、用 `<0 || ==7` 判断结束；函数用 `char` 接收 `fgetc`，`0xFF` 字节会被误判成 EOF（输入是字母，实际不触发）。

#### `int run_game(FILE *input)`

- **功能**：正常/交互模式：预读当前块与下一块，循环"取最优落点 → 落子 → 消行 → 打印进度"。
- **输入**：方块输入流。
- **输出**：`0` 正常结束；`1` 放置失败。
- **注意**：每步打印 `rotation x` 与累计 `score` 两行；`y + block_h > 14` 时打印 `You lose!` 和整个棋盘（从 `row=19` 往下）；`next_block` 为 `-1`/`7`/`-666` 时结束循环。

### 7.4 模型读写

#### `int load_value_model(const char *path)`

- **功能**：从文本文件读 6 个权重进全局 `value_weights`。
- **输入**：文件路径。
- **输出**：`1` 成功；`0` 文件不存在（`errno==ENOENT`，调用方可用初始权重继续）；`-1` 打开失败、读取失败或读到非有限数。
- **注意**：先读进临时数组，全部有效才覆盖，避免半截损坏的模型污染当前权重。

#### `int save_value_model(const char *path)`

- **功能**：把 6 个权重按行写回文本文件。
- **输入**：文件路径。
- **输出**：`1` 成功；`0` 打不开或 `fclose` 失败。
- **注意**：每行 `%.12g`；会覆盖同名文件，不做备份。

### 7.5 追踪记录

#### `int write_training_trace_header(FILE *trace)`

- **功能**：写训练追踪 CSV 的表头。
- **输入**：已打开的可写流。
- **输出**：`1` 成功；`0` 写失败。
- **注意**：前 16 列固定，随后每个特征追加 5 列（顺序必须与 `record_training_data` 完全一致）。

#### `int write_evaluation_trace_header(FILE *trace)`

- **功能**：写测试追踪 CSV 的表头（与训练追踪是两套格式）。
- **输入**：已打开的可写流。
- **输出**：`1` 成功；`0` 写失败。
- **注意**：列是 `episode,games,mean_score,mean_survival_steps,top_out_rate`，之后每个特征追加 `featureN,weightN,contributionN`。

#### `int record_training_data(...)`

- **功能**：按表头格式写一行训练追踪数据。
- **输入**：追踪流、`episode`、`turn`、`shape`、`action` 指针、`epsilon`、`score`、`reward`、`terminal`、`current_value`、`next_value`、`target`、`delta`、`norm`、`board_cells`、`before[6]`、`updates[6]`、`after[6]`。
- **输出**：`1` 成功；`0` 写失败（并在 stderr 报错）。
- **注意**：`trace` 为 `NULL` 时直接返回 `1`，是 no-op——所以不记追踪时这条路径没有 IO 开销。数值一律 `%.17g`。

### 7.6 训练

#### `int random_below(int upper_bound)`

- **功能**：返回 `[0, upper_bound)` 的随机整数。
- **输入**：上界。
- **输出**：`int`。
- **注意**：用两次 `rand()` 拼宽（基数为 `RAND_MAX+1`）再取模，规避单次 `rand()` 范围窄的重复模式；`upper_bound` 传 0 会除零，调用方保证 `replay_size>=warmup`。

#### `void add_replay_transition(const replay_transition *transition)`

- **功能**：把一条转移写进环形回放区。
- **输入**：转移指针（按值整体拷贝）。
- **输出**：无。
- **注意**：`replay_next` 回绕，`replay_size` 封顶 50000；满了之后覆盖最旧的一条，不做随机淘汰。

#### `int update_dqn_batch(const replay_transition batch[32],int batch_size,double alpha,double gamma,double updates[6],double *mean_absolute_td)`

- **功能**：对一个批次做一次梯度下降，就地更新 `value_weights`。
- **输入**：批次数组、实际批量、学习率、折扣率、`updates[6]`（输出每维权重增量）、`*mean_absolute_td`（输出平均 `|TD|`）。
- **输出**：`1` 成功；`0` 批量非法或中途出现非有限数。
- **注意**：`current_value` 是用**当前** `value_weights` **重算**的，而不是回放里存的 `action->value`；"Huber"在这里实际只是把 TD 误差裁剪到 `±100`，没有二次段；再按 `1+Σφᵢ²` 归一化后累加梯度，最后 `Δwᵢ = alpha·Σ(clipped_td·φᵢ/norm)/batch_size`。

#### `int train_on_transition(FILE *trace,int episode,unsigned long long turn,int shape,const training_action *action,double epsilon,int score,int board_cells,const replay_transition *transition)`

- **功能**：处理一条环境转移的完整训练步骤：算 TD、写回放、可能触发批量更新与目标同步、写追踪行。
- **输入**：追踪流、局号、步号、方块、动作指针、`epsilon`、得分、棋盘方块数、转移指针。
- **输出**：`1` 成功；`0` 数值异常或写盘失败。
- **注意**：`current_value` 直接取 `action->value`（选择时算好的那个），与 `update_dqn_batch` 里重算的 current 不是同一个值——所以 CSV 里的 `td_error` 与批量更新用的误差会有细微差异。回放攒够 64 条才开始更新；每 500 次更新同步一次 `target_weights`。

#### `int train_value_episode(FILE *block_input,FILE *trace,int episode,double epsilon,int *has_training_data,int *episode_score,int *survival_steps)`

- **功能**：跑一整局训练（清空棋盘 → 逐步落子 → 直到顶出或输入耗尽）。
- **输入**：输入流、追踪流、局号、`epsilon`、三个输出指针。
- **输出**：`1` 正常结束；`0` 生成动作或放置失败。
- **注意**：开局读不到方块时直接返回 `1` 且 `*has_training_data=0`，主循环据此 `break`；奖励为 `score + 1.0`，顶出再扣 `300·exp(-(turn-1)/60)`；下一块读不到（`<0` 或 `==7`）也按终局处理。

#### `int train_value_model(int episodes,const char *input_path,FILE *trace,int *completed_episodes,const char *test_path,FILE *evaluation_trace,int evaluation_interval)`

- **功能**：训练主循环：初始化目标权重与回放区，按局调用 `train_value_episode`，处理周期评估与日志。
- **输入**：总局数、输入路径（`NULL` 或 `"-"` 表示 `stdin`）、追踪流、输出实际完成局数、测试路径、测试追踪流、评估间隔。
- **输出**：`1` 成功；`0` 打开输入失败、单局失败或周期评估失败。
- **注意**：`srand(time(NULL))` 在这里播种子；每局重开输入文件（`stdin` 除外）；`epsilon = 0.35 - 0.30·episode/episodes`，下限 `0.05`；每 100 局打印一次窗口平均得分与步数。

### 7.7 评估

#### `int run_evaluation_episode(FILE *input,int step_limit,evaluation_result *result)`

- **功能**：在某一片段上跑一局贪心评估。
- **输入**：已 `fseek` 到片段起点的流、步数上限、输出结构（`score/survival_steps/top_out/features`）。
- **输出**：`1` 正常；`0` 无法生成或放置动作。
- **注意**：`result->features` 是逐步 `normalize_feature` 之后的**求和**，不是平均（平均在 `evaluate_value_model` 里按总步数除）；顶出或达到 `step_limit` 立即停。

#### `int evaluate_value_model(const char *path,const char *label,int sample_count,FILE *trace,int episode)`

- **功能**：把测试文件按字节切成 `sample_count` 段，每段跑一局评估，汇总并打印、可选写追踪行。
- **输入**：测试文件路径、报告标签（如"训练前测试"）、片段数、可空追踪流、写追踪用的局号。
- **输出**：`1` 成功；`0` 打不开、定位失败、文件为空、没有可用方块或写追踪失败。
- **注意**：每段 `step_limit = min(segment_size, 500)`；切分按字节、不看方块边界，片段可能落在半个记录上；文件字节数小于 `sample_count` 时自动调小片段数。

### 7.8 入口

#### `int main(int argc,char **argv)`

- **功能**：解析命令行、校验参数组合、加载/重置模型，然后分发到三种模式：独立测试、训练（可带周期评估）、正常对局。
- **输入**：命令行参数。
- **输出**：`0` 成功；`1` 参数错误、IO 错误或训练失败。
- **注意**：校验规则包括——`--trace` 只能配 `--train`、`--eval-trace` 必须配 `--test`、测试文件不得与训练输入同名、独立测试不能同时给 `-i`、`--fresh` 只能配 `--train`、多轮训练必须用文件输入、追踪文件不得指向 `q_weights.dat`。训练成功后无条件保存 `q_weights.dat` 并打印 6 个最终权重。

## 附录 A 源码索引

行号对应本文写作时的 [`eluose.c`](./eluose.c)（共 1085 行）。源码改动后请只更新这张表。

| 行号 | 内容 |
| --- | --- |
| 7–59 | `block[7][4][4][2]` 七种方块的四个旋转格坐标 |
| 60 | `board[25][15]` |
| 61–62 | `FEATURE_COUNT`、初始 `value_weights` |
| 63–81 | `block_r` / `block_w` / `block_h` |
| 82 | `can_place` 前置声明 |
| 84–88 | `reset_value_model` |
| 90–123 | `place_block` |
| 125–144 | `clear_lines` |
| 146–151 | `remove_block` |
| 153–163 | `can_place` |
| 164–171 | `simulate_fall` |
| 173–195 | `evaluate_board` |
| 196–200 | `training_action` |
| 202–206 | `score_features` |
| 208–234 | `extract_action_features` |
| 235–251 | `get_block` |
| 253–295 | `generate_action` |
| 297–308 | `find_best_position` |
| 310–328 | `load_value_model` |
| 330–342 | `save_value_model` |
| 344–354 | `write_training_trace_header` |
| 355–365 | `write_evaluation_trace_header` |
| 366–398 | `record_training_data` |
| 399–410 | `DQN_*` / `EVALUATION_*` 常量 |
| 412–417 | `replay_transition` |
| 419–423 | 回放区与目标权重全局变量 |
| 425–429 | `random_below` |
| 431–481 | `update_dqn_batch` |
| 483–487 | `add_replay_transition` |
| 489–531 | `train_on_transition` |
| 533–603 | `train_value_episode` |
| 605–610 | 两个前置声明 |
| 608–663 | `train_value_model` |
| 665–677 | 评估用结构体与 `normalize_feature` |
| 679–716 | `run_evaluation_episode` |
| 718–805 | `evaluate_value_model` |
| 807–845 | `run_game` |
| 847–1085 | `main` |
