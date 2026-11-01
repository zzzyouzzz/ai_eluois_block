# 俄罗斯方块线性 DQN 训练说明

本文对应源文件 [`eluose.c`](./eluose.c)：用 C 实现俄罗斯方块的落子与消行、6 维棋盘特征提取，以及一个"线性主干 + 一层 tanh 隐藏层"的价值函数近似器；训练采用一步 DQN + 经验回放。

> 先给结论：代码里的 `value_weights` 名字像"棋盘价值"，实际训练的对象是 `Q(s,a)`——奖励折在回归目标里，不是漏加。为什么这样是对的、为什么不建议"再加一个 R"，见 §6。

## 目录

- 快速开始
- 1. 背景：强化学习与本程序
  - 1.1 概念对照
  - 1.2 一步 DQN 与经验回放
  - 1.3 函数近似：线性主干 + 一层隐藏层
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
  - 2.10 对局与命令行
  - 2.11 参数配置区（文件开头）
  - 2.12 数据追踪模块（可整段删除）
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

# 训练：局数、是否从初始权重重训都在源码开头的「参数配置区」（§2.11）
.\eluose.exe --train -i .\1.txt

# 正常对局：贪心策略跑一段方块序列
.\eluose.exe -i .\3.txt
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

### 1.3 函数近似：线性主干 + 一层隐藏层

棋盘状态组合很多，程序没有为每个状态单独保存一项 Q 值，而是提取 6 个特征，用一组可训练参数计算动作价值。当前模型是"线性主干 + 一层 tanh 隐藏层"的残差结构：

```text
Q(s, a) ≈ w·φ  +  Σ_h W2_h · tanh( b_h + Σ_j W1_hj · φ_j · scale_j )
```

- **主干**（`value_weights`，6 个参数）就是原来那个线性模型，保留为直连路径。
- **隐藏层**（16 个 tanh 单元）并联在主干上，负责表示特征之间的交互，例如"洞数和井深必须同时很大、局面才真的差"这类线性项表达不了的关系。
- 隐藏层输入端先按固定常数缩放（`HIDDEN_INPUT_SCALE`），否则井深、洞数动辄上百，`tanh` 一开始就饱和、梯度消失。

用残差而不是纯 MLP 的原因：输出权重 `w2` 全为 0 时隐藏项恒为 0，整个网络**严格等于原来的线性模型**，已经训练好的 `q_weights.dat` 不会作废。代价是 `w2=0` 时 `∂Q/∂w1 ≡ 0`（表示部分永远学不动），所以训练入口会给 `w2` 一个很小的非零初值。

代码中的 `features` 是特征向量 `φ`，`value_weights` 是主干权重，隐藏层参数放在 `hidden`（在线网络）与 `target_hidden`（目标网络）里。模型文件 `q_weights.dat` 现在以 `mlp 16` 开头；旧的 6 行纯线性文件仍然能读（读进来等价于 `w2=0`）。

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
- `value_weights` 保存线性主干的 6 个参数。程序首次运行、且当前目录没有 `q_weights.dat` 时使用这里的初始权重；训练中原地更新，结束后连同隐藏层参数一起写入模型文件。隐藏层参数在 `hidden` / `target_hidden`（`HIDDEN_SIZE = 16`，每层 16×6 权重 + 16 偏置 + 16 输出权重）。
- `reset_value_model` 又用一份局部数组写了一遍同样的 6 个初值，`TRAIN_FRESH` 设成 `1` 走的就是它——**改初值时两处要一起改**。

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

`score_features(weights,layer,features)` 先算 6 项点积 `wᵀφ`，再叠加隐藏层项 `Σ_h w2_h·tanh(...)`。`static` 表示只在本文件内使用；权重和隐藏层都由调用方传入，所以函数本身不依赖全局模型状态——传 `value_weights, &hidden` 是在线网络，传 `target_weights, &target_hidden` 就是目标网络。分数越高，实际价值估计越大，策略越倾向选择该落点。单个隐单元的前向在 `hidden_unit_output` 里：`tanh(b_h + Σ_j w1_hj·φ_j·scale_j)`。

```c
static training_action generate_action(int shape,double epsilon,
                                       const double value_weights[FEATURE_COUNT],
                                       const hidden_layer *layer){
    training_action actions[40];
    int count=0;
```

函数最多枚举 40 个候选动作：旋转数最多为 4，每个旋转枚举 10 个横向位置。权重、隐藏层（`layer`）和探索概率由调用方传入；无可用动作时返回 `valid` 为 0 的结果。

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
            action->value=score_features(value_weights,layer,action->features);
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
    char tag[16];
    if(fscanf(fp,"%15s",tag)!=1) return -1;   /* 连标签都读不到 */
    int ok=0;
    if(strcmp(tag,"mlp")==0){
        int size=0;
        if(fscanf(fp,"%d",&size)==1&&size==HIDDEN_SIZE){
            ok=1;
            /* 依次读 6 个主干权重、16 个 b1、16×6 个 w1、16 个 w2 */
```

- 文件第一行是 `mlp 16`（隐藏层单元数），其后 135 行依次是主干 6 个、`b1` 16 个、`w1` 96 个、`w2` 16 个。
- **兼容旧格式**：如果第一行不是 `mlp`，就把它当作第一个主干权重，再读剩下 5 个；隐藏层全部置 0，等价于直接退化成原来的线性模型。
- 读到非有限数（无穷大、NaN）或参数个数与 `HIDDEN_SIZE` 不符时报格式错误。
- 先读进临时数组与临时结构体，确认全部有效后才覆盖正在使用的权重，避免半截损坏的模型污染当前参数。
- 返回 `1` 表示成功加载。

`save_value_model(path)`：以写入模式创建或覆盖文件，先写 `mlp 16`，再按主干 → `b1` → `w1` → `w2` 的顺序逐行写参数（`%.12g`，足够保留训练得到的小数），最后检查 `fclose` 的返回值。文件打不开或关闭失败都输出错误并返回 `0`。

模型文件默认名为 `q_weights.dat`，路径相对于程序运行时的当前工作目录。

### 2.7 回放区与批量更新

超参数集中在 `DQN_*` 宏里：

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

`replay_transition` 存一次环境转移需要的全部信息：

```c
typedef struct{
    training_action action;              // 状态-动作：含 features 与 value
    double next_features[FEATURE_COUNT]; // 贪心后继动作的特征
    double reward;
    int terminal;
}replay_transition;
```

`replay_buffer[50000]` 是环形缓冲（`replay_next` 回绕），`target_weights` 与 `target_hidden` 是目标网络参数，`dqn_update_count` 累计批量更新次数。`random_below` 用两次 `rand()` 拼出一个比 `RAND_MAX` 宽得多的无符号数再取模，避开单个 `rand()` 范围过窄带来的重复模式。

`update_dqn_batch` 的更新式为：

```text
clipped_td = clamp(target - current_value, -100, 100)
norm       = 1 + Σᵢ φᵢ²
Δwᵢ       = alpha × Σ_batch (clipped_td × φᵢ / norm) / 32
Δw2ₕ      = alpha × Σ_batch (clipped_td × tanh(zₕ) / norm) / 32
Δbₕ       = alpha × Σ_batch (clipped_td × w2ₕ × (1 - tanh²(zₕ)) / norm) / 32
Δw1ₕⱼ     = alpha × Σ_batch (clipped_td × w2ₕ × (1 - tanh²(zₕ)) × φⱼ × scaleⱼ / norm) / 32
```

- 除以 `norm` 是按特征范数归一化，避免 6 维特征同时很大时梯度爆炸。
- 所谓 Huber 在这里实际只体现为**对 TD 误差做 ±100 的裁剪**，并没有二次段。
- 梯度除以批量大小，`alpha` 是学习率。
- 隐藏层用同一套 `clipped_td` 与 `norm` 反传，`tanh'(z) = 1 - tanh²(z)`。主干的 `clipped_td` 与更新量不变，所以隐藏层的加入不会改动原来那 6 个权重的更新式；所有新参数先算完、逐个检查有限，再统一写回，避免只写了一半。

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

训练开始时将当前参数（主干 6 个 + 隐藏层）复制到目标网络，并清空回放区。之后的流程就是循环调用 `train_value_episode`：

- 每步按 epsilon-greedy 选择动作，执行落子和消行后保存 `features`、奖励、终局标志及下一方块的贪心动作特征。
- 回放区达到 64 条经验后，每步随机抽取 32 条进行更新；每 500 次批量更新同步一次目标权重，学习率为 `0.0001`。
- epsilon 随局数线性下降：`epsilon = START + (END - START) * episode / episodes`，下限 `END`；`START` / `END` 取自配置区的 `TRAIN_EPSILON_START` / `TRAIN_EPSILON_END`（默认 `0.35` → `0.05`）。

训练局数区间的平均得分与平均存活步数每 100 局输出一次。

### 2.10 对局与命令行

`run_game` 是交互/正常模式：预读当前块与下一块，调用 `find_best_position` 取落点，每步输出

```text
rotation x
score
```

当 `y + block_h[current_block][rotation] > 14` 时判定失败并打印棋盘（空格用 `" O"[board[row][column]]`，从 `row=19` 向下打印）。

`main` 统一解析运行模式和输入源，命令行只剩两个参数：

- `--train` 启用训练；训练局数不再从命令行读，而是源码「参数配置区」里的 `TRAIN_EPISODES`（默认 2000）。
- `-i 方块输入文件` 在训练和正常运行模式中指定方块序列；正常模式省略时使用 `stdin`，`-i -` 也表示 `stdin`。
- 其余运行参数全部集中在文件开头的「参数配置区」（见 §2.11）：是否从初始权重重训 `TRAIN_FRESH`、模型路径 `VALUE_MODEL_PATH`、训练追踪 `TRAINING_TRACE_PATH`。
- 参数顺序不限，例如 `eluose.exe -i .\1.txt --train` 与 `eluose.exe --train -i .\1.txt` 等价。
- 配置区里的追踪路径若指向模型文件，程序会拒绝启动。

多轮训练必须用 `-i` 指定可重复打开的文件；不指定文件或使用 `-i -` 时只允许训练一轮。每轮训练前棋盘清空，重新打开输入文件并从头读取。输入 EOF、`E` 或 `X` 会结束当前轮，之后下一轮仍从文件开头重新读取。训练会加载已有 `q_weights.dat`（不存在则使用源码默认权重），并保存更新后的权重。

### 2.11 参数配置区（文件开头）

`eluose.c` 开头、`#include` 之后是一整块「参数配置区」，所有可调参数都集中在那里，改完重新编译即可，不再从命令行传。三组共 15 个宏：

| 宏 | 默认值 | 作用 |
| --- | --- | --- |
| `TRAIN_EPISODES` | `2000` | 训练局数 |
| `TRAIN_FRESH` | `0` | `1` = 忽略已有模型、从源码里的初始权重开始；`0` = 接着 `VALUE_MODEL_PATH` 训练 |
| `VALUE_MODEL_PATH` | `"q_weights.dat"` | 模型文件名，读和写都用它 |
| `TRAINING_TRACE_PATH` | `"training_trace.csv"` | 每步训练追踪 CSV；空串表示不追踪 |
| `DQN_REPLAY_CAPACITY` | `50000` | 经验回放区容量 |
| `DQN_BATCH_SIZE` | `32` | 每次批量更新的样本数 |
| `DQN_REPLAY_WARMUP` | `64` | 回放区攒满多少条才开始更新 |
| `DQN_TARGET_UPDATE_INTERVAL` | `500` | 目标网络同步间隔 |
| `DQN_HUBER_DELTA` | `100.0` | TD 的梯度裁剪阈值 |
| `DQN_EARLY_END_PENALTY` / `DQN_EARLY_END_DECAY` | `300.0` / `60.0` | 顶出惩罚的幅度与衰减 |
| `DQN_GAMMA` | `0.99` | 折扣因子 |
| `DQN_LEARNING_RATE` | `0.0001` | 学习率 |
| `TRAIN_EPSILON_START` | `0.35` | 每批 epsilon 的起点（可用 `-DTRAIN_EPSILON_START=0.15` 覆盖） |
| `TRAIN_EPSILON_END` | `0.05` | 每批 epsilon 的终点，也是下限 |

想追踪训练，把 `TRAINING_TRACE_PATH` 改成 `"training_trace.csv"` 再编译，然后照常 `eluose.exe --train -i .\1.txt`。想做一次"从头开始"的训练（例如换一组初值），把 `TRAIN_FRESH` 改成 `1`。

### 2.12 数据追踪模块（可整段删除）

追踪代码全部收在两条 banner 注释之间：从 `/* ====...` 开头的那段说明起，到 `/* ------------------------ 数据追踪模块结束 ------------------------ */` 为止。模块自己持有 `FILE *`，所以 `train_value_model` / `train_value_episode` / `train_on_transition` 的签名里都没有追踪参数。训练代码里只剩下面几处插桩，每一处都是可单独删掉的整行：

| 位置 | 插桩语句 | 作用 |
| --- | --- | --- |
| `main` | `trace_attach(trace);` | 把输出流交给模块 |
| `main` | `trace_write_training_header()` | 写表头 |
| `train_value_episode` | `trace_begin_step(...)` / `trace_end_step();` | 抓更新前后的快照、写一行 |

原则只有一条：**"只为写 CSV 才需要"的计算不许留在训练代码里**。TD 五件套、棋盘方块数、权重前值快照、特征归一化与累加全部搬进模块；`update_dqn_batch` 也因此没了出参——以前的 `updates[6]` 换成"前后快照相减"，平均 `|TD|` 换成"逐条 TD 直接记"，函数本身只负责更新权重。

把模块整段删掉、再删掉上表里的插桩行，剩下的代码照常编译、照常训练。仓库里实测过：去掉追踪后同一段方块序列的得分、落子序列与带追踪时完全一致，训练得到的权重也逐位相同——追踪只在旁边记录，不参与任何计算。

## 3. 追踪 CSV 格式

`TRAINING_TRACE_PATH` 设成文件路径后，每个环境动作记录一行。前 16 列为：

```text
episode,turn,shape,rotation,x,y,epsilon,score,reward,terminal,current_value,next_value,target,td_error,feature_norm,board_cells
```

之后每个特征 `i` 都会追加 `feature_i,weight_before_i,contribution_i,weight_delta_i,weight_after_i` 五列：

| 列 | 含义 |
| --- | --- |
| `feature0` | 本次动作的特征值 `φ₀` |
| `weight_before0` | 更新前权重 `w₀` |
| `contribution0` | 该特征对当前估值的贡献 `w₀ × φ₀` |
| `weight_delta0` | 权重变化 `Δw₀ = weight_after0 − weight_before0` |
| `weight_after0` | 更新后权重 |

特征 1 到 5 同样记录。因为训练采用经验回放，一行表示一次实际环境动作；权重变化列记录此动作加入回放后执行的批量更新，**不代表单独由该行特征产生的更新**。`td_error` 是动作写入回放时计算的误差，后续重放会再次更新同一经验，因此 CSV 并非完整的逐次采样批次日志。

`shape` 是方块编号：0=I、1=T、2=O、3=J、4=L、5=S、6=Z。

## 4. 编译、运行与可视化

### 4.1 编译

在 Windows PowerShell 中，从仓库目录执行：

```powershell
gcc -std=c11 -O2 .\eluose.c -lm -o .\eluose.exe
```

`.vscode/tasks.json` 里另有一条 `gcc -g` 的单文件调试构建。

### 4.2 训练

训练局数、是否从初始权重重训、追踪文件都在源码「参数配置区」（§2.11）里，命令行只选模式与文件。

最基本的训练——重复读取 `1.txt` 训练 `TRAIN_EPISODES`（默认 2000）局：

```powershell
.\eluose.exe --train -i .\1.txt
```

要把训练追踪写进 CSV，先把配置区里 `TRAINING_TRACE_PATH` 改成文件名再重新编译，命令本身不变：

```powershell
.\eluose.exe --train -i .\1.txt
```

做一次"从初始权重重训"，把 `TRAIN_FRESH` 改成 `1` 后重新编译即可，命令行不需要额外参数。

每轮训练都会重开 `-i` 指定的文件并从头读取，棋盘顶出或文件结束时本轮结束。动作探索仍使用 epsilon-greedy 随机策略。多轮训练不能使用 `stdin`，因为输入流不能重新打开；若把 `TRAIN_EPISODES` 设为 1，可以省略 `-i`，用 `Get-Content .\1.txt | .\eluose.exe --train`。

### 4.3 追踪与出图

记录每一步特征与权重变化：先把配置区里的 `TRAINING_TRACE_PATH` 设为 `"training_trace.csv"` 重新编译，再照常训练：

```powershell
.\eluose.exe --train -i .\1.txt
```

CSV 可用 Excel 导入，也可用 Python、R 等工具统计。重点比较 `featureN`、`weight_beforeN`、`contributionN`、`weight_deltaN` 和 `weight_afterN`，同时观察 `td_error` 与 `feature_norm`。例如，若某维 `weight_deltaN` 长期同号且绝对值持续增大，说明该维在持续推动权重漂移；若 `td_error` 很大而多维特征同时非零，需结合每维的 `weight_deltaN` 判断具体贡献。

仓库中的 [`analyze_training.py`](./analyze_training.py) 会绘制中文训练图（滚动平均得分/存活步数、epsilon、六个权重、归一化特征、加权特征贡献、消行率和 TD 误差）。高度按棋盘高度归一化，消行特征按 4 格归一化，行变化、列变化、洞数和井深按落子后棋盘方块总数归一化，减少不同堆叠规模造成的量级偏差。加权贡献为 `weight_before × 归一化特征`，可观察权重变化是否伴随该特征对应贡献或特征值下降。脚本要求使用当前 DQN 版本生成的追踪 CSV；追踪格式不匹配时会报告缺失字段。旧版追踪文件需重新训练后再分析。需要 Python 与 `matplotlib`：

```powershell
python -m pip install matplotlib
python .\analyze_training.py .\training_trace.csv --output .\training_analysis.png
```

可用 `--window 50` 调整滚动均值窗口。

### 4.4 正常游戏模式

正常模式默认从标准输入读取方块序列，也可通过 `-i` 指定输入文件：

```powershell
Get-Content .\3.txt | .\eluose.exe
.\eluose.exe -i .\3.txt
```

模型文件第一行是 `mlp 16`，其后每行一个浮点参数，共 135 行（主干 6 + 偏置 16 + 隐藏权重 96 + 输出权重 16）。源文件中的 `*.txt` 忽略规则不会忽略 `q_weights.dat`。

### 4.5 多批次实验脚本 `run_experiments.ps1`

把"训练 N 局 → 跑一局贪心测试"重复若干批，模型在批与批之间延续，每批的测试得分记进 `test_scores.csv`。所有参数都在脚本开头：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `$TrainSet` / `$TestSet` | `.\1.txt` / `.\2.txt` | 训练集、测试集路径 |
| `$EpisodesPerBatch` | `200` | 每批训练局数（编译时用 `-DTRAIN_EPISODES` 覆盖，不改源码） |
| `$BatchCount` | `20` | 一共重复多少批 |
| `$EpsilonStart` / `$EpsilonStep` / `$EpsilonEnd` | `0.35` / `0.05` / `0.05` | 首批 epsilon 起点、每批测试后下调的步长、下限 |
| `$WorkDir` | `.\runs` | exe、模型、CSV 的工作目录 |

**epsilon 逐批下调**：每批都按自己的起点重新编译（`-DTRAIN_EPSILON_START`），批内仍按原逻辑从起点线性降到 `TRAIN_EPSILON_END`；一批测试完就把下一批的起点减 `$EpsilonStep`，不低于 `$EpsilonEnd`。这样后一批是接着前一批的探索水平继续。若不做这件事，每次重启进程 epsilon 都会跳回 0.35，图上就是每 200 局一个完全相同的锯齿，跨批趋势全被它盖住。

产物：

- `test_scores.csv`：`batch,episodes,cum_episodes,epsilon_start,test_score,test_pieces,test_ms`。测试是**一局确定性的贪心对局**——策略贪心、输入固定，同一模型结果恒定；它衡量的是"这个模型在这条测试序列上能撑多久"，不是多局平均。`test_pieces` 就是这一局走了多少步，`test_score / test_pieces` 可用来分离"消行效率"与"存活长度"。
- `trace_batchNNN.csv`：每批的训练追踪，出图时按顺序传入，局号会首尾相接。
- `q_weights.dat`：持续更新的模型。

出图：

```powershell
python .\analyze_training.py .\runs\trace_batch*.csv --test-log .\runs\test_scores.csv --output .\runs\analysis.png
```

## 5. 常见问题与注意事项

1. **每轮从文件头重读**：`-i` 指定的文件每轮重新打开；EOF、`E` 或 `X` 结束当前轮。多轮训练必须用文件输入，`stdin` 仅支持一轮。
2. **模型文件不带训练元信息**：`q_weights.dat` 是纯权重文件，不含随机种子、训练局数或超参数。程序加载后会继续训练；想比较不同实验，需自行备份。
3. **训练会覆盖模型**：默认在当前工作目录读写 `q_weights.dat`（路径由配置区 `VALUE_MODEL_PATH` 决定），实验前先备份。若想从源码默认权重开始（例如从旧版流程切到 DQN），把 `TRAIN_FRESH` 设为 `1` 再编译；想接着 DQN 训练就保持 `0`。
4. **模型里出现 `nan` 会被拒绝加载**：先备份并移走文件再训练，例如：
   ```powershell
   Move-Item .\q_weights.dat .\q_weights.nan.bak
   .\eluose.exe --train -i .\1.txt
   ```
5. **追踪文件名**：配置区里 `TRAINING_TRACE_PATH` 不要指向 `VALUE_MODEL_PATH`，程序会拒绝启动。
6. **训练变久不等于变强**：可见指标会波动。固定测试序列、比较训练前后的总分/存活步数/顶出率，比看单局得分可靠。
7. **探索仍在发生**：训练按输入序列读取方块，但 epsilon-greedy 仍会随机探索落点；追踪文件记录的是每一步实际训练的方块编号。
8. **顶出边界是硬编码的 14**：`y + block_h > 14`，训练与对局共用这一条规则；改棋盘尺寸或成功判据时，特征提取、候选生成、奖励和游戏逻辑要一起检查。
9. **改了特征维度就要重训**：隐藏层的输入维度固定为 6，一旦改特征维度（例如把 `feature1` 拆成 `fill_line` 与格子数两维，见 §6.6），旧权重文件就不能再用。旧的 6 行纯线性文件仍可加载（隐藏层按 0 处理），但那只是兼容旧文件，不是等价替换。
10. **几个不影响结果但值得知道的小问题**：`board` 声明 25×15 但只用 20×10；`block_w`/`block_h` 是跨度而非格子数，`features[0]` 因此比真实最高行小 1；`get_block` 用 `char` 接收 `fgetc` 的返回值，`0xFF` 字节会被误判成 EOF（输入是字母，实际不触发）。

## 6. 概念辨析：这个模型到底在学 V 还是 Q

这一节回答一个反复出现的疑问：TD 更新里用的是 `R(s,a)`，看起来像"把 Q 该用的奖励用在了 V 上"；那是不是该改成 `Q = R + V(s')`，或者给特征加一个动作维，让它成为真正的 Q？结论是：**当前实现已经是正确的 Q-learning，不需要为此改代码**；但有一处设计取舍和两处真正的不一致值得单独讨论（§6.6）。

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
2. **特征里其实含动作信息**：`feature0 = y + block_h - 1`（落点高度）与 `feature1 = fill_line × 格子数`（这一手消了几行、方块有多大比例被满行吸收）都是 `(s,a)` 的函数，而不是棋盘 `s` 单独的函数。严格写应是 `φ(A_t, a)`。
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

### 6.6 一处设计取舍与两处真不一致

1. **`feature1` 是"消行效率"特征，不是奖励估计器**——这点要说清，免得被误当成缺陷。`fill_line × 格子数` 描述的是"这一手有多少工作量真正转化成了消行"：值越大，说明方块被满行吸收得越充分、留在堆上的部分越少。它**刻意不承担**还原 100/300/500/800 的职责——奖励通过 TD 目标进入估值，特征只提供偏好方向。所以"`feature1` 分不清消了几行"不是表示缺陷，而是设计上让它不表达行数。

   这个选择的已知代价是：有两种落点会落在同一个特征值上（奖励却差 200）：

   | 情形 | `fill_line` | 格子数（总） | `feature1` | 奖励 |
   | --- | --- | --- | --- | --- |
   | 消 1 行，方块 4 格都在该行 | 1 | 4 | 4 | 101 |
   | 消 2 行，共占 2 格（另外 2 格留在堆上） | 2 | 2 | 4 | 301 |
   | 消 2 行，共占 3 格 | 2 | 3 | 6 | 301 |
   | 消 3 行，共占 2 格 | 3 | 2 | 6 | 501 |

   其余组合不撞：`(1,1)=1`、`(2,4)=8`、`(3,3)=9`、`(3,4)=12`、`(4,4)=16`。撞车的两对都落在"行数少但格数多"与"行数多但格数少"之间，正是这个特征要压缩掉的那一个自由度；要不要区分它，取决于你希望模型学"单位方块消几行"还是"这一手消了几行"——前者是策略偏好，后者是奖励，两者不必都进特征。

2. **EOF 被当成终局**：`train_value_episode` 里读不到下一块就 `terminal = 1`、不 bootstrap，等于把"还能继续玩"的价值强行记成 0，序列尾部估值系统性偏低。这是截断（truncation）偏差，不是 MDP 的终止。

3. **target 网络的 max 双采样**：`argmax` 和取值都用 `target_weights`，是标准的 DQN 高估偏差（6 参数下影响小，但存在）。

### 6.7 复习用的一句话

> `Q = R + γV_after` 在确定性转移下是精确等式；代码把 Q 存在 afterstate 上，把 `R` 放进回归目标里，所以 `w·φ` **就是 Q**。
>
> 因此：不要给选择式再加 `R`（双重计奖）；不要加动作编号特征（afterstate 已是充分统计量）；`feature1` 是设计好的效率特征而非奖励估计；真正值得改的是 EOF 的截断处理。

## 7. 函数速查：输入、输出、功能

`eluose.c` 共 28 个函数（含 `main`）。下面逐个给出签名、输入、输出和容易误用的地方；行号见附录 A。

### 7.1 棋盘与方块操作

#### `void reset_value_model(void)`

- **功能**：把 `value_weights` 恢复成源码里的 6 个初始权重，并把隐藏层清零（`w2=0`，此时严格等价于纯线性模型）。
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

#### `double score_features(const double weights[6],const hidden_layer *layer,const double features[6])`

- **功能**：估值 = 6 维点积 `wᵀφ`，再加隐藏层项 `Σ_h w2_h·tanh(·)`。
- **输入**：主干权重数组、隐藏层指针（在线或目标）、特征数组。
- **输出**：`double` 估值，可为负。
- **注意**：无状态，权重与隐藏层都由调用方给：传 `value_weights, &hidden` 就是在线网络，传 `target_weights, &target_hidden` 就是目标网络。

#### `double hidden_unit_output(const hidden_layer *layer,int unit,const double features[6])`

- **功能**：算一个隐单元的输出 `tanh(b_h + Σ_j w1_hj·φ_j·scale_j)`。
- **输入**：隐藏层指针、单元下标、特征数组。
- **输出**：`[-1, 1]` 之间的 `double`。
- **注意**：输入先乘 `HIDDEN_INPUT_SCALE` 再进 `tanh`；前向打分与反向传播都复用这个函数，保证两边算的是同一个 `z`。

#### `int extract_action_features(int shape,int rotation,int x,int y,int block_height,double features[6])`

- **功能**：算出"这一手落下去"对应的 6 维特征。
- **输入**：位置参数、方块高度（调用方传 `block_h[shape][rotation]`）、输出数组。
- **输出**：`1` 成功；`0` 不能放置（此时 `features` 已被清零）。
- **注意**：内部**真的落子并消行**（调 `place_block` → 统计 → `clear_lines`），最后靠 `memcpy` 快照整体回滚；`place_block` 返回的消行得分被丢弃，所以特征里没有原始奖励，只有 `fill_line × 格子数` 这个近似。

### 7.3 动作生成与对局

#### `training_action generate_action(int shape,double epsilon,const double weights[6],const hidden_layer *layer)`

- **功能**：枚举所有候选落点、算特征与价值，再按 epsilon-greedy 选一个。
- **输入**：方块编号、探索概率、用于打分的主干权重、隐藏层指针。
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

- **功能**：从文本文件读全部模型参数进 `value_weights` 与 `hidden`。
- **输入**：文件路径。
- **输出**：`1` 成功；`0` 文件不存在（`errno==ENOENT`，调用方可用初始权重继续）；`-1` 打开失败、读取失败或读到非有限数。
- **注意**：先读进临时数组与临时结构体，全部有效才覆盖；以 `mlp 16` 开头的是新格式，否则按旧的 6 行纯线性文件解析（隐藏层按 0 处理）。

#### `int save_value_model(const char *path)`

- **功能**：把全部参数按行写回文本文件（`mlp 16` → 主干 6 → `b1` 16 → `w1` 96 → `w2` 16）。
- **输入**：文件路径。
- **输出**：`1` 成功；`0` 打不开或 `fclose` 失败。
- **注意**：每行 `%.12g`，共 135 行；会覆盖同名文件，不做备份。

#### `void fill_hidden_w1(void)`

- **功能**：给 `w1` 填确定性的、各单元互不相同的初值。
- **输入**：无。
- **输出**：无。
- **注意**：16 个单元如果初值相同，梯度会一直相同，等于只训练了一个单元。

#### `void reset_hidden_layer(void)`

- **功能**：`w1` 重新取值、`b1` 与 `w2` 清零，并把 `hidden` 同步给 `target_hidden`。
- **输入**：无。
- **输出**：无。
- **注意**：`w2=0` 时隐藏项恒为 0，`score_features` 退化成 6 项点积；测试里可以用它构造纯线性基准。

#### `void seed_hidden_layer(void)`

- **功能**：给 `b1`、`w2` 一点非零小初值，作为训练起点。
- **输入**：无。
- **输出**：无。
- **注意**：`w2=0` 时 `∂Q/∂w1 ≡ 0`，表示部分永远学不动，所以 `TRAIN_FRESH=1` 训练开始前必须调它。

### 7.5 数据追踪（可整段删除）

本组函数全部写在 `eluose.c` 里两条 banner 注释之间，**只读训练状态、只写 CSV**，不参与任何决策。训练代码里只留下几处单行插桩调用；把整段模块连同这些调用行一起删掉，训练与对局的行为、输出、随机数序列完全不变（已用"删掉后重新编译并跑对照"验证）。

#### `static FILE *trace_training_file`

模块自己持有输出流，因此训练函数的签名里不再出现 `FILE *trace` 参数。

#### `void trace_attach(FILE *training)`

- **功能**：登记输出流。
- **输入**：训练追踪流（可以是 `NULL`）。
- **输出**：无。
- **注意**：`NULL` 表示不追踪，后续写函数遇到 `NULL` 直接 no-op，没有 IO 开销。

#### `int trace_write_training_header(void)`

- **功能**：写 CSV 的表头（46 列）。
- **输出**：`1` 成功；`0` 写失败（调用方据此退出）。
- **注意**：列名与顺序是 [`analyze_training.py`](./analyze_training.py) 的输入契约，改格式必须同步脚本。

#### `int trace_begin_step(...)`

- **功能**：在一步训练**更新权重之前**抓快照：局号、步号、方块、动作、`epsilon`、即时得分、TD 五件套（`current_value` / `next_value` / `target` / `td_error` / `norm`）、棋盘方块数、6 个权重的前值。
- **输入**：`episode`、`turn`、`shape`、`action`、`epsilon`、`score`、`transition`。
- **输出**：`1`（不追踪时同样返回 `1`）。
- **注意**：这里的 `next_value` 用目标网络 `target_weights` / `target_hidden` 计算，与训练内部一致；这些量原先散落在训练函数里，现在全部收进模块。

#### `int trace_end_step(void)`

- **功能**：在权重更新**之后**补上 `weight_after` 与 `weight_delta`（= 更新后权重 − 快照里的权重），写出一整行。
- **输出**：`1`；写失败返回 `0`。
- **注意**：数值一律 `%.17g`；`weight_delta` 由前后两次快照相减得到，所以 `update_dqn_batch` 不需要再传出 `updates[]`。

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

#### `int update_dqn_batch(const replay_transition batch[32],int batch_size,double alpha,double gamma)`

- **功能**：对一个批次做一次梯度下降，就地更新 `value_weights` 与隐藏层参数。
- **输入**：批次数组、实际批量、学习率、折扣率。
- **输出**：`1` 成功；`0` 批量非法或中途出现非有限数。
- **注意**：`current_value` 是用**当前** `value_weights` **重算**的，而不是回放里存的 `action->value`；"Huber"在这里实际只是把 TD 误差裁剪到 `±100`，没有二次段；再按 `1+Σφᵢ²` 归一化后累加梯度，最后 `Δwᵢ = alpha·Σ(clipped_td·φᵢ/norm)/batch_size`；隐藏层用同一套 `clipped_td` / `norm` 反传（`Δw2`、`Δb1`、`Δw1`），主干那 6 个权重的更新式不变。

#### `int train_on_transition(const replay_transition *transition)`

- **功能**：处理一条环境转移的完整训练步骤：写回放、可能触发批量更新与目标同步。
- **输入**：转移指针。
- **输出**：`1` 成功；`0` 数值异常。
- **注意**：只做训练本身，不做任何追踪——CSV 需要的 `current_value`、`episode`、`epsilon` 等由调用点前面的 `trace_begin_step` 自己抓。回放攒够 64 条才开始更新；每 500 次更新同步一次 `target_weights` 与 `target_hidden`。

#### `int train_value_episode(FILE *block_input,int episode,double epsilon,int *has_training_data)`

- **功能**：跑一整局训练（清空棋盘 → 逐步落子 → 直到顶出或输入耗尽）。
- **输入**：输入流、局号、`epsilon`、输出指针 `has_training_data`。
- **输出**：`1` 正常结束；`0` 生成动作或放置失败。
- **注意**：开局读不到方块时直接返回 `1` 且 `*has_training_data=0`，主循环据此 `break`；奖励为 `score + 1.0`，顶出再扣 `300·exp(-(turn-1)/60)`；下一块读不到（`<0` 或 `==7`）也按终局处理。

#### `int train_value_model(int episodes,const char *input_path,int *completed_episodes)`

- **功能**：训练主循环：初始化目标权重与回放区，按局调用 `train_value_episode`。
- **输入**：总局数（`TRAIN_EPISODES`）、输入路径（`NULL` 或 `"-"` 表示 `stdin`）、输出实际完成局数。
- **输出**：`1` 成功；`0` 打开输入失败或单局失败。
- **注意**：`srand(time(NULL))` 在这里播种子；每局重开输入文件（`stdin` 除外）；`epsilon` 从 `TRAIN_EPSILON_START` 线性降到 `TRAIN_EPSILON_END`（默认 `0.35` → `0.05`，后者同时是下限）。训练过程不打印进度（只有结束时一行 `训练完成：N 局`），得分与进度看追踪 CSV，或跑 `run_experiments.ps1` 得到的测试得分表。

### 7.7 入口

#### `int main(int argc,char **argv)`

- **功能**：解析命令行（只有 `--train` / `-i`）、校验参数组合、加载或重置模型，然后分发到两种模式：训练、正常对局。
- **输入**：命令行参数；其余运行参数来自文件开头的「参数配置区」。
- **输出**：`0` 成功；`1` 参数错误、IO 错误或训练失败。
- **注意**：校验规则包括——`TRAIN_EPISODES>1` 时必须用 `-i` 给文件输入、配置区里的追踪路径不得指向 `VALUE_MODEL_PATH`。训练成功后无条件保存 `VALUE_MODEL_PATH` 并打印 6 个最终权重。

## 附录 A 源码索引

行号对应本文写作时的 [`eluose.c`](./eluose.c)（共 1010 行）。源码改动后请只更新这张表。

| 行号 | 内容 |
| --- | --- |
| 8–68 | 参数配置区：运行参数（`TRAIN_EPISODES` / `TRAIN_FRESH` / `VALUE_MODEL_PATH` / `TRAINING_TRACE_PATH`）、`DQN_*`、探索（`TRAIN_EPSILON_START` / `TRAIN_EPSILON_END`） |
| 69–121 | `block[7][4][4][2]` 七种方块的四个旋转格坐标 |
| 122 | `board[25][15]` |
| 123 | `FEATURE_COUNT` |
| 124 | 初始 `value_weights` |
| 126–143 | 隐藏层：注释、`HIDDEN_SIZE`、`hidden_layer` 类型、`hidden` / `target_hidden` / `target_weights`、`HIDDEN_INPUT_SCALE` |
| 144 | `block_r` |
| 145–153 | `block_w` |
| 154–162 | `block_h` |
| 163 | `can_place` 前置声明 |
| 165–173 | `fill_hidden_w1` |
| 174–184 | `reset_hidden_layer` |
| 185–192 | `seed_hidden_layer` |
| 193–199 | `reset_value_model` |
| 200–234 | `place_block` |
| 235–255 | `clear_lines` |
| 256–262 | `remove_block` |
| 263–273 | `can_place` |
| 274–282 | `simulate_fall` |
| 283–305 | `evaluate_board` |
| 306–311 | `training_action` |
| 312–319 | `hidden_unit_output` |
| 320–328 | `score_features` |
| 329–355 | `extract_action_features` |
| 356–373 | `get_block` |
| 374–418 | `generate_action` |
| 419–431 | `find_best_position` |
| 432–482 | `load_value_model` |
| 483–504 | `save_value_model` |
| 505–511 | `replay_transition` |
| 512–515 | 回放区与目标网络全局变量 |
| 517–522 | `random_below` |
| 523–611 | `update_dqn_batch` |
| 612–616 | `add_replay_transition` |
| 617–628 | 数据追踪模块起始 banner 与说明 |
| 629 | `trace_training_file` |
| 631–634 | `trace_attach` |
| 635–647 | `trace_write_training_header` |
| 648–656 | `trace_count_board_cells` |
| 657–676 | `trace_step_state` / `trace_step` |
| 678–717 | `trace_begin_step` |
| 718–749 | `trace_end_step` |
| 750 | 数据追踪模块结束 banner |
| 752–769 | `train_on_transition` |
| 770–831 | `train_value_episode` |
| 832–867 | `train_value_model` |
| 868–906 | `run_game` |
| 908–1010 | `main` |
