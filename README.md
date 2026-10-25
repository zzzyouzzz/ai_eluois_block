# 俄罗斯方块价值函数与 Q-learning 训练说明

本文对应源文件 [`eluose.c`](./eluose.c)，重点说明其中用于训练棋盘价值函数的代码。代码使用 C 语言实现俄罗斯方块落子、棋盘特征提取和一个线性函数近似器。

## 1. 相关强化学习知识

### 1.1 强化学习的基本概念

强化学习让智能体通过与环境交互来学习决策。俄罗斯方块中，可以作如下对应：

| 强化学习概念 | 本程序中的含义 |
| --- | --- |
| 智能体（Agent） | 选择旋转角度和横向位置的落子策略 |
| 环境（Environment） | 棋盘、方块下落规则、消行和游戏结束规则 |
| 状态（State） | 当前棋盘与当前待放置方块 |
| 动作（Action） | 当前方块的旋转角度、横坐标和最终落点 |
| 奖励（Reward） | 消行得分，以及顶出棋盘时的惩罚 |
| 策略（Policy） | 根据动作价值选择落点的方法 |
| 回合（Episode） | 从空棋盘开始，直到游戏结束或达到回合上限的一局训练 |

智能体执行动作后，棋盘发生变化，得到下一个状态。训练的目标不是只挑眼前看起来最好的落点，而是让当前选择也考虑后续可能得到的回报。

### 1.2 Q 值、折扣回报和 TD 更新

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

### 1.3 本程序的线性函数近似

棋盘状态组合很多，程序没有为每个状态单独保存一项 Q 值，而是提取 6 个特征，用可训练权重计算动作价值：

```text
Q(s, a) ≈ w₀φ₀ + w₁φ₁ + ... + w₅φ₅
```

代码中的 `features` 是特征向量 `φ`，`value_weights` 是权重 `w`。每次训练通过 TD 误差调整权重。因而它是**线性函数近似的 Q-learning 风格实现**，不是保存状态—动作表格的 tabular Q-learning，也不是神经网络。

特征描述的是“把当前方块放到某个位置后”的棋盘：

| 下标 | 特征 | 代码含义 |
| --- | --- | --- |
| `0` | 落点高度 | `y + block_height - 1` |
| `1` | 消行特征 | 完成行数乘以当前方块落在这些完成行中的格子数 |
| `2` | 行方向变化 `drow` | 相邻列在同一行上的占用状态差异计数 |
| `3` | 列方向变化 `dcol` | 同一列相邻行上的占用状态差异计数 |
| `4` | 洞数 `holes` | 某格为空但其上方已出现方块时的计数 |
| `5` | 井深 `well` | 左右两侧被占据而本格为空时累计的井深 |

初始权重为 `{-45, 34, -32, -93, -79, -34}`，是原先启发式评估系数的起点。训练后权重会变化；每个特征对局面好坏的影响由训练结果决定。

### 1.4 探索与利用

训练时需要平衡两种行为：

- **利用（exploitation）**：选择当前估值最高的动作。
- **探索（exploration）**：随机选一个合法动作，尝试当前估值较低但可能有潜力的局面。

程序使用 epsilon-greedy 策略：以概率 `epsilon` 随机探索，否则选择估值最高的动作。训练过程逐渐降低 `epsilon`，让前期多尝试，后期更多利用已学到的权重。

## 2. 训练代码逐行解析

以下按 [`eluose.c`](./eluose.c) 中训练所依赖的函数顺序解析。行号按当前仓库文件标注；源文件后续修改后，行号可能变化。

### 2.1 权重与特征数量（第 62–63 行）

```c
enum { FEATURE_COUNT = 6 };
double value_weights[FEATURE_COUNT] = {-45.0, 34.0, -32.0, -93.0, -79.0, -34.0};
```

- `FEATURE_COUNT` 固定特征数量为 6，特征抽取、价值计算、模型读写和参数更新都使用这个数量。
- `value_weights` 保存线性模型参数。程序首次运行、且当前目录没有 `q_weights.dat` 时使用这里的初始权重。
- 训练会原地更新这些权重，并在训练结束后保存到模型文件。

### 2.2 价值函数 `score_features`（第 190–194 行）

```c
static double score_features(const double features[FEATURE_COUNT]){
    double score=0.0;
    for(int i=0;i<FEATURE_COUNT;i++)score+=value_weights[i]*features[i];
    return score;
}
```

逐行说明：

1. `static` 表示该函数只在当前 C 源文件中使用。
2. 参数 `features` 是一个长度为 6 的特征数组。
3. `score` 从 `0.0` 开始，使用 `double` 保留训练中的小数精度。
4. 循环逐项累加“权重 × 特征值”，即计算向量点积 `wᵀφ`。
5. 返回该动作的价值估计。通常分数越高，策略越倾向于选择该落点。

### 2.3 提取动作特征 `extract_action_features`（第 195–217 行）

```c
static int extract_action_features(int shape,int rotation,int x,int y,int block_height,
                                   double features[FEATURE_COUNT]){
```

函数输入方块形状、旋转角、横坐标、落点纵坐标和方块高度，输出落子后的特征。返回 `1` 表示成功，返回 `0` 表示不能放置。

```c
    for(int i=0;i<FEATURE_COUNT;i++)features[i]=0.0;
```

先把输出特征清零，避免无效或未赋值的特征残留旧数据。

```c
    int fill[4],full_rows[4],fill_line;
    if(place_block(shape,rotation,x,y,fill,full_rows,&fill_line)<0)return 0;
```

- `fill` 接收方块涉及的各局部行的占用格数。
- `full_rows` 接收本次放置后完成的棋盘行号。
- `fill_line` 接收完成行数。
- `place_block` 临时把方块放到全局棋盘并计算消行结果。放置失败则直接返回 `0`。
- 这里提取特征时只临时放置，不调用 `clear_lines`；真正执行动作时才消行。

```c
    int drow=0,dcol=0,holes=0,well=0;
    evaluate_board(&drow,&dcol,&holes,&well);
```

在临时放入方块后的棋盘上计算四项棋盘结构特征。局部变量先清零，因为 `evaluate_board` 是通过指针累加结果。

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
- 因此该特征不是简单的消行行数，而是“消行行数 × 当前方块在完成行中的格子数”的等价计数方式。

```c
    features[1]=fill_line*block_cells_in_full_rows;
    features[2]=drow;
    features[3]=dcol;
    features[4]=holes;
    features[5]=well;
```

将消行特征和四个棋盘结构指标放入向量中。特征下标与权重下标一一对应。

```c
    remove_block(shape,rotation,x,y);
    return 1;
}
```

- 移除刚才临时放置的方块，恢复棋盘，以免评估候选落点时互相污染。
- 返回 `1` 表示特征成功提取。

### 2.4 游戏实际选择落点时的评估 `evaluate_place`（第 219–224 行）

```c
int evaluate_place(int shape,int rotation,int x,int y,int block_height){
    double features[FEATURE_COUNT];
    if(y+block_height>14)return -9000000;
    if(!extract_action_features(shape,rotation,x,y,block_height,features))return -9000000;
    return (int)score_features(features);
}
```

1. 创建本次候选落点的特征数组。
2. 若放置高度超过程序约定的安全边界 `14`，直接返回很低的分数，让搜索策略避开该落点。
3. 特征提取失败时也返回很低的分数。
4. 否则计算线性价值并转为 `int`，供 `find_best_position` 比较候选位置。

训练更新使用 `double`，而正常落点搜索的接口返回整数，因此搜索时会截去小数部分。

### 2.5 训练动作结构 `training_action`（第 261–266 行）

```c
typedef struct{
    int rotation,x,y,top_out;
    double features[FEATURE_COUNT];
    double value;
}training_action;
```

结构体保存一个候选动作及其评估结果：

- `rotation`：旋转角编号。
- `x`, `y`：横坐标和落点纵坐标。
- `top_out`：该落点是否达到程序定义的顶出边界。
- `features`：该动作对应的棋盘特征。
- `value`：根据当前权重计算的动作价值，用于 epsilon-greedy 选择。

### 2.6 枚举合法候选动作 `collect_actions`（第 267–285 行）

```c
static int collect_actions(int shape,training_action actions[40]){
    int count=0;
```

最多接收 40 个候选动作：旋转数最多为 4，每个旋转枚举 10 个横向位置，因此数组容量 `40` 足够。

```c
    for(int rotation=0;rotation<block_r[shape];rotation++){
        for(int x=0;x<10;x++){
```

按照当前方块实际支持的旋转数，枚举旋转角与棋盘横坐标。

```c
            if(!can_place(shape,rotation,x,16))continue;
```

先检查方块能否从高处 `y=16` 放入棋盘。不能放入的动作跳过。

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

如果放置后高度超过 `14`，标记为顶出。训练中这类动作会被赋予极低价值，并在执行后终止本局。

```c
            extract_action_features(shape,rotation,x,y,block_h[shape][rotation],
                                    action->features);
```

提取候选落点特征。这里候选已由合法性检查和模拟下落确定为可放置。

```c
            action->value=action->top_out?-1000000.0:score_features(action->features);
```

- 顶出动作直接给极低的临时价值，避免贪心策略选择它。
- 其他动作使用当前权重计算线性价值。
- 这只是动作选择时的当前估值，之后还会根据 TD 误差更新权重。

```c
        }
    }
    return count;
}
```

完成枚举后返回候选动作数。若没有合法候选动作，返回 `0`。

### 2.7 读取模型 `load_value_model`（第 287–305 行）

```c
static int load_value_model(const char *path){
    FILE *fp=fopen(path,"r");
```

以文本模式打开模型文件。

```c
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
```

- 临时读入 6 个权重。
- `fscanf` 失败或读到非有限数（如无穷大、NaN）时，报告模型格式错误。
- 先读到临时数组，确保所有值有效后才覆盖正在使用的权重。

```c
    fclose(fp);
    for(int i=0;i<FEATURE_COUNT;i++)value_weights[i]=loaded[i];
    return 1;
}
```

关闭文件，将验证过的参数复制到全局模型，返回 `1` 表示成功加载。

### 2.8 保存模型 `save_value_model`（第 307–319 行）

```c
static int save_value_model(const char *path){
    FILE *fp=fopen(path,"w");
```

以写入模式创建或覆盖模型文件。

```c
    if(!fp){
        fprintf(stderr,"无法保存价值模型 %s\n",path);
        return 0;
    }
```

文件无法创建时输出错误并返回失败。

```c
    for(int i=0;i<FEATURE_COUNT;i++)fprintf(fp,"%.12g\n",value_weights[i]);
```

每行写入一个权重，使用足够的有效数字保留训练得到的小数。

```c
    if(fclose(fp)!=0){
        fprintf(stderr,"保存价值模型失败：%s\n",path);
        return 0;
    }
    return 1;
}
```

关闭文件并检查关闭错误。成功返回 `1`。模型文件默认名为 `q_weights.dat`，路径相对于程序运行时的当前工作目录。

### 2.9 训练追踪表头 `write_training_trace_header`

训练函数可选接收一个 CSV 文件指针。启用追踪后，程序先写表头。前 15 列为：

```text
episode,turn,shape,rotation,x,y,epsilon,score,reward,terminal,current_value,next_value,target,td_error,feature_norm
```

之后每个特征 `i` 都会追加 `feature_i,weight_before_i,contribution_i,weight_delta_i,weight_after_i` 五列，例如特征 0。`record_training_data` 独立负责写入一条完整更新记录；不启用追踪时不会写 CSV：

| 列 | 含义 |
| --- | --- |
| `feature0` | 本次动作的特征值 `φ₀` |
| `weight_before0` | 更新前权重 `w₀` |
| `contribution0` | 该特征对当前估值的贡献 `w₀ × φ₀` |
| `weight_delta0` | 本次权重变化 `Δw₀` |
| `weight_after0` | 更新后权重 |

特征 1 到 5 同样记录。由此可以区分权重变化是由较大的 TD 误差、该特征的取值，还是它们的乘积造成。每行对应一次训练更新。

`shape` 是方块编号：0=I、1=T、2=O、3=J、4=L、5=S、6=Z。

### 2.10 核心训练函数 `train_value_model`

```c
const double alpha=0.15,gamma=0.90;
srand((unsigned int)time(NULL));
```

- `alpha` 是学习率，控制每次权重更新的幅度。较大时学习快但可能不稳定，较小时更新平缓。
- `gamma` 是未来价值折扣率。程序用它折扣下一步的最大价值。
- `srand` 使用当前时间初始化伪随机数生成器，用于 epsilon-greedy 动作探索。训练方块不再随机生成，而是按输入序列调用 `get_block` 读取。

#### 开始每一局并清空棋盘（第 335–337 行）

```c
for(int episode=0;episode<episodes;episode++){
    for(int i=0;i<25;i++)
        for(int j=0;j<15;j++)board[i][j]=0;
```

- 外层循环训练指定局数。
- 每局开始时清空整个 `board[25][15]` 数组，保证上一局状态不会泄漏到新一局。

#### 设置探索概率并读取起始方块

```c
double epsilon=0.25-0.20*(double)episode/episodes;
if(epsilon<0.05)epsilon=0.05;
int shape=get_block(block_input);
```

- `epsilon` 从约 `0.25` 随训练进度线性下降到 `0.05`。
- 最低值限制为 `0.05`，保留少量探索。
- `get_block` 从训练输入流按原顺序读取方块。棋盘顶出后开始下一局，并继续消费同一输入流。
- 输入结束、遇到 `E` 或 `X` 时停止训练；不足请求局数时按实际完成局数报告。

#### 每局的最大步数（第 341 行）

```c
for(int turn=0;turn<1000;turn++){
```

每局最多训练 1000 次落子，避免单局无限运行。达到上限就结束本局并进入下一局。

#### 获取可选动作和 epsilon-greedy 选择（第 342–353 行）

```c
training_action actions[40];
int count=collect_actions(shape,actions);
if(count==0)break;
```

建立候选数组、枚举合法落点。没有合法动作时结束本局。

```c
int selected=0;
if((double)rand()/RAND_MAX<epsilon){
    selected=rand()%count;
}else{
    for(int i=1;i<count;i++)
        if(actions[i].value>actions[selected].value)selected=i;
}
```

- `rand()/RAND_MAX` 生成 `[0,1]` 范围的随机比例。
- 随机比例小于 `epsilon` 时从候选中随机选一个，进行探索。
- 否则扫描候选动作并选择价值最高的动作，进行利用。
- 如果多个候选价值相同，保留先出现的候选。

#### 执行动作和处理消行（第 353–360 行）

```c
training_action action=actions[selected];
int fill[4],full_rows[4],fill_line;
int score=place_block(shape,action.rotation,action.x,action.y,
                      fill,full_rows,&fill_line);
if(score<0)break;
clear_lines(full_rows,fill_line);
```

- 复制选中动作，保存本次选择时的特征。
- `place_block` 真正将方块写入棋盘，并计算消行得分。
- 放置失败时结束本局。
- `clear_lines` 删除已满行并将上方行下移。这一步是实际环境状态转移的一部分；与特征提取时临时放置后只评估、随后撤销不同。

```c
int terminal=action.top_out;
double reward=0.34*score-(terminal?200.0:0.0);
```

- `terminal` 表示本次动作是否达到顶出边界。
- 消行得分乘以 `0.34` 后作为正向奖励。
- 顶出时再减去 `200`，形成终局惩罚。

```c
double next_value=0.0;
int next_shape=get_block(block_input);
```

初始化下一步价值为零，并通过 `get_block(block_input)` 读取序列中的下一个方块。

#### 估算后继动作价值（第 361–377 行）

```c
if(!terminal){
    training_action next_actions[40];
    int next_count=collect_actions(next_shape,next_actions);
    int has_next_action=0;
```

非终局且输入仍有方块时，根据刚更新过的棋盘和输入序列中的下一个方块枚举后继动作。`has_next_action` 用来区分“最大价值确实为 0”和“没有任何非顶出动作”。

```c
    for(int i=0;i<next_count;i++){
        if(!next_actions[i].top_out&&
           (!has_next_action||next_actions[i].value>next_value)){
            next_value=next_actions[i].value;
            has_next_action=1;
        }
    }
```

遍历后继候选，只考虑非顶出动作，取当前模型估值最大的动作，即 `max_a' Q(next_state,a')`。

```c
    if(!has_next_action)reward-=200.0;
    else next_value=gamma*next_value;
}
```

- 若没有可选的非顶出后继动作，额外扣除 `200`，将这种即将失败的局面视为不利状态。
- 否则乘以折扣率 `gamma`，得到折扣后的下一状态价值。

#### 计算 TD 误差并更新权重（第 379–423 行）

```c
double current_value=score_features(action.features);
double target=reward+(terminal?0.0:next_value);
double delta=target-current_value;
```

- `current_value` 使用执行动作时保存的特征和当前权重重新计算当前估值。
- `target` 是 TD 目标：终局只使用即时奖励，非终局再加折扣后的后继价值。
- `delta` 是目标减当前估值，即 TD 误差。

```c
double norm=1.0;
for(int i=0;i<FEATURE_COUNT;i++)
    norm+=action.features[i]*action.features[i];
```

计算特征向量平方范数，并加 `1.0` 避免除零。归一化能降低特征数值较大时单次更新过猛的风险。

程序先计算候选新权重，再统一应用；追踪文件记录每维的更新前后值和变化量。若 TD 值或候选权重不是有限数，训练会报错退出，并且不会保存模型。

逐个更新权重：

```text
wᵢ ← wᵢ + alpha * delta * φᵢ / (1 + Σφⱼ²)
```

这相当于对线性估值进行归一化的半梯度 TD 更新。若某特征在当前动作中为零，它对应的权重不会被这一步直接修改。

```c
if(terminal)break;
shape=next_shape;
```

- 顶出时结束当前训练局。
- 否则将输入序列中的下一个方块设为下一步当前方块，继续训练。

### 2.11 命令行参数

`main` 统一解析运行模式和输入源：

- `--train [局数]` 启用训练；局数可省略（默认 2000），范围为 1 到 1,000,000。
- `-i 方块输入文件` 在训练和正常运行模式中指定方块序列；省略时使用 `stdin`，`-i -` 也表示 `stdin`。
- `--trace CSV文件` 可选，仅用于训练模式，记录每步训练数据。
- 参数顺序不限，例如 `eluose.exe -i .\1.txt --train 1000` 与 `eluose.exe --train 1000 -i .\1.txt` 等价。

训练会加载已有 `q_weights.dat`（不存在则使用源码默认权重），按输入序列训练，并保存更新后的权重。EOF、`E` 或 `X` 会结束输入；实际完成局数可能小于指定上限。

## 3. 编译和运行

在 Windows PowerShell 中，从仓库目录执行：

```powershell
gcc -std=c11 -O2 .\eluose.c -o .\eluose.exe
```

训练模式默认最多重复 2000 局，从标准输入读取方块序列：

```powershell
Get-Content .\1.txt | .\eluose.exe --train
```

指定训练局数，例如 10000 局：

```powershell
.\eluose.exe --train 10000 -i .\1.txt
```

记录每一步特征与权重变化（例如最多训练 1000 局）：

```powershell
.\eluose.exe --train 1000 --trace .\training_trace.csv -i .\1.txt
```

训练依次使用输入中的方块；棋盘顶出时进入下一局并继续读取后续方块。遇到 EOF、`E` 或 `X` 时停止，因此实际完成局数可能少于指定上限。动作探索仍使用 epsilon-greedy 随机策略。`-i` 可用于指定训练或正常模式的输入文件；省略时默认使用 `stdin`，传入 `-i -` 也表示 `stdin`。

CSV 可用 Excel 导入，也可用 Python、R 等工具统计。重点比较 `featureN`、`weight_beforeN`、`contributionN`、`weight_deltaN` 和 `weight_afterN`，同时观察 `td_error` 与 `feature_norm`。例如，若某维 `weight_deltaN` 长期同号且绝对值持续增大，说明该维在持续推动权重漂移；若 `td_error` 很大而多维特征同时非零，需结合每维的 `weight_deltaN` 判断具体贡献。

仓库中的 [`analyze_training.py`](./analyze_training.py) 可将追踪数据汇总并绘制训练得分、权重轨迹、每步权重变化和各特征量级。需要 Python 与 `matplotlib`：

```powershell
python -m pip install matplotlib
python .\analyze_training.py .\training_trace.csv --output .\training_analysis.png
```

训练会在**当前工作目录**读取已有的 `q_weights.dat`（若存在）并保存训练后的模型。想从初始权重重新训练时，应先备份或移走旧模型文件。
如果模型文件中已有 `nan`，程序会拒绝加载。先备份并移走该文件，再开始训练，例如：

```powershell
Move-Item .\q_weights.dat .\q_weights.nan.bak
.\eluose.exe --train 1000 -i .\1.txt --trace .\training_trace.csv
```

追踪文件应使用不同于 `q_weights.dat` 的文件名。

正常模式默认从标准输入读取方块序列，也可通过 `-i` 指定输入文件：

```powershell
Get-Content .\3.txt | .\eluose.exe
.\eluose.exe -i .\3.txt
```

模型文件每行一个浮点权重，共 6 行。源文件中的 `*.txt` 忽略规则不会忽略 `q_weights.dat`。

## 4. 使用与解读注意事项

1. 训练方块从 `stdin` 或 `-i` 指定的输入源读取；遇到 EOF、`E` 或 `X` 会停止训练。
2. `q_weights.dat` 是纯权重文件，不包含随机种子、训练局数或超参数。程序加载后会继续训练；想比较不同实验，需自行备份模型。
3. 训练按输入序列读取方块，但 epsilon-greedy 仍会随机探索落点；追踪文件记录每一步实际训练的方块编号。
4. 训练局数增加不保证策略一定变强。可固定测试方块序列，在训练前后比较总分、存活步数或顶出次数。
5. 特征计算是在方块临时放置后、实际消行清理前进行；下一状态则是在真实消行清理后计算。理解训练行为时要注意这两个棋盘时点不同。
6. 代码将顶出边界固定为 `y + block_height > 14`，并将游戏最高高度也按此规则判负；修改棋盘尺寸或边界规则时，应同步检查特征提取、候选生成、奖励和游戏逻辑。
