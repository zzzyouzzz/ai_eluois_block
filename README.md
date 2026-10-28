# 俄罗斯方块线性 DQN 训练说明

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
| 回合（Episode） | 从空棋盘开始，直到游戏结束或输入序列耗尽的一局训练 |

智能体执行动作后，棋盘发生变化，得到下一个状态。训练的目标不是只挑眼前看起来最好的落点，而是让当前选择也考虑后续可能得到的回报。

### 1.2 DQN 目标与经验回放

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

程序使用一步 DQN 目标，并把转移放入有限大小的经验回放区，随机抽取小批次训练。目标权重每隔一段更新步数才从在线权重复制一次，以降低目标快速漂移。批量更新使用 Huber TD 梯度和特征范数归一化；终局转移不 bootstrap。超参数在源码的 `DQN_*` 常量中设置。

### 1.3 本程序的线性函数近似

棋盘状态组合很多，程序没有为每个状态单独保存一项 Q 值，而是提取 6 个特征，用可训练权重计算动作价值：

```text
Q(s, a) ≈ w₀φ₀ + w₁φ₁ + ... + w₅φ₅
```

代码中的 `features` 是特征向量 `φ`，`value_weights` 是权重 `w`。当前仍是 6 个输入、单个线性输出的动作价值模型；DQN 的经验回放和目标权重机制不改变 `q_weights.dat` 的六个参数格式。

特征描述的是“把当前方块放到某个位置后”的棋盘：

| 下标 | 特征 | 代码含义 |
| --- | --- | --- |
| `0` | 落点高度 | `y + block_height - 1` |
| `1` | 消行特征 | 完成行数乘以当前方块落在这些完成行中的格子数 |
| `2` | 行方向变化 `drow` | 清除满行后的棋盘上，相邻列在同一行的占用状态差异计数 |
| `3` | 列方向变化 `dcol` | 清除满行后的棋盘上，同一列相邻行的占用状态差异计数 |
| `4` | 洞数 `holes` | 清除满行后的棋盘上，某格为空但其上方已出现方块时的计数 |
| `5` | 井深 `well` | 清除满行后的棋盘上，左右两侧被占据而本格为空时累计的井深 |

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
static double score_features(const double value_weights[FEATURE_COUNT],
                             const double features[FEATURE_COUNT]){
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
5. 权重由调用方传入，函数不依赖全局模型状态。
6. 返回该动作的价值估计。通常分数越高，策略越倾向于选择该落点。

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

### 2.4 训练动作结构 `training_action`

```c
typedef struct{
    int valid,rotation,x,y,top_out;
    double features[FEATURE_COUNT];
    double value;
}training_action;
```

结构体保存动作生成函数选出的动作及其评估结果：

- `valid`：是否成功生成可放置的动作。
- `rotation`：旋转角编号。
- `x`, `y`：横坐标和落点纵坐标。
- `top_out`：该落点是否达到程序定义的顶出边界。
- `features`：该动作对应的棋盘特征。
- `value`：根据当前权重计算的动作价值，用于 epsilon-greedy 选择。

### 2.5 生成并选择动作 `generate_action`

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

### 2.6 读取模型 `load_value_model`（第 287–305 行）

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

### 2.7 保存模型 `save_value_model`（第 307–319 行）

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

### 2.8 训练追踪表头 `write_training_trace_header`

`--trace` 为每个环境动作记录一行。前 15 列为：

```text
episode,turn,shape,rotation,x,y,epsilon,score,reward,terminal,current_value,next_value,target,td_error,feature_norm
```

之后每个特征 `i` 都会追加 `feature_i,weight_before_i,contribution_i,weight_delta_i,weight_after_i` 五列。因为训练采用经验回放，一行表示一次实际环境动作；权重变化列记录此动作加入回放后执行的批量更新，不代表单独由该行特征产生的更新。

| 列 | 含义 |
| --- | --- |
| `feature0` | 本次动作的特征值 `φ₀` |
| `weight_before0` | 更新前权重 `w₀` |
| `contribution0` | 该特征对当前估值的贡献 `w₀ × φ₀` |
| `weight_delta0` | 本次批量更新的权重变化 `Δw₀` |
| `weight_after0` | 更新后权重 |

特征 1 到 5 同样记录。`td_error` 是动作写入回放时计算的误差；后续重放会再次更新同一经验，因此 CSV 并非完整的逐次采样批次日志。

`shape` 是方块编号：0=I、1=T、2=O、3=J、4=L、5=S、6=Z。

### 2.9 核心训练函数 `train_value_model`

训练开始时将当前 6 个权重复制到目标权重，并清空最多 50,000 条经验的回放区。每步按 epsilon-greedy 选择动作，执行落子和消行后保存 `features`、奖励、终局标志及下一方块的贪心动作特征。回放区达到 64 条经验后，每步随机抽取 32 条进行更新；每 500 次批量更新同步一次目标权重，学习率为 `0.0001`。

单样本目标为：

```text
target = reward                         (terminal)
target = reward + 0.99 * Q_target(next)  (otherwise)
```

奖励为实际消行分数加每步 `1` 分存活奖励；顶出时再扣除 `300 * exp(-(turn - 1) / 60)`，使越早结束的局受到越重惩罚。有限输入耗尽视为截断终局，不额外施加顶出惩罚。批量更新使用截断在 `[-100, 100]` 的 TD 梯度和特征范数归一化，以降低大 TD 误差导致权重骤变的风险。

训练局数区间的平均得分与平均存活步数每 100 局输出一次。`--test` 会在训练前后将测试文件分成 100 个不重叠片段，对每段运行贪心策略，报告平均得分、存活步数和顶出率；每局最多使用该片段的方块数。不指定 `--train` 时，它也可单独评估当前模型。

### 2.10 命令行参数

`main` 统一解析运行模式和输入源：

- `--train [局数]` 启用训练；局数可省略（默认 2000），范围为 1 到 1,000,000。
- `-i 方块输入文件` 在训练和正常运行模式中指定方块序列；正常模式省略时使用 `stdin`，`-i -` 也表示 `stdin`。
- `--test 测试文件` 训练时比较训练前后的贪心表现；不带 `--train` 时单独评估当前模型。建议测试集与 `-i` 训练集分离。
- `--fresh` 训练时忽略已有 `q_weights.dat`，从源码中的初始 6 个权重开始；适合从旧版训练流程切换到 DQN 时使用。
- `--trace CSV文件` 可选，仅用于训练模式，记录每步训练数据。
- 参数顺序不限，例如 `eluose.exe -i .\1.txt --train 1000` 与 `eluose.exe --train 1000 -i .\1.txt` 等价。

多轮训练必须用 `-i` 指定可重复打开的文件；不指定文件或使用 `-i -` 时只允许训练一轮。每轮训练前棋盘清空，重新打开输入文件并从头读取。输入 EOF、`E` 或 `X` 会结束当前轮，之后下一轮仍从文件开头重新读取。训练会加载已有 `q_weights.dat`（不存在则使用源码默认权重），并保存更新后的权重。

## 3. 编译和运行

在 Windows PowerShell 中，从仓库目录执行：

```powershell
gcc -std=c11 -O2 .\eluose.c -lm -o .\eluose.exe
```

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

只评估当前模型：

```powershell
.\eluose.exe --test .\2.txt
```

记录每一步特征与权重变化（例如最多训练 1000 局）：

```powershell
.\eluose.exe --train 1000 --trace .\training_trace.csv -i .\1.txt
```

每轮训练都会重开 `-i` 指定的文件并从头读取，棋盘顶出或文件结束时本轮结束；训练轮数由 `--train` 指定。动作探索仍使用 epsilon-greedy 随机策略。多轮训练不能使用 `stdin`，因为输入流不能重新打开；单轮训练可省略 `-i` 或使用 `-i -`。使用 `--test` 时，测试策略始终为贪心，不进行探索。

CSV 可用 Excel 导入，也可用 Python、R 等工具统计。重点比较 `featureN`、`weight_beforeN`、`contributionN`、`weight_deltaN` 和 `weight_afterN`，同时观察 `td_error` 与 `feature_norm`。例如，若某维 `weight_deltaN` 长期同号且绝对值持续增大，说明该维在持续推动权重漂移；若 `td_error` 很大而多维特征同时非零，需结合每维的 `weight_deltaN` 判断具体贡献。

仓库中的 [`analyze_training.py`](./analyze_training.py) 可将追踪数据汇总并绘制训练得分、权重轨迹、每步权重变化和各特征量级。需要 Python 与 `matplotlib`：

```powershell
python -m pip install matplotlib
python .\analyze_training.py .\training_trace.csv --output .\training_analysis.png
```

训练默认会在**当前工作目录**读取已有的 `q_weights.dat`（若存在）并保存训练后的模型。切换旧版训练流程时，建议使用 `--fresh` 从源码默认权重开始；若希望继续 DQN 训练，则不要加此参数。训练会覆盖 `q_weights.dat`，实验前请先备份。
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

1. 每轮训练从头读取 `-i` 指定的文件；EOF、`E` 或 `X` 会结束当前轮。多轮训练必须使用文件输入，stdin 仅支持一轮。
2. `q_weights.dat` 是纯权重文件，不包含随机种子、训练局数或超参数。程序加载后会继续训练；想比较不同实验，需自行备份模型。
3. 训练按输入序列读取方块，但 epsilon-greedy 仍会随机探索落点；追踪文件记录每一步实际训练的方块编号。
4. 训练局数增加不保证策略一定变强。可固定测试方块序列，在训练前后比较总分、存活步数或顶出次数。
5. 行变化、洞数和井深特征按消除满行后的棋盘计算；高度和消行特征仍描述当前放置动作。提取候选特征后会恢复原棋盘。
6. 代码将顶出边界固定为 `y + block_height > 14`，并将游戏最高高度也按此规则判负；修改棋盘尺寸或边界规则时，应同步检查特征提取、候选生成、奖励和游戏逻辑。
