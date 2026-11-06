#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <string.h>

/* ==================================================================
   参数配置区
   所有可调参数集中在这里：改完重新编译即可生效。命令行只剩下
   --train / -i 两个"选模式、指文件"的参数。

   组装实验时可以用编译参数临时覆盖运行参数，例如
   gcc -DTRAIN_EPISODES=200 -DTRAINING_TRACE_PATH='"batch.csv"' ...
   （见 run_experiments.ps1），不改源码也能换局数与追踪文件名。
   ================================================================== */

/* ---------- 运行参数 ---------- */

/* 训练局数（命令行不再接受数字） */
#ifndef TRAIN_EPISODES
#define TRAIN_EPISODES 20000
#endif
/* 是否忽略已保存的模型、从源码里的初始值开始训练：1 = 忽略，0 = 沿用
   模型文件。初始值就是上面 value_weights 的 6 个数，选 1 时不读文件。 */
#ifndef TRAIN_FRESH
#define TRAIN_FRESH 0
#endif
/* 随机种子：0 = 用当前时间（每次训练都不一样，默认行为）；
   非 0 = 固定种子，同样参数多次训练得到同一条轨迹。
   sweep_hyperparams.py 给同一组对比实验传同一个种子，把随机噪声消掉，
   差异才真的是参数造成的。 */
#ifndef TRAIN_SEED
#define TRAIN_SEED 0
#endif
/* 模型文件路径（训练结束写这里，平时从这里读） */
#ifndef VALUE_MODEL_PATH
#define VALUE_MODEL_PATH "q_weights.dat"
#endif
/* 每步训练数据写入的追踪 CSV；空字符串 "" 表示不追踪 */
#ifndef TRAINING_TRACE_PATH
#define TRAINING_TRACE_PATH "training_trace.csv"
#endif

/* ---------- DQN 训练参数 ----------
   全部带 #ifndef：不然命令行 -D 会被下面的 #define 覆盖掉（编译器只给一句
   redefined 警告），sweep_hyperparams.py 枚举出来的就是一串一模一样的结果。 */

/* 经验回放区容量（条） */
#ifndef DQN_REPLAY_CAPACITY
#define DQN_REPLAY_CAPACITY 50000
#endif
/* 每次批量更新抽多少条样本 */
#ifndef DQN_BATCH_SIZE
#define DQN_BATCH_SIZE 24
#endif
/* 回放区攒满多少条才开始更新 */
#ifndef DQN_REPLAY_WARMUP
#define DQN_REPLAY_WARMUP 5000
#endif
/* 每多少次批量更新同步一次目标网络 */
#ifndef DQN_TARGET_UPDATE_INTERVAL
#define DQN_TARGET_UPDATE_INTERVAL 500
#endif
/* 折扣因子 */
#ifndef DQN_GAMMA
#define DQN_GAMMA 0.99
#endif
/* 学习率：Adam 下每一步大约就走这么多，和梯度量级无关（可用 -D 覆盖） */
#ifndef DQN_LEARNING_RATE
#define DQN_LEARNING_RATE 0.0001
#endif
/* 奖励缩放：棋盘上的消行分 0/100/300/500/800 乘这个系数才是训练奖励，
   0.01 即奖励变成 0~8，TD 和梯度的量级同步缩小 100 倍。 */
#ifndef REWARD_SCALE
#define REWARD_SCALE 0.01
#endif
/* 奖励整形里三个惩罚项的系数（单位都是"奖励值"，1 = 消一行）：
   顶出一次扣多少、每个洞扣多少、每格堆高扣多少。
   都做成宏是为了能 -D 覆盖：sweep_hyperparams.py 就是靠这个枚举的。 */
#ifndef PENALTY_TOP_OUT
#define PENALTY_TOP_OUT 0.0
#endif
#ifndef PENALTY_HOLE
#define PENALTY_HOLE 0.0
#endif
#ifndef PENALTY_HEIGHT
#define PENALTY_HEIGHT 0.0
#endif
/* Adam 的超参 */
#ifndef ADAM_BETA1
#define ADAM_BETA1 0.9
#endif
#ifndef ADAM_BETA2
#define ADAM_BETA2 0.999
#endif
#ifndef ADAM_EPSILON
#define ADAM_EPSILON 1e-8
#endif
/* 学习率衰减：Adam 的步长不会自己变小，走够几十万步权重就会一路漂移。
   0 = 不衰减（之前的行为）；N = 每走 N 步学习率减半，
   即 alpha_t = alpha / (1 + step/N)。 */
#ifndef DQN_LR_DECAY_STEPS
#define DQN_LR_DECAY_STEPS 0
#endif
/* 差分目标的基线 ρ（平均每步奖励）的学习率：
   ρ ← ρ + DQN_RHO_RATE · 批内平均 TD。
   ρ 复用 value_bias 这个变量存放，不再进入 Q 值本身。 */
#ifndef DQN_RHO_RATE
#define DQN_RHO_RATE 0.01
#endif

/* ---------- 探索（epsilon 退火） ---------- */

/* 每局训练用的 epsilon 从 START 线性降到 END，END 同时是下限。
 * 退火只占前 TRAIN_EPSILON_DECAY_RATIO 这一段（默认 9/10）：
 * 前面一路线性降到最低，最后 1/10 的局停在 END 上不动，让策略在收尾阶段
 * 有一整段纯利用的稳定期，而不是到最后一局才刚落到最低探索率。
 * run_experiments.ps1 把上一批的 END 当成下一批的 START 传进来，于是 epsilon
 * 跨批首尾相接、整体平滑下降，而不是每批都重新从高处探索一遍。 */
#ifndef TRAIN_EPSILON_START
#define TRAIN_EPSILON_START 1
#endif
#ifndef TRAIN_EPSILON_END
#define TRAIN_EPSILON_END 0.05
#endif
/* 退火段占总局数的比例：0.9 = 前 90% 的局降到底，最后 10% 保持在 END。
 * 设成 1.0 就退回到"整段线性、最后一局才落地"的老行为。 */
#ifndef TRAIN_EPSILON_DECAY_RATIO
#define TRAIN_EPSILON_DECAY_RATIO 0.8
#endif

const int block[7][4][4][2] = {
    // I型
    {
        {{0,0}, {0,1}, {0,2}, {0,3}},  // 0°
        {{0,0}, {1,0}, {2,0}, {3,0}},  // 90°
        {{0,0}, {0,1}, {0,2}, {0,3}}, // 180°
        {{0,0}, {1,0}, {2,0}, {3,0}}  // 270°
    },
    // T型
    {
        {{0,0}, {0,1}, {0,2}, {1,1}},
        {{0,0}, {1,0}, {1,1}, {2,0}},
        {{1,0}, {1,1}, {1,2}, {0,1}},
        {{0,1}, {1,0}, {1,1}, {2,1}}
    },
    // O型
    {
        {{0,0}, {0,1}, {1,0}, {1,1}},
        {{0,0}, {0,1}, {1,0}, {1,1}},  
        {{0,0}, {0,1}, {1,0}, {1,1}},
        {{0,0}, {0,1}, {1,0}, {1,1}}

    },
    // J型
    {
        {{0,0}, {0,1}, {0,2}, {1,0}},
        {{0,0}, {1,0}, {2,0}, {2,1}},
        {{1,0}, {1,1}, {1,2}, {0,2}},
        {{0,0}, {0,1}, {1,1}, {2,1}}
    },
    // L型
    {
        {{0,0}, {0,1}, {0,2}, {1,2}},
        {{0,0}, {0,1}, {1,0}, {2,0}},
        {{0,0}, {1,0}, {1,1}, {1,2}},
        {{0,1}, {1,1}, {2,0}, {2,1}}  
    },
    // S型
    {
        {{0,0}, {0,1}, {1,1}, {1,2}},
        {{0,1}, {1,0}, {1,1}, {2,0}},
        {{0,0}, {0,1}, {1,1}, {1,2}},
        {{0,1}, {1,0}, {1,1}, {2,0}}
    },
    // Z型
    {
        {{0,1}, {0,2}, {1,1}, {1,0}},
        {{0,0}, {1,1}, {1,0}, {2,1}},
        {{0,1}, {0,2}, {1,1}, {1,0}},  
        {{0,0}, {1,1}, {1,0}, {2,1}}
    }

};
int board[25][15]={0};
enum { FEATURE_COUNT = 6 };
double value_weights[FEATURE_COUNT] = {-45.0,34.0,-32.0,-93.0,-79.0,-34.0};
/* 常数偏置：Q = w·φ + b。没有它，整体几百上千的偏移只能靠特征硬凑。 */
double value_bias = 0.0;

/* 价值模型：6 个权重 + 1 个偏置（目标网络各一份） */
static double target_weights[FEATURE_COUNT];
static double target_bias = 0.0;

/* Adam 的状态：一阶矩、二阶矩和步数。
   Adam 对每个参数单独自适应步长，所以不需要再按 1+Σφ² 除梯度，
   特征之间量级差十倍也不会把某个权重按死。 */
static double adam_m[FEATURE_COUNT];
static double adam_v[FEATURE_COUNT];
static long long adam_step = 0;

static void reset_adam_state(void){
    memset(adam_m,0,sizeof(adam_m));
    memset(adam_v,0,sizeof(adam_v));
    adam_step = 0;
}

int block_r[7]={2,4,1,4,4,2,2};//每个方块的旋转数
int block_w[7][4]={
    {3,0,3,0},
    {2,1,2,1},
    {1,1,1,1},
    {2,1,2,1},
    {2,1,2,1},
    {2,1,2,1},
    {2,1,2,1}
};//每个方块的宽度
int block_h[7][4]={
    {0,3,0,3},
    {1,2,1,2},
    {1,1,1,1},
    {1,2,1,2},
    {1,2,1,2},
    {1,2,1,2},
    {1,2,1,2}
};
int can_place(int shape,int rotation,int x,int y);

static void reset_value_model(void){
    const double initial_weights[FEATURE_COUNT]=
        {0,0,0,0,0,0};
    memcpy(value_weights,initial_weights,sizeof(value_weights));
    value_bias=0.0;
    reset_adam_state();
}

int place_block(int shape,int rotation,int x,int y,int fill[4],
                int full_rows[4],int *fill_line){
    for(int j=0;j<4;j++){
        fill[j]=0;
        full_rows[j]=-1;
    }
    *fill_line=0;
    if(shape<0||shape>=7||rotation<0||rotation>=4||
       !can_place(shape,rotation,x,y))return -1;

    int occupied_rows[4]={0};
    for(int i=0;i<4;i++){
        int dy=block[shape][rotation][i][0];
        int dx=block[shape][rotation][i][1];
        board[y+dy][x+dx]=1;
        occupied_rows[dy]=1;
    }

    for(int j=0;j<4;j++){
        if(!occupied_rows[j])continue;
        for(int column=0;column<10;column++)
            fill[j]+=board[y+j][column];
        if(fill[j]==10)
            full_rows[(*fill_line)++]=y+j;
    }

    switch(*fill_line){//根据消除的行数计算得分
        case 1: return 100;
        case 2: return 300;
        case 3: return 500;
        case 4: return 800;
        default: return 0;
    }
}

void clear_lines(const int full_rows[4],int fill_line){
    int line_index=0;
    int write_row=0;
    for(int read_row=0;read_row<20;read_row++){
        if(line_index<fill_line&&read_row==full_rows[line_index]){
            line_index++;
            continue;
        }
        if(write_row!=read_row){
            for(int column=0;column<10;column++)
                board[write_row][column]=board[read_row][column];
        }
        write_row++;
    }
    while(write_row<20){
        for(int column=0;column<10;column++)
            board[write_row][column]=0;
        write_row++;
    }
}

void remove_block(int shape,int rotation,int x,int y){
    for(int i=0;i<4;i++){
        int dy=block[shape][rotation][i][0];
        int dx=block[shape][rotation][i][1];
        board[y+dy][x+dx]=0;
    }
}
int can_place(int shape,int rotation,int x,int y){//判断是否可以放置
    if(x<0||x>=10||y<0||y>=20)return 0;
    for(int i=0;i<4;i++){
        int dy=block[shape][rotation][i][0];
        int dx=block[shape][rotation][i][1];
        if(dx+x>=10||dy+y>=20||board[y+dy][x+dx]==1){
            return 0;
        }   
    }
    return 1;
}
int simulate_fall(int shape,int rotation,int x){//模拟下落
    int y=16;
    //if(!can_place(shape,rotation,x,y))return -1;
    while(y>=0&&can_place(shape,rotation,x,y)){
        y--;
    }
    return y+1;
}

void evaluate_board(int* drow,int* dcol,int* holes,int* well){//评估棋盘
    for(int j=0;j<10;j++){
        int nwell=0;
        int istop=0;
        for(int i=19;i>=0;i--){
            (*drow)+=(board[i][j]!=board[i][j-1>=0?j-1:0]);
            (*dcol)+=(board[i][j]!=board[i-1>=0?i-1:0][j]);
            if(board[i][j]==0){
                int right=j+1<10?board[i][j+1]:1;
                int left=j-1>=0?board[i][j-1]:1;
                if(left&&right){
                    nwell++;
                    (*well)+=nwell;
                }
                (*holes)+=istop;
            }
            else{
                nwell=0; 
                istop=1;
            }
        }
    }
}
/* 一个候选：局面编码 + 落法 + 价值。
   价值不提前算，generate_action 里按当时的权重现算。 */
typedef struct{
    double feature[FEATURE_COUNT];
    int shape;
    int r;                      /* 旋转 */
    int x;
    int y;
    int top_out;
    double value;
}state_and_action;

static double score_features(const double weights[FEATURE_COUNT],double bias,
                             const double features[FEATURE_COUNT]){
    double score=bias;
    for(int i=0;i<FEATURE_COUNT;i++)score+=weights[i]*features[i];
    return score;
}
static int extract_action_features(int shape,int rotation,int x,int y,int block_height,
                                   double features[FEATURE_COUNT]){
    for(int i=0;i<FEATURE_COUNT;i++)features[i]=0.0;
    int saved_board[25][15];
    memcpy(saved_board,board,sizeof(board));
    int fill[4],full_rows[4],fill_line;
    if(place_block(shape,rotation,x,y,fill,full_rows,&fill_line)<0)return 0;
    int block_cells_in_full_rows=0;
    for(int i=0;i<4;i++){
        if(fill[i]==10){
            for(int cell=0;cell<4;cell++)
                block_cells_in_full_rows+=
                    block[shape][rotation][cell][0]==i;
        }
    }
    clear_lines(full_rows,fill_line);
    int drow=0,dcol=0,holes=0,well=0;
    evaluate_board(&drow,&dcol,&holes,&well);
    features[0]=y+block_height-1;
    features[1]=fill_line*block_cells_in_full_rows;
    features[2]=drow;
    features[3]=dcol;
    features[4]=holes;
    features[5]=well;
    memcpy(board,saved_board,sizeof(board));
    return 1;
}
int get_block(FILE *fp){
    char block;
    while((block=fgetc(fp))!='I'&&block!='T'&&block!='O'&&block!='J'&&block!='L'&&block!='S'&&block!='Z'&&block!='E'&&block!='X'){
        if(block==EOF)return -666;
        continue;
    }
    switch(block){
        case 'I': return 0;
        case 'T': return 1;
        case 'O': return 2; 
        case 'J': return 3;
        case 'L': return 4;
        case 'S': return 5;
        case 'Z': return 6;
        case 'E': return 7;
        default: return -1;
    }
}
#define MAX_ACTIONS 40

/* 从候选里挑一个动作。它只做两件事：
   1) 用 score_features 把每个候选的 value 现算出来；
   2) 以 epsilon 概率随机，否则在非顶出的候选里取 value 最大。
   候选（feature/r/x/y/top_out）由 environment_step 负责填好，这里不生成。
   返回选中的那一个（拷贝）；没有候选时返回 r=-1 的无效动作。 */
static state_and_action generate_action(state_and_action actions[MAX_ACTIONS],
                                        int count,double epsilon,
                                        const double weights[FEATURE_COUNT],
                                        double bias){
    state_and_action invalid={0};
    invalid.r=-1;
    if(count<=0||count>MAX_ACTIONS)return invalid;

    for(int i=0;i<count;i++)
        actions[i].value=score_features(weights,bias,actions[i].feature);

    int selected=-1;
    if((double)rand()/((double)RAND_MAX+1.0)<epsilon){
        selected=rand()%count;
    }else{
        for(int i=0;i<count;i++){
            if(!actions[i].top_out&&
               (selected<0||actions[i].value>actions[selected].value))
                selected=i;
        }
        if(selected<0)selected=0;
    }
    return actions[selected];
}


/* 环境交互。
   action==NULL：不落子，只读入第一个方块，把它的候选写进 next[]；
   否则：落子、消行、读入下一个方块，把新候选直接覆盖到 next[]
         （调用方手里的 action 是副本，所以可以安全覆盖）。
   返回新候选个数（0 表示这一局结束），*reward 是这一步的奖励。
   *board_score（可为 NULL）回传这一步的棋盘得分 0/100/300/500/800，
   和奖励分开：奖励里有惩罚项，评"打得好不好"必须看纯棋盘分。 */
static int environment_step(const state_and_action *action,FILE *input,
                            state_and_action next[MAX_ACTIONS],double *reward,
                            int *board_score){
    if(reward)*reward=0.0;
    if(board_score)*board_score=0;
    if(action){
        int fill[4],full_rows[4],fill_line;
        int score=place_block(action->shape,action->r,action->x,action->y,
                              fill,full_rows,&fill_line);
        if(score<0)return -1;
        clear_lines(full_rows,fill_line);
        if(board_score)*board_score=score;
        if(reward)*reward=(double)score*REWARD_SCALE
            -(action->top_out?PENALTY_TOP_OUT:0.0)
            -PENALTY_HOLE*(double)action->feature[4]
            -PENALTY_HEIGHT*(double)action->feature[0];
    }
    /* 枚举下一个方块的全部落点，直接写进 next[]：只填 feature 与落法，不算价值 */
    int shape=get_block(input);
    memset(next,0,sizeof(state_and_action)*MAX_ACTIONS);
    if(shape<0||shape>=7)return 0;
    int count=0;
    for(int rotation=0;rotation<block_r[shape];rotation++){
        for(int x=0;x<10;x++){
            if(!can_place(shape,rotation,x,16))continue;
            int y=simulate_fall(shape,rotation,x);
            if(y<0||y>=20)continue;
            state_and_action *candidate=&next[count];
            if(!extract_action_features(shape,rotation,x,y,
                                        block_h[shape][rotation],
                                        candidate->feature))
                continue;
            candidate->shape=shape;
            candidate->r=rotation;
            candidate->x=x;
            candidate->y=y;
            candidate->top_out=y+block_h[shape][rotation]>14;
            count++;
        }
    }
    return count;
}
static int load_value_model(const char *path){
    FILE *fp=fopen(path,"r");
    if(!fp){
        if(errno==ENOENT)return 0;
        fprintf(stderr,"无法读取价值模型 %s\n",path);
        return -1;
    }
    double loaded[FEATURE_COUNT];
    double loaded_bias=0.0;
    char tag[64];
    int ok=0;
    if(fscanf(fp,"%63s",tag)!=1){
        fprintf(stderr,"价值模型格式错误：%s\n",path);
        fclose(fp);
        return -1;
    }
    if(strcmp(tag,"mlp")==0){
        /* 旧版带隐藏层的文件："mlp 16" 之后就是 6 个主干权重，其余忽略 */
        int size=0;
        if(fscanf(fp,"%d",&size)!=1){
            fprintf(stderr,"价值模型格式错误：%s\n",path);
            fclose(fp);
            return -1;
        }
        ok=1;
        for(int i=0;i<FEATURE_COUNT&&ok;i++)
            ok=fscanf(fp,"%lf",&loaded[i])==1&&isfinite(loaded[i]);
        /* 旧文件没有偏置，按 0 处理 */
    }else{
        ok=sscanf(tag,"%lf",&loaded[0])==1&&isfinite(loaded[0]);
        for(int i=1;i<FEATURE_COUNT&&ok;i++)
            ok=fscanf(fp,"%lf",&loaded[i])==1&&isfinite(loaded[i]);
        double extra=0.0;
        if(ok&&fscanf(fp,"%lf",&extra)==1&&isfinite(extra))loaded_bias=extra;
    }
    fclose(fp);
    if(!ok){
        fprintf(stderr,"价值模型格式错误：%s\n",path);
        return -1;
    }
    for(int i=0;i<FEATURE_COUNT;i++)value_weights[i]=loaded[i];
    value_bias=loaded_bias;
    return 1;
}

static int save_value_model(const char *path){
    FILE *fp=fopen(path,"w");
    if(!fp){
        fprintf(stderr,"无法保存价值模型 %s\n",path);
        return 0;
    }
    for(int i=0;i<FEATURE_COUNT;i++)fprintf(fp,"%.12g\n",value_weights[i]);
    fprintf(fp,"%.12g\n",value_bias);
    if(fclose(fp)!=0){
        fprintf(stderr,"保存价值模型失败：%s\n",path);
        return 0;
    }
    return 1;
}

/* 训练超参已统一移到文件开头的「参数配置区」 */

typedef struct{
    state_and_action action;                 /* 这一步选中的落法 */
    int next_count;                          /* 下一状态的候选数，0 = 终局 */
    state_and_action next[MAX_ACTIONS];
    double reward;                           /* 训练奖励（含惩罚项，已缩放） */
    int board_score;                         /* 纯棋盘分 0/100/300/500/800 */
}replay_transition;


static replay_transition replay_buffer[DQN_REPLAY_CAPACITY];
static int replay_size;
static int replay_next;
static unsigned long long dqn_update_count;

static int random_below(int upper_bound){
    unsigned long long random_value=
        (unsigned long long)rand()*((unsigned long long)RAND_MAX+1U)+rand();
    return (int)(random_value%(unsigned int)upper_bound);
}

/* batch 里只放指针，直接指向回放槽位——不再把 3.2 KB 的记录一条条拷进来 */
static int update_dqn_batch(const replay_transition *batch[DQN_BATCH_SIZE],
                            int batch_size,double alpha,double gamma){
    if(batch_size<1||batch_size>DQN_BATCH_SIZE){
        fprintf(stderr,"DQN 批次大小无效\n");
        return 0;
    }

    double gradients[FEATURE_COUNT]={0.0};
    double td_sum=0.0;
    for(int sample=0;sample<batch_size;sample++){
        const replay_transition *transition=batch[sample];
        /* 差分目标：V(s) = w·φ(s)，ρ 放在 value_bias 里，
           td = (r − ρ) + γ·V(s') − V(s)，平均 TD 用来更新 ρ */
        double current_value=score_features(value_weights,value_bias,
                                            transition->action.feature);
        double next_value=0.0;
    if(transition->next_count>0){
        state_and_action next_best=generate_action((state_and_action *)transition->next,
                                                   transition->next_count,0.0,
                                                   target_weights,target_bias);
        if(next_best.r>=0&&!next_best.top_out)next_value=next_best.value;
    }
        double target=transition->reward+gamma*next_value;
        double td_error=target-current_value;//td_error=reward+gamma*(next_best.value+target_bias)-(current_value+value_bias)没有任何问题
        td_sum+=td_error;
        if(!isfinite(current_value)||!isfinite(next_value)||
           !isfinite(target)||!isfinite(td_error)){
            fprintf(stderr,"DQN 训练数值异常\n");
            return 0;
        }

        /* Adam 自己会按参数量纲缩放，这里不再除 1+Σφ² */
        for(int i=0;i<FEATURE_COUNT;i++)
            gradients[i]+=td_error*transition->action.feature[i];
    }

    /* 一个批次走一步 Adam：矩估计 + 偏差修正 + 逐参数自适应步长 */
    adam_step++;
    double step_alpha=alpha;
#if DQN_LR_DECAY_STEPS > 0
    step_alpha=alpha/(1.0+(double)adam_step/(double)DQN_LR_DECAY_STEPS);
#endif
    double correction1=1.0-pow(ADAM_BETA1,(double)adam_step);
    double correction2=1.0-pow(ADAM_BETA2,(double)adam_step);
    double after[FEATURE_COUNT];
    for(int i=0;i<FEATURE_COUNT;i++){
        double gradient=gradients[i]/(double)batch_size;
        adam_m[i]=ADAM_BETA1*adam_m[i]+(1.0-ADAM_BETA1)*gradient;
        adam_v[i]=ADAM_BETA2*adam_v[i]+(1.0-ADAM_BETA2)*gradient*gradient;
        double m_hat=adam_m[i]/correction1;
        double v_hat=adam_v[i]/correction2;
        after[i]=value_weights[i]+step_alpha*m_hat/(sqrt(v_hat)+ADAM_EPSILON);
        if(!isfinite(after[i])){
            fprintf(stderr,"DQN 权重数值异常：feature=%d\n",i);
            return 0;
        }
    }
    /* ρ 按平均 TD 做一步 Robbins-Monro：td>0 说明 ρ 偏小，往上抬 */
    double rho_step=DQN_RHO_RATE*td_sum/(double)batch_size;
    if(!isfinite(rho_step)){
        fprintf(stderr,"DQN 基线数值异常\n");
        return 0;
    }
    value_bias+=rho_step;

    for(int i=0;i<FEATURE_COUNT;i++)value_weights[i]=after[i];
    return 1;
}

/* 占一个回放槽位：返回可写的指针，游标与计数一起推进。
   调用方直接把这一条转移写进槽里，省掉一次整块拷贝。 */
static replay_transition *replay_reserve(void){
    replay_transition *slot=&replay_buffer[replay_next];
    replay_next=(replay_next+1)%DQN_REPLAY_CAPACITY;
    if(replay_size<DQN_REPLAY_CAPACITY)replay_size++;
    return slot;
}


/* ==================================================================
   数据追踪模块：与训练逻辑解耦，只读训练状态、只写训练 CSV
   ------------------------------------------------------------------
   训练代码里只有下面几处插桩（每处一行，不保留任何局部变量，
   所有"为了写 CSV 才需要"的计算都锁在本模块内部）：
     main                : trace_attach() + trace_write_training_header()
     train_value_episode : trace_begin_step(...) / trace_end_step()
   把本注释到"数据追踪模块结束"之间的代码整段删除，再删掉上述调用行，
   训练的行为、输出、随机数序列完全不变。
   ================================================================== */

static FILE *trace_training_file;

/* 相对 TD 误差的分母下限，单位是奖励（1 = 消一行）。
   开局权重全是 0，V(s') 恰好等于 0，|TD|/|V| 会炸成无穷；
   把分母钳在这条地板上，只有最开始那一两步会碰到，之后一律是真比值。 */
#define TRACE_REL_TD_FLOOR 1e-3

static void trace_attach(FILE *training){
    trace_training_file=training;
}

static int trace_write_training_header(void){
    FILE *trace=trace_training_file;
    if(!trace)return 1;
    if(fprintf(trace,
               "episode,turn,shape,rotation,x,y,epsilon,score,reward,terminal,"
               "current_value,next_value,target,td_error,feature_norm,board_cells")<0)return 0;
    for(int i=0;i<FEATURE_COUNT;i++){
        if(fprintf(trace,",feature%d,weight_before%d,contribution%d,weight_delta%d,weight_after%d",
                   i,i,i,i,i)<0)return 0;
    }
    if(fprintf(trace,",bias_before,bias_after,rel_td_error")<0)return 0;
    return fprintf(trace,"\n")>=0;
}

static int trace_count_board_cells(void){
    int cells=0;
    for(int row=0;row<20;row++)
        for(int column=0;column<10;column++)
            cells+=board[row][column]!=0;
    return cells;
}

/* 一步训练行所需的全部快照，begin 填好、end 写出 */
typedef struct{
    int active;
    int episode;
    unsigned long long turn;
    int shape;
    state_and_action action;
    double epsilon;
    int score;
    double reward;
    int terminal;
    double current_value;
    double next_value;
    double target;
    double td_error;
    double rel_td;
    double norm;
    int board_cells;
    double before[FEATURE_COUNT];
    double bias_before;
}trace_step_state;

static trace_step_state trace_step;

static int trace_begin_step(int episode,unsigned long long turn,
                            const state_and_action *action,double epsilon,
                            const replay_transition *transition){
    trace_step.active=0;
    if(!trace_training_file)return 1;

    /* 要的是棋盘上的原始消行分（0/100/300/500/800）。不能拿 reward 除回去：
       奖励里还有顶出/洞/高度三个惩罚项，除出来的分数会被惩罚污染成负数 */
    int score=transition->board_score;

    /* 与 update_dqn_batch 同一套差分口径：V = w·φ，ρ = value_bias */
    double current_value=action->value;
    double next_value=0.0;
    if(transition->next_count>0){
        state_and_action next_best=generate_action((state_and_action *)transition->next,
                                                   transition->next_count,0.0,
                                                   target_weights,target_bias);
        if(next_best.r>=0&&!next_best.top_out)next_value=next_best.value;
    }
    double target=transition->reward+
        (transition->next_count>0?DQN_GAMMA*next_value:0.0);
    double td_error=target-current_value;
    double norm=1.0;
    for(int i=0;i<FEATURE_COUNT;i++)
        norm+=action->feature[i]*action->feature[i];
    if(!isfinite(current_value)||!isfinite(next_value)||
       !isfinite(target)||!isfinite(td_error)||!isfinite(norm)){
        fprintf(stderr,"训练数值异常：episode=%d turn=%llu\n",episode,turn);
        return 0;
    }

    /* 判断训练是否在发散要看相对误差：|TD| 的绝对量会随价值尺度 V 一起变大，
        只有 |TD|/|V| 这种无量纲比值才是"学得好不好"的证据。
        分母只在地板处钳位（V 恰好为 0 的最初一两步），不是给每步加常数——
        加法地板会把早期的小 V 稀释成 1.0，比值被压小、趋势直接反过来。 */
    double rel_td_denom=fabs(current_value);
    if(rel_td_denom<TRACE_REL_TD_FLOOR)rel_td_denom=TRACE_REL_TD_FLOOR;
    double rel_td=fabs(td_error)/rel_td_denom;

    trace_step.active=1;
    trace_step.episode=episode;
    trace_step.turn=turn;
    trace_step.shape=action->shape;
    trace_step.action=*action;
    trace_step.epsilon=epsilon;
    trace_step.score=score;
    trace_step.reward=transition->reward;
    trace_step.terminal=transition->next_count<=0;
    trace_step.current_value=current_value;
    trace_step.next_value=next_value;
    trace_step.target=target;
    trace_step.td_error=td_error;
    trace_step.rel_td=rel_td;
    trace_step.norm=norm;
    trace_step.board_cells=trace_count_board_cells();
    memcpy(trace_step.before,value_weights,sizeof(trace_step.before));
    trace_step.bias_before=value_bias;
    return 1;
}

static int trace_end_step(void){
    if(!trace_training_file||!trace_step.active)return 1;
    trace_step.active=0;
    if(fprintf(trace_training_file,
               "%d,%llu,%d,%d,%d,%d,%.17g,%d,%.17g,%d,"
               "%.17g,%.17g,%.17g,%.17g,%.17g,%d",
               trace_step.episode,trace_step.turn,trace_step.shape,
               trace_step.action.r,trace_step.action.x,
               trace_step.action.y,trace_step.epsilon,trace_step.score,
               trace_step.reward,trace_step.terminal,trace_step.current_value,
               trace_step.next_value,trace_step.target,trace_step.td_error,
               trace_step.norm,trace_step.board_cells)<0){
        fprintf(stderr,"写入训练追踪文件失败\n");
        return 0;
    }
    for(int i=0;i<FEATURE_COUNT;i++){
        double delta=value_weights[i]-trace_step.before[i];
        if(fprintf(trace_training_file,",%.17g,%.17g,%.17g,%.17g,%.17g",
                   trace_step.action.feature[i],trace_step.before[i],
                   trace_step.before[i]*trace_step.action.feature[i],
                   delta,value_weights[i])<0){
            fprintf(stderr,"写入训练追踪文件失败\n");
            return 0;
        }
    }
    if(fprintf(trace_training_file,",%.17g,%.17g,%.17g",
               trace_step.bias_before,value_bias,trace_step.rel_td)<0){
        fprintf(stderr,"写入训练追踪文件失败\n");
        return 0;
    }
    if(fprintf(trace_training_file,"\n")<0){
        fprintf(stderr,"写入训练追踪文件失败\n");
        return 0;
    }
    return 1;
}

/* ------------------------ 数据追踪模块结束 ------------------------ */

/* 这一条转移已经躺在回放区里（train_value_episode 用 replay_reserve 拿的槽），
   这里只负责抽样更新与目标网络同步 */
static int train_on_transition(void){
    if(replay_size>=DQN_REPLAY_WARMUP){
        const replay_transition *batch[DQN_BATCH_SIZE];
        for(int i=0;i<DQN_BATCH_SIZE;i++)
            batch[i]=&replay_buffer[random_below(replay_size)];
        if(!update_dqn_batch(batch,DQN_BATCH_SIZE,DQN_LEARNING_RATE,
                             DQN_GAMMA))
            return 0;
        dqn_update_count++;
        if(dqn_update_count%DQN_TARGET_UPDATE_INTERVAL==0){
            memcpy(target_weights,value_weights,sizeof(target_weights));
            target_bias=value_bias;
        }
    }
    return 1;
}

static int train_value_episode(FILE *block_input,int episode,
                               double epsilon,int *has_training_data){
    memset(board,0,sizeof(board));
    *has_training_data=0;

    /* 环境先吐出第一个方块的候选：还没有落子，所以没有奖励 */
    state_and_action first_candidates[MAX_ACTIONS];
    state_and_action *candidates=first_candidates;
    double reward=0.0;
    int count=environment_step(NULL,block_input,candidates,&reward,NULL);
    if(count<=0)return 1;               /* 输入为空：这一局没有可训练的数据 */

    for(unsigned long long turn=1;;turn++){
        /* 到这里才真正算 Q，并把选中的落法复制出来 */
        state_and_action action=generate_action(candidates,count,epsilon,
                                                value_weights,value_bias);
        if(action.r<0){
            fprintf(stderr,"无法生成训练动作：episode=%d turn=%llu\n",
                    episode,turn);
            return 0;
        }

        /* 先占一个回放槽，再让环境把下一个方块的候选直接写进槽里的 next[]：
           全程没有“候选数组 → transition → 回放”的整块拷贝 */
        replay_transition *transition=replay_reserve();
        transition->action=action;
        transition->reward=reward;
        transition->board_score=0;
        count=environment_step(&action,block_input,transition->next,&reward,
                               &transition->board_score);
        if(count<0){
            fprintf(stderr,"训练动作无法放置：episode=%d turn=%llu\n",
                    episode,turn);
            return 0;
        }
        transition->reward=reward;
        if(action.top_out)count=0;      /* 这一步会顶出：按终局处理 */
        transition->next_count=count;
        candidates=transition->next;    /* 下一轮直接在这份候选上挑 */

        trace_begin_step(episode,turn,&action,epsilon,transition);
        if(!train_on_transition())
            return 0;
        trace_end_step();
        *has_training_data=1;
        if(count==0)break;              /* 候选为空 = 这一局结束 */
    }
    return 1;
}
static int train_value_model(int episodes,const char *input_path,
                             int *completed_episodes){
    srand(TRAIN_SEED?(unsigned int)TRAIN_SEED:(unsigned int)time(NULL));
    memcpy(target_weights,value_weights,sizeof(target_weights));
    target_bias=value_bias;
    replay_size=0;
    replay_next=0;
    dqn_update_count=0;
    *completed_episodes=0;

    /* 方块流只打开一次。以前是每局 fopen/fclose，等于每局都从文件第 1 个
       方块重新开始，100 万方块的输入每局只用到开头几十个；现在句柄跨局
       持有，上一局停在哪、下一局就从哪接着读，读穿整份文件才回绕。
       读取接口没动：get_block / environment_step 仍然只收一个 FILE *。 */
    FILE *block_input=stdin;
    int owns_input=0;
    if(input_path&&strcmp(input_path,"-")!=0){
        block_input=fopen(input_path,"r");
        if(!block_input){
            fprintf(stderr,"无法打开训练输入文件 %s\n",input_path);
            return 0;
        }
        owns_input=1;
    }

    int wraps=0,success=1;
    for(int episode=0;episode<episodes;episode++){
        /* 上一局把文件读到 EOF 才回绕。文本流只允许 seek 到 0 这类魔法值，
           不能拿 ftell 返回的偏移去 fseek（Windows 上还带 CRLF 转换）。 */
        if(owns_input&&feof(block_input)){
            if(fseek(block_input,0,SEEK_SET)!=0){
                fprintf(stderr,"无法回绕训练输入文件 %s\n",input_path);
                success=0;
                break;
            }
            clearerr(block_input);
            wraps++;
        }

        /* 前 TRAIN_EPSILON_DECAY_RATIO（默认 9/10）的局线性降到 END，
           剩下 1/10 的局钳在 END 上：收尾有一段纯利用的稳定期。
           上一批的 END 就是下一批的 START，多批次实验里 epsilon 仍首尾相接。 */
        double epsilon=TRAIN_EPSILON_START;
        if(episodes>1){
            double decay_episodes=(double)(episodes-1)*TRAIN_EPSILON_DECAY_RATIO;
            if(decay_episodes<1.0)decay_episodes=1.0;
            double progress=(double)episode/decay_episodes;
            if(progress>1.0)progress=1.0;
            epsilon+=(TRAIN_EPSILON_END-TRAIN_EPSILON_START)*progress;
        }
        if(epsilon<TRAIN_EPSILON_END)epsilon=TRAIN_EPSILON_END;

        int has_training_data=0;
        if(!train_value_episode(block_input,episode+1,epsilon,
                                &has_training_data)){
            success=0;
            break;
        }
        /* get_block 把 EOF 和读错误都报成 -666，只能在这一层区分：
           feof 为假而 ferror 为真就是真读失败，别当成"输入读完了"。 */
        if(owns_input&&ferror(block_input)){
            fprintf(stderr,"读取训练输入文件失败：%s\n",input_path);
            success=0;
            break;
        }
        if(!has_training_data)break;
        (*completed_episodes)++;
    }

    if(owns_input&&fclose(block_input)!=0){
        fprintf(stderr,"关闭训练输入文件失败：%s\n",input_path);
        return 0;
    }
    if(success&&wraps>0)
        printf("训练输入顺序读完 %d 遍（回绕 %d 次）\n",wraps+1,wraps);
    return success;
}

static int run_game(FILE *input){
    clock_t start=clock();
    int score=0;

    state_and_action candidates[MAX_ACTIONS];
    double reward=0.0;
    int step_score=0;
    int count=environment_step(NULL,input,candidates,&reward,&step_score);

    while(count>0){
        int best=-1;
        for(int i=0;i<count;i++){
            if(candidates[i].top_out)continue;
            candidates[i].value=score_features(value_weights,value_bias,
                                               candidates[i].feature);
            if(best<0||candidates[i].value>candidates[best].value)best=i;
        }
        if(best<0){                      /* 没有安全落点：顶出 */
            printf("You lose!\n");
            for(int row=19;row>=0;row--){
                for(int column=0;column<10;column++)
                    printf("%c"," O"[board[row][column]]);
                printf("\n");
            }
            break;
        }

        state_and_action action=candidates[best];
        count=environment_step(&action,input,candidates,&reward,&step_score);
        if(count<0){
            fprintf(stderr,"无法放置方块\n");
            return 1;
        }
        /* 打印的是纯棋盘得分（消行分之和），不掺奖励里的惩罚项：
           惩罚系数一改，掺进去的分数就没法跨实验比了 */
        score+=step_score;
        printf("%d %d\n%d\n",action.r,action.x,score);
        fflush(stdout);

        if(action.top_out||count==0){
            printf("You lose!\n");
            for(int row=19;row>=0;row--){
                for(int column=0;column<10;column++)
                    printf("%c"," O"[board[row][column]]);
                printf("\n");
            }
            break;
        }
    }
    printf("耗时：%ld ms\n",(long)(clock()-start));
    return 0;
}
int main(int argc,char **argv){
    int training=0;
    const int fresh_model=TRAIN_FRESH;
    const char *input_path=NULL;
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i],"--train")==0){
            if(training){
                fprintf(stderr,"训练模式参数重复\n");
                return 1;
            }
            training=1;
        }else if(strcmp(argv[i],"-i")==0){
            if(input_path||i+1>=argc||argv[i+1][0]=='\0'||
               (argv[i+1][0]=='-'&&strcmp(argv[i+1],"-")!=0)){
                fprintf(stderr,"-i 后必须指定输入文件，且只能指定一次\n");
                return 1;
            }
            input_path=argv[++i];
        }else{
            fprintf(stderr,
                    "用法：%s --train -i 训练输入文件\n"
                    "      %s -i 方块输入文件\n"
                    "训练局数、是否从初始权重开始、追踪文件等参数"
                    "已定义在源码开头的「参数配置区」。\n",
                    argv[0],argv[0]);
            return 1;
        }
    }
    if(TRAINING_TRACE_PATH[0]&&
       strcmp(TRAINING_TRACE_PATH,VALUE_MODEL_PATH)==0){
        fprintf(stderr,"训练追踪文件不能指向模型文件 %s\n",VALUE_MODEL_PATH);
        return 1;
    }
    if(training&&TRAIN_EPISODES>1&&
       (!input_path||strcmp(input_path,"-")==0)){
        fprintf(stderr,"多轮训练必须使用 -i 指定可重复读取的输入文件\n");
        return 1;
    }

    FILE *input=stdin;
    if(!training&&input_path&&strcmp(input_path,"-")!=0){
        input=fopen(input_path,"r");
        if(!input){
            fprintf(stderr,"无法打开方块输入文件 %s\n",input_path);
            return 1;
        }
    }
    if(fresh_model){
        reset_value_model();
        printf("从源码里的初始参数开始（与 %s 相同）\n",
               VALUE_MODEL_PATH);
    }else{
        int loaded=load_value_model(VALUE_MODEL_PATH);
        if(loaded<0){
            if(input!=stdin)fclose(input);
            return 1;
        }
        if(loaded==0){
            printf("找不到模型文件 %s，从源码里的初始参数开始\n",VALUE_MODEL_PATH);
        }
    }

    if(training){
        FILE *trace=NULL;
        if(TRAINING_TRACE_PATH[0]){
            trace=fopen(TRAINING_TRACE_PATH,"w");
            if(!trace){
                fprintf(stderr,"无法创建训练追踪文件 %s\n",TRAINING_TRACE_PATH);
                return 1;
            }
        }
        trace_attach(trace);
        if(!trace_write_training_header()){
            fprintf(stderr,"无法写入追踪文件表头：%s\n",
                    TRAINING_TRACE_PATH[0]?TRAINING_TRACE_PATH:"（未启用）");
            if(trace)fclose(trace);
            return 1;
        }
        int completed_episodes=0;
        if(!train_value_model(TRAIN_EPISODES,input_path,&completed_episodes)){
            if(trace)fclose(trace);
            return 1;
        }
        if(trace&&fclose(trace)!=0){
            fprintf(stderr,"关闭训练追踪文件失败：%s\n",TRAINING_TRACE_PATH);
            return 1;
        }
        if(completed_episodes==0){
            fprintf(stderr,"训练输入中没有可用方块\n");
            return 1;
        }
        if(!save_value_model(VALUE_MODEL_PATH))return 1;
        printf("训练完成：%d 局，模型已保存到 %s\n",completed_episodes,
               VALUE_MODEL_PATH);
        for(int i=0;i<FEATURE_COUNT;i++)
            printf("w%d=%.6f ",i,value_weights[i]);
        printf("b=%.6f\n",value_bias);
        return 0;
    }

    int result=run_game(input);
    if(input!=stdin&&fclose(input)!=0){
        fprintf(stderr,"关闭方块输入文件失败：%s\n",input_path);
        return 1;
    }
    return result;
}
