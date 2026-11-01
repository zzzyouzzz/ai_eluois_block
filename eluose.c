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
#define TRAIN_EPISODES 2000
#endif
/* 是否忽略已保存的模型、从源码里的初始权重开始训练：1 = 忽略，0 = 沿用 */
#ifndef TRAIN_FRESH
#define TRAIN_FRESH 0
#endif
/* 模型文件路径（训练结束写这里，平时从这里读） */
#ifndef VALUE_MODEL_PATH
#define VALUE_MODEL_PATH "q_weights.dat"
#endif
/* 每步训练数据写入的追踪 CSV；空字符串 "" 表示不追踪 */
#ifndef TRAINING_TRACE_PATH
#define TRAINING_TRACE_PATH "training_trace.csv"
#endif

/* ---------- DQN 训练参数 ---------- */

/* 经验回放区容量（条） */
#define DQN_REPLAY_CAPACITY 50000
/* 每次批量更新抽多少条样本 */
#define DQN_BATCH_SIZE 32
/* 回放区攒满多少条才开始更新 */
#define DQN_REPLAY_WARMUP 64
/* 每多少次批量更新同步一次目标网络 */
#define DQN_TARGET_UPDATE_INTERVAL 500
/* 梯度裁剪阈值：|TD| 超过它就按 ±100 截断（防止个别样本把权重带飞） */
#define DQN_HUBER_DELTA 100.0
/* 顶出惩罚：reward -= 300·exp(-(步数-1)/60)，越早顶出扣得越狠 */
#define DQN_EARLY_END_PENALTY 300.0
#define DQN_EARLY_END_DECAY 60.0
/* 折扣因子 */
#define DQN_GAMMA 0.99
/* 学习率 */
#define DQN_LEARNING_RATE 0.0001

/* ---------- 探索（epsilon 退火） ---------- */

/* 每局训练用的 epsilon 从 START 线性降到 END，END 同时是下限。
   多批次实验可以逐批下调 START（run_experiments.ps1 就是这么做的），
   从而让后一批"接着"前一批的探索水平，而不是每批重新探索一遍。 */
#ifndef TRAIN_EPSILON_START
#define TRAIN_EPSILON_START 0.35
#endif
#ifndef TRAIN_EPSILON_END
#define TRAIN_EPSILON_END 0.05
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
double value_weights[FEATURE_COUNT] = {-45.0, 34.0, -32.0, -93.0, -79.0, -34.0};

/* ---- 价值模型：线性主干 + 一层 tanh 隐藏层（残差连接） ----
   Q = w·φ + Σ_h w2[h]·tanh( b1[h] + Σ_j w1[h][j]·φ_j·HIDDEN_INPUT_SCALE[j] )
   w2 初始为 0 时隐藏层贡献恒为 0，网络严格退化为原来的线性模型。 */
enum { HIDDEN_SIZE = 16 };
typedef struct{
    double w1[HIDDEN_SIZE][FEATURE_COUNT];
    double b1[HIDDEN_SIZE];
    double w2[HIDDEN_SIZE];
}hidden_layer;
static hidden_layer hidden;
static hidden_layer target_hidden;
static double target_weights[FEATURE_COUNT];

/* 隐藏层输入端缩放：6 维特征量级差得很大（高度 0..20，井深可到上百），
   不压到 O(1) 附近的话 tanh 一开始就饱和、梯度消失。 */
static const double HIDDEN_INPUT_SCALE[FEATURE_COUNT]={
    1.0/20.0,1.0/16.0,1.0/50.0,1.0/50.0,1.0/50.0,1.0/50.0
};
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

/* w1 用确定性"哈希"散布，保证 16 个隐单元互不相同；
   如果所有隐单元初值相同，它们的梯度会一直相同，等于只训练了一个单元。 */
static void fill_hidden_w1(void){
    for(int h=0;h<HIDDEN_SIZE;h++)
        for(int j=0;j<FEATURE_COUNT;j++)
            hidden.w1[h][j]=0.5*(double)(((h*13+j*7)%29)-14)/14.0;
}

/* 隐藏层重置为"不改变输出"的状态：w2 全 0 时 Q 严格等于线性主干 */
static void reset_hidden_layer(void){
    fill_hidden_w1();
    for(int h=0;h<HIDDEN_SIZE;h++){
        hidden.b1[h]=0.0;
        hidden.w2[h]=0.0;
    }
    target_hidden=hidden;
}

/* 训练起点：w2 必须给一点非零初值。
   w2=0 时 dQ/dw1 恒为 0，隐藏层的表示部分永远学不动（死初始化）。 */
static void seed_hidden_layer(void){
    for(int h=0;h<HIDDEN_SIZE;h++){
        hidden.b1[h]=0.05*(double)(((h*5)%7)-3)/3.0;
        hidden.w2[h]=0.02*(double)(((h*3)%5)-2)/2.0;
    }
    target_hidden=hidden;
}

static void reset_value_model(void){
    const double initial_weights[FEATURE_COUNT]=
        {-45.0,34.0,-32.0,-93.0,-79.0,-34.0};
    memcpy(value_weights,initial_weights,sizeof(value_weights));
    reset_hidden_layer();
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
typedef struct{
    int valid,rotation,x,y,top_out;
    double features[FEATURE_COUNT];
    double value;
}training_action;

static double hidden_unit_output(const hidden_layer *layer,int unit,
                                 const double features[FEATURE_COUNT]){
    double z=layer->b1[unit];
    for(int j=0;j<FEATURE_COUNT;j++)
        z+=layer->w1[unit][j]*features[j]*HIDDEN_INPUT_SCALE[j];
    return tanh(z);
}

static double score_features(const double value_weights[FEATURE_COUNT],
                             const hidden_layer *layer,
                             const double features[FEATURE_COUNT]){
    double score=0.0;
    for(int i=0;i<FEATURE_COUNT;i++)score+=value_weights[i]*features[i];
    for(int h=0;h<HIDDEN_SIZE;h++)
        score+=layer->w2[h]*hidden_unit_output(layer,h,features);
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
static training_action generate_action(int shape,double epsilon,
                                       const double value_weights[FEATURE_COUNT],
                                       const hidden_layer *layer){
    training_action actions[40];
    int count=0;
    training_action invalid={0};
    if(shape<0||shape>=7)return invalid;

    for(int rotation=0;rotation<block_r[shape];rotation++){
        for(int x=0;x<10;x++){
            if(!can_place(shape,rotation,x,16))continue;
            int y=simulate_fall(shape,rotation,x);
            if(y<0||y>=20)continue;
            training_action *action=&actions[count++];
            action->valid=1;
            action->rotation=rotation;
            action->x=x;
            action->y=y;
            action->top_out=y+block_h[shape][rotation]>14;
            if(!extract_action_features(shape,rotation,x,y,
                                        block_h[shape][rotation],
                                        action->features)){
                action->valid=0;
                count--;
                continue;
            }
            action->value=score_features(value_weights,layer,action->features);
        }
    }
    if(count==0)return invalid;

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

void find_best_position(int shape,int* rotation,int* x,int* y){
    training_action action=generate_action(shape,0.0,value_weights,&hidden);
    if(!action.valid){
        *rotation=0;
        *x=0;
        *y=-1;
        return;
    }
    *rotation=action.rotation;
    *x=action.x;
    *y=action.y;
}

static int load_value_model(const char *path){
    FILE *fp=fopen(path,"r");
    if(!fp){
        if(errno==ENOENT)return 0;
        fprintf(stderr,"无法读取价值模型 %s\n",path);
        return -1;
    }
    double loaded[FEATURE_COUNT];
    hidden_layer loaded_hidden;
    memset(&loaded_hidden,0,sizeof(loaded_hidden));
    char tag[16];
    if(fscanf(fp,"%15s",tag)!=1){
        fprintf(stderr,"价值模型格式错误：%s\n",path);
        fclose(fp);
        return -1;
    }
    int ok=0;
    if(strcmp(tag,"mlp")==0){
        int size=0;
        if(fscanf(fp,"%d",&size)==1&&size==HIDDEN_SIZE){
            ok=1;
            for(int i=0;i<FEATURE_COUNT&&ok;i++)
                ok=fscanf(fp,"%lf",&loaded[i])==1&&isfinite(loaded[i]);
            for(int h=0;h<HIDDEN_SIZE&&ok;h++)
                ok=fscanf(fp,"%lf",&loaded_hidden.b1[h])==1&&
                   isfinite(loaded_hidden.b1[h]);
            for(int h=0;h<HIDDEN_SIZE&&ok;h++)
                for(int j=0;j<FEATURE_COUNT&&ok;j++)
                    ok=fscanf(fp,"%lf",&loaded_hidden.w1[h][j])==1&&
                       isfinite(loaded_hidden.w1[h][j]);
            for(int h=0;h<HIDDEN_SIZE&&ok;h++)
                ok=fscanf(fp,"%lf",&loaded_hidden.w2[h])==1&&
                   isfinite(loaded_hidden.w2[h]);
        }
    }else if(sscanf(tag,"%lf",&loaded[0])==1&&isfinite(loaded[0])){
        /* 旧格式：只有 6 个线性权重，隐藏层保持全 0（等价于原模型） */
        ok=1;
        for(int i=1;i<FEATURE_COUNT&&ok;i++)
            ok=fscanf(fp,"%lf",&loaded[i])==1&&isfinite(loaded[i]);
    }
    if(!ok){
        fprintf(stderr,"价值模型格式错误：%s\n",path);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    for(int i=0;i<FEATURE_COUNT;i++)value_weights[i]=loaded[i];
    hidden=loaded_hidden;
    return 1;
}

static int save_value_model(const char *path){
    FILE *fp=fopen(path,"w");
    if(!fp){
        fprintf(stderr,"无法保存价值模型 %s\n",path);
        return 0;
    }
    fprintf(fp,"mlp %d\n",HIDDEN_SIZE);
    for(int i=0;i<FEATURE_COUNT;i++)fprintf(fp,"%.12g\n",value_weights[i]);
    for(int h=0;h<HIDDEN_SIZE;h++)fprintf(fp,"%.12g\n",hidden.b1[h]);
    for(int h=0;h<HIDDEN_SIZE;h++)
        for(int j=0;j<FEATURE_COUNT;j++)
            fprintf(fp,"%.12g\n",hidden.w1[h][j]);
    for(int h=0;h<HIDDEN_SIZE;h++)fprintf(fp,"%.12g\n",hidden.w2[h]);
    if(fclose(fp)!=0){
        fprintf(stderr,"保存价值模型失败：%s\n",path);
        return 0;
    }
    return 1;
}

/* 训练超参已统一移到文件开头的「参数配置区」 */

typedef struct{
    training_action action;
    double next_features[FEATURE_COUNT];
    double reward;
    int terminal;
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

static int update_dqn_batch(const replay_transition batch[DQN_BATCH_SIZE],
                            int batch_size,double alpha,double gamma){
    if(batch_size<1||batch_size>DQN_BATCH_SIZE){
        fprintf(stderr,"DQN 批次大小无效\n");
        return 0;
    }

    double gradients[FEATURE_COUNT]={0.0};
    double grad_w2[HIDDEN_SIZE]={0.0};
    double grad_b1[HIDDEN_SIZE]={0.0};
    double grad_w1[HIDDEN_SIZE][FEATURE_COUNT];
    memset(grad_w1,0,sizeof(grad_w1));
    for(int sample=0;sample<batch_size;sample++){
        const replay_transition *transition=&batch[sample];
        double current_value=score_features(value_weights,&hidden,
                                            transition->action.features);
        double next_value=transition->terminal?0.0:
            score_features(target_weights,&target_hidden,
                           transition->next_features);
        double target=transition->reward+
            (transition->terminal?0.0:gamma*next_value);
        double td_error=target-current_value;
        if(!isfinite(current_value)||!isfinite(next_value)||
           !isfinite(target)||!isfinite(td_error)){
            fprintf(stderr,"DQN 训练数值异常\n");
            return 0;
        }

        double clipped_td=td_error;
        if(clipped_td>DQN_HUBER_DELTA)clipped_td=DQN_HUBER_DELTA;
        if(clipped_td<-DQN_HUBER_DELTA)clipped_td=-DQN_HUBER_DELTA;
        double norm=1.0;
        for(int i=0;i<FEATURE_COUNT;i++)
            norm+=transition->action.features[i]*
                  transition->action.features[i];
        for(int i=0;i<FEATURE_COUNT;i++)
            gradients[i]+=clipped_td*transition->action.features[i]/norm;

        /* 隐藏层反向传播：dQ/dw2[h]=tanh(z_h)，
           dQ/dw1[h][j]=w2[h]·tanh'(z_h)·x_j，dQ/db1[h]=w2[h]·tanh'(z_h) */
        for(int h=0;h<HIDDEN_SIZE;h++){
            double out=hidden_unit_output(&hidden,h,
                                          transition->action.features);
            double slope=1.0-out*out;
            double upstream=clipped_td*hidden.w2[h]*slope/norm;
            grad_w2[h]+=clipped_td*out/norm;
            grad_b1[h]+=upstream;
            for(int j=0;j<FEATURE_COUNT;j++)
                grad_w1[h][j]+=upstream*
                    transition->action.features[j]*HIDDEN_INPUT_SCALE[j];
        }
    }

    double after[FEATURE_COUNT];
    for(int i=0;i<FEATURE_COUNT;i++){
        after[i]=value_weights[i]+alpha*gradients[i]/batch_size;
        if(!isfinite(after[i])){
            fprintf(stderr,"DQN 权重数值异常：feature=%d\n",i);
            return 0;
        }
    }
    double delta_w2[HIDDEN_SIZE],delta_b1[HIDDEN_SIZE];
    double delta_w1[HIDDEN_SIZE][FEATURE_COUNT];
    for(int h=0;h<HIDDEN_SIZE;h++){
        delta_w2[h]=alpha*grad_w2[h]/batch_size;
        delta_b1[h]=alpha*grad_b1[h]/batch_size;
        for(int j=0;j<FEATURE_COUNT;j++)
            delta_w1[h][j]=alpha*grad_w1[h][j]/batch_size;
        if(!isfinite(delta_w2[h])||!isfinite(delta_b1[h])){
            fprintf(stderr,"DQN 权重数值异常：hidden=%d\n",h);
            return 0;
        }
    }
    for(int h=0;h<HIDDEN_SIZE;h++)
        for(int j=0;j<FEATURE_COUNT;j++)
            if(!isfinite(hidden.w1[h][j]+delta_w1[h][j])){
                fprintf(stderr,"DQN 权重数值异常：hidden=%d feature=%d\n",h,j);
                return 0;
            }

    for(int i=0;i<FEATURE_COUNT;i++)value_weights[i]=after[i];
    for(int h=0;h<HIDDEN_SIZE;h++){
        hidden.w2[h]+=delta_w2[h];
        hidden.b1[h]+=delta_b1[h];
        for(int j=0;j<FEATURE_COUNT;j++)hidden.w1[h][j]+=delta_w1[h][j];
    }
    return 1;
}

static void add_replay_transition(const replay_transition *transition){
    replay_buffer[replay_next]=*transition;
    replay_next=(replay_next+1)%DQN_REPLAY_CAPACITY;
    if(replay_size<DQN_REPLAY_CAPACITY)replay_size++;
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
    training_action action;
    double epsilon;
    int score;
    double reward;
    int terminal;
    double current_value;
    double next_value;
    double target;
    double td_error;
    double norm;
    int board_cells;
    double before[FEATURE_COUNT];
}trace_step_state;

static trace_step_state trace_step;

static int trace_begin_step(int episode,unsigned long long turn,int shape,
                            const training_action *action,double epsilon,int score,
                            const replay_transition *transition){
    trace_step.active=0;
    if(!trace_training_file)return 1;

    double current_value=action->value;
    double next_value=transition->terminal?0.0:
        score_features(target_weights,&target_hidden,transition->next_features);
    double target=transition->reward+
        (transition->terminal?0.0:DQN_GAMMA*next_value);
    double td_error=target-current_value;
    double norm=1.0;
    for(int i=0;i<FEATURE_COUNT;i++)
        norm+=action->features[i]*action->features[i];
    if(!isfinite(current_value)||!isfinite(next_value)||
       !isfinite(target)||!isfinite(td_error)||!isfinite(norm)){
        fprintf(stderr,"训练数值异常：episode=%d turn=%llu\n",episode,turn);
        return 0;
    }

    trace_step.active=1;
    trace_step.episode=episode;
    trace_step.turn=turn;
    trace_step.shape=shape;
    trace_step.action=*action;
    trace_step.epsilon=epsilon;
    trace_step.score=score;
    trace_step.reward=transition->reward;
    trace_step.terminal=transition->terminal;
    trace_step.current_value=current_value;
    trace_step.next_value=next_value;
    trace_step.target=target;
    trace_step.td_error=td_error;
    trace_step.norm=norm;
    trace_step.board_cells=trace_count_board_cells();
    memcpy(trace_step.before,value_weights,sizeof(trace_step.before));
    return 1;
}

static int trace_end_step(void){
    if(!trace_training_file||!trace_step.active)return 1;
    trace_step.active=0;
    if(fprintf(trace_training_file,
               "%d,%llu,%d,%d,%d,%d,%.17g,%d,%.17g,%d,"
               "%.17g,%.17g,%.17g,%.17g,%.17g,%d",
               trace_step.episode,trace_step.turn,trace_step.shape,
               trace_step.action.rotation,trace_step.action.x,
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
                   trace_step.action.features[i],trace_step.before[i],
                   trace_step.before[i]*trace_step.action.features[i],
                   delta,value_weights[i])<0){
            fprintf(stderr,"写入训练追踪文件失败\n");
            return 0;
        }
    }
    if(fprintf(trace_training_file,"\n")<0){
        fprintf(stderr,"写入训练追踪文件失败\n");
        return 0;
    }
    return 1;
}

/* ------------------------ 数据追踪模块结束 ------------------------ */

static int train_on_transition(const replay_transition *transition){
    add_replay_transition(transition);
    if(replay_size>=DQN_REPLAY_WARMUP){
        replay_transition batch[DQN_BATCH_SIZE];
        for(int i=0;i<DQN_BATCH_SIZE;i++)
            batch[i]=replay_buffer[random_below(replay_size)];
        if(!update_dqn_batch(batch,DQN_BATCH_SIZE,DQN_LEARNING_RATE,
                             DQN_GAMMA))
            return 0;
        dqn_update_count++;
        if(dqn_update_count%DQN_TARGET_UPDATE_INTERVAL==0){
            memcpy(target_weights,value_weights,sizeof(target_weights));
            target_hidden=hidden;
        }
    }
    return 1;
}

static int train_value_episode(FILE *block_input,int episode,
                               double epsilon,int *has_training_data){
    memset(board,0,sizeof(board));
    *has_training_data=0;
    int shape=get_block(block_input);
    for(unsigned long long turn=1;;turn++){
        training_action action=generate_action(shape,epsilon,value_weights,&hidden);
        if(!action.valid){
            fprintf(stderr,"无法生成训练动作：episode=%d turn=%llu\n",
                    episode,turn);
            return 0;
        }

        int fill[4],full_rows[4],fill_line;
        int score=place_block(shape,action.rotation,action.x,action.y,
                              fill,full_rows,&fill_line);
        if(score<0){
            fprintf(stderr,"训练动作无法放置：episode=%d turn=%llu\n",
                    episode,turn);
            return 0;
        }
        clear_lines(full_rows,fill_line);

        int terminal=action.top_out;
        double reward=score+1.0;
        if(action.top_out)
            reward-=DQN_EARLY_END_PENALTY*
                    exp(-(double)(turn-1)/DQN_EARLY_END_DECAY);

        int next_shape=-1;
        replay_transition transition={0};
        transition.action=action;
        transition.reward=reward;
        if(!terminal){
            next_shape=get_block(block_input);
            terminal=next_shape<0||next_shape==7;
            if(!terminal){
                training_action next_action=
                    generate_action(next_shape,0.0,target_weights,&target_hidden);
                if(!next_action.valid){
                    fprintf(stderr,
                            "无法生成后继动作：episode=%d turn=%llu\n",
                            episode,turn);
                    return 0;
                }
                memcpy(transition.next_features,next_action.features,
                       sizeof(transition.next_features));
            }
        }
        transition.terminal=terminal;

        trace_begin_step(episode,turn,shape,&action,epsilon,score,&transition);
        if(!train_on_transition(&transition))
            return 0;
        trace_end_step();
        *has_training_data=1;
        if(terminal)break;
        shape=next_shape;
    }
    return 1;
}

static int train_value_model(int episodes,const char *input_path,
                             int *completed_episodes){
    srand((unsigned int)time(NULL));
    memcpy(target_weights,value_weights,sizeof(target_weights));
    replay_size=0;
    replay_next=0;
    dqn_update_count=0;
    *completed_episodes=0;

    for(int episode=0;episode<episodes;episode++){
        FILE *block_input=stdin;
        if(input_path&&strcmp(input_path,"-")!=0){
            block_input=fopen(input_path,"r");
            if(!block_input){
                fprintf(stderr,"无法打开训练输入文件 %s\n",input_path);
                return 0;
            }
        }

        double epsilon=TRAIN_EPSILON_START+
                       (TRAIN_EPSILON_END-TRAIN_EPSILON_START)*
                       (double)episode/episodes;
        if(epsilon<TRAIN_EPSILON_END)epsilon=TRAIN_EPSILON_END;
        int has_training_data=0;
        int success=train_value_episode(block_input,episode+1,epsilon,&has_training_data);
        if(block_input!=stdin&&fclose(block_input)!=0){
            fprintf(stderr,"关闭训练输入文件失败：%s\n",input_path);
            return 0;
        }
        if(!success)return 0;
        if(!has_training_data)break;
        (*completed_episodes)++;
    }
    return 1;
}

static int run_game(FILE *input){
    clock_t start=clock();
    int score=0;
    int current_block=get_block(input);
    int next_block=get_block(input);

    while(current_block>=0&&current_block<7){
        int rotation=0,x=0,y=16;
        find_best_position(current_block,&rotation,&x,&y);
        int fill[4],full_rows[4],fill_line;
        int block_score=place_block(current_block,rotation,x,y,
                                    fill,full_rows,&fill_line);
        if(block_score<0){
            fprintf(stderr,"无法放置方块\n");
            return 1;
        }
        score+=block_score;
        clear_lines(full_rows,fill_line);

        printf("%d %d\n%d\n",rotation,x,score);
        fflush(stdout);

        if(y+block_h[current_block][rotation]>14){
            printf("You lose!\n");
            for(int row=19;row>=0;row--){
                for(int column=0;column<10;column++)
                    printf("%c"," O"[board[row][column]]);
                printf("\n");
            }
            break;
        }

        if(next_block==-1||next_block==7||next_block==-666)break;
        current_block=next_block;
        next_block=get_block(input);
    }
    printf("耗时：%d ms\n",clock()-start);
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
    if(!fresh_model&&load_value_model(VALUE_MODEL_PATH)<0){
        if(input!=stdin)fclose(input);
        return 1;
    }
    if(fresh_model){
        reset_value_model();
        printf("使用源码中的初始权重（线性主干 + 隐藏层）\n");
    }

    if(training){
        if(fresh_model)seed_hidden_layer();
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
            printf("w%d=%.6f%c",i,value_weights[i],i+1==FEATURE_COUNT?'\n':' ');
        printf("隐藏层：%d 个 tanh 单元（残差连接，w2 全 0 时等价于线性模型）\n",
               HIDDEN_SIZE);
        return 0;
    }

    int result=run_game(input);
    if(input!=stdin&&fclose(input)!=0){
        fprintf(stderr,"关闭方块输入文件失败：%s\n",input_path);
        return 1;
    }
    return result;
}
