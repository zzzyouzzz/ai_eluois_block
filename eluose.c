#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <string.h>
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

static double score_features(const double value_weights[FEATURE_COUNT],
                             const double features[FEATURE_COUNT]){
    double score=0.0;
    for(int i=0;i<FEATURE_COUNT;i++)score+=value_weights[i]*features[i];
    return score;
}
static int extract_action_features(int shape,int rotation,int x,int y,int block_height,
                                   double features[FEATURE_COUNT]){
    for(int i=0;i<FEATURE_COUNT;i++)features[i]=0.0;
    int fill[4],full_rows[4],fill_line;
    if(place_block(shape,rotation,x,y,fill,full_rows,&fill_line)<0)return 0;
    int drow=0,dcol=0,holes=0,well=0;
    evaluate_board(&drow,&dcol,&holes,&well);
    features[0]=y+block_height-1;
    int block_cells_in_full_rows=0;
    for(int i=0;i<4;i++){
        if(fill[i]==10){
            for(int cell=0;cell<4;cell++)
                block_cells_in_full_rows+=
                    block[shape][rotation][cell][0]==i;
        }
    }
    features[1]=fill_line*block_cells_in_full_rows;
    features[2]=drow;
    features[3]=dcol;
    features[4]=holes;
    features[5]=well;
    remove_block(shape,rotation,x,y);
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
                                       const double value_weights[FEATURE_COUNT]){
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
            action->value=score_features(value_weights,action->features);
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
    training_action action=generate_action(shape,0.0,value_weights);
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
}

static int save_value_model(const char *path){
    FILE *fp=fopen(path,"w");
    if(!fp){
        fprintf(stderr,"无法保存价值模型 %s\n",path);
        return 0;
    }
    for(int i=0;i<FEATURE_COUNT;i++)fprintf(fp,"%.12g\n",value_weights[i]);
    if(fclose(fp)!=0){
        fprintf(stderr,"保存价值模型失败：%s\n",path);
        return 0;
    }
    return 1;
}

static int write_training_trace_header(FILE *trace){
    if(fprintf(trace,
               "episode,turn,shape,rotation,x,y,epsilon,score,reward,terminal,"
               "current_value,next_value,target,td_error,feature_norm")<0)return 0;
    for(int i=0;i<FEATURE_COUNT;i++){
        if(fprintf(trace,",feature%d,weight_before%d,contribution%d,weight_delta%d,weight_after%d",
                   i,i,i,i,i)<0)return 0;
    }
    return fprintf(trace,"\n")>=0;
}

static int record_training_data(FILE *trace,int episode,unsigned long long turn,int shape,
                                const training_action *action,double epsilon,
                                int score,double reward,int terminal,
                                double current_value,double next_value,
                                double target,double delta,double norm,
                                const double before[FEATURE_COUNT],
                                const double updates[FEATURE_COUNT],
                                const double after[FEATURE_COUNT]){
    if(!trace)return 1;
    if(fprintf(trace,"%d,%llu,%d,%d,%d,%d,%.17g,%d,%.17g,%d,"
                     "%.17g,%.17g,%.17g,%.17g,%.17g",
               episode,turn,shape,action->rotation,action->x,action->y,
               epsilon,score,reward,terminal,current_value,next_value,
               target,delta,norm)<0){
        fprintf(stderr,"写入训练追踪文件失败\n");
        return 0;
    }
    for(int i=0;i<FEATURE_COUNT;i++){
        if(fprintf(trace,",%.17g,%.17g,%.17g,%.17g,%.17g",
                   action->features[i],before[i],
                   before[i]*action->features[i],updates[i],after[i])<0){
            fprintf(stderr,"写入训练追踪文件失败\n");
            return 0;
        }
    }
    if(fprintf(trace,"\n")<0){
        fprintf(stderr,"写入训练追踪文件失败\n");
        return 0;
    }
    return 1;
}

static int train_value_episode(FILE *block_input,FILE *trace,int episode,
                               double epsilon,int *has_training_data){
    const double alpha=0.01,gamma=0.90;
    for(int i=0;i<25;i++)
        for(int j=0;j<15;j++)board[i][j]=0;
    *has_training_data=0;
    int shape=get_block(block_input);
    if(shape<0||shape==7)return 1;

    for(unsigned long long turn=1;;turn++){
        training_action action=generate_action(shape,epsilon,value_weights);
        if(!action.valid)return 1;
        int fill[4],full_rows[4],fill_line;
        int score=place_block(shape,action.rotation,action.x,action.y,
                              fill,full_rows,&fill_line);
        if(score<0)break;
        clear_lines(full_rows,fill_line);

        int terminal=action.top_out;
        double reward=score-(action.top_out?200.0:0.0);
        double next_value=0.0;
        int next_shape=-1;
        if(!terminal){
            next_shape=get_block(block_input);
            if(next_shape<0||next_shape==7){
                terminal=1;
            }else{
                training_action next_action=
                    generate_action(next_shape,0.0,value_weights);
                if(!next_action.valid||next_action.top_out){
                    reward-=200.0;
                    terminal=1;
                }else{
                    next_value=gamma*next_action.value;
                }
            }
        }

        double current_value=action.value;
        double target=reward+next_value;
        double delta=target-current_value;
        double norm=1.0;
        for(int i=0;i<FEATURE_COUNT;i++)
            norm+=action.features[i]*action.features[i];
        double before[FEATURE_COUNT],after[FEATURE_COUNT],updates[FEATURE_COUNT];
        for(int i=0;i<FEATURE_COUNT;i++){
            before[i]=value_weights[i];
            updates[i]=alpha*delta*action.features[i];
            after[i]=before[i]+updates[i];
            if(!isfinite(after[i])){
                fprintf(stderr,
                        "权重数值异常：episode=%d turn=%llu feature=%d\n",
                        episode,turn,i);
                return 0;
            }
        }
        if(!record_training_data(trace,episode,turn,shape,&action,
                                 epsilon,score,reward,terminal,current_value,
                                 next_value,target,delta,norm,before,updates,after))
            return 0;
        for(int i=0;i<FEATURE_COUNT;i++)value_weights[i]=after[i];

        *has_training_data=1;
        if(terminal)break;
        shape=next_shape;
    }
    return 1;
}

static int train_value_model(int episodes,const char *input_path,FILE *trace,
                             int *completed_episodes){
    srand((unsigned int)time(NULL));
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

        double epsilon=0.25-0.20*(double)episode/episodes;
        if(epsilon<0.05)epsilon=0.05;
        int has_training_data=0;
        int success=train_value_episode(block_input,trace,episode+1,epsilon,
                                        &has_training_data);
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
    long episodes=2000;
    int training=0;
    const char *input_path=NULL;
    const char *trace_path=NULL;
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i],"--train")==0){
            if(training){
                fprintf(stderr,"训练模式参数重复\n");
                return 1;
            }
            training=1;
            if(i+1<argc&&argv[i+1][0]!='-'){
                char *end=NULL;
                errno=0;
                long parsed=strtol(argv[i+1],&end,10);
                if(end!=argv[i+1]&&*end=='\0'){
                    if(errno||parsed<1||parsed>1000000){
                        fprintf(stderr,"训练局数必须是 1 到 1000000 之间的整数\n");
                        return 1;
                    }
                    episodes=parsed;
                    i++;
                }
            }
        }else if(strcmp(argv[i],"-i")==0){
            if(input_path||i+1>=argc||argv[i+1][0]=='\0'||
               (argv[i+1][0]=='-'&&strcmp(argv[i+1],"-")!=0)){
                fprintf(stderr,"-i 后必须指定输入文件，且只能指定一次\n");
                return 1;
            }
            input_path=argv[++i];
        }else if(strcmp(argv[i],"--trace")==0){
            if(trace_path||i+1>=argc||argv[i+1][0]=='\0'){
                fprintf(stderr,"--trace 后必须指定追踪文件，且只能指定一次\n");
                return 1;
            }
            trace_path=argv[++i];
        }else{
            fprintf(stderr,
                    "用法：%s [--train [局数]] [-i 方块输入文件] [--trace 追踪文件]\n",
                    argv[0]);
            return 1;
        }
    }
    if(trace_path&&!training){
        fprintf(stderr,"--trace 只能与 --train 一起使用\n");
        return 1;
    }
    if(trace_path&&strcmp(trace_path,"q_weights.dat")==0){
        fprintf(stderr,"追踪文件不能覆盖 q_weights.dat\n");
        return 1;
    }
    if(training&&episodes>1&&
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
    if(load_value_model("q_weights.dat")<0){
        if(input!=stdin)fclose(input);
        return 1;
    }

    if(training){
        FILE *trace=NULL;
        if(trace_path){
            trace=fopen(trace_path,"w");
            if(!trace){
                fprintf(stderr,"无法创建训练追踪文件 %s\n",trace_path);
                return 1;
            }
            if(!write_training_trace_header(trace)){
                fprintf(stderr,"无法写入训练追踪文件表头 %s\n",trace_path);
                fclose(trace);
                return 1;
            }
        }
        int completed_episodes=0;
        if(!train_value_model((int)episodes,input_path,trace,&completed_episodes)){
            if(trace)fclose(trace);
            return 1;
        }
        if(trace&&fclose(trace)!=0){
            fprintf(stderr,"关闭训练追踪文件失败：%s\n",trace_path);
            return 1;
        }
        if(completed_episodes==0){
            fprintf(stderr,"训练输入中没有可用方块\n");
            return 1;
        }
        if(!save_value_model("q_weights.dat"))return 1;
        printf("训练完成：%d 局，模型已保存到 q_weights.dat\n",
               completed_episodes);
        for(int i=0;i<FEATURE_COUNT;i++)
            printf("w%d=%.6f%c",i,value_weights[i],i+1==FEATURE_COUNT?'\n':' ');
        return 0;
    }

    int result=run_game(input);
    if(input!=stdin&&fclose(input)!=0){
        fprintf(stderr,"关闭方块输入文件失败：%s\n",input_path);
        return 1;
    }
    return result;
}
