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
static double score_features(const double features[FEATURE_COUNT]){
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
int evaluate_place(int shape,int rotation,int x,int y,int block_height){//评估放置
    double features[FEATURE_COUNT];
    if(y+block_height>14)return -9000000;
    if(!extract_action_features(shape,rotation,x,y,block_height,features))return -9000000;
    return (int)score_features(features);
}
void find_best_position(int shape,int* rotation,int* x,int* y){//找到最佳放置位置
    int best_eval=-0x7fffffff;
    int best_priority=0x7fffffff;
    for(int i=0;i<block_r[shape];i++){
        for(int j=0;j<10-block_w[shape][i];j++){
            int best_y=simulate_fall(shape,i,j);
            int eval=evaluate_place(shape,i,j,best_y,block_h[shape][i]);
            int priority=100*j+i;
            if(eval>best_eval||(eval==best_eval&&priority<best_priority)){
                best_eval=eval;
                best_priority=priority;
                *rotation=i;
                *x=j;
                *y=best_y;
            }
        }
    }
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
typedef struct{
    int rotation,x,y,top_out;
    double features[FEATURE_COUNT];
    double value;
}training_action;

static int collect_actions(int shape,training_action actions[40]){
    int count=0;
    for(int rotation=0;rotation<block_r[shape];rotation++){
        for(int x=0;x<10;x++){
            if(!can_place(shape,rotation,x,16))continue;
            int y=simulate_fall(shape,rotation,x);
            if(y<0||y>=20)continue;
            training_action *action=&actions[count++];
            action->rotation=rotation;
            action->x=x;
            action->y=y;
            action->top_out=y+block_h[shape][rotation]>14;
            extract_action_features(shape,rotation,x,y,block_h[shape][rotation],
                                    action->features);
            action->value=action->top_out?-1000000.0:score_features(action->features);
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

static void train_value_model(int episodes){
    const double alpha=0.15,gamma=0.90;
    srand((unsigned int)time(NULL));
    for(int episode=0;episode<episodes;episode++){
        for(int i=0;i<25;i++)
            for(int j=0;j<15;j++)board[i][j]=0;
        double epsilon=0.25-0.20*(double)episode/episodes;
        if(epsilon<0.05)epsilon=0.05;
        int shape=rand()%7;
        for(int turn=0;turn<1000;turn++){
            training_action actions[40];
            int count=collect_actions(shape,actions);
            if(count==0)break;

            int selected=0;
            if((double)rand()/RAND_MAX<epsilon){
                selected=rand()%count;
            }else{
                for(int i=1;i<count;i++)
                    if(actions[i].value>actions[selected].value)selected=i;
            }
            training_action action=actions[selected];
            int fill[4],full_rows[4],fill_line;
            int score=place_block(shape,action.rotation,action.x,action.y,
                                  fill,full_rows,&fill_line);
            if(score<0)break;
            clear_lines(full_rows,fill_line);
            int terminal=action.top_out;
            double reward=0.34*score-(terminal?200.0:0.0);
            double next_value=0.0;
            int next_shape=rand()%7;

            if(!terminal){
                training_action next_actions[40];
                int next_count=collect_actions(next_shape,next_actions);
                int has_next_action=0;
                for(int i=0;i<next_count;i++){
                    if(!next_actions[i].top_out&&
                       (!has_next_action||next_actions[i].value>next_value)){
                        next_value=next_actions[i].value;
                        has_next_action=1;
                    }
                }
                if(!has_next_action)reward-=200.0;
                else next_value=gamma*next_value;
            }

            double current_value=score_features(action.features);
            double target=reward+(terminal?0.0:next_value);
            double delta=target-current_value;
            double norm=1.0;
            for(int i=0;i<FEATURE_COUNT;i++)
                norm+=action.features[i]*action.features[i];
            for(int i=0;i<FEATURE_COUNT;i++)
                value_weights[i]+=alpha*delta*action.features[i]/norm;

            if(terminal)break;
            shape=next_shape;
        }
    }
}

int main(int argc,char **argv){
    if(argc>1&&strcmp(argv[1],"--train")==0){
        long episodes=2000;
        if(argc>2){
            char *end=NULL;
            errno=0;
            episodes=strtol(argv[2],&end,10);
            if(errno||end==argv[2]||*end!='\0'||episodes<1||episodes>1000000){
                fprintf(stderr,"用法：%s --train [episodes]\n",argv[0]);
                return 1;
            }
        }
        if(load_value_model("q_weights.dat")<0)return 1;
        train_value_model((int)episodes);
        if(!save_value_model("q_weights.dat"))return 1;
        printf("训练完成：%ld 局，模型已保存到 q_weights.dat\n",episodes);
        for(int i=0;i<FEATURE_COUNT;i++)
            printf("w%d=%.6f%c",i,value_weights[i],i+1==FEATURE_COUNT?'\n':' ');
        return 0;
    }
    
    if(load_value_model("q_weights.dat")<0)return 1;
    clock_t start = clock();
    int score=0;
    int current_block=-1,next_block=-1;
    FILE *fp = stdin;
    if(argc>1){
        fp = fopen(argv[1], "r");
        if(!fp){
            fprintf(stderr,"无法打开方块序列文件 3.txt\n");
            return 1;
        }
    }
    current_block=get_block(fp);
    next_block=get_block(fp);
    while(current_block!=-1){
        int rotation=0,x=0,y=16;
        find_best_position(current_block,&rotation,&x,&y);
        int fill[4],full_rows[4],fill_line;
        int block_score=place_block(current_block,rotation,x,y,
                                    fill,full_rows,&fill_line);
        if(block_score<0){
            fprintf(stderr,"无法放置方块\n");
            fclose(fp);
            return 1;
        }
        score+=block_score;
        clear_lines(full_rows,fill_line);
        
        /*for(int i=19;i>=0;i--){
            for(int j=0;j<10;j++){
                printf("%c"," O"[board[i][j]]); 
            }printf("\n");
        }*/
        printf("%d %d\n%d\n",rotation,x,score);
        fflush(stdout);

        if(y+block_h[current_block][rotation]>14){
            printf("You lose!\n") ;
            for(int i=19;i>=0;i--){
                for(int j=0;j<10;j++){
                    printf("%c"," O"[board[i][j]]); 
                }printf("\n");
            }
            break;
        }
        //下一个方块
        
        if(next_block!=-1){
            current_block=next_block;
            next_block=get_block(fp);
            
            if(next_block==7){
                break;
            }
            if(next_block==-666){
                break; 
            }
        }else{
            current_block=next_block;
        }

    }
    fclose(fp);
    printf("耗时：%d ms\n", clock()-start);
    return 0;
}
