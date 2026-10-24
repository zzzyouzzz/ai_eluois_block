#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
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
typedef struct{//记录每个方块的位置
    int x;
    int y;
    int shape;
    int rotation;
    int eval;
}block_info;

int place_block(int shape,int rotation,int x,int y){
    if(x<0||x>=10||y<0||y>=20)return -1;//放置错误，直接退出
    int fill[4]={0};//记录每一行是否已经被填满
    int i;
    for(i=0;i<4;i++){
        int dy=block[shape][rotation][i][0];
        int dx=block[shape][rotation][i][1];
        /*if(dx+x>=10||dy+y>=20){//放置错误，直接退出
            break;
        }
        if(board[y+dy][x+dx]==1){//放置错误，直接退出
            break;
        }*/
        
        board[y+dy][x+dx]=1;
        if(fill[dy]==0){
            for(int j=0;j<10;j++){
               fill[dy]+=board[y+dy][j]; 
            }
        }
        else{
            fill[dy]+=1;
        }
    }
    /* while(i<4&&i>=0){//如果放置错误，撤销放置
        i--;
        int dy=block[shape][rotation][i][0];
        int dx=block[shape][rotation][i][1];
        board[y+dy][x+dx]=0;
        
    }
    if(i==-1)return -1;//放置错误，直接退出*/
   
    int fill_line=0;
    for(int j=0;j<4;j++){
        if(fill[j]==10){//如果这一行已经被填满，将这一行上面的所有行下移
            int a;
            for(int k=y+j-fill_line;k<19;k++){
                for(int m=0;m<10;m++){
                    board[k][m]=board[k+1][m];
                }
            }
            fill_line++;
        }
    }
    switch(fill_line){//根据消除的行数计算得分
        case 1: return 100;
        case 2: return 300;
        case 3: return 500;
        case 4: return 800;
        default: return 0;
    }
}
int simulate_place(int shape,int rotation,int x,int y){//模拟放置
    //if(x<0||x>=10||y<0||y>=20)return -1;//放置错误，直接退出
    int fill[4]={0};//记录每一行是否已经被填满
    int fill_line=0;
    int block_num[4]={0};
    int num=0;
    int i;
    for(i=0;i<4;i++){
        int dy=block[shape][rotation][i][0];
        int dx=block[shape][rotation][i][1];
        /* if(dx+x>=10||dy+y>=20){//放置错误，直接退出
            break;
        }
        if(board[y+dy][x+dx]==1){//放置错误，直接退出
            break;
        }*/
       
        board[y+dy][x+dx]=1;
        if(fill[dy]==0){
            block_num[dy]+=1;
            for(int j=0;j<10;j++){
               fill[dy]+=board[y+dy][j]; 
            }
        }else{
            fill[dy]+=1;
            block_num[dy]+=1;
        }
        if(fill[dy]==10){
            fill_line++;
            num+=block_num[dy];
        }
    }
    /*while(i<4&&i>=0){//如果放置错误，撤销放置
        i--;
        int dy=block[shape][rotation][i][0];
        int dx=block[shape][rotation][i][1];
        board[y+dy][x+dx]=0;
        
    }
    if(i==0)return -1;//放置错误，直接退出*/
    
    return fill_line*num;
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
int evaluate_place(int shape,int rotation,int x,int y,int block_height){//评估放置
    if(y+block_height>14)return -9000000;
    int fill_line=simulate_place(shape,rotation,x,y);
    //if(fill_line==-1)return -10000000;
    int drow=0,dcol=0,holes=0,well=0,land_height=y+block_height-1;
    evaluate_board(&drow,&dcol,&holes,&well);

    int eval=0;
    eval-=45*land_height;
    eval+=34*fill_line;
    eval-=32*drow;
    eval-=93*dcol;
    eval-=79*holes;
    eval-=34*well;


    remove_block(shape,rotation,x,y);

    return eval;
}
void find_best_position(int shape,int* rotation,int* x,int* y,int next_shape){//找到最佳放置位置
    int best_eval=-1000000000;
    int best_priority=1000000000;
    block_info first_block[5]={0};
    int p=0;
    /*for(int i=0;i<5;i++){
        first_block[i].eval=-1000000000;
    }*/
    
    for(int i=0;i<block_r[shape];i++){
        for(int j=0;j<10-block_w[shape][i];j++){
            int best_y=simulate_fall(shape,i,j);
            //if(best_y==-1)continue;
            int eval=evaluate_place(shape,i,j,best_y,block_h[shape][i]);
            int priority=100*j+i;
            if(eval>best_eval){
                int index=p%5;
                first_block[index].x=j;
                first_block[index].y=best_y;
                first_block[index].shape=shape;
                first_block[index].rotation=i;
                first_block[index].eval=eval;
                best_eval=eval;
                best_priority=priority;
                p++;
            }
            else if(eval==best_eval&&priority<best_priority){
                int index=(p-1)%5;
                first_block[index].x=j;
                first_block[index].y=best_y;
                first_block[index].shape=shape;
                first_block[index].rotation=i;
                first_block[index].eval=eval;
                best_eval=eval;
                best_priority=priority;
            }
            else if(eval<=best_eval&&p<5){
                int index=p-1;
                while(eval<first_block[index].eval&&index>=0){
                    first_block[index+1]=first_block[index];
                    index--;
                }

                index++;
                first_block[index].x=j;
                first_block[index].y=best_y;
                first_block[index].shape=shape;
                first_block[index].rotation=i;
                first_block[index].eval=eval;
                p++;
            }
        }
        
    }
    best_eval=-1000000000;
    for(int i=0;i<5;i++){
        //if(first_block[i].eval==-1000000000)continue;
        int next_eval=-1000000000;
        int next_priority=1000000000;
        //if(!can_place(first_block[i].shape,first_block[i].rotation,first_block[i].x,first_block[i].y))continue;
        simulate_place(first_block[i].shape,first_block[i].rotation,first_block[i].x,first_block[i].y);
        for(int j=0;j<block_r[next_shape];j++){
            for(int m=0;m<10-block_w[next_shape][j];m++){
                int next_y=simulate_fall(next_shape,j,m);
                if(next_y==-1)continue;
                int eval=evaluate_place(next_shape,j,m,next_y,block_h[next_shape][j]); 
                int priority=100*m+j*10;
                if(eval>next_eval){
                    next_eval=eval;
                    next_priority=priority; 
                }
                if(eval==next_eval&&priority<next_priority){
                    next_eval=eval;
                    next_priority=priority; 
                }
            }
        }
        remove_block(first_block[i].shape,first_block[i].rotation,first_block[i].x,first_block[i].y);
        if(first_block[i].eval+next_eval/2>best_eval){
            best_eval=first_block[i].eval+next_eval/2;
            *rotation=first_block[i].rotation;
            *x=first_block[i].x;
            *y=first_block[i].y;
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
int main(){
    clock_t start = clock();
    int score=0;
    int current_block=-1,next_block=-1;
    FILE *fp = fopen("3.txt", "r");
    current_block=get_block(fp);
    next_block=get_block(fp);
    while(current_block!=-1){
        int rotation=0,x=0,y=16;
        find_best_position(current_block,&rotation,&x,&y,next_block);
        score+=place_block(current_block,rotation,x,y); 
        
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


