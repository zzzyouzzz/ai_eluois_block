#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define main eluose_application_main
#include "../eluose.c"
#undef main

static const char *feature_names[FEATURE_COUNT]={
    "高度","消行特征","行方向变化","列方向变化","洞数","井深"
};
static char current_case[128];
static int line_number;

static void fail(const char *message){
    fprintf(stderr,"FAIL [%s] at data line %d: %s\n",
            current_case,line_number,message);
    exit(EXIT_FAILURE);
}

static int read_line(FILE *fp,char line[256]){
    while(fgets(line,256,fp)){
        line_number++;
        size_t length=strlen(line);
        while(length>0&&(line[length-1]=='\n'||line[length-1]=='\r'))
            line[--length]='\0';
        char *start=line;
        while(isspace((unsigned char)*start))start++;
        if(*start=='\0'||*start=='#')continue;
        if(start!=line)memmove(line,start,strlen(start)+1);
        return 1;
    }
    return 0;
}

static void expect_line(FILE *fp,const char *expected,char line[256]){
    if(!read_line(fp,line)){
        fprintf(stderr,"Unexpected end of test data; expected %s\n",expected);
        exit(EXIT_FAILURE);
    }
    if(strcmp(line,expected)!=0){
        fprintf(stderr,"FAIL [%s] at data line %d: expected '%s', got '%s'\n",
                current_case,line_number,expected,line);
        exit(EXIT_FAILURE);
    }
}

static void read_board(FILE *fp,int picture[25][15],
                       int action_picture[25][15],char line[256]){
    for(int row=19;row>=0;row--){
        if(!read_line(fp,line))fail("incomplete board picture");
        if(strlen(line)!=10)fail("board rows must contain exactly 10 cells");
        for(int column=0;column<10;column++){
            if(line[column]!='O'&&line[column]!='.'&&line[column]!='X')
                fail("board cells must be O, X or .");
            picture[row][column]=line[column]=='O';
            action_picture[row][column]=line[column]=='X';
        }
    }
}

static int parse_shape(char shape){
    switch(shape){
        case 'I': return 0;
        case 'T': return 1;
        case 'O': return 2;
        case 'J': return 3;
        case 'L': return 4;
        case 'S': return 5;
        case 'Z': return 6;
        default: return -1;
    }
}

static void copy_board(int target[25][15],int source[25][15]){
    memcpy(target,source,sizeof(board));
}

static void read_expected_features(FILE *fp,double expected[FEATURE_COUNT],
                                   char line[256]){
    expect_line(fp,"FEATURES",line);
    for(int i=0;i<FEATURE_COUNT;i++){
        char name[64],extra;
        double value;
        if(!read_line(fp,line))fail("missing expected feature");
        if(sscanf(line,"%63s %lf %c",name,&value,&extra)!=2)
            fail("feature must be: 中文特征名 数值");
        if(strcmp(name,feature_names[i])!=0)
            fail("expected features must follow the documented order");
        expected[i]=value;
    }
}

static void verify_action_picture(int action_picture[25][15],int shape,
                                  int rotation,int x,int y){
    int expected[25][15]={{0}};
    if(rotation<0||rotation>=4)fail("action rotation is out of range");
    for(int cell=0;cell<4;cell++){
        int row=y+block[shape][rotation][cell][0];
        int column=x+block[shape][rotation][cell][1];
        if(row<0||row>=20||column<0||column>=10)
            fail("X-marked action cells must fit on the board");
        expected[row][column]=1;
    }
    for(int row=0;row<20;row++)
        for(int column=0;column<10;column++)
            if(expected[row][column]!=action_picture[row][column])
                fail("X cells do not match the action's shape, rotation and position");
}

static int run_case(FILE *fp,char line[256]){
    int initial[25][15]={{0}};
    int original[25][15];
    int action_picture[25][15]={{0}};
    double expected[FEATURE_COUNT];
    double features[FEATURE_COUNT];
    char shape_char,extra;
    int rotation,x,y,block_height,valid;

    if(sscanf(line,"CASE %127s %c",current_case,&extra)!=1)
        fail("case header must be: CASE name");
    expect_line(fp,"BOARD",line);
    read_board(fp,initial,action_picture,line);

    if(!read_line(fp,line)||
       sscanf(line,"ACTION %c %d %d %d %d %c",
              &shape_char,&rotation,&x,&y,&block_height,&extra)!=5)
        fail("action must be: ACTION shape rotation x y block_height");
    int shape=parse_shape(shape_char);
    if(shape<0)fail("unknown shape");
    verify_action_picture(action_picture,shape,rotation,x,y);

    if(!read_line(fp,line)||
       sscanf(line,"VALID %d %c",&valid,&extra)!=1||
       (valid!=0&&valid!=1))
        fail("validity must be VALID 0 or VALID 1");
    read_expected_features(fp,expected,line);
    expect_line(fp,"END",line);

    copy_board(board,initial);
    copy_board(original,board);
    int result=extract_action_features(shape,rotation,x,y,block_height,features);
    int passed=1;
    if(result!=valid){
        fprintf(stderr,"FAIL [%s]: expected validity %d, got %d\n",
                current_case,valid,result);
        passed=0;
    }
    for(int i=0;i<FEATURE_COUNT;i++){
        if(fabs(features[i]-expected[i])>1e-9){
            fprintf(stderr,
                    "FAIL [%s] at data line %d: %s expected %.12g, got %.12g\n",
                    current_case,line_number,feature_names[i],expected[i],
                    features[i]);
            passed=0;
        }
    }
    for(int row=0;row<25;row++)
        for(int column=0;column<15;column++)
            if(board[row][column]!=original[row][column]){
                fprintf(stderr,"FAIL [%s]: board was not restored\n",current_case);
                return 0;
            }
    if(passed)printf("PASS: %s\n",current_case);
    return passed;
}

int main(int argc,char **argv){
    const char *path=argc>1?argv[1]:"tests\\feature_cases.cases";
    if(argc>2){
        fprintf(stderr,"Usage: %s [test-data-file]\n",argv[0]);
        return EXIT_FAILURE;
    }
    FILE *fp=fopen(path,"r");
    if(!fp){
        fprintf(stderr,"Cannot open test data: %s\n",path);
        return EXIT_FAILURE;
    }
    char line[256];
    int cases=0;
    int failures=0;
    while(read_line(fp,line)){
        if(strncmp(line,"CASE ",5)!=0){
            fprintf(stderr,"Unexpected data at line %d: %s\n",line_number,line);
            fclose(fp);
            return EXIT_FAILURE;
        }
        if(!run_case(fp,line))failures++;
        cases++;
    }
    fclose(fp);
    if(cases==0){
        fprintf(stderr,"No feature test cases found\n");
        return EXIT_FAILURE;
    }
    if(failures){
        fprintf(stderr,"%d of %d action feature tests failed.\n",failures,cases);
        return EXIT_FAILURE;
    }
    printf("All %d action feature tests passed.\n",cases);
    return EXIT_SUCCESS;
}
