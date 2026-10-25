#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define main eluose_application_main
#include "../eluose.c"
#undef main

static char current_case[128];
static int line_number;

typedef struct{
    int shape;
    int rotation;
    int x;
    int y;
}test_action;

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

static void read_board(FILE *fp,int picture[25][15],char line[256]){
    if(!read_line(fp,line))fail("missing board picture");
    if(strcmp(line,"EMPTY")==0)return;

    for(int row=19;row>=0;row--){
        if(row!=19&&!read_line(fp,line))fail("incomplete board picture");
        if(strlen(line)!=10)fail("board rows must contain exactly 10 cells");
        for(int column=0;column<10;column++){
            if(line[column]!='O'&&line[column]!='.')
                fail("board cells must be O or .");
            picture[row][column]=line[column]=='O';
        }
    }
}

static int parse_action(const char *line,test_action *action){
    char shape;
    char extra;
    int fields=sscanf(line," %c %d %d %d %c",
                      &shape,&action->rotation,&action->x,&action->y,&extra);
    if(fields!=4)return 0;
    switch(shape){
        case 'I': action->shape=0; break;
        case 'T': action->shape=1; break;
        case 'O': action->shape=2; break;
        case 'J': action->shape=3; break;
        case 'L': action->shape=4; break;
        case 'S': action->shape=5; break;
        case 'Z': action->shape=6; break;
        default: action->shape=7; break;
    }
    return 1;
}

static void copy_board(int target[25][15],int source[25][15]){
    for(int row=0;row<25;row++)
        for(int column=0;column<15;column++)
            target[row][column]=source[row][column];
}

static void verify_board(int expected[25][15]){
    for(int row=0;row<25;row++){
        for(int column=0;column<15;column++){
            if(board[row][column]!=expected[row][column]){
                fprintf(stderr,
                        "FAIL [%s] at data line %d: board[%d][%d] expected "
                        "%d, got %d\n",
                        current_case,line_number,row,column,
                        expected[row][column],board[row][column]);
                exit(EXIT_FAILURE);
            }
        }
    }
}

static void run_case(FILE *fp,char line[256]){
    int initial[25][15]={{0}};
    int expected[25][15]={{0}};

    expect_line(fp,"BEFORE",line);
    read_board(fp,initial,line);
    copy_board(board,initial);

    if(!read_line(fp,line))fail("missing ACTIONS section");
    int action_count;
    char extra;
    if(sscanf(line,"ACTIONS %d %c",&action_count,&extra)!=1||
       action_count<1||action_count>100)
        fail("ACTIONS must specify between 1 and 100 placements");

    int total_score=0;
    for(int i=0;i<action_count;i++){
        test_action action;
        int fill[4],full_rows[4],fill_line=0;
        if(!read_line(fp,line))fail("missing placement action");
        if(!parse_action(line,&action))
            fail("placement must be: shape rotation x y");

        int score=place_block(action.shape,action.rotation,action.x,action.y,
                              fill,full_rows,&fill_line);
        if(score>=0){
            total_score+=score;
            clear_lines(full_rows,fill_line);
        }
    }

    if(!read_line(fp,line))fail("missing EXPECTED_SCORE section");
    int expected_score;
    if(sscanf(line,"EXPECTED_SCORE %d %c",&expected_score,&extra)!=1)
        fail("expected score must be: EXPECTED_SCORE integer");
    if(total_score!=expected_score)
        fail("total score does not match EXPECTED_SCORE");

    expect_line(fp,"AFTER",line);
    read_board(fp,expected,line);
    verify_board(expected);
    expect_line(fp,"END",line);
}

int main(int argc,char **argv){
    const char *path=argc>1?argv[1]:"tests/game_cases.txt";
    if(argc>2){
        fprintf(stderr,"Usage: %s [test-data-file]\n",argv[0]);
        return EXIT_FAILURE;
    }

    FILE *fp=fopen(path,"r");
    if(!fp){
        fprintf(stderr,"Cannot open test data: %s\n",path);
        return EXIT_FAILURE;
    }

    int cases=0;
    char line[256];
    while(read_line(fp,line)){
        if(strncmp(line,"CASE ",5)!=0){
            fprintf(stderr,"Expected CASE at data line %d, got '%s'\n",
                    line_number,line);
            fclose(fp);
            return EXIT_FAILURE;
        }
        size_t name_length=strlen(line+5);
        if(name_length==0||name_length>=sizeof(current_case)){
            fprintf(stderr,"Invalid case name at data line %d\n",line_number);
            fclose(fp);
            return EXIT_FAILURE;
        }
        memcpy(current_case,line+5,name_length+1);
        run_case(fp,line);
        printf("PASS: %s\n",current_case);
        cases++;
    }

    if(fclose(fp)!=0){
        fprintf(stderr,"Failed to close test data: %s\n",path);
        return EXIT_FAILURE;
    }
    if(cases==0){
        fprintf(stderr,"No test cases found in %s\n",path);
        return EXIT_FAILURE;
    }
    printf("All %d game tests passed.\n",cases);
    return EXIT_SUCCESS;
}
