#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(TD_N) && TD_N < 3
#undef TD_N
#endif
#ifndef TD_N
#define TD_N 3
#endif

#define main eluose_application_main
#include "../eluose.c"
#undef main

static void fail(const char *message){
    fprintf(stderr,"FAIL: %s\n",message);
    exit(EXIT_FAILURE);
}

static void expect_close(double actual,double expected,const char *message){
    if(fabs(actual-expected)>1e-9){
        fprintf(stderr,"FAIL: %s (expected %.12g, got %.12g)\n",
                message,expected,actual);
        exit(EXIT_FAILURE);
    }
}

static training_transition make_transition(unsigned long long turn,
                                           double value,double reward,
                                           int feature_index){
    training_transition transition={0};
    transition.turn=turn;
    transition.action.valid=1;
    transition.action.value=value;
    transition.action.features[feature_index]=1.0;
    transition.reward=reward;
    return transition;
}

static void reset_weights(void){
    for(int i=0;i<FEATURE_COUNT;i++)value_weights[i]=0.0;
}

static void test_full_queue_with_bootstrap(void){
    reset_weights();
    training_transition queue[TD_N]={
        make_transition(1,0.5,2.0,0),
        make_transition(2,0.0,3.0,1),
        make_transition(3,1.0,5.0,2)
    };

    if(!update_training_transitions(NULL,1,0.1,0.5,queue,3,4.0))
        fail("three-transition update failed");

    expect_close(value_weights[0],0.2375,"first transition weight");
    expect_close(value_weights[1],0.325,"second transition weight");
    expect_close(value_weights[2],0.3,"third transition weight");
    puts("PASS: queue-length discounted rewards and bootstrap");
}

static void test_shorter_queue_without_bootstrap(void){
    reset_weights();
    training_transition queue[TD_N]={
        make_transition(1,1.0,4.0,0),
        make_transition(2,0.5,6.0,1)
    };

    if(!update_training_transitions(NULL,2,0.1,0.25,queue,2,0.0))
        fail("two-transition terminal update failed");

    expect_close(value_weights[0],0.225,"first terminal transition weight");
    expect_close(value_weights[1],0.275,"second terminal transition weight");
    puts("PASS: actual queue length with zero bootstrap");
}

static void test_empty_queue_is_noop(void){
    reset_weights();
    training_transition queue[TD_N]={{0}};

    if(!update_training_transitions(NULL,3,0.1,0.9,queue,0,10.0))
        fail("empty queue should be a successful no-op");

    for(int i=0;i<FEATURE_COUNT;i++)
        expect_close(value_weights[i],0.0,"empty queue changed weights");
    puts("PASS: empty queue");
}

int main(void){
    test_full_queue_with_bootstrap();
    test_shorter_queue_without_bootstrap();
    test_empty_queue_is_noop();
    puts("All training update tests passed.");
    return EXIT_SUCCESS;
}
