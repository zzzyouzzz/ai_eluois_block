#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

static void reset_weights(void){
    for(int i=0;i<FEATURE_COUNT;i++){
        value_weights[i]=0.0;
        target_weights[i]=0.0;
    }
    value_bias=0.0;
    target_bias=0.0;
    reset_adam_state();
}

/* Adam 的第一步：m̂ = g、v̂ = g²，于是 m̂/(√v̂+ε) = sign(g)，
   权重正好挪 ±学习率。三个用例都用学习率 0.1，所以期望值都是 0.1。 */
static void test_terminal_target_uses_only_reward(void){
    reset_weights();
    replay_transition samples[DQN_BATCH_SIZE];
    const replay_transition *batch[DQN_BATCH_SIZE]={&samples[0]};
    memset(samples,0,sizeof(samples));
    samples[0].action.feature[0]=1.0;
    samples[0].reward=10.0;
    samples[0].next_count=0;             /* 终局：没有候选 */
    samples[0].next[0].feature[1]=100.0; /* 终局时应该被忽略 */
    target_weights[1]=100.0;

    if(!update_dqn_batch(batch,1,0.1,0.5))
        fail("terminal DQN update failed");

    expect_close(value_weights[0],0.1,"terminal reward update");
    expect_close(value_weights[1],0.0,"terminal target included bootstrap");
    puts("PASS: terminal DQN target");
}

static void test_bootstrap_target_updates_current_action(void){
    reset_weights();
    replay_transition samples[DQN_BATCH_SIZE];
    const replay_transition *batch[DQN_BATCH_SIZE]={&samples[0]};
    memset(samples,0,sizeof(samples));
    samples[0].action.feature[0]=2.0;
    samples[0].next_count=1;
    samples[0].next[0].feature[1]=1.0;
    target_weights[1]=4.0;
    samples[0].reward=3.0;

    if(!update_dqn_batch(batch,1,0.1,0.5))
        fail("bootstrapped DQN update failed");

    expect_close(value_weights[0],0.1,"reward and bootstrap update");
    expect_close(value_weights[1],0.0,"update changed unrelated feature");
    puts("PASS: target-network bootstrap");
}

static void test_step_size_does_not_depend_on_td(void){
    reset_weights();
    replay_transition samples[DQN_BATCH_SIZE];
    const replay_transition *batch[DQN_BATCH_SIZE]={&samples[0]};
    memset(samples,0,sizeof(samples));
    samples[0].action.feature[0]=1.0;
    samples[0].reward=1000.0;
    samples[0].next_count=0;

    if(!update_dqn_batch(batch,1,0.1,0.5))
        fail("clipped DQN update failed");

    /* TD=1000 若按梯度缩放会得到 100，Adam 对它做了逐参数缩放，
       一步仍然只走学习率 0.1——这正是"特征量级差十倍也能一起学"的原因 */
    expect_close(value_weights[0],0.1,"Adam step size is independent of TD");
    puts("PASS: Adam step size is scale invariant");
}

int main(void){
    test_terminal_target_uses_only_reward();
    test_bootstrap_target_updates_current_action();
    test_step_size_does_not_depend_on_td();
    puts("All DQN update tests passed.");
    return EXIT_SUCCESS;
}
