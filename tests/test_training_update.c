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
    reset_hidden_layer();   /* 隐藏层归零，使 Q 退化为纯线性模型 */
}

static void test_terminal_target_uses_only_reward(void){
    reset_weights();
    replay_transition batch[DQN_BATCH_SIZE];
    memset(batch,0,sizeof(batch));
    batch[0].action.features[0]=1.0;
    batch[0].reward=10.0;
    batch[0].terminal=1;
    batch[0].next_features[1]=100.0;
    target_weights[1]=100.0;

    if(!update_dqn_batch(batch,1,0.1,0.5))
        fail("terminal DQN update failed");

    expect_close(value_weights[0],0.5,"terminal reward update");
    expect_close(value_weights[1],0.0,"terminal target included bootstrap");
    puts("PASS: terminal DQN target");
}

static void test_bootstrap_target_updates_current_action(void){
    reset_weights();
    replay_transition batch[DQN_BATCH_SIZE];
    memset(batch,0,sizeof(batch));
    batch[0].action.features[0]=2.0;
    batch[0].next_features[1]=1.0;
    batch[0].reward=3.0;
    target_weights[1]=4.0;

    if(!update_dqn_batch(batch,1,0.1,0.5))
        fail("bootstrapped DQN update failed");

    expect_close(value_weights[0],0.2,"reward and bootstrap update");
    expect_close(value_weights[1],0.0,"update changed unrelated feature");
    puts("PASS: target-network bootstrap");
}

static void test_huber_gradient_is_clipped(void){
    reset_weights();
    replay_transition batch[DQN_BATCH_SIZE];
    memset(batch,0,sizeof(batch));
    batch[0].action.features[0]=1.0;
    batch[0].reward=1000.0;
    batch[0].terminal=1;

    if(!update_dqn_batch(batch,1,0.1,0.5))
        fail("clipped DQN update failed");

    /* TD=1000 若不被截断会得到 50，得到 5 说明梯度确实被截到 ±100 */
    expect_close(value_weights[0],5.0,"Huber gradient clip");
    puts("PASS: Huber gradient clipping");
}

int main(void){
    test_terminal_target_uses_only_reward();
    test_bootstrap_target_updates_current_action();
    test_huber_gradient_is_clipped();
    puts("All DQN update tests passed.");
    return EXIT_SUCCESS;
}
