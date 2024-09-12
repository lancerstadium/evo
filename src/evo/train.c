#include <evo.h>
#include <evo/util/sys.h>
#include <string.h>
#include <math.h>

// ==================================================================================== //
//                                  train: loss func
// ==================================================================================== //

// Loss Function: MSE
float loss_mse(float *output, float *target, int size) {
    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = output[i] - target[i];
        loss += diff * diff;
    }
    return loss / size;
}

// Loss Function: cross entropy
float loss_cross_entropy(float *output, float *target, int size) {
    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        loss -= target[i] * logf(output[i] + 1e-10);  // 加1e-10防止log(0)
    }
    return loss / size;
}

// ==================================================================================== //
//                                  train: update sgd
// ==================================================================================== //

typedef struct {
    float momentum;                                     // 动量系数 (用于 SGD)
    float weight_decay;                                 // 权重衰减系数 (L2 正则化)
} trainer_opt_sgd_t;

// ==================================================================================== //
//                                  train: update adam
// ==================================================================================== //

typedef struct {
    float *m;                                           /* 1st moment estimation value  */
    float *v;                                           /* 2nd moment estimation value  */
    float beta1;                                        /* 1st moment estimation coeff  */
    float beta2;                                        /* 2nd moment estimation coeff  */
    int nparam;                                         /* param number                 */ 
} trainer_opt_adam_t;

void update_adam(trainer_t *t, float *param, float *grad, int step) {
    trainer_opt_adam_t *state = (trainer_opt_adam_t *)t->priv;
    float beta1 = state->beta1;
    float beta2 = state->beta2;
    float learning_rate = t->learning_rate;
    float epsilon = t->epsilon;

    for (int i = 0; i < state->nparam; i++) {
        state->m[i] = beta1 * state->m[i] + (1 - beta1) * grad[i];
        state->v[i] = beta2 * state->v[i] + (1 - beta2) * grad[i] * grad[i];

        // 偏差修正
        float m_hat = state->m[i] / (1 - powf(beta1, step));
        float v_hat = state->v[i] / (1 - powf(beta2, step));

        param[i] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}



// ==================================================================================== //
//                                  train: API
// ==================================================================================== //

trainer_t * trainer_new(trainer_loss_type_t loss_type, trainer_opt_type_t opt_type) {
    trainer_t* trn = sys_malloc(sizeof(trainer_t));
    memset(trn, 0, sizeof(trainer_t));
    trn->loss_type = loss_type;
    trn->opt_type = opt_type;
    switch(loss_type) {
        case TRAINER_LOSS_CROSS_ENTROPY:
            trn->loss = loss_cross_entropy;
            break;
        case TRAINER_LOSS_MSE:
        default:
            trn->loss = loss_mse;
            break;
    }
    switch(opt_type) {
        case TRAINER_OPT_ADAM:
            break;
        case TRAINER_OPT_SGD:
        default:
            break;
    }
    return trn;
}


void trainer_free(trainer_t* trn) {
    if(trn) {
        if(trn->priv) {
            free(trn->priv);
            trn->priv = NULL;
        }
        free(trn);
        trn = NULL;
    }
}