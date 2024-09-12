#include <evo.h>
#include <evo/util/sys.h>
#include <evo/util/log.h>
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

void loss_grad_mse(float *output, float *target, float *grad, int size) {
    for (int i = 0; i < size; i++) {
        grad[i] = 2.0f * (output[i] - target[i]) / size;
    }
}

// Loss Function: cross entropy
float loss_cross_entropy(float *output, float *target, int size) {
    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        loss -= target[i] * logf(output[i] + 1e-10);  // 加1e-10防止log(0)
    }
    return loss / size;
}

void loss_grad_cross_entropy(float *output, float *target, float *grad, int size) {
    for (int i = 0; i < size; i++) {
        grad[i] = - (target[i] / (output[i] + 1e-10)) + ((1.0f - target[i]) / (1.0f - output[i] + 1e-10));
    }
}

// ==================================================================================== //
//                                  train: update sgd
// ==================================================================================== //

void update_sgd(trainer_t *trn, tensor_t* ts) {
    if(!trn || !ts || !ts->grad || ts->type != TENSOR_TYPE_FLOAT32) return;

    float learning_rate = trn->learning_rate;

    float *td = ts->datas;
    float *gd = ts->grad->datas;

    for (int i = 0; i < ts->ndata; i++) {
        td[i] -= learning_rate * gd[i];
    }
}

// ==================================================================================== //
//                                  train: update sgdm
// ==================================================================================== //

typedef struct {
    float *v;                                           /* 2nd moment estimation value  */
    float momentum;                                     /* 2nd moment estimation coeff  */
} trainer_opt_sgdm_t;

void update_sgdm(trainer_t *trn, tensor_t* ts) {
    if(!trn || !ts || !ts->grad || ts->type != TENSOR_TYPE_FLOAT32) return;
    trainer_opt_sgdm_t *state = (trainer_opt_sgdm_t *)trn->priv;
    float learning_rate = trn->learning_rate;
    float momentum = state->momentum;
    float *td = ts->datas;
    float *gd = ts->grad->datas;

    for (int i = 0; i < ts->ndata; i++) {
        state->v[i] = momentum * state->v[i] + learning_rate * gd[i];
        td[i] -= state->v[i];
    }
}

// ==================================================================================== //
//                                  train: update adam
// ==================================================================================== //

typedef struct {
    float *m;                                           /* 1st moment estimation value  */
    float *v;                                           /* 2nd moment estimation value  */
    float beta1;                                        /* 1st moment estimation coeff  */
    float beta2;                                        /* 2nd moment estimation coeff  */
} trainer_opt_adam_t;

void update_adam(trainer_t *trn, tensor_t* ts) {
    if(!trn || !trn->priv || !ts || !ts->grad || ts->type != TENSOR_TYPE_FLOAT32) return;
    trainer_opt_adam_t *state = (trainer_opt_adam_t *)trn->priv;
    float beta1 = state->beta1;
    float beta2 = state->beta2;
    float learning_rate = trn->learning_rate;
    float epsilon = trn->epsilon;

    float *td = ts->datas;    
    float *gd = ts->grad->datas;

    for (int i = 0; i < ts->ndata; i++) {
        state->m[i] = beta1 * state->m[i] + (1 - beta1) * gd[i];
        state->v[i] = beta2 * state->v[i] + (1 - beta2) * gd[i] * gd[i];

        float m_hat = state->m[i] / (1 - powf(beta1, trn->step));
        float v_hat = state->v[i] / (1 - powf(beta2, trn->step));

        td[i] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

// ==================================================================================== //
//                                  train: API
// ==================================================================================== //

trainer_t * trainer_new(float learning_rate, float epsilon, trainer_loss_type_t loss_type, trainer_opt_type_t opt_type) {
    trainer_t* trn = sys_malloc(sizeof(trainer_t));
    memset(trn, 0, sizeof(trainer_t));
    if(learning_rate > 0) trn->learning_rate = learning_rate;
    if(epsilon > 0) trn->epsilon = epsilon;
    trn->step = 0;
    trn->loss_type = loss_type;
    trn->opt_type = opt_type;
    switch(loss_type) {
        case TRAINER_LOSS_CROSS_ENTROPY:
            trn->loss = loss_cross_entropy;
            trn->loss_grad = loss_grad_cross_entropy;
            break;
        case TRAINER_LOSS_MSE:
        default:
            trn->loss = loss_mse;
            trn->loss_grad = loss_grad_mse;
            break;
    }
    switch(opt_type) {
        case TRAINER_OPT_ADAM: {
            // adam optim init
            trainer_opt_adam_t* priv = sys_malloc(sizeof(trainer_opt_adam_t));
            memset(priv, 0, sizeof(trainer_opt_adam_t));
            trn->priv = priv;
            trn->update = update_adam;
            break;
        }
        case TRAINER_OPT_SGD:
        default: {
            trn->update = update_sgd;
            break;
        }
    }
    return trn;
}

float trainer_step(trainer_t* trn, model_t* mdl, tensor_t* trg) {
    if(!trn || !mdl || !mdl->graph || mdl->graph->ntensor <= 0 || !trg) return 0;
    // 1. set train mode & forward
    graph_prerun(mdl->graph);
    graph_set_mode(mdl->graph, 0);
    graph_run(mdl->graph);
    // 2. calculate loss & grad
    float loss = 0.0;
    tensor_t* out = mdl->graph->tensors[mdl->graph->ntensor - 1];
    if(out->type == TENSOR_TYPE_FLOAT32 && trg->type == TENSOR_TYPE_FLOAT32 && out->ndata == trg->ndata) {
        float* od = out->datas;
        float* td = trg->datas;
        if(trn->loss) loss = trn->loss(od, td, out->ndata);
        if (!out->grad) {
            char name_buf[54];
            sprintf(name_buf, "%s_grad", out->name);
            out->grad = tensor_new(name_buf, out->type);
            tensor_reshape(out->grad, out->ndim, out->dims);
        }
        float* gd = out->grad->datas;
        if(trn->loss_grad) trn->loss_grad(od, td, gd, out->ndata);
        // LOG_INFO("[%4d] Trainer Loss: %.4f\n", trn->step, loss);
    }
    // 3. backward & update param
    graph_set_mode(mdl->graph, 1);
    graph_run(mdl->graph);
    if(trn->update) {
        for(int i = 0; i < mdl->graph->ntensor; i++) {
            tensor_t* ts = mdl->graph->tensors[i];
            if(ts && ts->grad && ts->is_param) {
                trn->update(trn, ts);
            }
        }
    }
    graph_posrun(mdl->graph);
    trn->step++;
    return loss;
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