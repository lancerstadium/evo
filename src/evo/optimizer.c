#include <evo.h>
#include <evo/util/sys.h>
#include <evo/util/log.h>
#include <string.h>
#include <math.h>


// ref: https://zhuanlan.zhihu.com/p/416979875

// ==================================================================================== //
//                                  optimizer: loss func
// ==================================================================================== //

// Loss Function: MSE
float loss_mse(float *output, float *target, int size) {
    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = output[i] - target[i];
        loss += diff * diff;
    }
    return loss / size;  // 对应 PyTorch 的 MSE 实现，这里不需要除以2
}

// Gradient of MSE Loss
void loss_grad_mse(float *output, float *target, float *grad, int size) {
    for (int i = 0; i < size; i++) {
        grad[i] = 2 * (output[i] - target[i]) / size;  // PyTorch 中 MSE 的梯度是 2 倍误差
    }
}

// Loss Function: Cross Entropy (general case, no Softmax involved)
float loss_cross_entropy(float *output, float *target, int size) {
    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        // 加上一个较小的值以避免 log(0) 导致的数值不稳定问题
        loss -= target[i] * logf(output[i] + 1e-10);  
    }
    return loss / size;
}

// Gradient of Cross Entropy Loss (general case)
void loss_grad_cross_entropy(float *output, float *target, float *grad, int size) {
    for (int i = 0; i < size; i++) {
        // 使用标准的交叉熵梯度公式
        grad[i] = - target[i] / (output[i] + 1e-10);  // 加1e-10避免除以0
    }
}

// ==================================================================================== //
//                                  optimizer: update sgd
// ==================================================================================== //

// ref: https://blog.csdn.net/weixin_39228381/article/details/108310520
void update_sgd(optimizer_t *opt, tensor_t* ts) {
    if(!opt || !ts || !ts->grad || ts->type != TENSOR_TYPE_FLOAT32) return;
    float lr = opt->lr;
    // float momentum = priv->momentum;
    float *td = ts->datas;
    float *gd = ts->grad->datas;

    for (int i = 0; i < ts->ndata; i++) {
        td[i] -= lr * gd[i];
    }
}

// ==================================================================================== //
//                                  optimizer: update adam
// ==================================================================================== //

void update_adam(optimizer_t *opt, tensor_t* ts) {
    if(!opt || !ts || !ts->grad || ts->type != TENSOR_TYPE_FLOAT32) return;
    float beta1 = opt->adam.beta1;
    float beta2 = opt->adam.beta2;
    float lr = opt->lr;
    float epsilon = opt->epsilon;

    float *td = ts->datas;    
    float *gd = ts->grad->datas;

    float m[ts->ndata];
    float v[ts->ndata];

    for (int i = 0; i < ts->ndata; i++) {

        m[i] = beta1 * m[i] + (1 - beta1) * gd[i];
        v[i] = beta2 * v[i] + (1 - beta2) * gd[i] * gd[i];

        float m_hat = m[i] / (1 - powf(beta1, opt->cur_step));
        float v_hat = v[i] / (1 - powf(beta2, opt->cur_step));

        td[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

// ==================================================================================== //
//                                  optimizer: API
// ==================================================================================== //

optimizer_t * optimizer_new(float lr, float epsilon, optimizer_loss_type_t loss_type, optimizer_type_t type) {
    optimizer_t* opt = sys_malloc(sizeof(optimizer_t));
    memset(opt, 0, sizeof(optimizer_t));
    opt->cur_step = 0;
    opt->cur_loss = -1.0;
    opt->lr = lr > 0 ? lr : 1e-2;
    opt->lr_decay = 0.0;
    opt->wt_decay = 0.0;
    opt->epsilon = epsilon > 0 ? epsilon : 1e-8f;
    opt->loss_type = loss_type;
    opt->type = type;
    switch(loss_type) {
        case OPTIMIZER_LOSS_TYPE_CROSS_ENTROPY:
            opt->loss = loss_cross_entropy;
            opt->loss_grad = loss_grad_cross_entropy;
            break;
        case OPTIMIZER_LOSS_TYPE_MSE:
        default:
            opt->loss = loss_mse;
            opt->loss_grad = loss_grad_mse;
            break;
    }
    switch(type) {
        case OPTIMIZER_TYPE_ADAM:{
            // adam optim init
            opt->update = update_adam;
            opt->adam.beta1 = 0.9f;             /* default: */
            opt->adam.beta2 = 0.999f;           /* default: */
            break;
        }
        case OPTIMIZER_TYPE_SGD:
        default: {
            // agd optim init
            opt->update = update_sgd;
            opt->sgd.momentum = 0.0f;          /* default: */
            opt->sgd.dampening = 0.0f;         /* default: */
            break;
        }
    }
    return opt;
}

float optimizer_loss(optimizer_t* opt, model_t* mdl, tensor_t* trg, bool no_grad) {
    if(!opt || !mdl || !mdl->graph || mdl->graph->ntensor <= 0 || !trg) return -1.0f;
    float loss = -1.0f;
    tensor_t* out = mdl->graph->tensors[mdl->graph->ntensor - 1];
    if(out->type == TENSOR_TYPE_FLOAT32 && trg->type == TENSOR_TYPE_FLOAT32 && out->ndata == trg->ndata) {
        float* od = out->datas;
        float* td = trg->datas;
        if(opt->loss) loss = opt->loss(od, td, out->ndata);
        if (!out->grad) {
            char name_buf[54];
            sprintf(name_buf, "%s_grad", out->name);
            out->grad = tensor_new(name_buf, out->type);
            tensor_reshape(out->grad, out->ndim, out->dims);
        }
        float* gd = out->grad->datas;
        if(!no_grad && opt->loss_grad) {
            opt->loss_grad(od, td, gd, out->ndata);
        }
    }
    return loss;
}

void optimizer_update_grad(optimizer_t* opt, model_t* mdl) {
    if(!opt || !mdl || !mdl->graph || mdl->graph->ntensor <= 0) return;
    if(opt->update) {
        for(int i = 0; i < mdl->graph->ntensor; i++) {
            tensor_t* ts = mdl->graph->tensors[i];
            if(ts && ts->grad && ts->is_param) {
                opt->update(opt, ts);
            }
        }
    }
}

void optimizer_zero_grad(optimizer_t* opt, model_t* mdl) {
    if(!opt || !mdl || !mdl->graph || mdl->graph->ntensor <= 0) return;
    for(int i = 0; i < mdl->graph->ntensor; i++) {
        tensor_t* ts = mdl->graph->tensors[i];
        tensor_fill_zero(ts->grad);
    }
}

float optimizer_step(optimizer_t* opt, model_t* mdl, tensor_t* trg) {
    if(!opt || !mdl || !mdl->graph || mdl->graph->ntensor <= 0 || !trg) return -1.0f;
    // 1. set optimizer mode & forward
    graph_prerun(mdl->graph);
    graph_set_mode(mdl->graph, 0);
    graph_run(mdl->graph);
    // 2. calculate loss & grad
    float loss = optimizer_loss(opt, mdl, trg, false);
    // 3. backward & update param
    graph_set_mode(mdl->graph, 1);
    graph_run(mdl->graph);
    optimizer_update_grad(opt, mdl);
    graph_posrun(mdl->graph);
    graph_set_mode(mdl->graph, 0);
    opt->cur_loss = loss;
    opt->cur_step++;
    return loss;
}


void optimizer_free(optimizer_t* opt) {
    if(opt) {
        if(opt->priv) {
            switch(opt->type) {
                case OPTIMIZER_TYPE_ADAM:{
                    // TODO: free
                    break;
                }
                default: break;
            }
            free(opt->priv);
            opt->priv = NULL;
        }
        free(opt);
        opt = NULL;
    }
}