#include <evo/resolver.h>
#include <evo/util/log.h>
#include <string.h>

void Expand_init(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin == 2) || !(nd->nout == 1) || (nd->in[0]->ndim == 0) || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
}

void Expand_reshape(node_t* nd) {
    if (!nd || !nd->in || !nd->out) return;
    if(nd->nin < 2 || nd->in[1]->type != TENSOR_TYPE_INT64) return;

    tensor_t* x = nd->in[0];            /* 输入张量 */
    tensor_t* shape_tensor = nd->in[1]; /* 目标形状的张量 */
    tensor_t* y = nd->out[0];           /* 输出张量 */

    int x_ndim = x->ndim;                    /* 输入张量的维度数量 */
    int shape_ndim = shape_tensor->ndata;    /* 目标形状的维度数量，目标形状的数值个数 */
    int64_t* target_shape = (int64_t*)shape_tensor->datas; /* 目标形状的数组 */

    /* 初始化输出张量的维度数组 */
    int y_dims[shape_ndim];  /* 使用栈上数组来存储输出张量的维度 */

    /* 从后往前处理广播，符合NumPy的广播规则 */
    for (int i = 0; i < shape_ndim; i++) {
        int x_dim = (i >= (shape_ndim - x_ndim)) ? x->dims[i - (shape_ndim - x_ndim)] : 1;
        int shape_dim = target_shape[i];  /* 目标形状中的维度 */

        if (x_dim != shape_dim && x_dim != 1 && shape_dim != 1) {
            /* 如果维度不兼容，广播失败，退出 */
            return;
        }
        y_dims[i] = (x_dim == 1) ? shape_dim : x_dim;
    }

    /* 更新输出张量的形状 */
    tensor_reshape(y, shape_ndim, y_dims);
}


void Expand_forward(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    if(nd->nin < 2 || nd->in[1]->type != TENSOR_TYPE_INT64) return;

    tensor_t* x = nd->in[0];    /* 输入张量 */
    tensor_t* y = nd->out[0];   /* 输出张量 */

    int x_ndim = x->ndim;       /* 输入张量的维度数量 */
    int y_ndim = y->ndim;       /* 输出张量的维度数量 */
    
    int element_size = tensor_type_sizeof(x->type);  /* 获取元素的字节大小 */

    /* 遍历输出张量 y 的每一个元素 */
    for (int i = 0; i < y->ndata; i++) {
        /* 计算 y 中每个元素对应的 x 中的索引 */
        int64_t x_idx = 0;
        int64_t y_idx = i;

        /* 从后向前遍历 y 和 x 的每个维度 */
        for (int j = 0; j < y_ndim; j++) {
            int y_dim_idx = (y_idx / y->strides[j]) % y->dims[j];  /* y 中第 j 维的索引 */
            
            /* 处理广播情况 */
            if (j >= y_ndim - x_ndim) {
                int x_dim = x->dims[j - (y_ndim - x_ndim)];  /* 对应的 x 维度 */
                
                /* 如果 x 维度不为 1，计算 x 的索引 */
                if (x_dim != 1) {
                    x_idx += (y_dim_idx % x_dim) * x->strides[j - (y_ndim - x_ndim)];
                }
            }

            y_idx %= y->strides[j];  /* 更新 y_idx 为下一维度计算 */
        }

        /* 使用 memcpy 将 x 中的值复制到 y 中 */
        memcpy(y->datas + i * element_size, x->datas + x_idx * element_size, element_size);
    }
}

void Expand_exit(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    return;
}

void op_Expand_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Expand_init;
    nd->op->reshape     = Expand_reshape;
    nd->op->forward     = Expand_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Expand_exit;
}