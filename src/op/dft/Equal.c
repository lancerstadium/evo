#include <evo/resolver.h>
#include <string.h>

void Equal_init(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
}

void Equal_reshape(node_t* nd) {
    if (!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[0]->ndim == 1) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    tensor_t* A = nd->in[0];  /* 第一个输入张量 */
    tensor_t* B = nd->in[1];  /* 第二个输入张量 */
    tensor_t* C = nd->out[0]; /* 输出张量 */

    int A_ndim = A->ndim;     /* 输入张量A的维度 */
    int B_ndim = B->ndim;     /* 输入张量B的维度 */
    int max_ndim = (A_ndim > B_ndim) ? A_ndim : B_ndim; /* 输出张量的最大维度 */

    /* 初始化输出张量的维度数组 */
    int C_dims[max_ndim];

    /* 从后往前处理广播，符合NumPy的广播规则 */
    for (int i = 0; i < max_ndim; i++) {
        int A_dim = (i >= (max_ndim - A_ndim)) ? A->dims[i - (max_ndim - A_ndim)] : 1;
        int B_dim = (i >= (max_ndim - B_ndim)) ? B->dims[i - (max_ndim - B_ndim)] : 1;

        if (A_dim != B_dim && A_dim != 1 && B_dim != 1) {
            /* 如果维度不兼容，广播失败，退出 */
            return;
        }
        C_dims[i] = (A_dim == 1) ? B_dim : ((B_dim == 1) ? A_dim : A_dim);
    }

    /* 更新输出张量的形状 */
    C->type = TENSOR_TYPE_BOOL;
    tensor_reshape(C, max_ndim, C_dims);
}

void Equal_forward(node_t* nd) {
    if (!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[0]->ndim == 1) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED
        || nd->in[0]->type != nd->in[1]->type) {
        return;
    }
    tensor_t* A = nd->in[0];  /* 第一个输入张量 */
    tensor_t* B = nd->in[1];  /* 第二个输入张量 */
    tensor_t* C = nd->out[0]; /* 输出张量 */

    int A_ndim = A->ndim;     /* 输入张量A的维度 */
    int B_ndim = B->ndim;     /* 输入张量B的维度 */
    int max_ndim = (A_ndim > B_ndim) ? A_ndim : B_ndim; /* 输出张量的最大维度 */

    /* 获取输入张量的大小 */
    int64_t C_size = C->ndata;  /* 输出张量的大小 */

    /* 获取元素的位宽 */
    int A_elem_size = tensor_type_sizeof(A->type);
    int B_elem_size = tensor_type_sizeof(B->type);

    /* 遍历输出张量，逐个元素进行比较 */
    for (int i = 0; i < C_size; i++) {
        /* 计算A和B在广播情况下的索引 */
        int64_t A_idx = 0;
        int64_t B_idx = 0;
        int64_t C_idx = i;

        for (int j = 0; j < max_ndim; j++) {
            int A_dim = (j >= (max_ndim - A_ndim)) ? A->dims[j - (max_ndim - A_ndim)] : 1;
            int B_dim = (j >= (max_ndim - B_ndim)) ? B->dims[j - (max_ndim - B_ndim)] : 1;
            int C_stride = C->strides[j];

            int C_dim_idx = (C_idx / C_stride) % C->dims[j];

            if (A_dim != 1) {
                int A_stride = A->strides[j - (max_ndim - A_ndim)];
                A_idx += (C_dim_idx % A_dim) * A_stride;
            }

            if (B_dim != 1) {
                int B_stride = B->strides[j - (max_ndim - B_ndim)];
                B_idx += (C_dim_idx % B_dim) * B_stride;
            }
        }

        /* 比较 A 和 B 中的元素 */
        bool equal_result = memcmp((char*)A->datas + A_idx * A_elem_size,
                                   (char*)B->datas + B_idx * B_elem_size,
                                   A_elem_size) == 0;

        /* 将结果写入输出张量C，使用uint8_t表示bool */
        ((uint8_t*)C->datas)[i] = equal_result ? 1 : 0;
    }
}


void Equal_exit(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    return;
}


void op_Equal_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Equal_init;
    nd->op->reshape     = Equal_reshape;
    nd->op->forward     = Equal_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Equal_exit;
}