#include <evo/resolver.h>
#include <string.h>

typedef struct {
    int64_t* starts;
    int nstart;
    int64_t* ends;
    int nend;
    int64_t* axes;
    int naxe;
    int64_t* steps;
    int nstep;
} operator_pdata_t;

void Slice_init(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin >= 3) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) || (nd->in[2]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED || nd->in[2]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t* pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        memset(pdat, 0, sizeof(operator_pdata_t));

        pdat->nstart = node_get_attr_ints(nd, "starts", &pdat->starts);
        pdat->nend = node_get_attr_ints(nd, "ends", &pdat->ends);
        pdat->naxe = node_get_attr_ints(nd, "axes", &pdat->axes);
        pdat->nstep = node_get_attr_ints(nd, "steps", &pdat->steps);
        nd->priv = pdat;
    }
}

void Slice_reshape(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin >= 3) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) || (nd->in[2]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED || nd->in[2]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    tensor_t* input = nd->in[0];      /* 输入张量 */
    tensor_t* output = nd->out[0];    /* 输出张量 */
    operator_pdata_t* pdat = (operator_pdata_t*)nd->priv;  /* 私有数据，包含 starts, ends 等信息 */

    int ndim = input->ndim;           /* 输入张量的维度数量 */
    int* input_shape = input->dims;  /* 输入张量的形状 */
    int output_shape[ndim];       /* 输出张量的形状 */

    /* 初始化输出张量形状为输入张量的形状 */
    memcpy(output_shape, input_shape, ndim * sizeof(int64_t));

    /* 如果 axes 是 NULL，默认处理所有轴 */
    if (!pdat->axes) {
        pdat->axes = malloc(ndim * sizeof(int64_t));
        for (int i = 0; i < ndim; i++) {
            pdat->axes[i] = i;
        }
        pdat->naxe = ndim;
    }

    /* 处理每个轴上的切片 */
    for (int i = 0; i < pdat->naxe; i++) {
        int axis = pdat->axes[i];

        /* 获取当前轴上的 start, end 和 step */
        int64_t start = pdat->starts[i];
        int64_t end = pdat->ends[i];
        int64_t step = (pdat->steps && pdat->nstep > i) ? pdat->steps[i] : 1;

        /* 处理负数的 start 和 end 值 */
        if (start < 0) {
            start += input_shape[axis];
        }
        if (end < 0) {
            end += input_shape[axis];
        }

        /* 限制 start 和 end 的范围 */
        start = (start < 0) ? 0 : (start >= input_shape[axis]) ? input_shape[axis] : start;
        end = (end < 0) ? 0 : (end >= input_shape[axis]) ? input_shape[axis] : end;

        /* 计算该轴上输出的长度 */
        int64_t len = (end - start + step - 1) / step;
        len = (len < 0) ? 0 : len;

        /* 更新输出张量的形状 */
        output_shape[axis] = len;
    }

    /* 设置输出张量的形状 */
    tensor_reshape(output, ndim, output_shape);
}

void Slice_forward(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin >= 3) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) || (nd->in[2]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED || nd->in[2]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    tensor_t* input = nd->in[0];      /* 输入张量 */
    tensor_t* output = nd->out[0];    /* 输出张量 */
    operator_pdata_t* pdat = (operator_pdata_t*)nd->priv;  /* 私有数据，包含 starts, ends 等信息 */

    int ndim = input->ndim;           /* 输入张量的维度数量 */
    int* output_shape = output->dims; /* 输出张量的形状 */

    int elem_size = tensor_type_sizeof(input->type);  /* 获取张量中每个元素的大小 */

    /* 初始化索引数组 */
    int64_t input_idx[ndim];
    memset(input_idx, 0, sizeof(input_idx));

    /* 初始化输出张量当前的位置 */
    int64_t output_idx[ndim];
    memset(output_idx, 0, sizeof(output_idx));

    /* 逐元素处理输出张量，复制对应位置的数据 */
    int64_t output_size = output->ndata;  /* 输出张量的元素数量 */
    for (int64_t i = 0; i < output_size; i++) {
        /* 根据输出索引，计算对应的输入张量索引 */
        for (int j = 0; j < pdat->naxe; j++) {
            int axis = pdat->axes[j];
            int64_t start = pdat->starts[j];
            int64_t step = (pdat->steps && pdat->nstep > j) ? pdat->steps[j] : 1;
            input_idx[axis] = start + output_idx[axis] * step;
        }

        /* 计算输入和输出的内存偏移量 */
        int64_t input_offset = 0;
        int64_t output_offset = 0;
        for (int j = 0; j < ndim; j++) {
            input_offset += input_idx[j] * input->strides[j];
            output_offset += output_idx[j] * output->strides[j];
        }

        /* 复制数据 */
        memcpy((char*)output->datas + output_offset * elem_size,
               (char*)input->datas + input_offset * elem_size,
               elem_size);

        /* 更新输出索引，处理多维度索引的进位 */
        for (int j = ndim - 1; j >= 0; j--) {
            output_idx[j]++;
            if (output_idx[j] < output_shape[j]) {
                break;
            }
            output_idx[j] = 0;
        }
    }
}


void Slice_exit(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}

void op_Slice_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Slice_init;
    nd->op->reshape     = Slice_reshape;
    nd->op->forward     = Slice_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Slice_exit;
}