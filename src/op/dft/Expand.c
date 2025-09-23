#include <evo/resolver.h>
#include <evo/util/log.h>
#include <string.h>
void Expand_init(node_t* nd) {
    if (!nd || !nd->in) 
        return;
}
void Expand_reshape(node_t* nd) {
    if (!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 2) || !(nd->nout == 1) || (nd->in[0]->ndim == 0) || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    if(nd->in[1]->type != TENSOR_TYPE_INT64) return;
    tensor_t* x = nd->in[0];            /* 输入张量 */
    tensor_t* y = nd->in[1];            /* 形状张量 */
    tensor_t* z = nd->out[0];           /* 输出张量 */
    int x_ndim = x->ndim;                   /* 输入张量的维度 */
    int y_ndata = y->ndata;                 /* 目标形状的维度 */
    int* x_dims = (int*)x->dims;            /* 输入形状的数组 */
    int64_t* y_datas = (int64_t*)y->datas;  /* 目标形状的数组 */
    int max = x_ndim>y_ndata ? x_ndim : y_ndata;    /* 输出张量的维度 */
    int z_dims[max];                                /* 输出形状的数组 */
    int xd = x_ndim, yd = y_ndata;
    for(int i=max;i>0;){
        int xdim = xd>0 ? x_dims[--xd] : 1;
        int ydim = yd>0 ? y_datas[--yd] : 1;
        if((xdim!=ydim)&&(xdim!=1)&&(ydim!=1)){
            return;
        }
        z_dims[--i] = xdim==1 ? ydim : xdim; 
    }
    z->type = x->type;
    tensor_reshape(z, max, z_dims);
}
void Expand_forward_int64(node_t* nd) {
    tensor_t* x = nd->in[0];    /* 输入张量 */
    tensor_t* y = nd->out[0];   /* 输出张量 */
    int64_t* xdatas = (int64_t*)x->datas;
    int64_t* ydatas = (int64_t*)y->datas;
    int indim[y->ndim],outdim[y->ndim],bsdim[y->ndim];
    int xs=1,ys=1,gs=0;
    for(int xd = x->ndim-1,yd=y->ndim-1; yd>=0; xd--,yd--){
        int xdim= xd>=0 ? x->dims[xd] : 1;
        xs *= xdim;
        ys *= y->dims[yd];
        if((xdim==1 && y->dims[yd]>1) || yd==0){
            indim[gs]=xs;
            outdim[gs]=ys;
            bsdim[gs++]=y->dims[yd]/xdim;
        }
    }
    int cp_len = indim[0];
    int cp_byte = cp_len*(sizeof(*xdatas));
    int offs[indim[gs-1]/indim[0]];
    for(int i=0; i<indim[gs-1]/indim[0]; i++){
        int in_off = i*cp_len, out_off = 0;
        for(int j=gs-2,sy=in_off; j>=0;j--){
            out_off += sy/indim[j]*outdim[j];
            sy = sy % indim[j];
        }
        memcpy(ydatas + out_off, xdatas + in_off, cp_byte);
        offs[i] = out_off;
    }
    for(int i=0; i<gs; i++){
        for(int j=0; j<indim[gs-1]/indim[0]; j++){
            if(offs[j] % outdim[i] == 0){   
                cp_len = outdim[i] / bsdim[i];
                cp_byte = cp_len*(sizeof(*xdatas));
                int from = offs[j];
                int at = from + cp_len;
                int end = from + outdim[i];
                while(at + cp_len <= end){
                    memcpy(ydatas+at, ydatas+from, cp_byte);
                    at += cp_len;
                    cp_len <<= 1;
                    cp_byte <<= 1;
                }
                while(at < end){
                    cp_len >>= 1;
                    cp_byte >>= 1;
                    if(at + cp_len <= end){
                        memcpy(ydatas+at, ydatas+from, cp_byte);
                        at += cp_len;
                    }
                }
            }
        }
    }
}
void Expand_forward_float(node_t* nd) {
    tensor_t* x = nd->in[0];    /* 输入张量 */
    tensor_t* y = nd->out[0];   /* 输出张量 */
    float* xdatas = (float*)x->datas;
    float* ydatas = (float*)y->datas;
    int indim[y->ndim],outdim[y->ndim],bsdim[y->ndim];
    int xs=1,ys=1,gs=0;
    for(int xd = x->ndim-1,yd=y->ndim-1; yd>=0; xd--,yd--){
        int xdim= xd>=0 ? x->dims[xd] : 1;
        xs *= xdim;
        ys *= y->dims[yd];
        if((xdim==1 && y->dims[yd]>1) || yd==0){
            indim[gs]=xs;
            outdim[gs]=ys;
            bsdim[gs++]=y->dims[yd]/xdim;
        }
    }
    int cp_len = indim[0];
    int cp_byte = cp_len*(sizeof(*xdatas));
    int offs[indim[gs-1]/indim[0]];
    for(int i=0; i<indim[gs-1]/indim[0]; i++){
        int in_off = i*cp_len, out_off = 0;
        for(int j=gs-2,sy=in_off; j>=0;j--){
            out_off += sy/indim[j]*outdim[j];
            sy = sy % indim[j];
        }
        memcpy(ydatas + out_off, xdatas + in_off, cp_byte);
        offs[i] = out_off;
    }
    for(int i=0; i<gs; i++){
        for(int j=0; j<indim[gs-1]/indim[0]; j++){
            if(offs[j] % outdim[i] == 0){   
                cp_len = outdim[i] / bsdim[i];
                cp_byte = cp_len*(sizeof(*xdatas));
                int from = offs[j];
                int at = from + cp_len;
                int end = from + outdim[i];
                while(at + cp_len <= end){
                    memcpy(ydatas+at, ydatas+from, cp_byte);
                    at += cp_len;
                    cp_len <<= 1;
                    cp_byte <<= 1;
                }
                while(at < end){
                    cp_len >>= 1;
                    cp_byte >>= 1;
                    if(at + cp_len <= end){
                        memcpy(ydatas+at, ydatas+from, cp_byte);
                        at += cp_len;
                    }
                }
            }
        }
    }
}
void Expand_forward(node_t* nd) {
    if (!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 2) || !(nd->nout == 1) || (nd->in[0]->ndim == 0) || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
     if(nd->nin < 2 || nd->in[1]->type != TENSOR_TYPE_INT64) return;
    switch(nd->in[0]->type){
        case TENSOR_TYPE_INT64:
            Expand_forward_int64(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            Expand_forward_float(nd);
            break;
        default:
            break;
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