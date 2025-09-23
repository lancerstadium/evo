#include <evo/resolver.h>
#include <evo/util/math.h>
float fmodx(float a, float b) {
    int c = a/b;
    return a - (b*c);
}
void Tri_forward_float32(node_t* nd) {
    tensor_t* img_t = nd->in[0];
    tensor_t* lut_t = nd->in[1];
    tensor_t* out_t = nd->out[0];
    float* lut = (float*)lut_t->datas;
    float* img = (float*)img_t->datas; 
    float* out = (float*)out_t->datas;
    int W = img_t->dims[2];
    int H = img_t->dims[3];
    int dim = lut_t->dims[lut_t->ndim-1];
    int shi = dim*dim*dim;
    float bin = 1.000001/(dim-1);
    int img_size = H*W, index;
    int r_p,g_p,b_p;
    float r,g,b,    r_o,g_o,b_o; 
    int pos[8]; float off[8]; 
    for(index=0; index<img_size; index++){
        r = img[index];         g = img[index+img_size];    b = img[index+img_size*2];
        r_p = r/bin;            g_p = g/bin;                b_p = b/bin;
        r_o = fmodx(r,bin)/bin; g_o = fmodx(g,bin)/bin;     b_o = fmodx(b,bin)/bin;
        pos[0] = r_p + g_p*dim + b_p*dim*dim;               //  r,  g,  b,
        pos[1] = pos[0] + dim*dim;                          //  r,  g,  b+1,
        pos[2] = pos[0] + dim;                              //  r,  g+1,b,
        pos[3] = pos[1] + dim;                              //  r,  g+1,b+1,
        pos[4] = pos[0] + 1;                                //  r+1,g,  b,
        pos[5] = pos[1] + 1;                                //  r+1,g,  b+1,
        pos[6] = pos[2] + 1;                                //  r+1,g+1,b,
        pos[7] = pos[3] + 1;                                //  r+1,g+1,b+1,
        off[7] = r_o*g_o*b_o;                               //  r,  g,  b,      rgb = 7
        off[6] = r_o*g_o - off[7];                          //  r,  g,  1-b,    rg = 7+6
        off[5] = r_o*b_o - off[7];                          //  r,  1-g,b,      rb = 7+5
        off[4] = r_o - off[7] - off[6] - off[5];            //  r,  1-g,1-b,    
        off[3] = g_o*b_o - off[7];                          //  1-r,g,  b,      gb = 7+3
        off[2] = g_o - off[7] - off[6] - off[3];            //  1-r,g,  1-b,
        off[1] = b_o - off[7] - off[5] - off[3];            //  1-r,1-g,b,
        off[0] = 1 - r_o - g_o + off[7] + off[6] - off[1];  //  1-r,1-g,1-b,    1-r-g-b+rg+rb+gb-rgb
        for(int i=0; i<4; i++){
            out[index+img_size*i]=0.0;
            for(int j=0; j<8; j++)
                out[index+img_size*i] += off[j]*lut[pos[j]+shi*i];
        }
    }
}
void Tri_init(node_t *nd) {
    if (!nd || !nd->in) return;
}
void Tri_reshape(node_t *nd) {
    tensor_t *img = nd->in[0];
    tensor_t *out = nd->out[0];
    out->type = img->type;
    int new_dims[img->ndim];
    new_dims[0] = img->dims[0];
    new_dims[1] = 4;
    new_dims[2] = img->dims[2];
    new_dims[3] = img->dims[3];
    tensor_reshape(out, img->ndim, new_dims);
}
void Tri_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    switch (nd->in[0]->type) {
        // case TENSOR_TYPE_INT8:
        //     Tri_forward_int8(nd);
        //     break;
        // case TENSOR_TYPE_INT16:
        //     Tri_forward_int16(nd);
        //     break;
        // case TENSOR_TYPE_INT32:
        //     Tri_forward_int32(nd);
        //     break;
        // case TENSOR_TYPE_INT64:
        //     Tri_forward_int64(nd);
        //     break;
        // case TENSOR_TYPE_UINT8:
        //     Tri_forward_uint8(nd);
        //     break;
        // case TENSOR_TYPE_UINT16:
        //     Tri_forward_uint16(nd);
        //     break;
        // case TENSOR_TYPE_UINT32:
        //     Tri_forward_uint32(nd);
        //     break;
        // case TENSOR_TYPE_UINT64:
        //     Tri_forward_uint64(nd);
        //     break;
        // case TENSOR_TYPE_FLOAT16:
        //     Tri_forward_float16(nd);
        //     break;
        // case TENSOR_TYPE_BFLOAT16:
        //     Tri_forward_bfloat16(nd);
        //     break;
        case TENSOR_TYPE_FLOAT32:
            Tri_forward_float32(nd);
            break;
        // case TENSOR_TYPE_FLOAT64:
        //     Tri_forward_float64(nd);
        //     break;
        default:
            break;
    }
}
void Tri_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    return;
}
void op_Tri_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Tri_init;
    nd->op->reshape     = Tri_reshape;
    nd->op->forward     = Tri_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Tri_exit;
}