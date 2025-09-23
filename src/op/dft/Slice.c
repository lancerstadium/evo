#include <evo/resolver.h>
#include <string.h>
typedef struct {
    int64_t* starts;
    int64_t* ends;
    int64_t* steps;
}operator_pdata_t;
void Slice_init(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    operator_pdata_t* pdat=sys_malloc(sizeof(operator_pdata_t));
    if (pdat) {
        memset(pdat,0,sizeof(operator_pdata_t));
        nd->priv=pdat;
    } else {
        free(pdat);
        return;
    }
    
}
void Slice_reshape(node_t* nd) {
    if (!nd || !nd->in) return;
    if (!(nd->nin >= 3) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) || (nd->in[2]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED || nd->in[2]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    tensor_t* input = nd->in[0];     
    tensor_t* output = nd->out[0];    
    int ndim = input->ndim;   
    int adim=ndim;
    int* in_shape = input->dims;  
    int out_shape[ndim];       
    memcpy(out_shape, in_shape, ndim * sizeof(int64_t));
    operator_pdata_t* info=(operator_pdata_t*)nd->priv;
    info->starts=sys_malloc(ndim * sizeof(int64_t));
    info->ends=sys_malloc(ndim * sizeof(int64_t));
    info->steps=sys_malloc(ndim * sizeof(int64_t));
    for(int i=0; i<ndim; i++){
        info->starts[i]=0;
        info->ends[i]=in_shape[i];
        info->steps[i]=1;
    }
    if(nd->nin>3){
        adim=nd->in[3]->ndata;
        int64_t* axes=nd->in[3]->datas;
        for(int i=0; i<adim;i++)
            axes[i]=axes[i]<0 ? axes[i]+ndim : axes[i];
        int64_t tmp[adim];  
        memcpy(tmp,nd->in[1]->datas,adim*sizeof(int64_t));
        for(int i=0; i<adim; i++)
            info->starts[axes[i]]=tmp[i];
        memcpy(tmp,nd->in[2]->datas,adim*sizeof(int64_t));
        for(int i=0; i<adim; i++)
            info->ends[axes[i]]=tmp[i];
        if(nd->nin>4){
            memcpy(tmp,nd->in[4]->datas,adim*sizeof(int64_t));
            for(int i=0; i<adim; i++)
                info->steps[axes[i]]=tmp[i];
        }
    }    
    for(int i=0; i<ndim; i++){
        int64_t step=info->steps[i],start,end;
        start = info->starts[i]<0 ? info->starts[i]+in_shape[i] : info->starts[i];
        start = start<0 ? -1 : (start>in_shape[i] ? in_shape[i] : start);
        end = info->ends[i]<0 ? info->ends[i]+in_shape[i] : info->ends[i];
        end = end<0 ? -1 : (end>in_shape[i] ? in_shape[i] : end);
        int len = (end-start+(step>0?-1:1))/step+1;
        out_shape[i]=len;
        info->starts[i]=start;
        info->ends[i]=end;
    }
    output->type=input->type;
    tensor_reshape(output, ndim, out_shape);
}
void slice_dg_int64(int64_t* arr, int64_t** res, int dim, int ndim, int pos, int64_t* start, int64_t* end, int64_t* step, int* stride){
    if(dim==ndim)
        *(*res)++ = arr[pos];
    else {
        if(step[dim]>0)
            for(int i=start[dim]; i<end[dim]; i+=step[dim])
                slice_dg_int64(arr,res,dim+1,ndim,pos+i*stride[dim],start,end,step,stride); 
        else    
            for(int i=start[dim]; i>end[dim]; i+=step[dim])
                slice_dg_int64(arr,res,dim+1,ndim,pos+i*stride[dim],start,end,step,stride); 
    }
}
void slice_dg_float32(float* arr, float** res, int dim, int ndim, int pos, int64_t* start, int64_t* end, int64_t* step, int* stride){
    if(dim==ndim)
        *(*res)++ = arr[pos];
    else {
        if(step[dim]>0)
            for(int i=start[dim]; i<end[dim]; i+=step[dim])
                slice_dg_float32(arr,res,dim+1,ndim,pos+i*stride[dim],start,end,step,stride);
        else
            for(int i=start[dim]; i>end[dim]; i+=step[dim])
                slice_dg_float32(arr,res,dim+1,ndim,pos+i*stride[dim],start,end,step,stride);
    }
}
void Slice_forward_int64(node_t* nd) {
    tensor_t* input = nd->in[0];     
    tensor_t* output = nd->out[0];    
    int ndim = input->ndim;          
    int64_t* starts=NULL,*ends=NULL,*steps=NULL; 
    operator_pdata_t* info=(operator_pdata_t *)nd->priv;
    starts = info->starts;
    ends = info->ends;
    steps = info->steps;
    int64_t* in_data=input->datas;
    int64_t* out_data=output->datas;
    int* in_stride=input->strides; 
    slice_dg_int64(in_data,&out_data,0,ndim,0,starts,ends,steps,in_stride);
}
void Slice_forward_float32(node_t* nd) {
    tensor_t* input = nd->in[0];    
    tensor_t* output = nd->out[0];  
    int ndim = input->ndim;         
    int64_t* starts=NULL,*ends=NULL,*steps=NULL;     
    operator_pdata_t* info=(operator_pdata_t *)nd->priv;
    starts = info->starts;
    ends = info->ends;
    steps = info->steps;
    float *in_data=input->datas;
    float *out_data=output->datas;
    int* in_stride=input->strides; 
    slice_dg_float32(in_data,&out_data,0,ndim,0,starts,ends,steps,in_stride);
}
void Slice_forward(node_t *nd) {
    if (!nd || !nd->in) return;
    if (!(nd->nin >= 3) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) || (nd->in[2]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED || nd->in[2]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_INT64:
            Slice_forward_int64(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            Slice_forward_float32(nd);
            break;
        default:
            break;
    }
}
void Slice_exit(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *info = (operator_pdata_t *)nd->priv;
    if (info)
        free(info);
    nd->priv = NULL;
}
void op_Slice_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Slice_init;
    nd->op->reshape     = Slice_reshape;
    nd->op->forward     = Slice_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Slice_exit;
}