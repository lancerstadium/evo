#include "sob.h"
#include <evo.h>


UnitTest_fn_def(test_lut3d) {
    runtime_t *rt = runtime_new("onnx");
    runtime_load(rt, "model/lut3d_96/model.onnx");
    char instr[100];
    char outstr[100];
    int64_t start[3]={1,0,5};
    tensor_t* ts = model_get_tensor(rt->mdl, "input0");
    tensor_t* out = model_get_tensor(rt->mdl, "output0");
    image_t* cat;
    tensor_t* ts_t, *ts_f, *res_f, *res2;
    graph_prerun(rt->mdl->graph);
    int num=0;
    for(int num=0; num<10; num++){
        snprintf(instr, sizeof(instr), "model/lut3d_96/test_data_set_0/input_%d.jpg", num);
        snprintf(outstr, sizeof(outstr), "model/lut3d_96/test_data_set_0/result_%d.tiff", num);
        cat = image_load(instr);
        ts_t = tensor_nhwc2nchw(cat->raw);
        ts_f = tensor_cast(ts_t, TENSOR_TYPE_FLOAT32);
        for(int i = 0; i < ts_f->ndata; i++) {
            ((float*)(ts_f->datas))[i] /= 255.0f;
        }
        tensor_reshape(ts, ts_f->ndim, ts_f->dims);
        tensor_apply(ts, ts_f->datas, ts_f->ndata*sizeof(float));
        int out_dims[4] = {ts_f->dims[0],4,ts_f->dims[2],ts_f->dims[3]}; 
        tensor_reshape(out, ts_f->ndim, out_dims);
        double time_st = 0.0, time_ed = 0.0;
        // time_st = sys_time();
        graph_run(rt->mdl->graph);
        // time_ed = sys_time();
        // fprintf(stderr,"run time:%f\n",time_ed-time_st);
        graph_exec_report_level(rt->mdl->graph, 1); // Exec dump
        int Height = ts->dims[2];
        int Width = ts->dims[3];
    
        res_f = tensor_nchw2nhwc(out);
        float* ddd = res_f->datas;
        for(int i = 0; i < res_f->ndata; i++) {
            ddd[i] =  (ddd[i] < 0 ? 0 : (ddd[i] > 1.0 ? 1.0 : ddd[i])) * 255;
        } 
        image_t* img = image_from_tensor(res_f);
        image_save(img, outstr);

    }
    runtime_free(rt);
    return NULL;
}
UnitTest_fn_def(test_all) {
    device_reg("cpu");
    UnitTest_add(test_lut3d);
    device_unreg("cpu");
    return NULL;
}

UnitTest_run(test_all);