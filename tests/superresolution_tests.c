#include "sob.h"
#include <evo.h>


UnitTest_fn_def(test_model) {
    model_t* mdl = model_load("model/edsr_v1/model.onnx");
    
    // load
    image_t* cat = image_load("picture/DIV2K/00004.png");
    // image_to_grey(cat);
    // image_crop_center(cat, 256, 256);

    tensor_t* ts_w = tensor_nhwc2nchw(cat->raw);
    tensor_t* ts_f = tensor_cast(ts_w, TENSOR_TYPE_FLOAT32);

    // tensor_dump(ts_f);
    model_set_tensor(mdl, "input", ts_f);
    
    graph_prerun(mdl->graph);
    graph_run(mdl->graph);
    graph_posrun(mdl->graph);
    // graph_dump(mdl->graph);
    tensor_t* out_f = model_get_tensor(mdl, "output");
    for(int i = 0; i < out_f->ndata; i++) {
        if (((float*)(out_f->datas))[i] < 0) {
            ((float*)(out_f->datas))[i] = 0;
        } else if(((float*)(out_f->datas))[i] > 255.0f) {
            ((float*)(out_f->datas))[i] = 255.0f;
        }
    }
    // tensor_dump(out_f);
    // tensor_t* out_c = tensor_nchw2nhwc(out_f);
    image_t* out_img = image_from_tensor(out_f);
    image_save(out_img, "superresolution-out.jpg");
    model_save(mdl, "superresolution.dot");
    // // tensor_dump2(model_get_tensor(mdl, "/layer1/block/block.5/Clip_output_0"));


    graph_exec_report_level(mdl->graph, 1); // Exec dump
    

    return NULL;
}

UnitTest_fn_def(test_all) {
    device_reg("cpu");
    UnitTest_add(test_model);
    device_unreg("cpu");
    return NULL;
}

UnitTest_run(test_all);