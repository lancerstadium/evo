#include "sob.h"
#include <evo.h>

#define WIDTH 720
#define HEIGHT 720
#define BG_COLOR 0xFF181818


canvas_t* cat_cav = NULL;
canvas_t* kitten(canvas_t* cav, float dt) {
    if(!cav) return NULL;

    // 0. init canvas
    canvas_fill(cav, BG_COLOR);
    if(dt == 0.0) {
        image_t* cat = image_load("picture/kitten.jpg");
        image_to_grey(cat);
        image_halftone_ostromoukhov(cat);
        image_save(cat, "cat-ht.jpg");
        cat_cav = canvas_from_image(cat);
    }


    // 1. cat photo
    {
        canvas_draw(cav, 0, 0, cat_cav->width, cat_cav->height, cat_cav->pixels);
    }

    return cav;
}

UnitTest_fn_def(test_floyd_steinberg) {
    // image_to_grey(img);
    renderer_t* rd = renderer_new(WIDTH, HEIGHT, RENDERER_TYPE_LINUX);
    renderer_run(rd, kitten);
    renderer_free(rd);
    return NULL;
}

UnitTest_fn_def(test_model) {
    model_t* mdl = model_load("model/halftone_v1/model.onnx");
    
    // load
    image_t* cat = image_load("picture/VOC2012/00005.png");
    image_to_grey(cat);
    // image_crop_center(cat, 256, 256);

    tensor_t* ts_f = tensor_cast(cat->raw, TENSOR_TYPE_FLOAT32);
    for(int i = 0; i < ts_f->ndata; i++) {
        ((float*)(ts_f->datas))[i] /= 255.0f;
    }
    // tensor_dump(ts_f);
    model_set_tensor(mdl, "input", ts_f);
    graph_prerun(mdl->graph);
    graph_run(mdl->graph);
    graph_posrun(mdl->graph);
    // graph_dump(mdl->graph);
    tensor_t* out_f = model_get_tensor(mdl, "output");
    tensor_t* out_hf = tensor_new("halftone", TENSOR_TYPE_FLOAT32);
    tensor_reshape(out_hf, 4, (int[]){1,1,256,256});
    for(int i = 0; i < out_hf->ndata; i++) {
        ((float*)(out_hf->datas))[i] = (((float*)(out_f->datas))[i + 256 * 256]) * 255.0f;
    }
    tensor_dump(out_hf);
    image_t* out_img = image_from_tensor(out_hf);
    image_save(cat, "halftone-gray.jpg");
    image_save_channel(out_img, "halftone-out.jpg", 0);
    model_save(mdl, "halftone.dot");
    tensor_dump2(model_get_tensor(mdl, "input"));
    
    graph_exec_report_level(mdl->graph, 1); // Exec dump
    

    return NULL;
}

UnitTest_fn_def(test_all) {
    device_reg("cpu");
    // UnitTest_add(test_floyd_steinberg);
    UnitTest_add(test_model);
    device_unreg("cpu");
    return NULL;
}

UnitTest_run(test_all);