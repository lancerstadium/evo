#include "sob.h"
#include <evo.h>
#include <evo/resolver.h>
#include <math.h>


image_t* out_img = NULL;
canvas_t* out_cav = NULL;
canvas_t* kitten(canvas_t* cav, float dt) {
    if(!cav) return NULL;

    // 0. init canvas
    canvas_fill(cav, 0x0);
    if(dt == 0.0) {
        out_cav = canvas_from_image(out_img);
    }


    // 1. cat photo
    {
        canvas_draw(cav, 0, 0, out_cav->width, out_cav->height, out_cav->pixels);
    }

    return cav;
}


float KL_calc(tensor_t* t1, tensor_t* t2) {
    float sum = 0.0f;
    float* d1 = t1->datas;
    float* d2 = t2->datas;
    float p, q;
    for(int i = 0; i < t1->ndata / 2; i++) {
        p = d1[i * 2 + 1];
        q = d2[i * 2 + 1];
        if(p != 0 && q != 0) sum = sum + p * logf(p / q);
    }
    return sum;
}

void KL_quant(tensor_t* ts, const char* path, bool save_fig) {
    // 1. Find max abs
    float* data = ts->datas;
    float max = data[0], min = data[0];
    for(int i = 0; i < ts->ndata; i++) {
        if(data[i] > max) max = data[i];
        if(data[i] < min) min = data[i];
    }
    if(max < 0) max = -max;
    if(min < 0) min = -min;
    float max_abs = max > min ? max : min;

    float search_start_scale = 0.7;
    float search_step = 0.01;
    int NUM = 60;
    tensor_t* ts_hist, *q_ts_hist, *q_ts_hist_ext;
    tensor_t* q_tss[NUM];
    float kl[NUM];

    for(int x = 0; x < NUM; x++){
        // 2. Get distbute P
        float T = max_abs * (search_start_scale + x * search_step);
        float interval = max_abs / 2048;
        ts_hist = tensor_hist(ts, -T, T, 2048);
        float scale = T / 127;
        int8_t zero_point = 0;

        // 3. Quant P to Q
        tensor_t* sc_ts = tensor_new_float32("scale", (int[]){1}, 1,(float[]){scale}, 1);
        tensor_t* zp_ts = tensor_new("zero_point", TENSOR_TYPE_INT8);
        tensor_reshape(zp_ts, 1, (int[]){1});
        tensor_apply(zp_ts, (int8_t[]){zero_point}, 1);
        q_tss[x] = tensor_quant(ts, sc_ts, zp_ts, 1, TENSOR_TYPE_INT8);
        q_ts_hist = tensor_hist(q_tss[x], -127, 127, 128);

        // 4. Extend Q to 2048
        q_ts_hist_ext = tensor_new("extend_hist", TENSOR_TYPE_FLOAT32);
        tensor_reshape(q_ts_hist_ext, 2, (int[]){2048, 2});
        int ext_sc = 2048 / 128;
        int ext_cnt;
        float* thd = ts_hist->datas;                // 2048 * 2
        float* qthd = q_ts_hist->datas;             // 128  * 2
        float* qthed = q_ts_hist_ext->datas;        // 2048 * 2
        for(int i = 0; i < q_ts_hist->ndata / 2; i++) { // 128  * 2
            ext_cnt = 0;
            for(int j = 0; j < ext_sc; j++) {
                ext_cnt = ext_cnt + (thd[(i * ext_sc + j) * 2 + 1] == 0 ? 0 : 1);
            }
            for(int j = 0; j < ext_sc; j++) {
                qthed[(i * ext_sc + j) * 2 + 0] = qthd[i * 2 + 0];
                qthed[(i * ext_sc + j) * 2 + 1] = thd[(i * ext_sc + j) * 2 + 1] == 0 ? 0 : (qthd[i * 2 + 1] / ext_cnt);
            }
        }
        // if(x == 0)
        //     tensor_dump2(ts_hist);

        kl[x] = KL_calc(ts_hist, q_ts_hist_ext);
        // fprintf(stderr, "T: %f, KL: %f\n", T, kl[x]);
    }

    int idx_kl = 0;
    int min_kl = kl[0];
    for(int i = 0; i < NUM; i++) {
        if(kl[i] < min_kl) {
            min_kl = kl[i];
            idx_kl = i;
        }
    }
    fprintf(stderr, "Idx: %d, Min KL : %f\n", idx_kl, kl[idx_kl]);
    tensor_save(q_tss[idx_kl], path, "a");

    if(save_fig) {
        figure_t* fig = figure_new_2d("hist", FIGURE_TYPE_VECTOR, FIGURE_PLOT_TYPE_BAR, ts_hist);
        // fig->axiss[0]->is_auto_scale = false;
        // fig->axiss[0]->range_min = -0.003;
        // fig->axiss[0]->range_max = 0.003;
        // fig->axiss[1]->is_auto_scale = false;
        // fig->axiss[1]->range_min = 0;
        // fig->axiss[1]->range_max = 3;
        fig->plot_vec[0]->bar.bwidth = 3;
        figure_save(fig, "hist.svg");
    }
}

UnitTest_fn_def(test_model) {
    model_t* mdl = model_load("model/edsr_v1/model_16_v2.onnx");
    int SIZE = 256;
    int SCALE = 2;
    int INSIZE = 16;
    int CLIP = SIZE / INSIZE;
    int BLOCK = SIZE / CLIP;
    int NEW_SIZE = SIZE * SCALE;
    int NEW_BLOCK = NEW_SIZE / CLIP;
    // load
    image_t* cat_img;
    tensor_t *ts_w, *ts_f, *out_f;
    tensor_t *ts_out = tensor_new("out", TENSOR_TYPE_FLOAT32);
    tensor_reshape(ts_out, 4, (int[]){1, 3, NEW_SIZE, NEW_SIZE});
    cat_img = image_load("picture/DIV2K/00005_256.png");
    ts_w = tensor_nhwc2nchw(cat_img->raw);
    ts_f = tensor_cast(ts_w, TENSOR_TYPE_FLOAT32);
    tensor_save(ts_f, "input.w", "w");
    // fprintf(stderr, "Mem Usage: %s\n", sys_mem_size(sys_mem_usage()));
    for(int i = 0; i < CLIP; i++) {
        for(int j = 0; j < CLIP; j++) {
            cat_img = image_load("picture/DIV2K/00005_256.png");
            // image_to_grey(cat_img);
            image_crop(cat_img, i * BLOCK, j * BLOCK, BLOCK, BLOCK);
            // Preprocess
            ts_w = tensor_nhwc2nchw(cat_img->raw);
            ts_f = tensor_cast(ts_w, TENSOR_TYPE_FLOAT32);
            model_set_tensor(mdl, "input", ts_f);
            if(i == 0 && j == 0) {
                // tensor_save(ts_f, "input.w", "w");
                graph_prerun(mdl->graph);
            }
            graph_run(mdl->graph);
            if(i == CLIP && j == CLIP) graph_posrun(mdl->graph);
            // graph_dump1(mdl->graph);
            out_f = model_get_tensor(mdl, "output");
            for(int i = 0; i < out_f->ndata; i++) {
                if (((float*)(out_f->datas))[i] < 0) {
                    ((float*)(out_f->datas))[i] = 0;
                } else if(((float*)(out_f->datas))[i] > 255.0f) {
                    ((float*)(out_f->datas))[i] = 255.0f;
                }
            }
            // if(i == 0 && j == 0) {
            //     tensor_save(out_f, "output.w", "w");
            // }
            // Reform Image
            for(int c = 0; c < 3; c++) {
                for(int m = 0; m < NEW_BLOCK; m++) {
                    for(int n = 0; n < NEW_BLOCK; n++) {
                        ((float*)(ts_out->datas))[(c * NEW_SIZE * NEW_SIZE) + (j * NEW_BLOCK + m) * NEW_SIZE + (i * NEW_BLOCK + n)] = ((float*)(out_f->datas))[c * NEW_BLOCK * NEW_BLOCK + m * NEW_BLOCK + n];
                    }
                }
            }
            tensor_free(ts_w);
            tensor_free(ts_f);
            image_free(cat_img);
        }
    }
    fprintf(stderr, "Mem Usage: %s\n", sys_mem_size(sys_mem_usage()));
    // mem_arena_t* ma = mem_arena_from_graph(mdl->graph);
    // mem_arena_solve(ma, MEM_ARENA_TYPE_LSTF);
    // mem_arena_dump(ma);
    // mem_arena_save_svg(ma, "mem.svg", 1024);
    out_img = image_from_tensor(ts_out);
    // tensor_save(ts_out, "output.w", "w");
    // tensor_dump(ts_out);
    image_save(out_img, "superresolution-out.jpg");
    // graph_dump3(mdl->graph);
    tensor_free(ts_out);
    // model_save(mdl, "superresolution.etm");
    image_free(out_img);    
    model_save(mdl, "superresolution.dot");


    tensor_t* mid;
    float scale = 0;
    int zero_point = 0;
    tensor_t* sc_ts = tensor_new("scale", TENSOR_TYPE_FLOAT32);
    tensor_t* zp_ts;
    tensor_reshape(sc_ts, 1, (int[]){1});
    tensor_t* mid_q;
    for(int i = 0; i < mdl->graph->ntensor; i++) {
        mid = mdl->graph->tensors[i];
        if(!mid->is_param) {
            // bool save_fig = false;
            // if(i == 35) {
            //     save_fig = true;
            // }
            // KL_quant(mid, "edsr.w", save_fig);
            if(strstr(mid->name, "bias") != NULL) {
                // Quant_solve(mid->datas, mid->ndata, &scale, &zero_point, 32, 1, 1);
                // zp_ts = tensor_new("zero_point", TENSOR_TYPE_INT32);
                // tensor_reshape(zp_ts, 1, (int[]){1});
                // tensor_apply(sc_ts, (float[]){scale}, 1 * sizeof(float));
                // tensor_apply(zp_ts, (int32_t[]){zero_point}, 1 * sizeof(int32_t));
                // mid_q = tensor_quant(mid, sc_ts, zp_ts, 1, TENSOR_TYPE_INT32);
            } else {
                // Quant_solve(mid->datas, mid->ndata, &scale, &zero_point, 8, 1, 0);
                // zp_ts = tensor_new("zero_point", TENSOR_TYPE_INT8);
                // tensor_reshape(zp_ts, 1, (int[]){1});
                // tensor_apply(sc_ts, (float[]){scale}, 1 * sizeof(float));
                // tensor_apply(zp_ts, (int8_t[]){zero_point}, 1 * sizeof(int8_t));
                // mid_q = tensor_quant(mid, sc_ts, zp_ts, 1, TENSOR_TYPE_INT8);
                // fprintf(stderr, "%-18s\t: %f\t%d\n", mid->name, scale, zero_point);
                // tensor_save(mid_q, "tiny_edsr_q.w", "a");
            }
        }
    }
    // tensor_t* ts_hist = model_get_tensor(mdl, "37");
    // tensor_save(ts_hist, "hist.w", "w");
    // graph_exec_report_level(mdl->graph, 1); // Exec dump
    // renderer_t* rd = renderer_new(SIZE, SIZE, RENDERER_TYPE_LINUX);
    // renderer_run(rd, kitten);
    // renderer_free(rd);
    model_free(mdl);
    return NULL;
}


UnitTest_fn_def(test_all) {
    device_reg("cpu");
    fprintf(stderr, "Mem Usage: %s\n", sys_mem_size(sys_mem_usage()));
    UnitTest_add(test_model);
    device_unreg("cpu");
    fprintf(stderr, "Mem Usage: %s\n", sys_mem_size(sys_mem_usage()));
    return NULL;
}

UnitTest_run(test_all);