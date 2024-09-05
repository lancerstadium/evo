#include <evo.h>
#include <evo/util/log.h>
#include <evo/util/sys.h>
#include <string.h>


model_t * model_new(const char *name) {
    model_t * mdl = (model_t*)sys_malloc(sizeof(model_t));
    if(name) {
        mdl->name = sys_strdup(name);
    } else {
        mdl->name = NULL;
    }
    mdl->sez = NULL;
    mdl->scd = scheduler_get_default();         /* Default: sync scheduler  */
    mdl->dev = internal_device_find("cpu");     /* Default: device cpu      */
    // load model
    mdl->model_proto = NULL;
    mdl->model_size = 0;
    // init graph
    mdl->graph = graph_new(mdl);
    // init tensor map
    mdl->tensor_map = hashmap_create();
    return mdl;
}


tensor_t* model_get_tensor(model_t *mdl, const char *name) {
    if(mdl && mdl->tensor_map) {
        tensor_t * t = NULL;
        int res = hashmap_get(mdl->tensor_map, hashmap_str_lit(name), (uintptr_t*)&t);
        if(res != 0 && t && strcmp(name, t->name) == 0) {
            return t;
        }
    }
    return NULL;
}

void model_set_tensor(model_t *mdl, const char *name, tensor_t* ts) {
    if(mdl && mdl->tensor_map) {
        tensor_t * t = NULL;
        int res = hashmap_get(mdl->tensor_map, hashmap_str_lit(name), (uintptr_t*)&t);
        if(res != 0 && t && strcmp(name, t->name) == 0) {
            tensor_copy(t, ts);
        }
    }
}

tensor_t* model_eval(model_t* mdl, tensor_t* ts) {
    if(!mdl || !mdl->graph) return NULL;
    mdl->graph->mode = 0;
    if(mdl->graph->ntensor > 0) {
        tensor_t* first = mdl->graph->tensors[0];
        tensor_copy(first, ts);
        graph_prerun(mdl->graph);
        graph_run(mdl->graph);
        graph_posrun(mdl->graph);
        return mdl->graph->tensors[mdl->graph->ntensor - 1];
    }
    return NULL;
}

void cross_entropy(tensor_t* out, tensor_t* ref) {
    if(!out || !ref 
        || out->type != ref->type 
        || out->ndata != ref->ndata) return;
    if (!out->grad) {
        char name_buf[54];
        sprintf(name_buf, "%s_grad", out->name);
        out->grad = tensor_new(name_buf, out->type);
        tensor_reshape(out->grad, out->ndim, out->dims);
    }
    tensor_t* grad = out->grad;
    float* dy = (float*)out->datas;
    float* dg = (float*)grad->datas;
    float* dr = (float*)ref->datas;
    for(int i = 0; i < grad->ndata; i++) {
        dg[i] = dy[i] - dr[i];
    }
}

void model_train(model_t* mdl, tensor_t* ref) {
    if(!mdl || !mdl->graph) return;
    mdl->graph->mode = 1;                   // train mode
    if(mdl->graph->ntensor > 0) {  
        graph_prerun(mdl->graph);
        graph_run(mdl->graph);
        graph_posrun(mdl->graph);
        // calculate grad
        tensor_t* out = mdl->graph->tensors[mdl->graph->ntensor - 1];
        cross_entropy(out, ref);
    }
}

void model_train_label(model_t* mdl, int label) {
    if (!mdl || !mdl->graph) return;
    tensor_t* out = mdl->graph->tensors[mdl->graph->ntensor - 1];
    tensor_t* ref = tensor_new("label_ref", out->type);
    tensor_reshape(ref, out->ndim, out->dims);
    if(label >= ref->ndata || label < 0) return;
    float* ref_data = (float*)ref->datas;
    ref_data[label] = 1.0f;
    model_train(mdl, ref);
    tensor_free(ref);
}

static int tensor_map_print(const void* key, size_t ksize, uintptr_t value, void* usr) {
    LOG_INFO("%s,", (char*)key);
    return 0;
}

void model_show_tensors(model_t *mdl) {
    LOG_INFO("%s[%d] = [", mdl->name, hashmap_size(mdl->tensor_map));
    hashmap_iterate(mdl->tensor_map, tensor_map_print, NULL);
    LOG_INFO("]\n");
}

void model_free(model_t *mdl) {
    if(mdl) {
        if(mdl->name) free(mdl->name);
        if(mdl->model_proto) mdl->sez->unload(mdl);
        if(mdl->tensor_map) hashmap_free(mdl->tensor_map);
        mdl->name = NULL;
        mdl->model_proto = NULL;
        mdl->tensor_map = NULL;
        sys_free(mdl);
    }
    mdl = NULL;
}