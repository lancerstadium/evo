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
    mdl->vmodel = NULL;
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
        graph_set_mode(mdl->graph, 0);
        graph_prerun(mdl->graph);
        graph_run(mdl->graph);
        graph_posrun(mdl->graph);
        return mdl->graph->tensors[mdl->graph->ntensor - 1];
    }
    return NULL;
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

model_t* model_load(const char* path) {
    char* ext = sys_get_file_ext(path);
    serializer_t* sez = NULL;
    sez = serializer_get(ext);
    if(sez && sez->save) {
        return sez->load_file(sez, path);
    }
    return NULL;
}

void model_save(model_t *mdl, const char* path) {
    if(!mdl) return;
    char* ext = sys_get_file_ext(path);
    serializer_t* sez = NULL;
    sez = serializer_get(ext);
    if(sez && sez->save) {
        sez->save(mdl, path);
    }
}

void model_free(model_t *mdl) {
    if(mdl) {
        if(mdl->name) free(mdl->name);
        if(mdl->vmodel) mdl->sez->unload(mdl);
        if(mdl->tensor_map) hashmap_free(mdl->tensor_map);
        mdl->name = NULL;
        mdl->vmodel = NULL;
        mdl->tensor_map = NULL;
        sys_free(mdl);
    }
    mdl = NULL;
}