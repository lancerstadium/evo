#include "../evo.h"
#include "../util/log.h"
#include "../util/sys.h"
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
    mdl->dev = device_registry_find("cpu");     /* Default: device cpu      */
    // load model
    mdl->model = NULL;
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

static int tensor_map_print(const void* key, size_t ksize, uintptr_t value, void* usr) {
    LOG_INFO("%s,", (char*)key);
    return 0;
}

void model_dump_tensor(model_t *mdl) {
    LOG_INFO("%s[%d] = [", mdl->name, hashmap_size(mdl->tensor_map));
    hashmap_iterate(mdl->tensor_map, tensor_map_print, NULL);
    LOG_INFO("]\n");
}

void model_free(model_t *mdl) {
    if(mdl) {
        if(mdl->name) free(mdl->name);
        if(mdl->model) mdl->sez->unload(mdl);
        if(mdl->tensor_map) hashmap_free(mdl->tensor_map);
    }
    mdl = NULL;
}