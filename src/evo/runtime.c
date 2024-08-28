#include <evo.h>
#include <util/log.h>
#include <util/sys.h>
#include <string.h>

runtime_t* runtime_new(const char* fmt) {
    device_reg(EVO_DFT_DEV);
    runtime_t * rt = (runtime_t*)sys_malloc(sizeof(runtime_t));
    rt->dev_reg_vec = internal_context_info.dev_vec;
    rt->sez = serializer_new(fmt);
    rt->mdl = NULL;
    return rt;
}

model_t* runtime_load(runtime_t *rt, const char *path) {
    if(!rt) return NULL;
    if(rt->sez && rt->sez->load_model) {
        if(rt->mdl) {
            runtime_unload(rt);
        }
        rt->mdl = rt->sez->load_model(rt->sez, path);
        return rt->mdl;
    }
    return NULL;
}

model_t* runtime_load_raw(runtime_t *rt, const void* buf, size_t size) {
    if(!rt) return NULL;
    if(rt->sez && rt->sez->load_model) {
        if(rt->mdl) {
            runtime_unload(rt);
        }
        rt->mdl = rt->sez->load(rt->sez, buf, size);
        return rt->mdl;
    }
    return NULL;
}

tensor_t* runtime_load_tensor(runtime_t *rt, const char *path) {
    if(!rt) return NULL;
    if(rt->sez && rt->sez->load_tensor) {
        tensor_t * ts = rt->sez->load_tensor(path);
        return ts;
    }
    return NULL;
}

void runtime_set_tensor(runtime_t *rt, const char *name, tensor_t *ts) {
    if(!rt) return;
    tensor_t * trg = runtime_get_tensor(rt, name);
    tensor_copy(trg, ts);
}

tensor_t* runtime_get_tensor(runtime_t *rt, const char *name) {
    if(!rt) return NULL;
    if(rt->mdl) {
        return model_get_tensor(rt->mdl, name);
    }
    return NULL;
}

void runtime_run(runtime_t *rt) {
    if(!rt) return;
    if(rt->mdl && rt->mdl->graph) {
        graph_prerun(rt->mdl->graph);
        graph_run(rt->mdl->graph);
        graph_posrun(rt->mdl->graph);
    }
}

void runtime_dump_graph(runtime_t *rt) {
    if(!rt) return;
    if(rt->mdl && rt->mdl->graph) {
        graph_dump(rt->mdl->graph);
    }
}

void runtime_unload(runtime_t *rt) {
    if(!rt) return;
    if(rt->mdl && rt->mdl->sez && rt->mdl->sez->unload) {
        rt->mdl->sez->unload(rt->mdl);
    }
    rt->mdl = NULL;
}

void runtime_free(runtime_t *rt) {
    if(!rt) return;
    runtime_unload(rt);
    if(rt->sez) serializer_free(rt->sez);
    sys_free(rt);
    rt = NULL;
    device_unreg(EVO_DFT_DEV);
}


device_t* runtime_reg_dev(runtime_t *rt, const char *name) {
    return device_reg(name);
}

void runtime_unreg_dev(runtime_t *rt, const char *name) {
    device_unreg(name);
}