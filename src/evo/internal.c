#include <evo.h>
#include <evo/util/log.h>
#include <string.h>

// ==================================================================================== //
//                                  internal: model
// ==================================================================================== //

model_t* internal_model_init() {
    device_t* dft_dev = internal_device_find_nowarn(EVO_DFT_DEV);
    if(dft_dev == NULL) {
        dft_dev = device_reg(EVO_DFT_DEV);
    }
    internal_context_info.mdl = malloc(sizeof(model_t));
    internal_context_info.mdl->name = "internal_context_info.mdl",
    internal_context_info.mdl->dev = dft_dev,
    internal_context_info.mdl->scd = scheduler_get_default(),
    internal_context_info.mdl->sez = serializer_new("onnx"),
    internal_context_info.mdl->graph = graph_new(internal_context_info.mdl),
    internal_context_info.mdl->tensor_map = hashmap_create();
    internal_context_info.mdl->model_proto = NULL;
    internal_context_info.mdl->model_size = 0;
    return internal_context_info.mdl;
}

void internal_model_set(model_t *mdl) {
    internal_context_info.mdl = mdl;
}

model_t* internal_model_get() {
    return internal_context_info.mdl;
}

// ==================================================================================== //
//                                  internal: device registry
// ==================================================================================== //

int internal_device_init(const char* name) {
    if(!internal_context_info.dev_vec) {
        internal_context_info.dev_vec = vector_create();
        if(!internal_context_info.dev_vec) {
            LOG_CRIT("Can not init device %s, create the vector failed.\n", name);
            return -1;
        }
    }
    return 0;
}

device_t* internal_device_find(const char* name) {
    if(!internal_context_info.dev_vec) {
        LOG_CRIT("Can not find any device, module was not inited.\n");
        return NULL;
    }
    int cnt = vector_size(internal_context_info.dev_vec);
    if(cnt == 0) {
        LOG_CRIT("Can not find any device, module was empty.\n");
        return NULL;
    }
    for(int i = 0; i < cnt; i++) {
        device_t* dev = internal_context_info.dev_vec[i];
        if(strcmp(dev->name, name) == 0) {
            return dev;
        }
    }
    LOG_CRIT("Can not find device %s, module was empty.\n", name);
    return NULL;
}

device_t* internal_device_find_nowarn(const char* name) {
    if(!internal_context_info.dev_vec) {
        return NULL;
    }
    int cnt = vector_size(internal_context_info.dev_vec);
    if(cnt == 0) {
        return NULL;
    }
    for(int i = 0; i < cnt; i++) {
        device_t* dev = internal_context_info.dev_vec[i];
        if(strcmp(dev->name, name) == 0) {
            return dev;
        }
    }
    return NULL;
}

device_t* internal_device_get(int idx) {
    int cnt = vector_size(internal_context_info.dev_vec);
    if(idx >= 0 && idx < cnt) {
        device_t* dev = internal_context_info.dev_vec[idx];
        return dev;
    } else {
        return NULL;
    }
}
device_t* internal_device_get_default() {
    return internal_device_find(EVO_DFT_DEV);
}

void internal_device_release() {
    while(vector_size(internal_context_info.dev_vec) > 0) {
        device_t* dev = internal_context_info.dev_vec[0];
        device_unreg_dev(dev);
    }
    vector_free(internal_context_info.dev_vec);
    internal_context_info.dev_vec = NULL;
}

int device_reg_dev(device_t* dev) {
    if(!dev) {
        LOG_CRIT("Device is null, reg fail\n");
        return -1;
    }
    internal_device_init(dev->name);
    if(!internal_context_info.dev_vec) {
        LOG_CRIT("Device %s register fail, module was not be inited.\n", dev->name);
        return -1;
    }
    device_t * find_dev = internal_device_find_nowarn(dev->name);
    if(find_dev) {
        return 0;
    }
    // interface init
    vector_add(&internal_context_info.dev_vec, dev);
    if (dev->itf && dev->itf->init)
        dev->itf->init(dev);
    int last = vector_size(internal_context_info.dev_vec);
    if (last > 0) {
        LOG_INFO("Device %s init success!\n", internal_context_info.dev_vec[last - 1]->name);
    }
    return 0;
}

int device_unreg_dev(device_t* dev) {
    if (!dev) return -1;
    int cnt = vector_size(internal_context_info.dev_vec);
    if (cnt == 0) {
        LOG_CRIT("Can not remove any device, module was empty.\n");
        return -1;
    }
    for (int i = 0; i < cnt; i++) {
        if (strcmp(internal_context_info.dev_vec[i]->name, dev->name) == 0) {
            vector_remove(internal_context_info.dev_vec, i);
            // interface release
            if (dev->itf && dev->itf->release) dev->itf->release(dev);
            LOG_INFO("Device %s release success!\n", internal_context_info.dev_vec[i]->name);
            return 0;
        }
    }
    return -1;
}

int device_unreg(const char* name) {
    if (!name) return -1;
    int cnt = vector_size(internal_context_info.dev_vec);
    if (cnt == 0) {
        LOG_CRIT("Can not remove any device, module was empty.\n");
        return -1;
    }
    for (int i = 0; i < cnt; i++) {
        if (strcmp(internal_context_info.dev_vec[i]->name, name) == 0) {
            vector_remove(internal_context_info.dev_vec, i);
            // interface release
            if (internal_context_info.dev_vec[i]->itf && internal_context_info.dev_vec[i]->itf->release) internal_context_info.dev_vec[i]->itf->release(internal_context_info.dev_vec[i]);
            LOG_INFO("Device %s release success!\n", internal_context_info.dev_vec[i]->name);
            return 0;
        }
    }
    return -1;
}