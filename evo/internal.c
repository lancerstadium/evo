
#include "evo.h"
#include "log.h"
#include <string.h>

// ==================================================================================== //
//                                  internal: device
// ==================================================================================== //

EVO_UNUSED static int device_registry_init(const char* name) {
    if(!internal_device_registry) {
        internal_device_registry = vector_create();
        if(internal_device_registry) {
            LOG_CRIT("Can not init device %s, create the vector failed.\n", name);
            return -1;
        }
    }
    return 0;
}

device_t* device_registry_find(const char* name) {
    if(!internal_device_registry) {
        LOG_CRIT("Can not find any device, module was not inited.\n");
        return NULL;
    }
    int cnt = vector_size(&internal_device_registry);
    if(cnt == 0) {
        LOG_CRIT("Can not find any device, module was empty.\n");
        return NULL;
    }
    for(int i = 0; i < cnt; i++) {
        device_t* dev = &internal_device_registry[i];
        if(strcmp(dev->name, name) == 0) {
            return dev;
        }
    }
    LOG_CRIT("Can not find device %s, module was empty.\n", name);
    return NULL;
}

device_t* device_registry_get(int idx) {
    int cnt = vector_size(internal_device_registry);
    if(idx >= 0 && idx < cnt) {
        device_t* dev = &internal_device_registry[idx];
        return dev;
    } else {
        return NULL;
    }
}

void device_registry_release() {
    while(vector_size(internal_device_registry) > 0) {
        device_t* dev = &internal_device_registry[0];
        device_unreg(dev);
    }
    vector_free(internal_device_registry);
    internal_device_registry = NULL;
}

int device_reg(device_t* dev) {
    if(!dev) return -1;
    device_registry_init(dev->name);
    if(!internal_device_registry) {
        LOG_CRIT("Tengine: Can not register %s, module was not be inited.\n", dev->name);
        return -1;
    }
    /// TODO: interface init
    vector_add(&internal_device_registry, *dev);
    return 0;
}

int device_unreg(device_t* dev) {
    if(!dev) return -1;
    int cnt = vector_size(&internal_device_registry);
    if(cnt == 0) {
        LOG_CRIT("Can not remove any device, module was empty.\n");
        return -1;
    }
    for(int i = 0; i < cnt; i++) {
        if(strcmp(internal_device_registry[i].name, dev->name) == 0) {
            vector_remove(internal_device_registry, i);
            /// TODO: interface release
            return 0;
        }
    }
    return -1;
}