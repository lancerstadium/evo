
#include "../evo.h"
#include "../util/log.h"
#include <string.h>

// ==================================================================================== //
//                                  internal: device
// ==================================================================================== //

int device_registry_init(const char* name) {
    if(!internal_device_registry) {
        internal_device_registry = vector_create();
        if(!internal_device_registry) {
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
    int cnt = vector_size(internal_device_registry);
    if(cnt == 0) {
        LOG_CRIT("Can not find any device, module was empty.\n");
        return NULL;
    }
    for(int i = 0; i < cnt; i++) {
        device_t* dev = internal_device_registry[i];
        if(strcmp(dev->name, name) == 0) {
            return dev;
        }
    }
    LOG_CRIT("Can not find device %s, module was empty.\n", name);
    return NULL;
}

static device_t* device_registry_find_nowarn(const char* name) {
    if(!internal_device_registry) {
        return NULL;
    }
    int cnt = vector_size(internal_device_registry);
    if(cnt == 0) {
        return NULL;
    }
    for(int i = 0; i < cnt; i++) {
        device_t* dev = internal_device_registry[i];
        if(strcmp(dev->name, name) == 0) {
            return dev;
        }
    }
    return NULL;
}

device_t* device_registry_get(int idx) {
    int cnt = vector_size(internal_device_registry);
    if(idx >= 0 && idx < cnt) {
        device_t* dev = internal_device_registry[idx];
        return dev;
    } else {
        return NULL;
    }
}
device_t* device_registry_get_default() {
    return device_registry_find(EVO_DFT_DEV);
}

void device_registry_release() {
    while(vector_size(internal_device_registry) > 0) {
        device_t* dev = internal_device_registry[0];
        device_unreg_dev(dev);
    }
    vector_free(internal_device_registry);
    internal_device_registry = NULL;
}

int device_reg_dev(device_t* dev) {
    if(!dev) {
        LOG_CRIT("Device is null, reg fail\n");
        return -1;
    }
    device_registry_init(dev->name);
    if(!internal_device_registry) {
        LOG_CRIT("Device %s register fail, module was not be inited.\n", dev->name);
        return -1;
    }
    device_t * find_dev = device_registry_find_nowarn(dev->name);
    if(find_dev) {
        return 0;
    }
    // interface init
    vector_add(&internal_device_registry, dev);
    if (dev->itf && dev->itf->init)
        dev->itf->init(dev);
    int last = vector_size(internal_device_registry);
    if (last > 0) {
        LOG_INFO("Device %s init success!\n", internal_device_registry[last - 1]->name);
    }
    return 0;
}

int device_unreg_dev(device_t* dev) {
    if (!dev) return -1;
    int cnt = vector_size(internal_device_registry);
    if (cnt == 0) {
        LOG_CRIT("Can not remove any device, module was empty.\n");
        return -1;
    }
    for (int i = 0; i < cnt; i++) {
        if (strcmp(internal_device_registry[i]->name, dev->name) == 0) {
            vector_remove(internal_device_registry, i);
            // interface release
            if (dev->itf && dev->itf->release) dev->itf->release(dev);
            LOG_INFO("Device %s release success!\n", internal_device_registry[i]->name);
            return 0;
        }
    }
    return -1;
}

int device_unreg(const char* name) {
    if (!name) return -1;
    int cnt = vector_size(internal_device_registry);
    if (cnt == 0) {
        LOG_CRIT("Can not remove any device, module was empty.\n");
        return -1;
    }
    for (int i = 0; i < cnt; i++) {
        if (strcmp(internal_device_registry[i]->name, name) == 0) {
            vector_remove(internal_device_registry, i);
            // interface release
            if (internal_device_registry[i]->itf && internal_device_registry[i]->itf->release) internal_device_registry[i]->itf->release(internal_device_registry[i]);
            LOG_INFO("Device %s release success!\n", internal_device_registry[i]->name);
            return 0;
        }
    }
    return -1;
}