
#include "evo.h"
#include "sys.h"
#include "log.h"
#include "dev/cpu/def.h"
#include <string.h>


device_t * device_new(const char *name) {
    device_t *dev = (device_t *)malloc(sizeof(device_t));
    if(name) {
        dev->name = name;
    } else {
        dev->name = "cpu";      /* Default: cpu */
    }
    dev->rsv = NULL;
    dev->itf = NULL;
    dev->alc = NULL;
    dev->opt = NULL;
    dev->scd = NULL;
    return dev;
}

device_t* device_reg(const char* name) {
    if(strcmp(name, "cpu") == 0) {
        return device_reg_cpu();
    } else {
        LOG_WARN("Device register input no name!\n");
        return NULL;
    }
}


op_t * device_find_op(device_t *dev, op_type_t t) {
    if(dev && dev->rsv) {
        op_t *trg_op = &dev->rsv->op_tbl[t];
        return trg_op;
    }
    return NULL;
}


