
#include "evo.h"
#include "sys.h"
#include "dev/cpu/def.h"


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



op_t * device_find_op(device_t *dev, op_type_t t) {
    if(dev && dev->rsv) {
        op_t *trg_op = &dev->rsv->op_tbl[t];
        return trg_op;
    }
    return NULL;
}


