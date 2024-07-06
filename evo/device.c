
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
    dev->itf = NULL;
    dev->alc = NULL;
    dev->opt = NULL;
    dev->scd = NULL;
    return dev;
}