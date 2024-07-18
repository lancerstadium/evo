
#include "tflite.h"


EVO_UNUSED context_t *load_tflite(struct serializer *s, const void *buf, int len) {
    return NULL;
}


EVO_UNUSED context_t *load_model_tflite(struct serializer *sez, const char *path) {
    return NULL;
}

EVO_UNUSED void unload_tflite(context_t *ctx);
EVO_UNUSED tensor_t *load_tensor_tflite(const char *path);
EVO_UNUSED graph_t *load_graph_tflite(context_t *ctx);
