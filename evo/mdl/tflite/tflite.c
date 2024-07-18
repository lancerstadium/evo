#include "tflite.h"
#include "tflite_builder.h"
#include "tflite_reader.h"
#include <stdio.h>


#undef ns
#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(etm, x)  // Specified in the schema.

EVO_UNUSED context_t *load_tflite(struct serializer *s, const void *buf, int len) {
    return NULL;
}


EVO_UNUSED context_t *load_model_tflite(struct serializer *sez, const char *path) {
    return NULL;
}

EVO_UNUSED void unload_tflite(context_t *ctx);
EVO_UNUSED tensor_t *load_tensor_tflite(const char *path);
EVO_UNUSED graph_t *load_graph_tflite(context_t *ctx);






#undef ns