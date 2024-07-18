#include "../mdl/onnx/onnx.h"
#include "../util/log.h"
#include <string.h>

// ==================================================================================== //
//                                      onnx
// ==================================================================================== //



static serializer_t onnx_serializer = {
    .fmt = "onnx",
    .load = load_onnx,
    .load_model = load_model_onnx,
    .load_tensor = load_tensor_onnx,
    .unload = unload_onnx,
    .load_graph = load_graph_onnx,
};

// ==================================================================================== //
//                                    serializer API
// ==================================================================================== //

serializer_t *serializer_new(const char *fmt) {
    if (strcmp(fmt, "onnx") == 0) {
        return &onnx_serializer;
    } else {  // default load by onnx
        LOG_WARN("Unsupport model format %s , use onnx as default\n", fmt);
        return &onnx_serializer;
    }
}

void serializer_free(serializer_t *sez) {
    if (sez) {
        sez->fmt = NULL;
        sez->load = NULL;
        sez->load_model = NULL;
        sez->load_graph = NULL;
        sez->unload = NULL;
        sez = NULL;
    }
}