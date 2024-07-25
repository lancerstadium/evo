#include "../mdl/onnx/onnx.h"
#include "../mdl/tflite/tflite.h"
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
//                                      tflite
// ==================================================================================== //

static serializer_t tflite_serializer = {
    .fmt = "tflite",
    .load = load_tflite,
    .load_model = load_model_tflite,
    .load_tensor = NULL,
    .unload = unload_tflite,
    .load_graph = load_graph_tflite,
};


// ==================================================================================== //
//                                    serializer API
// ==================================================================================== //

serializer_t *serializer_new(const char *fmt) {
    if (strcmp(fmt, "onnx") == 0) {
        return &onnx_serializer;
    } else if(strcmp(fmt, "tflite") == 0) {
        return &tflite_serializer;
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