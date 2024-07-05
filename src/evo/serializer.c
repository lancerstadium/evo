
#include "evo.h"
#include "sys.h"


// ==================================================================================== //
//                                      onnx
// ==================================================================================== //

context_t * load_onnx (struct serializer* s, const void * buf, int len) {
    context_t * ctx = NULL;
    if(!buf || len <= 0)
        return NULL;
    ctx = context_new(NULL);
    ctx->sez = s;
    ctx->model = onnx__model_proto__unpack(NULL, len, buf);
    ctx->model_size = sizeof(Onnx__ModelProto);
    if (!ctx->model) {
        if (ctx)
            free(ctx);
        return NULL;
    }
    return ctx;
}



// ==================================================================================== //
//                                    serializer API
// ==================================================================================== //

serializer_t * serializer_new() {
    serializer_t * s = (serializer_t *)sys_malloc(sizeof(serializer_t));
    s->init = NULL;
    s->load = load_onnx;
    s->release = NULL;
    return s;
}