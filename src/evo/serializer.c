
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
    ctx->model_size = len;
    if (!ctx->model) {
        if (ctx)
            free(ctx);
        return NULL;
    }
    return ctx;
}

context_t * load_file_onnx (struct serializer* sez, const char* path) {
    context_t * ctx = NULL;
    FILE *fp;
    uint32_t len;
    unsigned int i;
    void * buf;
    fp = fopen(path, "rb");
    if (fp) {
        fseek(fp, 0L, SEEK_END);
        len = ftell(fp);
        fseek(fp, 0L, SEEK_SET);
        if(len > 0) {
            buf = malloc(len);
            if(buf) {
                for(i = 0; i < len; i += fread(buf + i, 1, len - i, fp));
                ctx = load_onnx(sez, buf, len);
                free(buf);
            }
        }
        fclose(fp);
    }
    return ctx;
}


// ==================================================================================== //
//                                    serializer API
// ==================================================================================== //

serializer_t * serializer_new() {
    serializer_t * s = (serializer_t *)sys_malloc(sizeof(serializer_t));
    // default load by onnx
    s->init = NULL;
    s->load = load_onnx;
    s->load_file = load_file_onnx;
    s->release = NULL;
    return s;
}