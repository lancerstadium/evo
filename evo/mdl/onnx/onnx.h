#include "../../evo.h"


#ifndef __EVO_MDL_ONNX_ONNX_H__
#define __EVO_MDL_ONNX_ONNX_H__


context_t *load_model_onnx(struct serializer *sez, const char *path);
void unload_onnx(context_t *ctx);
tensor_t *load_tensor_onnx(const char *path);
graph_t *load_graph_onnx(context_t *ctx);
context_t *load_onnx(struct serializer *s, const void *buf, int len);




#endif // __EVO_MDL_ONNX_ONNX_H__