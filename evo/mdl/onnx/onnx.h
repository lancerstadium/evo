#include "../../evo.h"


#ifndef __EVO_MDL_ONNX_ONNX_H__
#define __EVO_MDL_ONNX_ONNX_H__


model_t *load_onnx(struct serializer *s, const void *buf, size_t len);
model_t *load_model_onnx(struct serializer *sez, const char *path);
void unload_onnx(model_t *mdl);
tensor_t *load_tensor_onnx(const char *path);
graph_t *load_graph_onnx(model_t *mdl);





#endif // __EVO_MDL_ONNX_ONNX_H__