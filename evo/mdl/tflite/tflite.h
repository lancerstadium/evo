#include "../../evo.h"


#ifndef __EVO_MDL_TFLITE_TFLITE_H__
#define __EVO_MDL_TFLITE_TFLITE_H__


context_t *load_tflite(struct serializer *s, const void *buf, size_t len);
context_t *load_model_tflite(struct serializer *sez, const char *path);
void unload_tflite(context_t *ctx);
tensor_t *load_tensor_tflite(const char *path);
graph_t *load_graph_tflite(context_t *ctx);



#endif // __EVO_MDL_TFLITE_TFLITE_H__