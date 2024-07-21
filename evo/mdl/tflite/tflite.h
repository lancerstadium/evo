#include "../../evo.h"


#ifndef __EVO_MDL_TFLITE_TFLITE_H__
#define __EVO_MDL_TFLITE_TFLITE_H__


model_t *load_tflite(struct serializer *s, const void *buf, size_t len);
model_t *load_model_tflite(struct serializer *sez, const char *path);
void unload_tflite(model_t *mdl);
tensor_t *load_tensor_tflite(const char *path);
graph_t *load_graph_tflite(model_t *mdl);



#endif // __EVO_MDL_TFLITE_TFLITE_H__