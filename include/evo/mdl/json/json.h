#include <evo.h>


#ifndef __EVO_MDL_JSON_JSON_H__
#define __EVO_MDL_JSON_JSON_H__


model_t *load_json(struct serializer *s, const void *buf, size_t len);
model_t *load_model_json(struct serializer *sez, const char *path);
void unload_json(model_t *mdl);
tensor_t *load_tensor_bin(const char *path);
graph_t *load_graph_json(model_t *mdl);
void save_json(model_t *mdl, const char* path);


#endif // __EVO_MDL_JSON_JSON_H__