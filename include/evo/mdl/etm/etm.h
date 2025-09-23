#include <evo.h>


#ifndef __EVO_MDL_ETM_ETM_H__
#define __EVO_MDL_ETM_ETM_H__


model_t *load_etm(struct serializer *s, const void *buf, size_t len);
model_t *load_model_etm(struct serializer *sez, const char *path);
void unload_etm(model_t *mdl);
tensor_t *load_tensor_etm(const char *path);
graph_t *load_graph_etm(model_t *mdl);
void save_etm(model_t *mdl, const char* path);


#endif // __EVO_MDL_ETM_ETM_H__