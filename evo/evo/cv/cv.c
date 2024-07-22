#include "../../evo.h"
#include <string.h>

attribute_t* image_get_attr(image_t* img, const char *name) {
    attribute_t *attr;
    int i;
    if(img && name) {
        for(i = 0; i < vector_size(img->attr_vec); i++) {
            attr = img->attr_vec[i];
            if(strcmp(attr->name, name) == 0) {
                return attr;
            }
        }
    }
    return NULL;
}


void image_free(image_t* img) {
    if (img) {
        if(img->name) free(img->name);
        if(img->raw) tensor_free(img->raw);
        if(img->attr_vec) vector_free(img->attr_vec);
        free(img);
        img = NULL;
    }
}