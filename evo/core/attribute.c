#include "../evo.h"
#include "../util/sys.h"
#include <string.h>


attribute_t* attribute_undefined(char *name) {
    attribute_t *attr = sys_malloc(sizeof(attribute_t));
    memset(attr, 0, sizeof(attribute_t));
    if(attr && name) {
        attr->name = name;
        attr->type = ATTRIBUTE_TYPE_UNDEFINED;
        return attr;
    }
    return NULL;
}

attribute_t* attribute_float(char *name, float f) {
    attribute_t *attr = sys_malloc(sizeof(attribute_t));
    memset(attr, 0, sizeof(attribute_t));
    if(attr && name) {
        attr->name = name;
        attr->type = ATTRIBUTE_TYPE_FLOAT;
        attr->f = f;
        return attr;
    }
    return NULL;
}

attribute_t* attribute_int(char *name, int i) {
    attribute_t *attr = sys_malloc(sizeof(attribute_t));
    memset(attr, 0, sizeof(attribute_t));
    if(attr && name) {
        attr->name = name;
        attr->type = ATTRIBUTE_TYPE_INT;
        attr->i = i;
        return attr;
    }
    return NULL;
}

attribute_t* attribute_string(char *name, char *ss, size_t ns) {
    attribute_t *attr = sys_malloc(sizeof(attribute_t));
    memset(attr, 0, sizeof(attribute_t));
    if(attr && name) {
        attr->name = name;
        attr->type = ATTRIBUTE_TYPE_STRING;
        attr->ss = ss;
        attr->ns = ns;
        return attr;
    }
    return NULL;
}

attribute_t* attribute_floats(char *name, float *fs, size_t nf) {
    attribute_t *attr = sys_malloc(sizeof(attribute_t));
    memset(attr, 0, sizeof(attribute_t));
    if(attr && name) {
        attr->name = name;
        attr->type = ATTRIBUTE_TYPE_FLOATS;
        attr->fs = fs;
        attr->nf = nf;
        return attr;
    }
    return NULL;
}

attribute_t* attribute_ints(char *name, int64_t *is, size_t ni) {
    attribute_t *attr = sys_malloc(sizeof(attribute_t));
    memset(attr, 0, sizeof(attribute_t));
    if(attr && name) {
        attr->name = name;
        attr->type = ATTRIBUTE_TYPE_INTS;
        attr->is = is;
        attr->ni = ni;
        return attr;
    }
    return NULL;
}

attribute_t* attribute_bytes(char *name, uint8_t *bs, size_t nb) {
    attribute_t *attr = sys_malloc(sizeof(attribute_t));
    memset(attr, 0, sizeof(attribute_t));
    if(attr && name) {
        attr->name = name;
        attr->type = ATTRIBUTE_TYPE_BYTES;
        attr->bs = bs;
        attr->nb = nb;
        return attr;
    }
    return NULL;
}