#include <evo.h>
#include <evo/util/sys.h>
#include <string.h>


attribute_t* attribute_undefined(char *name) {
    attribute_t *attr = sys_malloc(sizeof(attribute_t));
    memset(attr, 0, sizeof(attribute_t));
    if(attr && name) {
        attr->name = sys_strdup(name);
        attr->type = ATTRIBUTE_TYPE_UNDEFINED;
        return attr;
    }
    attribute_free(attr);
    return NULL;
}

attribute_t* attribute_float(char *name, float f) {
    attribute_t *attr = sys_malloc(sizeof(attribute_t));
    memset(attr, 0, sizeof(attribute_t));
    if(attr && name) {
        attr->name = sys_strdup(name);
        attr->type = ATTRIBUTE_TYPE_FLOAT;
        attr->f = f;
        return attr;
    }
    attribute_free(attr);
    return NULL;
}

attribute_t* attribute_int(char *name, int i) {
    attribute_t *attr = sys_malloc(sizeof(attribute_t));
    memset(attr, 0, sizeof(attribute_t));
    if(attr && name) {
        attr->name = sys_strdup(name);
        attr->type = ATTRIBUTE_TYPE_INT;
        attr->i = i;
        return attr;
    }
    attribute_free(attr);
    return NULL;
}

attribute_t* attribute_string(char *name, char *ss, size_t ns) {
    attribute_t *attr = sys_malloc(sizeof(attribute_t));
    memset(attr, 0, sizeof(attribute_t));
    if(attr && name) {
        attr->name = sys_strdup(name);
        attr->type = ATTRIBUTE_TYPE_STRING;
        attr->ss = sys_strdup(ss);
        attr->ns = ns;
        return attr;
    }
    attribute_free(attr);
    return NULL;
}

attribute_t* attribute_floats(char *name, float *fs, size_t nf) {
    attribute_t *attr = sys_malloc(sizeof(attribute_t));
    memset(attr, 0, sizeof(attribute_t));
    if(attr && name) {
        attr->name = sys_strdup(name);
        attr->type = ATTRIBUTE_TYPE_FLOATS;
        attr->fs = fs;
        attr->nf = nf;
        return attr;
    }
    attribute_free(attr);
    return NULL;
}

attribute_t* attribute_ints(char *name, int64_t *is, size_t ni) {
    attribute_t *attr = sys_malloc(sizeof(attribute_t));
    memset(attr, 0, sizeof(attribute_t));
    if(attr && name) {
        attr->name = sys_strdup(name);
        attr->type = ATTRIBUTE_TYPE_INTS;
        attr->is = is;
        attr->ni = ni;
        return attr;
    }
    attribute_free(attr);
    return NULL;
}

attribute_t* attribute_bytes(char *name, uint8_t *bs, size_t nb) {
    attribute_t *attr = sys_malloc(sizeof(attribute_t));
    memset(attr, 0, sizeof(attribute_t));
    if(attr && name) {
        attr->name = sys_strdup(name);
        attr->type = ATTRIBUTE_TYPE_BYTES;
        attr->bs = bs;
        attr->nb = nb;
        return attr;
    }
    attribute_free(attr);
    return NULL;
}

attribute_t* attribute_tensor(char *name, tensor_t *t) {
    attribute_t *attr = sys_malloc(sizeof(attribute_t));
    memset(attr, 0, sizeof(attribute_t));
    if(attr && name) {
        attr->name = sys_strdup(name);
        attr->type = ATTRIBUTE_TYPE_TENSOR;
        attr->t = t;
        return attr;
    }
    attribute_free(attr);
    return NULL;
}


void attribute_free(attribute_t *attr) {
    if(!attr) return;
    if(attr->name) sys_free(attr->name);
    sys_free(attr);
    attr = NULL;
    return;
}