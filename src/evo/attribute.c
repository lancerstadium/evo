#include <evo.h>
#include <evo/util/sys.h>
#include <evo/util/log.h>
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
        attr->fs = malloc(sizeof(float) * nf);
        memcpy(attr->fs, fs, sizeof(float) * nf);
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
        attr->is = malloc(sizeof(int64_t) * ni);
        memcpy(attr->is, is, sizeof(int64_t) * ni);
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
        attr->bs = malloc(sizeof(uint8_t) * nb);
        memcpy(attr->bs, bs, sizeof(uint8_t) * nb);
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
    switch(attr->type) {
        case ATTRIBUTE_TYPE_BYTES: if(attr->bs) free(attr->bs); attr->bs = NULL; break;
        case ATTRIBUTE_TYPE_FLOATS: if(attr->fs) free(attr->fs); attr->fs = NULL; break;
        case ATTRIBUTE_TYPE_INTS: if(attr->is) free(attr->is); attr->is = NULL; break;
        case ATTRIBUTE_TYPE_STRING: if(attr->ss) free(attr->ss); attr->ss = NULL; break;
        default: break;
    }
    sys_free(attr);
    attr = NULL;
    return;
}

void attribute_dump(attribute_t *attr) {
    if(!attr) return;
    LOG_INFO("%s : ", attr->name ? attr->name : "(attr)");
    switch(attr->type) {
        case ATTRIBUTE_TYPE_BYTES: {    LOG_INFO("["); for(int i=0; i<attr->nb;i++) {LOG_INFO("%u%s", attr->bs[i], (i == attr->nb - 1) ? "" : ",");} LOG_INFO("]"); break; }
        case ATTRIBUTE_TYPE_FLOATS: {   LOG_INFO("["); for(int i=0; i<attr->nf;i++) {LOG_INFO("%.2f%s", attr->fs[i], (i == attr->nf - 1) ? "" : ",");} LOG_INFO("]"); break; }
        case ATTRIBUTE_TYPE_INTS: {     LOG_INFO("["); for(int i=0; i<attr->ni;i++) {LOG_INFO("%d%s", attr->is[i], (i == attr->ni - 1) ? "" : ",");} LOG_INFO("]"); break; }
        case ATTRIBUTE_TYPE_TENSOR:     tensor_dump(attr->t);           break;
        case ATTRIBUTE_TYPE_INT:        LOG_INFO("%d", attr->i);        break;
        case ATTRIBUTE_TYPE_FLOAT:      LOG_INFO("%.4f", attr->f);      break;
        case ATTRIBUTE_TYPE_UNDEFINED:  LOG_INFO("undefined");          break;
        case ATTRIBUTE_TYPE_STRING:     LOG_INFO("\'%s\'", attr->ss);   break;
        default: break;
    }
}

char* attribute_dump_value(attribute_t *attr) {
    if(!attr) return NULL;
    char* buf = malloc(48 * sizeof(char));
    sprintf(buf, "%s : ", attr->name ? attr->name : "(attr)");
    switch(attr->type) {
        case ATTRIBUTE_TYPE_BYTES: {    sprintf(buf,"%s[", buf); for(int i=0; i<attr->nb;i++) {sprintf(buf,"%s%u%s", buf, attr->bs[i], (i == attr->nb - 1) ? "" : ",");} sprintf(buf,"%s]", buf); break; }
        case ATTRIBUTE_TYPE_FLOATS: {   sprintf(buf,"%s[", buf); for(int i=0; i<attr->nf;i++) {sprintf(buf,"%s%.2f%s", buf, attr->fs[i], (i == attr->nf - 1) ? "" : ",");} sprintf(buf,"%s]", buf); break; }
        case ATTRIBUTE_TYPE_INTS: {     sprintf(buf,"%s[", buf); for(int i=0; i<attr->ni;i++) {sprintf(buf,"%s%ld%s", buf, attr->is[i], (i == attr->ni - 1) ? "" : ",");} sprintf(buf,"%s]", buf); break; }
        case ATTRIBUTE_TYPE_TENSOR:{    char* shape = tensor_dump_shape(attr->t); sprintf(buf,"%s%s", buf, shape); free(shape); break;}
        case ATTRIBUTE_TYPE_INT:        sprintf(buf,"%s%ld", buf, attr->i);        break;
        case ATTRIBUTE_TYPE_FLOAT:      sprintf(buf,"%s%.4f", buf, attr->f);      break;
        case ATTRIBUTE_TYPE_UNDEFINED:  sprintf(buf,"%sundefined", buf);          break;
        case ATTRIBUTE_TYPE_STRING:     sprintf(buf,"%s\'%s\'", buf, attr->ss);   break;
        default: break;
    }
    return buf;
}