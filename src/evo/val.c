
#include <evo/evo.h>



Val* Val_str(char* str) {
    size_t len = strlen(str);
    Val* v = malloc(sizeof(Val));
    v->b = malloc(len * sizeof(u8));
    v->len = len;
    for(size_t i = 0; i < len; i++) {
        v->b[i] = str[i];
    }
    return v;
}


char* Val_hex(Val v) {
    char* tmp = malloc((3 + v.len * 4)* sizeof(char));
    snprintf(tmp, 3, "0x");
    for(size_t i = 0; i < v.len; i++) {
        char hex[4];
        if(v.b[i]) {
            snprintf(hex, 4, "%02x ", v.b[i]);
        } else {
            snprintf(hex, 4, "00 ");
        }
        strcat(tmp, hex);
    }
    return tmp;
}

u8 Val_as_u8(Val v) {
    if(v.len < 1) {
        return 0;
    } else {
        return v.b[0];
    }
}

u16 Val_as_u16(Val v) {
    if(v.len < 2) {
        u16 tmp = 0;
        for(size_t i = 0; i < v.len; i++) {
            tmp |= v.b[i] << (i * 8);
        }
        return tmp;
    } else {
        return Val_get_u16(v, 0);
    }
}

u32 Val_as_u32(Val v) {
    if(v.len < 4) {
        u32 tmp = 0;
        for(size_t i = 0; i < v.len; i++) {
            tmp |= v.b[i] << (i * 8);
        }
        return tmp;
    } else {
        return Val_get_u32(v, 0);
    }
}

u64 Val_as_u64(Val v) {
    if(v.len < 8) {
        u64 tmp = 0;
        for(size_t i = 0; i < v.len; i++) {
            tmp |= (u64)v.b[i] << (i * 8);
        }
        return tmp;
    } else {
        return Val_get_u64(v, 0);
    }
}

char* Val_as_str(Val* v) {
    char* tmp = malloc((v->len + 1) * sizeof(char));
    for(size_t i = 0; i < v->len; i++) {
        tmp[i] = v->b[i];
    }
    tmp[v->len] = '\0';
    return tmp;
}


u8 Val_get_u8(Val v, size_t i) {
    Log_ast(v.len > i, "Val_get_u8: index %lu out of bounds %lu", i, v.len);
    return v.b[i];
}

u16 Val_get_u16(Val v, size_t i) {
    Log_ast(v.len > i + 1, "Val_get_u16: index %lu out of bounds %lu", i, v.len);
    u16 tmp = 0;
    tmp |= v.b[i];
    tmp |= v.b[i + 1] << 8;
    return tmp;
}

u32 Val_get_u32(Val v, size_t i) {
    Log_ast(v.len > i + 3, "Val_get_u32: index %lu out of bounds %lu", i, v.len);
    u32 tmp = 0;
    tmp |= v.b[i];
    tmp |= v.b[i + 1] << 8;
    tmp |= v.b[i + 2] << 16;
    tmp |= v.b[i + 3] << 24;
    return tmp;
}

u64 Val_get_u64(Val v, size_t i) {
    Log_ast(v.len > i + 7, "Val_get_u64: index %lu out of bounds %lu", i, v.len);
    u64 tmp = 0;
    tmp |= v.b[i];
    tmp |= v.b[i + 1] << 8;
    tmp |= v.b[i + 2] << 16;
    tmp |= v.b[i + 3] << 24;
    tmp |= (u64)v.b[i + 4] << 32;
    tmp |= (u64)v.b[i + 5] << 40;
    tmp |= (u64)v.b[i + 6] << 48;
    tmp |= (u64)v.b[i + 7] << 56;
    return tmp;
}

void Val_set_u8(Val v, size_t i, u8 val) {
    Log_ast(v.len > i, "Val_set_u8: index %lu out of bounds %lu", i, v.len);
    v.b[i] = val;
}

void Val_set_u16(Val v, size_t i, u16 val) {
    Log_ast(v.len > i + 1, "Val_set_u16: index %lu out of bounds %lu", i, v.len);
    v.b[i] = val & 0xFF;
    v.b[i + 1] = (val >> 8) & 0xFF;
}

void Val_set_u32(Val v, size_t i, u32 val) {
    Log_ast(v.len > i + 3, "Val_set_u32: index %lu out of bounds %lu", i, v.len);
    v.b[i] = val & 0xFF;
    v.b[i + 1] = (val >> 8) & 0xFF;
    v.b[i + 2] = (val >> 16) & 0xFF;
    v.b[i + 3] = (val >> 24) & 0xFF;
}

void Val_set_u64(Val v, size_t i, u64 val) {
    Log_ast(v.len > i + 7, "Val_set_u64: index %lu out of bounds %lu", i, v.len);
    v.b[i] = val & 0xFF;
    v.b[i + 1] = (val >> 8) & 0xFF;
    v.b[i + 2] = (val >> 16) & 0xFF;
    v.b[i + 3] = (val >> 24) & 0xFF;
    v.b[i + 4] = (val >> 32) & 0xFF;
    v.b[i + 5] = (val >> 40) & 0xFF;
    v.b[i + 6] = (val >> 48) & 0xFF;
    v.b[i + 7] = (val >> 56) & 0xFF;
}