
#include <evo/evo.h>



Val* Val_alloc(size_t len) {
    Val* v = malloc(sizeof(Val));
    v->len = len;
    v->b = malloc(len * sizeof(u8));
    memset(v->b, 0, len * sizeof(u8));
    return v;
}


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

Val* Val_new_u8(u8 val) {
    Val* v = malloc(sizeof(Val));
    v->len = 1;
    v->b = malloc(1 * sizeof(u8));
    v->b[0] = val;
    return v;
}

Val* Val_new_u16(u16 val) {
    Val* v = malloc(sizeof(Val));
    v->len = 2;
    v->b = malloc(2 * sizeof(u8));
    v->b[0] = val & 0xFF;
    v->b[1] = (val & 0xFF00) >> 8;
    return v;
}

Val* Val_new_u32(u32 val) {
    Val* v = malloc(sizeof(Val));
    v->len = 4;
    v->b = malloc(4 * sizeof(u8));
    v->b[0] = val & 0xFF;
    v->b[1] = (val & 0xFF00) >> 8;
    v->b[2] = (val & 0xFF0000) >> 16;
    v->b[3] = (val & 0xFF000000) >> 24;
    return v;
}

Val* Val_new_u64(u64 val) {
    Val* v = malloc(sizeof(Val));
    v->len = 8;
    v->b = malloc(8 * sizeof(u8));
    v->b[0] = val & 0xFF;
    v->b[1] = (val & 0xFF00) >> 8;
    v->b[2] = (val & 0xFF0000) >> 16;
    v->b[3] = (val & 0xFF000000) >> 24;
    v->b[4] = (val & 0xFF00000000) >> 32;
    v->b[5] = (val & 0xFF000000000) >> 40;
    v->b[6] = (val & 0xFF0000000000) >> 48;
    v->b[7] = (val & 0xFF00000000000) >> 56;
    return v;
}


char* Val_as_hex(Val *v) {
    Log_ast(v != NULL, "Val_as_hex: v is null");
    char* tmp = malloc((3 + v->len * 4)* sizeof(char));
    snprintf(tmp, 3, "0x");
    for(size_t i = 0; i < v->len; i++) {
        char hex[4];
        if(v->b[i]) {
            snprintf(hex, 4, "%02x ", v->b[i]);
        } else {
            snprintf(hex, 4, "00 ");
        }
        strcat(tmp, hex);
    }
    return tmp;
}

u8 Val_as_u8(Val *v) {
    if(v->len < 1) {
        return 0;
    } else {
        return v->b[0];
    }
}

u16 Val_as_u16(Val *v) {
    if(v->len < 2) {
        u16 tmp = 0;
        for(size_t i = 0; i < v->len; i++) {
            tmp |= v->b[i] << (i * 8);
        }
        return tmp;
    } else {
        return Val_get_u16(v, 0);
    }
}

u32 Val_as_u32(Val *v) {
    if(v->len < 4) {
        u32 tmp = 0;
        for(size_t i = 0; i < v->len; i++) {
            tmp |= v->b[i] << (i * 8);
        }
        return tmp;
    } else {
        return Val_get_u32(v, 0);
    }
}

u64 Val_as_u64(Val *v) {
    if(v->len < 8) {
        u64 tmp = 0;
        for(size_t i = 0; i < v->len; i++) {
            tmp |= (u64)v->b[i] << (i * 8);
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


u8 Val_get_u8(Val *v, size_t i) {
    Log_ast(v->len > i, "Val_get_u8: index %lu out of bounds %lu", i, v->len);
    return v->b[i];
}

u16 Val_get_u16(Val *v, size_t i) {
    Log_ast(v->len > i + 1, "Val_get_u16: index %lu out of bounds %lu", i, v->len);
    u16 tmp = 0;
    tmp |= v->b[i];
    tmp |= v->b[i + 1] << 8;
    return tmp;
}

u32 Val_get_u32(Val *v, size_t i) {
    Log_ast(v->len > i + 3, "Val_get_u32: index %lu out of bounds %lu", i, v->len);
    u32 tmp = 0;
    tmp |= v->b[i];
    tmp |= v->b[i + 1] << 8;
    tmp |= v->b[i + 2] << 16;
    tmp |= v->b[i + 3] << 24;
    return tmp;
}

u64 Val_get_u64(Val *v, size_t i) {
    Log_ast(v->len > i + 7, "Val_get_u64: index %lu out of bounds %lu", i, v->len);
    u64 tmp = 0;
    tmp |= v->b[i];
    tmp |= v->b[i + 1] << 8;
    tmp |= v->b[i + 2] << 16;
    tmp |= v->b[i + 3] << 24;
    tmp |= (u64)v->b[i + 4] << 32;
    tmp |= (u64)v->b[i + 5] << 40;
    tmp |= (u64)v->b[i + 6] << 48;
    tmp |= (u64)v->b[i + 7] << 56;
    return tmp;
}

void Val_set_u8(Val *v, size_t i, u8 val) {
    Log_ast(v && v->len > i, "Val_set_u8: index %lu out of bounds %lu", i, v->len);
    v->b[i] = val;
}

void Val_set_u16(Val *v, size_t i, u16 val) {
    Log_ast(v && v->len > i + 1, "Val_set_u16: index %lu out of bounds %lu", i, v->len);
    v->b[i] = val & 0xFF;
    v->b[i + 1] = (val >> 8) & 0xFF;
}

void Val_set_u32(Val *v, size_t i, u32 val) {
    Log_ast(v && v->len > i + 3, "Val_set_u32: index %lu out of bounds %lu", i, v->len);
    v->b[i] = val & 0xFF;
    v->b[i + 1] = (val >> 8) & 0xFF;
    v->b[i + 2] = (val >> 16) & 0xFF;
    v->b[i + 3] = (val >> 24) & 0xFF;
}

void Val_set_u64(Val *v, size_t i, u64 val) {
    Log_ast(v && v->len > i + 7, "Val_set_u64: index %lu out of bounds %lu", i, v->len);
    v->b[i] = val & 0xFF;
    v->b[i + 1] = (val >> 8) & 0xFF;
    v->b[i + 2] = (val >> 16) & 0xFF;
    v->b[i + 3] = (val >> 24) & 0xFF;
    v->b[i + 4] = (val >> 32) & 0xFF;
    v->b[i + 5] = (val >> 40) & 0xFF;
    v->b[i + 6] = (val >> 48) & 0xFF;
    v->b[i + 7] = (val >> 56) & 0xFF;
}


// val = v[hi:lo]
Val* Val_get_bit(Val *v, size_t hi, size_t lo) {
    Log_ast(v, "Val_get_bit: v is null");
    Log_ast(hi > lo, "Val_get_bit: hi > lo");
    Log_ast(hi < v->len * 8, "Val_get_bit: hi < v->len");
    Log_ast(lo < v->len * 8, "Val_get_bit: lo < v->len");
    size_t scl = hi - lo + 1;
    size_t len = scl / 8;
    if(scl % 8) len++;
    Val* tmp = Val_alloc(len);
    u64 val = Val_as_u64(v);
    u64 res = BITS(val, hi, lo);
    for(size_t i = 0; i < len; i++) {
        tmp->b[i] = (res >> (i * 8)) & 0xFF;
    }
    return tmp;
}

// v[hi:lo] = val
Val* Val_set_bit(Val *v, size_t hi, size_t lo, Val *val) {
    Log_ast(v, "Val_set_bit: v is null");
    Log_ast(val, "Val_set_bit: val is null");
    Log_ast(hi > lo, "Val_set_bit: hi > lo");
    Log_ast(hi < v->len * 8, "Val_set_bit: hi < v->len");
    Log_ast(lo < v->len * 8, "Val_set_bit: lo < v->len");

    size_t scl = hi - lo + 1;

    // clear bits
    u64 tmp = Val_as_u64(v);
    u64 mask = ~BITS(0xFFFFFFFFFFFFFFFF, hi, lo);
    tmp &= mask;

    // set bits
    u64 val_u64 = Val_as_u64(val);
    tmp |= ((val_u64 << lo) & BITS(0xFFFFFFFFFFFFFFFF, hi, lo));

    // set val to v
    for(size_t i = 0; i < v->len; i++) {
        v->b[i] = (tmp >> (i * 8)) & 0xFF;
    }
    return v;
}