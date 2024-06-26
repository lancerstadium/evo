
#include <evo/evo.h>



Val* Val_alloc(size_t len) {
    Val* v = malloc(sizeof(Val));
    v->len = len;
    v->b = malloc(len * sizeof(u8));
    memset(v->b, 0, len * sizeof(u8));
    return v;
}

void Val_free(Val* v) {
    Log_ast(v != NULL, "Val_free: v is null");
    free(v->b);
    free(v);
    v = NULL;
}

void Val_set(Val* v, size_t idx, u8 val) {
    Log_ast(v != NULL, "Val_set: v is null");
    Log_ast(idx < v->len, "Val_set: idx is out of bounds");
    v->b[idx] = val;
}

u8 Val_get(Val* v, size_t idx) {
    Log_ast(v != NULL, "Val_get: v is null");
    Log_ast(idx < v->len, "Val_get: idx is out of bounds");
    return v->b[idx];
}

void Val_concat(Val* v, Val* other) {
    Log_ast(v != NULL, "Val_concat: v is null");
    Log_ast(other != NULL, "Val_concat: other is null");
    v->b = realloc(v->b, (v->len + other->len) * sizeof(u8));
    memcpy(v->b + v->len, other->b, other->len * sizeof(u8));
    v->len += other->len;
}

void Val_push(Val* v, u8 val) {
    Log_ast(v != NULL, "Val_push: v is null");
    // malloc
    v->b = realloc(v->b, (v->len + 1) * sizeof(u8));
    v->b[v->len] = val;
    v->len++;
}

u8 Val_pop(Val* v) {
    Log_ast(v != NULL, "Val_pop: v is null");
    v->len--;
    return v->b[v->len];
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

char* Val_as_bin(Val* v) {

}

u8 Val_as_u8(Val *v, size_t i) {
    if(v->len < 1 + i) {
        return 0;
    } else {
        return v->b[i];
    }
}

u16 Val_as_u16(Val *v, size_t i) {
    if(v->len < 2 + i) {
        u16 tmp = 0;
        for(size_t j = 0; j < v->len; j++) {
            tmp |= v->b[j+i] << (j * 8);
        }
        return tmp;
    } else {
        return Val_get_u16(v, i);
    }
}

u32 Val_as_u32(Val *v, size_t i) {
    if(v->len < 4 + i) {
        u32 tmp = 0;
        for(size_t j = 0; j < v->len; j++) {
            tmp |= v->b[j+i] << (j * 8);
        }
        return tmp;
    } else {
        return Val_get_u32(v, i);
    }
}

u64 Val_as_u64(Val *v, size_t i) {
    if(v->len < 8 + i) {
        u64 tmp = 0;
        for(size_t j = 0; j < v->len; j++) {
            tmp |= (u64)v->b[j+i] << (j * 8);
        }
        return tmp;
    } else {
        return Val_get_u64(v, i);
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
    size_t start = lo / 8;
    size_t lo_ = lo - start * 8;
    size_t hi_ = hi - start * 8;
    u64 val = Val_as_u64(v, start);
    u64 res = BITS(val, hi_, lo_);
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

    size_t start = lo / 8;
    size_t lo_ = lo - start * 8;
    size_t hi_ = hi - start * 8;

    // clear bits
    u64 tmp = Val_as_u64(v, start);
    u64 mask = ~BITS(0xFFFFFFFFFFFFFFFF, hi_, lo_);
    tmp &= mask;

    // set bits
    u64 val_u64 = Val_as_u64(val, 0);
    tmp |= ((val_u64 << lo_) & BITS(0xFFFFFFFFFFFFFFFF, hi_, lo_));

    // set val to v
    for(size_t i = 0; i < v->len; i++) {
        v->b[i+start] = (tmp >> (i * 8)) & 0xFF;
    }
    return v;
}


u64 Val_get_map(Val *v, BitMap* map, size_t len) {
    Log_ast(v, "Val_get_map: v is null");
    Log_ast(map, "Val_get_map: map is null");
    u64 res = 0;
    size_t offset = 0;
    for(size_t i = 0; i < len; i++) {
        if(BitMap_chk(map, i)) {
            size_t scl = map[i].h - map[i].l + 1;
            Val* tmp = Val_get_bit(v, map[i].h, map[i].l);
            res |= Val_as_u64(tmp, 0) << offset;
            offset += scl;
            Val_free(tmp);
        }
    }
    return res;
}


Val* Val_set_map(Val *v, BitMap* map, size_t len, u64 val) {
    Log_ast(v, "Val_set_map: v is null");
    Log_ast(map, "Val_set_map: map is null");
    u64 val_ = val; 
    for(size_t i = 0; i < len; i++) {
        if(BitMap_chk(map, i)) {
            size_t scl = map[i].h - map[i].l + 1;
            u64 tmp = val_ & BITMASK(scl);
            Val_set_bit(v, map[i].h, map[i].l, Val_new_u64(tmp));
            val_ >>= scl;
        }
    }
    return v;
}