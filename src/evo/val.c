
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

Val* Val_inc(Val* v, size_t l) {
    u64 c = Val_as_u64(v, 0);
    c = c + (u64)l;
    Val_set_u64(v, 0, c);
    return v;
}


Val* Val_from(Val* val) {
    Log_ast(val != NULL, "Val_from: val is null");
    Val* v = malloc(sizeof(Val));
    v->len = val->len;
    v->b = malloc(v->len * sizeof(u8));
    for(size_t i=0; i< v->len; i++) {
        v->b[i] = val->b[i];
    }
    return v;
}

Val* Val_from_u32(u32* val, size_t len) {
    Log_ast(val != NULL, "Val_from: val is null");
    Val* v = malloc(sizeof(Val));
    v->len = len;
    v->b = malloc((v->len) * sizeof(u8));
    for(size_t i = 0; i < v->len; i++) {
        v->b[i] = (u8)(*((u8*)(val) + i));
    }
    return v;
}

void Val_copy(Val* v, Val* other) {
    Log_ast(v != NULL, "Val_copy: v is null");
    Log_ast(other != NULL, "Val_copy: other is null");
    if (v == other) return;
    v->len = other->len;
    v->b = malloc(v->len * sizeof(u8));
    memcpy(v->b, other->b, v->len * sizeof(u8));
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

u8* Val_get_ref(Val* v, size_t idx) {
    Log_ast(v != NULL, "Val_get: v is null");
    Log_ast(idx < v->len, "Val_get: idx is out of bounds");
    return &v->b[idx];
}

void Val_set_ref(Val* v, size_t idx, u8* val, size_t len) {
    Log_ast(v != NULL, "Val_set: v is null");
    Log_ast(idx < v->len, "Val_set: idx is out of bounds");
    memcpy(v->b + idx, val, MIN(len, v->len - idx));
    // for(size_t i = 0; (i < len) && (i + idx < v->len); i++) {
    //     v->b[i+idx] = val[i]; 
    // }
}

Val* Val_set_val(Val* v, size_t idx, Val* val) {
    Log_ast(v != NULL, "Val_set: v is null");
    Log_ast(val != NULL, "Val_set: val is null");
    Log_ast(idx < v->len, "Val_set: idx is out of bounds");
    Val_set_ref(v, idx, val->b, val->len);
    return v;
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
    v->b[5] = (val & 0xFF0000000000) >> 40;
    v->b[6] = (val & 0xFF000000000000) >> 48;
    v->b[7] = (val & 0xFF00000000000000) >> 56;
    return v;
}


char* Val_as_hex(Val *v, bool with_tag) {
    Log_ast(v != NULL, "Val_as_hex: v is null");
    char* tmp = malloc((3 + v->len * 4)* sizeof(char));
    if(with_tag) {
        snprintf(tmp, 3, "0x");
    } else {
        tmp[0] = '\0';
    }
    for(size_t i = 0; i < v->len; i++) {
        char hex[4];
        if(v->b[i]) {
            snprintf(hex, 4, "%02x", v->b[i]);
        } else {
            snprintf(hex, 4, "00");
        }
        strcat(tmp, hex);
        if(i < v->len - 1) {
            strcat(tmp, " ");
        }
    }
    return tmp;
}

char* Val_as_bin(Val* v) {
    Log_ast(v != NULL, "Val_as_bin: v is null");
    char* tmp = malloc((3 + v->len * 10)* sizeof(char));
    snprintf(tmp, 3, "0b");
    for(size_t i = 0; i < v->len; i++) {
        char bin[10];
        if(v->b[i]) {
            size_t cnt = 8;
            while(cnt-- > 0) {
                snprintf(bin, 2, "%1d", (v->b[i] >> cnt) & 0x1);
                strcat(tmp, bin);
            }
            // snprintf(bin, 10, "%08b ", v->b[i]);
        } else {
            snprintf(bin, 10, "00000000");
            strcat(tmp, bin);
        }
        if(i < v->len - 1) {
            strcat(tmp, " ");
        }
    }
    return tmp;
}


Val* Val_as_val(Val* v, size_t i, size_t len) {
    Val* res= Val_alloc(len);
    for(size_t j = 0; (j < len) && (i + j < v->len); j++) {
        res->b[j] = v->b[i+j];
    }
    return res;
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

i8 Val_as_i8(Val *v, size_t i) {
    if(v->len < 1 + i) {
        return 0;
    } else {
        return (i8)v->b[i];
    }
}

i16 Val_as_i16(Val *v, size_t i) {
    if(v->len < 2 + i) {
        i16 tmp = 0;
        for(size_t j = 0; j < v->len; j++) {
            tmp |= (i16)v->b[j+i] << (j * 8);
        }
        return tmp;
    } else {
        return (i16)Val_get_u16(v, i);
    }
}

i32 Val_as_i32(Val *v, size_t i) {
    if(v->len < 4 + i) {
        i32 tmp = 0;
        for(size_t j = 0; j < v->len; j++) {
            tmp |= (i32)v->b[j+i] << (j * 8);
        }
        return tmp;
    } else {
        return (i32)Val_get_u32(v, i);
    }
}

i64 Val_as_i64(Val *v, size_t i) {
    if(v->len < 8 + i) {
        i64 tmp = 0;
        for(size_t j = 0; j < v->len; j++) {
            tmp |= (i64)v->b[j+i] << (j * 8);
        }
        return tmp;
    } else {
        return (i64)Val_get_u64(v, i);
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
    tmp |= (u64)v->b[i + 3] << 24;
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
    Log_ast(hi < v->len * 8, "Val_get_bit: hi(%lu) < v->len(%lu) * 8", hi, v->len);
    Log_ast(lo < v->len * 8, "Val_get_bit: lo(%lu) < v->len(%lu) * 8", lo, v->len);
    size_t scl = hi - lo + 1;
    size_t len = scl / 8;
    if(scl % 8) len++;
    Val* tmp = Val_alloc(len);
    size_t start = lo / 8;
    size_t lo_ = lo - start * 8;
    u64 val = Val_as_u64(v, start);
    u64 mask = BITMASK(scl) << lo_;
    if (scl >= 64) {
        mask = ~0;
        scl = 64;
    }
    u64 res = (val & mask) >> lo_;
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
    size_t scl = hi_ - lo_ + 1;

    // clear bits
    u64 tmp = Val_as_u64(v, start);
    u64 mask = BITMASK(scl) << lo_;
    if (scl >= 64) {
        mask = ~0;
        scl = 64;
    }
    tmp &= ~mask;

    // set bits
    u64 val_u64 = Val_as_u64(val, 0);
    tmp |= ((val_u64 << lo_) & mask);

    // set val to v
    for(size_t i = 0; (i < v->len) && (i < 8); i++) {
        v->b[i+start] = (tmp >> (i * 8)) & 0xFF;
    }
    return v;
}

bool Val_eq_bit(Val *v, size_t hi, size_t lo, Val *val) {
    Log_ast(v, "Val_eq_bit: v is null");
    Log_ast(val, "Val_eq_bit: val is null");
    Log_ast(hi > lo, "Val_eq_bit: hi > lo");
    Log_ast(hi < v->len * 8, "Val_eq_bit: hi < v->len");
    Log_ast(lo < v->len * 8, "Val_eq_bit: lo < v->len");
    
    size_t scl = hi - lo + 1;
    bool is_eq = false;

    // Get bits
    Val* tmp = Val_get_bit(v, hi, lo);
    u64 tmp_u64 = Val_as_u64(tmp, 0);
    Val* val_ext = Val_get_bit(val, scl - 1, 0);
    u64 val_u64 = Val_as_u64(val_ext, 0);

    // Compare bits
    if(tmp_u64 == val_u64) is_eq = true;

    // Free
    Val_free(tmp);
    Val_free(val_ext);
    return is_eq;
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


Val* Val_ext_map(Val *v, BitMap* map, size_t len) {
    Log_ast(v, "Val_ext_map: v is null");
    Log_ast(map, "Val_ext_map: map is null");
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
    size_t len_ = offset / 8;
    if(offset % 8) len_++;
    Val* tmp = Val_alloc(len_);
    for(size_t i = 0; i < len_; i++) {
        tmp->b[i] = (res >> (i * 8)) & 0xFF;
    }
    return tmp;
}


Val* Val_set_map(Val *v, BitMap* map, size_t len, u64 val) {
    Log_ast(v, "Val_set_map: v is null");
    Log_ast(map, "Val_set_map: map is null");
    Log_ast(val, "Val_set_map: val is null");
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


Val* Val_wrt_map(Val *v, BitMap* map, size_t len, Val *val) {
    Log_ast(v != NULL, "Val_wrt_map: v is null");
    Log_ast(map != NULL, "Val_wrt_map: map is null");
    Log_ast(val != NULL, "Val_wrt_map: val is null");
    u64 val_ = Val_as_u64(val, 0);
    for(size_t i = 0; i < len; i++) {
        if(BitMap_chk(map, i)) {
            size_t scl = map[i].h - map[i].l + 1;
            u64 mask = BITMASK(scl);
            if (scl >= 64) {
                mask = ~0;
                scl = 64;
            }
            u64 tmp = val_ & mask;
            // UnitTest_msg("v: %s[%lu:%lu] <- 0x%lx (%lu)", Val_as_hex(v, 0), map[i].h, map[i].l, tmp, scl);
            Val_set_bit(v, map[i].h, map[i].l, Val_new_u64(tmp));
            // UnitTest_msg("v: %s[%lu:%lu] <- 0x%lx (%lu)", Val_as_hex(v, 0), map[i].h, map[i].l, tmp, scl);
            val_ >>= scl;
        }
    }
    return v;
}

// compare map <-> val (no position)
bool Val_eq_map(Val *v, BitMap* map, size_t len, Val *val) {
    Log_ast(v, "Val_eq_map: v is null");
    Log_ast(map, "Val_eq_map: map is null");
    Log_ast(val, "Val_eq_map: val is null");
    
    u64 v_ = Val_get_map(v, map, len);
    u64 val_ = Val_as_u64(val, 0);

    if(v_ == val_) return true;
    return false;
}

// compare map <-> map (yes position)
bool Val_cmp_map(Val *v, BitMap* map, size_t len, Val *val) {
    Log_ast(v, "Val_in_map: v is null");
    Log_ast(map, "Val_in_map: map is null");
    Log_ast(val, "Val_in_map: val is null");
    
    u64 v_ = Val_get_map(v, map, len);
    u64 val_ = Val_get_map(val, map, len);

    if(v_ == val_) return true;
    return false;
}