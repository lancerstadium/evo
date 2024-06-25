
#include <evo/evo.h>


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