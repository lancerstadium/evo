
#include "str.h"
#include <string.h>

ALLOC_DEF_METHOD(Str)



// ==================================================================================== //
//                                    utils API: str
// ==================================================================================== //

Str* str_new(char* s) {
    Str* p = malloc(sizeof(Str));
    p->s = s;
    p->len = strlen(s);
    return p;
}

Str* str_plus(Str* a, Str* b) {
    Str* p = malloc(sizeof(Str));
    p->s = malloc(a->len + b->len + 1);
    memcpy(p->s, a->s, a->len);
    memcpy(p->s + a->len, b->s, b->len);
    p->len = a->len + b->len;
    return p;
}

Str* str_plus_char(Str* a, char c) {
    Str* p = malloc(sizeof(Str));
    p->s = malloc(a->len + 2);
    memcpy(p->s, a->s, a->len);
    p->s[a->len] = c;
    p->s[a->len + 1] = '\0';
    p->len = a->len + 1;
    return p;
}

void str_free(Str* p) {
    free(p->s);
    free(p);
}

// 判断字符串是否以q开头
bool str_start_with(char* p, char* q) {
    return strncmp(p, q, strlen(q)) == 0;
}
