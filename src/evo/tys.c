
#include <evo/evo.h>



char* Ty_sym(Ty t) {
    char* tmp = malloc((24)* sizeof(char));
    tmp[0] = '\0';
    Ty* cur = &t;
    while(cur != NULL && strlen(tmp) < 24) {
        char sym[3];
        if(strlen(tmp) >= 1) {
            snprintf(sym, 2, "|");
            strcat(tmp, sym);
        }
        if(cur->sym) {
            snprintf(sym, 2, "%s", cur->sym);
        } else {
            snprintf(sym, 2, "x");
        }
        strcat(tmp, sym);
        cur = cur->or;
    }
    return tmp;
}


char* Tys_sym(Tys v) {
    char* tmp = malloc((1 + v.len * 6)* sizeof(char));
    tmp[0] = '\0';
    for(size_t i = 0; i < v.len; i++) {
        char sym[24];
        if(v.t[i].sym) {
            sprintf(sym, "%s ", Ty_sym(v.t[i]));
        } else {
            snprintf(sym, 3, "x ");
        }
        strcat(tmp, sym);
    }
    return tmp;
}