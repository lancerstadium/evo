
#include <evo/evo.h>


const char* TyKd_sym(TyKd k) {
    switch (k) {
    case TY_I:  return "I";
    case TY_S:  return ";";
    case TY_N:  return "N";
    case TY_i:  return "i";
    case TY_r:  return "r";
    case TY_m:  return "m";
    case TY_NONE:
    default:    return "";
    }
}

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
        if(cur->sym && cur->k != TY_S) {
            snprintf(sym, 2, "%s", cur->sym);
        } else if(cur->sym && cur->k == TY_S) {
            snprintf(sym, 2, ";");
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