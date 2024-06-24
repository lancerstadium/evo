
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
