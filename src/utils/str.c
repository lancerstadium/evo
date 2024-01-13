
#include "utils.h"

// 判断字符串是否以q开头
bool str_start_with(char* p, char* q) {
    return strncmp(p, q, strlen(q)) == 0;
}
