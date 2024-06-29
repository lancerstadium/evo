/***************************************************************************************
* Copyright (c) 2014-2022 Zihao Yu, Nanjing University
*
* NEMU is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
*
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
*
* See the Mulan PSL v2 for more details.
***************************************************************************************/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>

// this should be enough
static char buf[65536] = {};
static char code_buf[65536 + 128] = {}; // a little larger than `buf`
static int parentheses_off = 0;
static int gen_div_off = 0;
static int gen_max_cnt = 30;
static int gen_cnt = 0;
static char *code_format =
"#include <stdio.h>\n"
"int main() { "
"  unsigned result = %s; "
"  printf(\"%%u\", result); "
"  return 0; "
"}";

uint32_t choose(uint32_t n) {
    return rand() % n;
}

void gen(char c) {
    char* buf_buf = strdup(buf);
    int len = strlen(buf_buf);
    if(c == ')') { 
        if(parentheses_off > 0) { parentheses_off--; return; }
    }
    if((c == '(' && buf[len - 1] == 'u') || (c == '(' && buf[len - 1] == ')') || (c == '(' && buf[len - 1] == '(') || (c == '(' && buf[len - 1] == '/')) { parentheses_off++; return; }
    sprintf(buf, "%s%c", buf_buf, c);
}

void gen_num() {
    char* buf_buf = strdup(buf);
    int len = strlen(buf_buf);
    if(buf[len - 1] == 'u' || buf[len - 1] == ')') return;
    sprintf(buf, "%s%uu", buf_buf, choose(8000));
}

void gen_rand_op() {
    uint32_t pattern = choose(5);
    if(gen_div_off > 0) { pattern = choose(3); }
    switch(pattern) {
        case 0: gen('+'); gen_div_off=0; break;
        case 1: gen('-'); gen_div_off=0; break;
        case 2: gen('*'); break;
        case 3: gen('/'); gen_div_off=1; break;
        default: break;
    }
}

static void gen_rand_expr() {
  uint32_t pattern = choose(3);
  if (gen_cnt >= gen_max_cnt) {
      pattern = 0;
  }
  gen_cnt++;
  switch(pattern) {
    case 0: gen_num(); break;
    case 1: gen('('); gen_rand_expr(); gen(')'); break;
    default: gen_rand_expr(); gen_rand_op(); gen_rand_expr(); break;
  }
}

static void buffer_clear() {
  memset(buf, 0, sizeof(buf));
  memset(code_buf, 0, sizeof(code_buf));
  gen_cnt = 0;
}

int main(int argc, char *argv[]) {
  int seed = time(0);
  srand(seed);
  int loop = 1;
  if (argc > 1) {
    sscanf(argv[1], "%d", &loop);
  }
  int i;
  for (i = 0; i < loop; i ++) {
    gen_rand_expr();

    sprintf(code_buf, code_format, buf);

    FILE *fp = fopen("/tmp/.code.c", "w");
    assert(fp != NULL);
    fputs(code_buf, fp);
    fclose(fp);

    int ret = system("gcc /tmp/.code.c -o /tmp/.expr");
    if (ret != 0) continue;

    fp = popen("/tmp/.expr", "r");
    assert(fp != NULL);

    int result;
    ret = fscanf(fp, "%d", &result);
    pclose(fp);
    if((unsigned int)result < 2147483647) {
        printf("%u %s\n", result, buf);
    }
    buffer_clear();
  }
  return 0;
}
