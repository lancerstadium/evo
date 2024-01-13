
// ==================================================================================== //
//                                    evoc: lexer
// ==================================================================================== //

#include "evoc.h"

// ==================================================================================== //
//                                   Pri API: lexer
// ==================================================================================== //

// lexer词法分析：判断是否是标识符
static int is_ident(char c) {
    return ('a' <= c && c <= 'z') ||
           ('A' <= c && c <= 'Z') ||
           ('0' <= c && c <= '9') ||
           (c == '_');
}
// lexer词法分析：获取标点符号长度
static int read_punct(char *c) {
    if(str_start_with(c, "==") || str_start_with(c, "!=") ||
       str_start_with(c, "<=") || str_start_with(c, ">=")) {
        return 2;
    }
    return ispunct(*c) ? 1 : 0;
}
// lexer词法分析：创建新令牌
static Token* token_new(TokenType type, char* start, char* end) {
    Token *tok = (Token*)calloc(1, sizeof(Token));
    tok->type = type;
    tok->loc = start;
    tok->len = end - start;
    return tok;
}
// ==================================================================================== //
//                                    Pub API: lexer
// ==================================================================================== //

// lexer词法分析：判断令牌是否等于op
bool token_equal(Token *tok, char *op) {
    return memcmp(tok->loc, op, tok->len) == 0 && op[tok->len] == '\0';
}
// lexer词法分析：跳过令牌
Token* token_skip(Token *tok, char *op) {
    if(!token_equal(tok, op)) { // 如果不等于op，报错
        evoc_err_tok(tok, "expected `%s`", op);
    }
    return tok->next;
}
// lexer词法分析：识别令牌
Token* evoc_tokenize(char *p) {
    current_input = p;                      // 当前输入
    Token head = {};                        // 令牌链表头初始化
    Token *cur = &head;                     // 获取当前令牌

    while(*p) {
        // 如果是空格，跳过
        if(isspace(*p)) {                  
            p++;
            continue;
        }
        // 如果是数字
        if(isdigit(*p)) {                   
            cur = cur->next = token_new(TK_NUM, p, p);
            char *q = p;
            cur->val = strtoul(p, &p, 10);
            cur->len = p - q;
            continue;
        }
        // 如果是运算符
        int punct_len = read_punct(p);
        if(punct_len) {          
            cur = cur->next = token_new(TK_PUNCT, p, p + punct_len);
            p += cur->len;
            continue;
        }
        // // 如果是返回
        // if(strncmp(p, "return", 6) == 0) {
        //     cur->type = TK_RETURN;
        //     cur->loc = p;
        //     cur->len = 6;
        //     p += 6;
        //     continue;
        // }
        // // 如果是标识符
        // if(is_ident(*p)) {
        //     cur = cur->next = token_new(TK_IDENT, p, p);
        //     char *q = p;
        //     while(is_ident(*p)) p++;
        //     cur->len = p - q;
        //     continue;
        // }
        evoc_err_at(p, "invalid token: %c", *p); break;
    }
    cur = cur->next = token_new(TK_EOF, p, p);
    return head.next;
}

// ==================================================================================== //
//                                   Token Operations
// ==================================================================================== //


// // lexer词法分析：尝试消费下一个令牌
// bool consume(char* op) {
//     if(token->type == TK_EOF || strlen(op) != token->len || memcmp(token->str, op, token->len)) {
//         // 如果是结束令牌，或者不是期望的令牌，返回false
//         // log_error("expect `%c`, but got `%c`", op, token->str[0]);
//         return false;
//     }
//     // 否则，token下移，返回true
//     token = token->next;
//     return true;
// }
// Token* consume_ident() {
//     if(token->type != TK_IDENT) {
//         // log_error("expect identifier");
//         evoc_err_at(token->str, "expect identifier");
//     }
//     Token *t = token;
//     token = token->next;
//     return t;
// }
// // lexer词法分析：期望下一个令牌
// void expect(char* op) {
//     if(token->type != TK_RESERVED || strlen(op) != token->len || memcmp(token->str, op, token->len)) {
//         // log_error("expect `%c`, but got `%c`", op, token->str[0]);
//         evoc_err_at(token->str, "expect `%c`", op);
//     }
//     token = token->next;
// }

// // lexer词法分析：获取当前令牌的数字值
// int expect_number() {
//     if(token->type != TK_NUM) {
//         log_error("expect number");
//     }
//     int val = token->val;
//     token = token->next;
//     return val;
// }

// // lexer词法分析：到达令牌末尾
// bool at_end() {
//     return token->type == TK_EOF;
// }

// ==================================================================================== //
//                                    Data: Var
// ==================================================================================== //

// typedef struct Var Var;         // 变量结构体
// struct Var {
//     Var *next;                  // 下一个变量
//     char *name;                 // 变量名
//     int len;                    // 变量名长度
//     int offset;                 // 变量偏移
// };
// Var* local_vars;                // 局部变量链表
// // lexer词法分析：新建变量
// Var* var_new(char *name, Var *next, int len, int offset) {
//     Var *p = calloc(1, sizeof(Var));
//     p->next = next;
//     p->name = name;
//     p->len = len;
//     p->offset = offset;
//     return p;
// }
// // lexer词法分析：寻找变量
// Var* var_find(Token *tok) {
//     // 从局部变量链表中寻找：如果找到，返回变量指针，否则返回NULL
//     for(Var *p = local_vars; p; p = p->next) {
//         if(tok->len == p->len && memcmp(tok->loc, p->name, tok->len) == 0) {
//             return p;
//         }
//     }
//     return NULL;
// }
