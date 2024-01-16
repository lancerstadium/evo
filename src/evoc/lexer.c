
// ==================================================================================== //
//                                    evoc: lexer
// ==================================================================================== //

#include "evoc.h"

// ==================================================================================== //
//                                   Pri API: lexer
// ==================================================================================== //

// lexer词法分析：判断是否是标识符，不包括数字
static int is_ident_no_num(char c) {
    return ('a' <= c && c <= 'z') ||
           ('A' <= c && c <= 'Z') ||
           (c == '_');
}
// lexer词法分析：判断是否是标识符
static int is_ident(char c) {
    return is_ident_no_num(c) || ('0' <= c && c <= '9');
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
// lexer词法分析：判断令牌是否是关键词
static bool token_is_keyword(Token *tok) {
    static char * kw[] = {
        "return", "if", "else",
    };
    for(int i = 0; i < sizeof(kw) / sizeof(kw[0]); i++) {
        if(token_equal(tok, kw[i])) {
            return true;
        }
    }
    return false;
}
// lexer词法分析：转换关键字
static Token* token_convert_keywords(Token *tok) {
    for(Token *t = tok; t->type != TK_EOF; t = t->next) {
        if(token_is_keyword(t)) {
            t->type = TK_KEYWORD;
        }
    }
}
// ==================================================================================== //
//                                    Pub API: lexer
// ==================================================================================== //

// lexer词法分析：判断令牌是否等于op
bool token_equal(Token *tok, char *op) {
    return tok->len == strlen(op) && memcmp(tok->loc, op, tok->len) == 0;
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
        // 如果是标识符或关键字
        if(is_ident_no_num(*p)) {
            char* start = p;
            do{
                p++;
            }while(is_ident(*p));
            cur = cur->next = token_new(TK_IDENT, start, p);
            continue;
        }
        // 如果是运算符
        int punct_len = read_punct(p);
        if(punct_len) {          
            cur = cur->next = token_new(TK_PUNCT, p, p + punct_len);
            p += cur->len;
            continue;
        }
        evoc_err_at(p, "invalid token: %c", *p); break;
    }
    cur = cur->next = token_new(TK_EOF, p, p);
    token_convert_keywords(head.next);
    return head.next;
}
