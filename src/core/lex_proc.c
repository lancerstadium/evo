


#include "lexer.h"

// ==================================================================================== //
//                                  lexer: Char operate
// ==================================================================================== //

#define LEX_MACRO_HASHMAP_INIT_NUM 32

#define LEX_NEXTC(lex) (lex->next_char(lex))
#define LEX_PEEKC(lex) (lex->peek_char(lex))

#define LEX_GETC_IF(lex, buffer, c, EXPR)                             \
    do{                                                               \
        for (c = lex->peek_char(lex); EXPR; c = lex->peek_char(lex)){ \
            buffer_write(buffer, c);                                  \
            lex->next_char(lex);                                      \
        }                                                             \
    } while (0)

#define LEX_GETC_IF_NO_WS(lex, buffer, c, EXPR)                       \
    do{                                                               \
        for (c = lex->peek_char(lex); EXPR; c = lex->peek_char(lex)){ \
            if(c != ' ' && c != '\t') {                               \
                buffer_write(buffer, c);                              \
            }                                                         \
            lex->next_char(lex);                                      \
        }                                                             \
    } while (0)

#define LEX_ASSERT_NEXTC(lex, ec)            \
    do{                                      \
        char c = lex->next_char(lex);        \
        log_assert(c == ec, "get: `%c`", c); \
    } while (0)


#define CASE_OPERATOR_EXCLUDING_DIVISON \
    case '+':                           \
    case '-':                           \
    case '*':                           \
    case '>':                           \
    case '<':                           \
    case '^':                           \
    case '%':                           \
    case '!':                           \
    case '=':                           \
    case '~':                           \
    case '|':                           \
    case '&':                           \
    case '.':                           \
    case ':'                                            

#define CASE_NUMERIC \
    case '0':        \
    case '1':        \
    case '2':        \
    case '3':        \
    case '4':        \
    case '5':        \
    case '6':        \
    case '7':        \
    case '8':        \
    case '9'

#define CASE_SYMBOL \
    case '{':       \
    case '}':       \
    case ';':       \
    case '(':       \
    case ')':       \
    case '[':       \
    case ']':       \
    case ',':       \
    case '#':       \
    case '\\'

// ==================================================================================== //
//                                  lexer: String Identity
// ==================================================================================== //

/** 关键字 */
static const char* lex_keyword[] = {
    "mod", "use", "type", "impl", "fn", 
    "var", "const", "if", "else", "elif", 
    "while", "for", "do", "break", "continue",
    "switch", "case", "default", "return",
    "self", "pub", "pri"
};

static const char* lex_pre_keyword[] = {
    "def", "udf"
};

/** 数据类型 */
static const char* lex_datatype[] = {
    [DATA_TYPE_U8]          = "u8", 
    [DATA_TYPE_U16]         = "u16", 
    [DATA_TYPE_U32]         = "u32", 
    [DATA_TYPE_U64]         = "u64", 
    [DATA_TYPE_I8]          = "i8",  
    [DATA_TYPE_I16]         = "i16", 
    [DATA_TYPE_I32]         = "i32", 
    [DATA_TYPE_I64]         = "i64",
    [DATA_TYPE_F32]         = "f32", 
    [DATA_TYPE_F64]         = "f64", 
    [DATA_TYPE_CHAR]        = "char", 
    [DATA_TYPE_BOOL]        = "bool",
    [DATA_TYPE_SHORT]       = "short",
    [DATA_TYPE_INT]         = "int",
    [DATA_TYPE_LONG]        = "lint",
    [DATA_TYPE_LONG_LONG]   = "llint",
    [DATA_TYPE_UINT]        = "uint",
    [DATA_TYPE_ULONG]       = "ulint",
    [DATA_TYPE_ULONG_LONG]  = "ullint",
    [DATA_TYPE_FLOAT]       = "float",
    [DATA_TYPE_DOUBLE]      = "double",
    [DATA_TYPE_STR]         = "str",
    [DATA_TYPE_STRUCT]      = "struct",
    [DATA_TYPE_UNION]       = "union",
    [DATA_TYPE_DEFINED]     = "defined",
    [DATA_TYPE_ANY]         = "any",
    [DATA_TYPE_NONE]        = "none"
};
/** 单元运算符 */
static const char lex_single_op[] = {
    '+', '-', '/', '*', '=', '>', '<',
    '|', '&', '^', '%', '~', '!', '.',
    ':'
};
/** 二元运算符 */
static const char* lex_binary_op[] = {
    "+=", "-=", "*=", "/=", ">>", "<<",
    ">=", "<=", "||", "&&", "++", "--",
    "==", "!=", "!=", ":=", "~=", "->",
    ".*"
};
/** 三元运算符 */
static const char* lex_ternary_op[] = {
    "<=>"
};

#define LEX_KEYWORD_NUM         GET_ARR_LEN(lex_keyword)
#define LEX_PRE_KEYWORD_NUM     GET_ARR_LEN(lex_pre_keyword)
#define LEX_DATATYPE_NUM        GET_ARR_LEN(lex_datatype)
#define LEX_SINGLE_OP_NUM       GET_ARR_LEN(lex_single_op)
#define LEX_BINARY_OP_NUM       GET_ARR_LEN(lex_binary_op)
#define LEX_TERNARY_OP_NUM      GET_ARR_LEN(lex_ternary_op)

// ==================================================================================== //
//                                     lexer: declare
// ==================================================================================== //

static inline Token* lexer_handle_macro(LexProcess* lproc);

static inline Token* lexer_make_connect_ident_or_string(LexProcess* lproc);
// ==================================================================================== //
//                                  lexer: String Operation
// ==================================================================================== //

static inline bool is_keyword(const char* str) {
    bool is_key = false;
    int i;
    for(i = 0; i < LEX_KEYWORD_NUM; i++) {
        if(STR_EQ(str, lex_keyword[i])){
            is_key = true;
            return is_key;
        }
    }
    return is_key;
}

static inline int get_keyword_idx(const char* str) {
    int i;
    for(i = 0; i < LEX_KEYWORD_NUM; i++) {
        if(STR_EQ(str, lex_keyword[i])){
            return i;
        }
    }
    return -1;
}

static inline bool is_pre_keyword(const char* str) {
    bool is_pkw = false;
    int i;
    for(i = 0; i < LEX_PRE_KEYWORD_NUM; i++) {
        if(STR_EQ(str, lex_pre_keyword[i])){
            is_pkw = true;
            return is_pkw;
        }
    }
    return is_pkw;
}

static inline bool is_datatype(const char* str) {
    bool is_data = false;
    int i;
    for(i = 0; i < LEX_DATATYPE_NUM; i++) {
        if(STR_EQ(str, lex_datatype[i])){
            is_data = true;
            return is_data;
        }
    }
    return is_data;
}

static inline bool is_single_operator(char op) {
    bool is_single = false;
    int i;
    for(i = 0; i < LEX_SINGLE_OP_NUM; i++) {
        if(op == lex_single_op[i]){
            is_single = true;
            return is_single;
        }
    }
    return is_single;
}

static inline bool is_binary_operator(const char* op) {
    bool is_binary = false;
    int i;
    for(i = 0; i < LEX_BINARY_OP_NUM; i++) {
        if(STR_EQ(op, lex_binary_op[i])){
            is_binary = true;
            return is_binary;
        }
    }
    return is_binary;
}

static inline bool is_ternary_operator(const char* op) {
    bool is_ternary = false;
    int i;
    for(i = 0; i < LEX_TERNARY_OP_NUM; i++) {
        if(STR_EQ(op, lex_ternary_op[i])){
            is_ternary = true;
            return is_ternary;
        }
    }
    return is_ternary;
}

static inline bool is_valid_operator(const char* op) {
    switch(strlen(op)) {
        case 1: return is_single_operator(*op);
        case 2: return is_binary_operator(op);
        case 3: return is_ternary_operator(op);
        default: return false;
    }
}

static inline bool is_alpha(const char c) {
    if((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_') {
        return true;
    }
    return false;
}

// ==================================================================================== //
//                                  lexer: Read String
// ==================================================================================== //

static inline const char* lexer_read_operator(LexProcess* lproc) {
    char op = lproc->next_char(lproc);
    Buffer* buf = buffer_create();
    buffer_write(buf, op);
    op = lproc->peek_char(lproc);
    if(is_single_operator(op)) {        // 二元运算符
        buffer_write(buf, op);
        lproc->next_char(lproc);
        op = lproc->peek_char(lproc);
        if(is_single_operator(op)) {    // 三元运算符
            buffer_write(buf, op);
            lproc->next_char(lproc);
        }
    }
    const char* buf_ptr = buffer_ptr(buf);
    if(!is_valid_operator(buf_ptr)) {
        lexer_error("Invalid operator: `%s` at %s:%d:%d", buf_ptr, lproc->pos.filename, lproc->pos.line, lproc->pos.col);
    }
    return buf_ptr;
}

static const char* lexer_read_number_str(LexProcess* lproc) {
    const char* num = NULL;
    Buffer* buf = buffer_create();
    char c = lproc->peek_char(lproc);
    LEX_GETC_IF(lproc, buf, c, c >= '0' && c <= '9');

    buffer_write(buf, 0x00);    // 结束符号
    return buffer_ptr(buf);
}

static inline unsigned long long lexer_read_number(LexProcess* lproc) {
    const char* s = lexer_read_number_str(lproc);
    return atoll(s);
}


// ==================================================================================== //
//                                  lexer: Make Token
// ==================================================================================== //

static inline Token* lexer_create_token(LexProcess* lproc, Token* _token) {
    memcpy(&lproc->tmp_tok, _token, sizeof(Token));
    lproc->tmp_tok.edep = lproc->cur_expr_depth;
    lproc->tmp_tok.ldep = lproc->cur_list_depth;
    lproc->tmp_tok.sdep = lproc->cur_scope_depth;
    lproc->tmp_tok.pos = lproc->pos;
    return &lproc->tmp_tok;
}

static inline Token* lexer_make_string(LexProcess* lproc) {
    Buffer* buf = buffer_create();
    LEX_ASSERT_NEXTC(lproc, '"');
    char c = lproc->next_char(lproc);

    for(; c!= '"' && c != EOF; c = lproc->next_char(lproc)) {
        if(c == '\\') { // 转义字符
            lproc->next_char(lproc);
        }
        buffer_write(buf, c);
    }
    buffer_write(buf, 0x00);
    return lexer_create_token(lproc, &(Token){
        .type = TOKEN_TYPE_STRING,
        .sval = buffer_ptr(buf)
    });
}

static inline Token* lexer_make_newline(LexProcess* lproc) {
    LEX_ASSERT_NEXTC(lproc, '\n');
    return lexer_create_token(lproc, &(Token){
        .type = TOKEN_TYPE_NEWLINE
    });
}

static inline Token* lexer_make_operator_from_value(LexProcess* lproc, char* val) {
    return lexer_create_token(lproc, &(Token){
        .type = TOKEN_TYPE_OPERATOR,
        .sval = val
    });
}

static inline Token* lexer_make_operator(LexProcess* lproc) {
    return lexer_create_token(lproc, &(Token){
        .type = TOKEN_TYPE_OPERATOR,
        .sval = lexer_read_operator(lproc)
    });
}

static inline Token* lexer_make_number(LexProcess* lproc) {
    // 预处理
    if(&lproc->tmp_tok && lproc->tmp_tok.type == TOKEN_TYPE_IDENTIFIER && lproc->pre.macro_def == true) {
        Buffer* buf = buffer_create();
        char c = 0;
        LEX_GETC_IF(lproc, buf, c, c != ' ' && c != '\n' && c != EOF);
        return lexer_create_token(lproc, &(Token){
            .type = TOKEN_TYPE_STRING,
            .sval = buffer_ptr(buf)
        });
    }
    return lexer_create_token(lproc, &(Token){
        .type = TOKEN_TYPE_NUMBER,
        .llnum = lexer_read_number(lproc)
    });
}

static inline Token* lexer_make_ident(LexProcess* lproc, const char* buf1) {
    char c = lproc->peek_char(lproc);
    // 检查上一个token是否为宏定义：`#def`
    if(&lproc->tmp_tok && lproc->tmp_tok.type == TOKEN_TYPE_PRE_KEYWORD && STR_EQ(lproc->tmp_tok.sval, "def")) {
        lproc->pre.macro_def = true;
        if(lproc->pre.macro_sym_tbl == NULL) {
            lproc->pre.macro_sym_tbl = hashmap_create();
        }
        // 推掉`def`
        vector_pop(lproc->token_vec);
        Token* pop_tok = (Token*)vector_back(lproc->token_vec);
        lproc->tmp_tok = *pop_tok;
        c = 0;
        Buffer* buf2 = buffer_create();
        LEX_GETC_IF_NO_WS(lproc, buf2, c, c != '\n' && c != EOF);
        hashmap_set(lproc->pre.macro_sym_tbl, buf1, buffer_ptr(buf2));
        return lex_process_next_token(lproc);
    } else if(lproc->pre.macro_sym_tbl != NULL && hashmap_contains(lproc->pre.macro_sym_tbl, buf1)) {
        const char* macro_buf = hashmap_get(lproc->pre.macro_sym_tbl, buf1);
        return lexer_create_token(lproc, &(Token){
            .type = TOKEN_TYPE_IDENTIFIER,
            .sval = macro_buf
        });
    } else {        // 返回ident
        return lexer_create_token(lproc, &(Token){   
            .type = TOKEN_TYPE_IDENTIFIER,
            .sval = buf1
        });
    }
}

static inline Token* lexer_make_ident_or_keyword(LexProcess* lproc) {
    Buffer* buf = buffer_create();
    char c = lproc->peek_char(lproc);
    LEX_GETC_IF(lproc, buf, c, (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_');

    if(is_keyword(buffer_ptr(buf))) {               // 返回keyword
        return lexer_create_token(lproc, &(Token){
            .type = TOKEN_TYPE_KEYWORD,
            .sval = buffer_ptr(buf)
        });
    } else if(is_datatype(buffer_ptr(buf))) {       // 返回datatype
        return lexer_create_token(lproc, &(Token){
            .type = TOKEN_TYPE_DATATYPE,
            .sval = buffer_ptr(buf)
        });
    } else {
        return lexer_make_ident(lproc, buffer_ptr(buf));
    }
}

static inline Token* lexer_make_symbol(LexProcess* lproc) {
    char c = lproc->next_char(lproc);
    switch(c) {
        case '(' :  lproc->cur_expr_depth++;  break;
        case ')' :  lproc->cur_expr_depth--;  break;
        case '[' :  lproc->cur_list_depth++;  break;
        case ']' :  lproc->cur_list_depth--;  break;
        case '{' :  lproc->cur_scope_depth++; break;
        case '}' :  lproc->cur_scope_depth--; break;
    }
    return lexer_create_token(lproc, &(Token){   
        .type = TOKEN_TYPE_SYMBOL,
        .cval = c
    });
}

static inline Token* lexer_make_one_line_comment(LexProcess* lproc) {
    Buffer *buf = buffer_create();
    char c = 0;
    LEX_GETC_IF(lproc, buf, c, c != '\n' && c != EOF);
    return lexer_create_token(lproc, &(Token){
        .type = TOKEN_TYPE_COMMENT,
        .sval = buffer_ptr(buf)
    });
}

static inline Token* lexer_make_mutiline_comment(LexProcess* lproc) {
    Buffer *buf = buffer_create();
    char c = 0;
    while(1) {
        LEX_GETC_IF(lproc, buf, c, c != '*' && c != EOF && c != '\n');
        if(c == EOF) {
            lexer_error("EOF reached whilst in a multi-line comment. The comment was not terminated! \"%s\" \n", buffer_ptr(buf));
            exit(LEXER_ANALYSIS_ERROR);
        } else if(c == '*') {
            lproc->next_char(lproc);
            if(lproc->peek_char(lproc) == '/') {    // comment 结束
                lproc->next_char(lproc);
                break;
            }
        } else if(c == '\n') {
            lproc->next_char(lproc);
        }
    }
    return lexer_create_token(lproc, &(Token){
        .type = TOKEN_TYPE_COMMENT,
        .sval = buffer_ptr(buf)
    });
}

static inline Token* lexer_read_token_special(LexProcess* lproc) {
    char c = lproc->peek_char(lproc);
    if(isalpha(c) || c == '_') {
        return lexer_make_ident_or_keyword(lproc);
    } else {
        return NULL;
    }
}

static inline Token* lexer_make_quote(LexProcess* lproc) {
    LEX_ASSERT_NEXTC(lproc, '\'');
    char c = lproc->next_char(lproc);
    if(c == '\\') {
        c = lproc->next_char(lproc);
    }
    return lexer_create_token(lproc, &(Token){   
        .type = TOKEN_TYPE_NUMBER,
        .cval = c
    });
}

static inline Token* lexer_make_EOF(LexProcess* lproc) {
    return lexer_create_token(lproc, &(Token){   
        .type = TOKEN_TYPE_EOF,
    });
}


// ==================================================================================== //
//                                 lexer : preprocess
// ==================================================================================== //


static inline Token* lexer_handle_comment(LexProcess* lproc) {
    char c = lproc->peek_char(lproc);
    if(c == '/') {
        lproc->next_char(lproc);
        if(lproc->peek_char(lproc) == '/') {                // `//`
            lproc->next_char(lproc);
            return lexer_make_one_line_comment(lproc);
        } else if(lproc->peek_char(lproc) == '*') {         // `/**/`
            lproc->next_char(lproc);
            return lexer_make_mutiline_comment(lproc);
        } else {
            lproc->push_char(lproc, '/');                   // `/`
            return lexer_make_operator(lproc);
        }
    }
    return NULL;
}



static inline Token* lexer_make_connect_ident_or_string(LexProcess* lproc) {
    Buffer *buf = buffer_create();
    char c = 0;
    TokenType last_type = lproc->tmp_tok.type;
    buffer_printf(buf, "%s", lproc->tmp_tok.sval);
    vector_pop(lproc->token_vec);
    Token* pop_tok = (Token*)vector_back(lproc->token_vec);
    lproc->tmp_tok = *pop_tok;
    c = lproc->peek_char(lproc);
    for(;c == ' ' || c == '\t' || c == '\n'; ) {
        c = lproc->next_char(lproc);
        log_info("%c", c);
    }
    if(c == '"') {              // 如果有 ""，默认将前面转化为 string
        char c = lproc->next_char(lproc);
        for(; c!= '"' && c != EOF; c = lproc->next_char(lproc)) {
            if(c == '\\') { // 转义字符
                lproc->next_char(lproc);
            }
            buffer_write(buf, c);
        }
        buffer_write(buf, 0x00);
        return lexer_create_token(lproc, &(Token) {
            .type = TOKEN_TYPE_STRING,
            .sval = buffer_ptr(buf),
        });
    }
    LEX_GETC_IF(lproc, buf, c, c != ' ' && c != '\n' && c != EOF);
    return lexer_create_token(lproc, &(Token) {
        .type = last_type,
        .sval = buffer_ptr(buf),
    });
}

UNUSED static inline Token* lexer_make_feature_scope(LexProcess* lproc) {

}

static inline Token* lexer_make_macro_string(LexProcess* lproc) {
    Buffer *buf = buffer_create();
    char c = 0;
    LEX_GETC_IF(lproc, buf, c, c != ' ' && c != '#' &&  c != '\n' && c != EOF);
    return lexer_create_token(lproc, &(Token){
        .type = TOKEN_TYPE_STRING,
        .sval = buffer_ptr(buf),
    });
}
static inline Token* lexer_make_macro_keyword(LexProcess* lproc) {
    Buffer *buf = buffer_create();
    char c = 0;
    LEX_GETC_IF(lproc, buf, c, c != ' ' && c != '#' && c != '\n' && c != EOF);
    if(is_pre_keyword(buffer_ptr(buf))) {
        return lexer_create_token(lproc, &(Token){
            .type = TOKEN_TYPE_PRE_KEYWORD,
            .sval = buffer_ptr(buf)
        });
    } else if(lproc->pre.macro_sym_tbl != NULL && hashmap_contains(lproc->pre.macro_sym_tbl, buffer_ptr(buf))){
        const char* macro_buf = hashmap_get(lproc->pre.macro_sym_tbl, buffer_ptr(buf));
        return lexer_create_token(lproc, &(Token){
            .type = TOKEN_TYPE_STRING,
            .sval = macro_buf
        });
    } else {
        LEX_GETC_IF(lproc, buf, c, c != ' ' && c != '\n' && c != EOF);
        return lexer_create_token(lproc, &(Token){
            .type = TOKEN_TYPE_STRING,
            .sval = buffer_ptr(buf),
        });
    }
}

static inline Token* lexer_handle_macro(LexProcess* lproc) {
    char c = lproc->peek_char(lproc);
    if(c == '#') {
        lproc->next_char(lproc);
        while(lproc->peek_char(lproc) == ' ') {
            lproc->next_char(lproc);
        }
        if(lproc->peek_char(lproc) == '#') {                // `##`
            lproc->next_char(lproc);
            return lexer_make_connect_ident_or_string(lproc);
        } else if(lproc->peek_char(lproc) == '[') {         // `#[`
            lproc->next_char(lproc);
            return lexer_make_feature_scope(lproc);
        } else if(is_alpha(lproc->peek_char(lproc))) {      // `#alpha`
            return lexer_make_macro_keyword(lproc);
        } else {                                            // `#`
            return lexer_make_macro_string(lproc);
        }
    }
    return NULL;
}

static inline Token* lexer_preprocess_token(LexProcess* lproc) {
    Token* tok = NULL;
    if(tok = lexer_handle_comment(lproc)){
        return tok;
    }else if(tok = lexer_handle_macro(lproc)) {
        return tok;
    }
    return tok;
}


// ==================================================================================== //
//                              lexer Pub API: Char Operator
// ==================================================================================== //

char lex_process_next_char(LexProcess* lproc) {
    CompileProcess* cproc = lproc->compile_proc;
    char c = getc(cproc->cfile->fp);
    char* str = char_display(c);
    log_trace("get char `%s` (line %d col %d)", str, lproc->pos.line, lproc->pos.col);
    free(str);

    // 更新 pos
    lproc->last_pos = lproc->pos;
    lproc->pos.col += 1;

    if(c == '\n') {
        lproc->pos.line += 1;
        lproc->pos.col = 1;
    }

    return c;
}

char lex_process_peek_char(LexProcess* lproc) {
    CompileProcess* cproc = lproc->compile_proc;
    // 读取下一字符，留在原地
    char c = getc(cproc->cfile->fp);
    ungetc(c, cproc->cfile->fp);
    if(c == -1) {
        return -1;
    }
    // 输出
    char* str = char_display(c);
    log_trace("peek char `%s` (line %d col %d)", str, lproc->pos.line, lproc->pos.col);
    free(str);

    return c;
}

void lex_process_push_char(LexProcess* lproc, char c) {

    // 输出
    char* str = char_display(c);
    log_trace("push char `%s` (line %d col %d)", str, lproc->pos.line, lproc->pos.col);
    free(str);

    // 将字符 c 替换当前字符，并退一位
    CompileProcess* cproc = lproc->compile_proc;
    lproc->pos = lproc->last_pos;
    ungetc(c, cproc->cfile->fp);
    
}

LexProcess* lex_process_create(CompileProcess* cproc, void* priv) {
    LOG_TAG
    LexProcess* lproc = malloc(sizeof(LexProcess));
    *lproc = (LexProcess){
        .compile_proc = cproc,
        .cur_expr_depth = 0,
        .cur_list_depth = 0,
        .cur_scope_depth = 0,
        .pos = (Pos){.col = 1, .line = 1, .filename = cproc->cfile->path},
        .token_vec = vector_create(sizeof(Token)),
        .priv = priv,
        .next_char = lex_process_next_char,
        .peek_char = lex_process_peek_char,
        .push_char = lex_process_push_char
    };

    return lproc;
}

Vector* lex_process_tokens(LexProcess* lproc) {

    return lproc->token_vec;
}
void* lex_process_private(LexProcess* lproc) {

    return lproc->priv;
}

void lex_process_free(LexProcess* lproc) {
    LOG_TAG
    if(!lproc) {
        return;
    }
    vector_free(lproc->token_vec);
    free(lproc->priv);
    if(lproc->next_char) lproc->next_char = NULL;
    if(lproc->peek_char) lproc->peek_char = NULL;
    if(lproc->push_char) lproc->push_char = NULL;
    free(lproc);
}



Token* lex_process_next_token(LexProcess* lproc) {
    Token* tok = NULL;
    char c = lproc->peek_char(lproc);
    tok = lexer_preprocess_token(lproc);
    if(tok) {               // comment or macro
        return tok;
    }
    switch (c) {
        CASE_NUMERIC:
            tok = lexer_make_number(lproc);
            break;
        CASE_OPERATOR_EXCLUDING_DIVISON:
            tok = lexer_make_operator(lproc);
            break;
        CASE_SYMBOL:
            tok = lexer_make_symbol(lproc);
            break;
        case '\'':
            tok = lexer_make_quote(lproc);
            break;
        case '"':
            tok = lexer_make_string(lproc);
            break;
        case '\n':
            tok = lexer_make_newline(lproc);
            break;
        case ' ':
        case '\t':
            lproc->next_char(lproc);
            tok = lex_process_next_token(lproc);
            break;
        case EOF:
            tok = lexer_make_EOF(lproc);
            break;
        default:
            tok = lexer_read_token_special(lproc);
            if(!tok) {
                lexer_error("Invalid token char: `%c`\n", c);
                exit(LEXER_ANALYSIS_ERROR);
            }
            break;
    }
    return tok;
}

