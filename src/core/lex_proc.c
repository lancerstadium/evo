


#include "lexer.h"

// ==================================================================================== //
//                                  lexer: Char operate
// ==================================================================================== //

#define LEX_NEXTC(lex) (lex->next_char(lex))
#define LEX_PEEKC(lex) (lex->peek_char(lex))

#define LEX_GETC_IF(lex, buffer, c, EXP)                             \
    do{                                                              \
        for (c = lex->peek_char(lex); EXP; c = lex->peek_char(lex)){ \
            buffer_write(buffer, c);                                 \
            lex->next_char(lex);                                     \
        }                                                            \
    } while (0)

#define LEX_ASSERT_NEXTC(lex, ec)     \
    do{                               \
        char c = lex->next_char(lex); \
        log_assert(c == ec);          \
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
    case '(':                           \
    case '[':                           \
    case ',':                           \
    case '.':                           \
    case '?'

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
    case ':':       \
    case ';':       \
    case '#':       \
    case '\\':      \
    case ')':       \
    case ']'

// ==================================================================================== //
//                                  lexer: String Identity
// ==================================================================================== //

static const char* lex_keyword[] = {
    "unsigned", "signed", "char", "int",
    "double", "long", "void", "struct",
    "union", "static", "if", "else"
};

static const char lex_single_op[] = {
    '+', '-', '/', '*', '=', '>', '<',
    '|', '&', '^', '%', '~', '!'
};

static const char* lex_binary_op[] = {
    "+=", "-=", "*=", "/=", ">>", "<<",
    ">=", "<=", "||", "&&", "++", "--",
    "==", "!=", "!=", "->"
};

#define LEX_KEYWORD_NUM    GET_ARR_LEN(lex_keyword)
#define LEX_SINGLE_OP_NUM  GET_ARR_LEN(lex_single_op)
#define LEX_BINARY_OP_NUM  GET_ARR_LEN(lex_binary_op)

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

static inline bool is_valid_operator(const char* op) {
    return is_single_operator(*op) || is_binary_operator(op);
}

// ==================================================================================== //
//                                  lexer: Read String
// ==================================================================================== //

static inline const char* lexer_read_operator(LexProcess* lproc) {
    char op = lproc->next_char(lproc);
    Buffer* buf = buffer_create();
    buffer_write(buf, op);
    op = lproc->peek_char(lproc);
    if(is_single_operator(op)) {
        buffer_write(buf, op);
        lproc->next_char(lproc);
    }
    buffer_write(buf, 0x00);    // 结束符号
    
    const char* buf_ptr = buffer_ptr(buf);
    if(!is_valid_operator(buf_ptr)) {
        lexer_error("Invalid operator: `%s`", buf_ptr);
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
    token_create(_token);
    tmp_token.pos = lproc->pos;
    token_read(&tmp_token);
    return &tmp_token;
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
    return lexer_create_token(lproc, &(Token){
        .type = TOKEN_TYPE_NUMBER,
        .llnum = lexer_read_number(lproc)
    });
}

static inline Token* lexer_make_ident_or_keyword(LexProcess* lproc) {
    Buffer* buf = buffer_create();
    char c = lproc->peek_char(lproc);
    LEX_GETC_IF(lproc, buf, c, (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_');
    buffer_write(buf, 0x00);                        // 结束符号
    if(is_keyword(buffer_ptr(buf))) {               // 返回keyword
        return lexer_create_token(lproc, &(Token){
            .type = TOKEN_TYPE_KEYWORD,
            .sval = buffer_ptr(buf)
        });
    } else {
        return lexer_create_token(lproc, &(Token){   // 返回ident
            .type = TOKEN_TYPE_IDENTIFIER,
            .sval = buffer_ptr(buf)
        });
    }
}

static inline Token* lexer_make_symbol(LexProcess* lproc) {
    char c = lproc->next_char(lproc);
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
        LEX_GETC_IF(lproc, buf, c, c != '\n' && c != EOF);
        if(c == EOF) {
            lexer_error("EOF reached whilst in a multi-line comment. The comment was not terminated! \"%s\" \n", buffer_ptr(buf));
            exit(LEXER_ANALYSIS_ERROR);
        } else if(c == '*') {
            lproc->next_char(lproc);
            if(lproc->peek_char(lproc) == '/') {    // comment 结束
                lproc->next_char(lproc);
                break;
            }
        }
    }
    return lexer_create_token(lproc, &(Token){
        .type = TOKEN_TYPE_COMMENT,
        .sval = buffer_ptr(buf)
    });
}

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
    buffer_free(lproc->parenthesis_buffer);
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
    tok = lexer_handle_comment(lproc);
    if(tok) {               // comment
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

