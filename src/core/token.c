

#include "token.h"

Token tmp_token;

static const char* token_type_str[] = {
    [TOKEN_TYPE_IDENTIFIER]     = "identifier",
    [TOKEN_TYPE_KEYWORD]        = "keyword",
    [TOKEN_TYPE_OPERATOR]       = "operator",
    [TOKEN_TYPE_SYMBOL]         = "symbol",
    [TOKEN_TYPE_NUMBER]         = "number",
    [TOKEN_TYPE_STRING]         = "string",
    [TOKEN_TYPE_COMMENT]        = "comment",
    [TOKEN_TYPE_NEWLINE]        = "newline",
    [TOKEN_TYPE_PRE_KEYWORD]    = "pre-keyword",
    [TOKEN_TYPE_DATATYPE]       = "datatype",
    [TOKEN_TYPE_EOF]            = "EOF"
};

static inline void token_pos_read(Buffer* buf,  struct pos* pos) {
    buffer_printf(buf, "    Pos         : %s:%d:%d\n", pos->filename, pos->line, pos->col);
}

char* token_get_type_str(Token* tok) {
    return (char*)(token_type_str[tok->type] ? token_type_str[tok->type] : "Unknown");
}

void token_write_buffer(Token *tok, Buffer* buf) {
    if(!tok) return;
    if(buf->len == 0) {
        buffer_printf(buf, "\n%4d    ", 1);
    }
    const char* tmp_str = &buf->data[buf->len - 4];
    switch(tok->type) {
        case TOKEN_TYPE_SYMBOL:
            if(STR_EQ(tmp_str, "    ")) {
                switch(tok->cval) {
                    case ']': 
                    case ')': 
                    case '}': buf->len -= 4; break;
                }
            }
            switch(tok->cval) {
                case ',': buffer_printf(buf, "%c ", tok->cval); break;
                default : buffer_printf(buf, "%c", tok->cval); break;
            }
            break;
        case TOKEN_TYPE_NUMBER:
            buffer_printf(buf, _bcyan("%d"), tok->inum);
            break;
        case TOKEN_TYPE_COMMENT:
            break;
        case TOKEN_TYPE_IDENTIFIER:
            buffer_printf(buf, _bblue("%s"), tok->sval);
            break;
        case TOKEN_TYPE_OPERATOR:
            buffer_printf(buf, _bred(" %s "), tok->sval);
            break;
        case TOKEN_TYPE_PRE_KEYWORD:
            buffer_printf(buf, _yellow("#%s "), tok->sval);
            break;
        case TOKEN_TYPE_DATATYPE:
            if(buf->data[buf->len - 1] == '\n' || STR_EQ(tmp_str, "    ")) {
                buffer_printf(buf, _mag("%s "), datatype_str[tok->inum]);
            }else {
                buffer_printf(buf, _mag("%s"), datatype_str[tok->inum]);
            }
            break;
        case TOKEN_TYPE_STRING:
            buffer_printf(buf, _bgreen("\"%s\""), tok->sval);
            break;
        case TOKEN_TYPE_KEYWORD:
            if(buf->data[buf->len - 1] == '\n' || STR_EQ(tmp_str, "    ")) {
                buffer_printf(buf, _yellow("%s "), tok->sval);
            }else {
                buffer_printf(buf, _yellow("%s"), tok->sval);
            }
            break;
        case TOKEN_TYPE_EOF:
            break;
        case TOKEN_TYPE_NEWLINE:
            if(buf->data[buf->len - 1] != '\n') {
                buffer_printf(buf, "\n%4d    ", tok->pos.line);
                int sum_depth = tok->edep + tok->ldep + tok->sdep;
                for(int i = 0; i < sum_depth; i++) {
                    buffer_printf(buf, "    ");
                }
            }
            break;
        default:
            break;
    };
}

void token_read(Token *tok) {
    if(!tok) return;
    Buffer *buf = buffer_create();
    buffer_printf(buf, "\n  Read token: \n");
    buffer_printf(buf, "    Type        : %s\n", token_get_type_str(tok));
    buffer_printf(buf, "    Depth       : (%d) [%d] {%d}\n", tok->edep, tok->ldep, tok->sdep);

    switch(tok->type) {

        case TOKEN_TYPE_SYMBOL:
            buffer_printf(buf, "    Val         : %c\n", tok->cval);
            break;
        case TOKEN_TYPE_NUMBER:
            buffer_printf(buf, "    Val         : %d\n", tok->inum);
            break;
        case TOKEN_TYPE_STRING:
        case TOKEN_TYPE_IDENTIFIER:
        case TOKEN_TYPE_KEYWORD:
        case TOKEN_TYPE_OPERATOR:
        case TOKEN_TYPE_COMMENT:
        case TOKEN_TYPE_PRE_KEYWORD:
            buffer_printf(buf, "    Val         : %s\n", tok->sval);
            break;
        case TOKEN_TYPE_DATATYPE:
            buffer_printf(buf, "    Val         : %s\n", datatype_str[tok->inum]);
            break;
        case TOKEN_TYPE_EOF:
            break;
        case TOKEN_TYPE_NEWLINE:
            buffer_printf(buf, "    Val         : \\n\n");
            break;
        default:
            break;
    };

    token_pos_read(buf, &tok->pos);
    // 打印buf信息
    log_debug("%s", buf->data);
    buffer_free(buf);
}
