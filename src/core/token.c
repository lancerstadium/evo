

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

void token_read(Token *tok) {
    if(!tok) return;
    Buffer *buf = buffer_create();
    buffer_printf(buf, "\n  Read token: \n");
    buffer_printf(buf, "    Type        : %s\n", token_get_type_str(tok));

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
        case TOKEN_TYPE_DATATYPE:
            buffer_printf(buf, "    Val         : %s\n", tok->sval);
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
