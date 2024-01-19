

#include "token.h"

struct token tmp_token;

static inline void token_pos_read(Buffer* buf,  struct pos* pos) {
    buffer_printf(buf, "  Read token pos: \n");
    buffer_printf(buf, "    file   : %s\n", pos->filename);
    buffer_printf(buf, "    line   : %d\n", pos->line);
    buffer_printf(buf, "    col    : %d\n", pos->col);
    log_debug("%s", buf->data);
}

static inline void token_read(struct token *tok) {
    Buffer *buf = buffer_create();
    buffer_printf(buf, "\n  Read token: \n");

    switch(tok->type) {
        case TOKEN_TYPE_IDENTIFIER:
            buffer_printf(buf, "    type   : identifier\n");
            buffer_printf(buf, "    value  : %s\n", tok->sval);
            break;
        case TOKEN_TYPE_KEYWORD:
            buffer_printf(buf, "    type   : keyword\n");
            buffer_printf(buf, "    value  : %s\n", tok->sval);
            break;
        case TOKEN_TYPE_OPERATOR:
            buffer_printf(buf, "    type   : operator\n");
            buffer_printf(buf, "    value  : %s\n", tok->sval);
            break;
        case TOKEN_TYPE_SYMBOL:
            buffer_printf(buf, "    type   : symbol\n");
            buffer_printf(buf, "    value  : %c\n", tok->cval);
            break;
        case TOKEN_TYPE_NUMBER:
            buffer_printf(buf, "    type   : number\n");
            buffer_printf(buf, "    value  : %d\n", tok->inum);
            break;
        case TOKEN_TYPE_STRING:
            buffer_printf(buf, "    type   : string\n");
            buffer_printf(buf, "    value  : %s\n", tok->sval);
            break;
        case TOKEN_TYPE_COMMENT:
            buffer_printf(buf, "    type   : comment\n");
            buffer_printf(buf, "    value  : %s\n", tok->sval);
            break;
    };
    buffer_printf(buf, "    whitespace       : %d\n", tok->whitespace ? "true" : "false");
    buffer_printf(buf, "    between_brackets : %s\n", tok->between_brackets);

    token_pos_read(buf, &tok->pos);
    // 打印buf信息
    buffer_free(buf);
}

struct token* token_create(struct token* _token) {
    memcpy(&tmp_token, _token, sizeof(struct token));
    token_read(_token);
    return &tmp_token;
}