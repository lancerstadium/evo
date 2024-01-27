

#include "token.h"

Token tmp_token;



static inline void token_pos_read(Buffer* buf,  struct pos* pos) {
    buffer_printf(buf, "    Position         : %s:%d:%d\n", pos->filename, pos->line, pos->col);
}

char* token_get_type_str(Token* tok) {
    switch(tok->type) {
        case TOKEN_TYPE_IDENTIFIER:     return "identifier";
        case TOKEN_TYPE_KEYWORD:        return "keyword";
        case TOKEN_TYPE_OPERATOR:       return "operator";
        case TOKEN_TYPE_SYMBOL:         return "symbol";
        case TOKEN_TYPE_NUMBER:         return "number";
        case TOKEN_TYPE_STRING:         return "string";
        case TOKEN_TYPE_COMMENT:        return "comment";
        case TOKEN_TYPE_NEWLINE:        return "newline";
        case TOKEN_TYPE_PRE_KEYWORD:    return "pre-keyword";
        case TOKEN_TYPE_EOF:            return "EOF";
        default:                        return "unknown";
    };
}

void token_read(Token *tok) {
    if(!tok) return;
    Buffer *buf = buffer_create();
    buffer_printf(buf, "\n  Read token: \n");

    switch(tok->type) {
        case TOKEN_TYPE_IDENTIFIER:
            buffer_printf(buf, "    type             : %s\n", token_get_type_str(tok));
            buffer_printf(buf, "    value            : %s\n", tok->sval);
            break;
        case TOKEN_TYPE_KEYWORD:
            buffer_printf(buf, "    type             : %s\n", token_get_type_str(tok));
            buffer_printf(buf, "    value            : %s\n", tok->sval);
            break;
        case TOKEN_TYPE_OPERATOR:
            buffer_printf(buf, "    type             : %s\n", token_get_type_str(tok));
            buffer_printf(buf, "    value            : %s\n", tok->sval);
            break;
        case TOKEN_TYPE_SYMBOL:
            buffer_printf(buf, "    type             : %s\n", token_get_type_str(tok));
            buffer_printf(buf, "    value            : %c\n", tok->cval);
            break;
        case TOKEN_TYPE_NUMBER:
            buffer_printf(buf, "    type             : %s\n", token_get_type_str(tok));
            buffer_printf(buf, "    value            : %d\n", tok->inum);
            break;
        case TOKEN_TYPE_STRING:
            buffer_printf(buf, "    type             : %s\n", token_get_type_str(tok));
            buffer_printf(buf, "    value            : %s\n", tok->sval);
            break;
        case TOKEN_TYPE_COMMENT:
            buffer_printf(buf, "    type             : %s\n", token_get_type_str(tok));
            buffer_printf(buf, "    value            : %s\n", tok->sval);
            break;
        case TOKEN_TYPE_PRE_KEYWORD:
            buffer_printf(buf, "    type             : %s\n", token_get_type_str(tok));
            buffer_printf(buf, "    value            : %s\n", tok->sval);
            break;
        case TOKEN_TYPE_EOF:
            buffer_printf(buf, "    type             : %s\n", token_get_type_str(tok));
            break;
        case TOKEN_TYPE_NEWLINE:
            buffer_printf(buf, "    type             : %s\n", token_get_type_str(tok));
            buffer_printf(buf, "    value            : \\n\n");
            break;
        default:
            break;
    };
    buffer_printf(buf, "    whitespace       : %s\n", (tok->whitespace == false) ? "false" : "true");
    if(tok->between_brackets) {
        buffer_printf(buf, "    between_brackets : %s\n", tok->between_brackets);
    }

    token_pos_read(buf, &tok->pos);
    // 打印buf信息
    log_debug("%s", buf->data);
    buffer_free(buf);
}
