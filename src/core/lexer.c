

#include "lexer.h"



int lex(LexProcess* lproc) {
    LOG_TAG
    // 1. 逐个解析令牌
    Token* tok = lex_process_next_token(lproc);
    while(tok && tok->type != TOKEN_TYPE_EOF) {
        vector_push(lproc->token_vec, tok);
        tok = lex_process_next_token(lproc);
    }
    vector_push(lproc->token_vec, tok);
    // 2. 检查括号匹配
    while(!vector_empty(lproc->parens_vec)) {
        tok = vector_back(lproc->parens_vec);
        lexer_error("unclosed `%c` at %s:%d:%d", tok->cval, tok->pos.filename, tok->pos.line, tok->pos.col);
        vector_pop(lproc->parens_vec);
    }
    // 3. 打印输出 token
    Buffer* buf = buffer_create();
    buffer_printf(buf, "\n");
    for(int i = 0; i < lproc->token_vec->count; i++) {
        tok = vector_at(lproc->token_vec, i);
        // token_read(tok);
        token_write_buffer(tok, buf);
    }
    log_info(buffer_ptr(buf));
    return LEXER_ANALYSIS_OK;
}