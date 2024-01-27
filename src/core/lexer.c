

#include "lexer.h"



int lex(LexProcess* lproc) {
    LOG_TAG
    Token* tok = lex_process_next_token(lproc);
    while(tok && tok->type != TOKEN_TYPE_EOF) {
        vector_push(lproc->token_vec, tok);
        tok = lex_process_next_token(lproc);
    }
    int i = 0;
    for(int i = 0; i < lproc->token_vec->count; i++) {
        tok = vector_at(lproc->token_vec, i);
        token_read(tok);
    }
    return LEXER_ANALYSIS_OK;
}