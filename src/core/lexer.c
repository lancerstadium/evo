

#include "lexer.h"



int lex(LexProcess* lproc) {
    LOG_TAG
    Token* tok = lexer_read_next_token(lproc);
    while(tok && tok->type != TOKEN_TYPE_EOF) {
        vector_push(lproc->token_vec, tok);
        tok = lexer_read_next_token(lproc);
    } 
    return LEXER_ANALYSIS_OK;
}