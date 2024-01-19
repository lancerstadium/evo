


#include "lexer.h"



char lex_process_next_char(LexProcess* lproc) {
    CompileProcess* cproc = lproc->compile_proc;
    char c = getc(cproc->cfile.fp);

    char* str = char_display(c);
    log_debug("get char %s (line %d col %d)", str, lproc->pos.line, lproc->pos.col);
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
    char c = getc(cproc->cfile.fp);
    ungetc(c, cproc->cfile.fp);

    // 输出
    char* str = char_display(c);
    log_debug("peek char %s (line %d col %d)", str, lproc->pos.line, lproc->pos.col);
    free(str);

    return c;
}

void lex_process_push_char(LexProcess* lproc, char c) {

    // 输出
    char* str = char_display(c);
    log_debug("push char %s (line %d col %d)", str, lproc->pos.line, lproc->pos.col);
    free(str);

    // 将字符 c 替换当前字符，并退一位
    CompileProcess* cproc = lproc->compile_proc;
    lproc->pos = lproc->last_pos;
    ungetc(c, cproc->cfile.fp);
    
}

LexProcess* lex_process_create(CompileProcess* cproc, void* priv) {
    LOG_TAG
    LexProcess* lproc = malloc(sizeof(LexProcess));
    *lproc = (LexProcess){
        .compile_proc = cproc,
        .pos = (Pos){.col = 1, .line = 1, .filename = cproc->cfile.path},
        .token_vec = vector_create(sizeof(struct token)),
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
    free(lproc);
}