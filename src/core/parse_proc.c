
#include "parser.h"


// ==================================================================================== //
//                               parser Pri Data
// ==================================================================================== //

#define PARSE_MAX_OPS_IN_GROUP 12

typedef enum {
    PARSE_OPERATOR_ASSOC_LEFT2RIGHT,
    PARSE_OPERATOR_ASSOC_RIGHT2LEFT
} OperatorAssociativity;

typedef struct {
    char* ops[PARSE_MAX_OPS_IN_GROUP];
    OperatorAssociativity asc;
} OpGroup;

OpGroup op_level[] = {
    {.ops = {"++", "--", "()", "(", "[", "]", ".", "->", NULL},                             .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
    {.ops = {"*", "/", "%%", NULL},                                                         .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
    {.ops = {"+", "-", NULL},                                                               .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
    {.ops = {"<<", ">>", NULL},                                                             .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
    {.ops = {"<", "<=", ">", ">=", NULL},                                                   .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
    {.ops = {"==", "!=", NULL},                                                             .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
    {.ops = {"&", NULL},                                                                    .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
    {.ops = {"^", NULL},                                                                    .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
    {.ops = {"|", NULL},                                                                    .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
    {.ops = {"&&", NULL},                                                                   .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
    {.ops = {"||", NULL},                                                                   .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
    {.ops = {"?", ":", NULL},                                                               .asc = PARSE_OPERATOR_ASSOC_RIGHT2LEFT},
    {.ops = {"=", "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "^=", "|=", NULL},      .asc = PARSE_OPERATOR_ASSOC_RIGHT2LEFT},
    {.ops = {",", NULL},                                                                    .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
};

#define PARSE_OP_LEVEL_NUM  GET_ARR_LEN(op_level)

static inline int parser_get_op_level(const char* op, OpGroup** group_out) {
    *group_out = NULL;
    for(int i = 0; i < PARSE_OP_LEVEL_NUM; i++) {
        for(int b = 0; op_level[i].ops[b]; b++) {
            const char *_op = op_level[i].ops[b];
            if(STR_EQ(op, _op)) {
                *group_out = &op_level[i];
                return i;
            }
        }
    }
    return -1;
}

static inline bool parser_left_op_has_priority(const char* op_left, const char* op_right) {
    OpGroup* group_left = NULL;
    OpGroup* group_right = NULL;

    // 一样的运算符？
    if(STR_EQ(op_left, op_right)){
        return false;
    }

    int level_left = parser_get_op_level(op_left, &group_left);
    int level_right = parser_get_op_level(op_right, &group_right);
    if(group_left->asc == PARSE_OPERATOR_ASSOC_RIGHT2LEFT) {
        return false;
    }
    return level_left <= level_right;
}

static inline bool is_unary_operator(const char *op) {
    return STR_EQ(op, "-") || STR_EQ(op, "!") || STR_EQ(op, "~");
}

// ==================================================================================== //
//                            parser: Token -> Node
// ==================================================================================== //


static inline void parser_single_token2node(ParseProcess* pproc) {
    Token* tok = pproc->next_token(pproc);
    Node* nd = NULL;
    switch(tok->type) {
        case TOKEN_TYPE_NUMBER:
            nd = pproc->create_node(pproc, &(Node){
                .type = NODE_TYPE_NUM,
                .llnum = tok->llnum
            });
            break;
        case TOKEN_TYPE_IDENTIFIER:
            nd = pproc->create_node(pproc, &(Node){
                .type = NODE_TYPE_IDENT,
                .sval = tok->sval
            });
            break;
        case TOKEN_TYPE_EOF:
            nd = pproc->create_node(pproc, &(Node){
                .type = NODE_TYPE_EOF,
            });
            break;
        default:
            parser_error("Problem converting token to node. No valid node exists for token of type %i\n", tok->type);
            break;
    }
}

static inline void parser_excp_operator(ParseProcess* pproc, char* op) {
    Token* next_token = pproc->next_token(pproc);
    if(next_token == NULL || next_token->type != TOKEN_TYPE_OPERATOR || !STR_EQ(next_token->sval, op)) {
        parser_error("Expecting the symbol `%s` but `%s` was provided in %s:%d:%d", 
        op, next_token->sval, next_token->pos.filename, next_token->pos.line, next_token->pos.col);
    }
}

static inline void parser_excp_keyword(ParseProcess* pproc, char* kw) {
    Token* next_token = pproc->next_token(pproc);
    if(next_token == NULL || next_token->type != TOKEN_TYPE_KEYWORD || !STR_EQ(next_token->sval, kw)) {
        parser_error("Expecting the keyword `%s` but `%s` was provided in %s:%d:%d", 
        kw, next_token->sval, next_token->pos.filename, next_token->pos.line, next_token->pos.col);
    }
}

static inline void parser_excp_ident(ParseProcess* pproc, char* ident) {
    Token* next_token = pproc->next_token(pproc);
    if(next_token == NULL || next_token->type != TOKEN_TYPE_IDENTIFIER || !STR_EQ(next_token->sval, ident)) {
        parser_error("Expecting the ident `%s` but `%s` was provided in %s:%d:%d", 
        ident, next_token->sval, next_token->pos.filename, next_token->pos.line, next_token->pos.col);
    }
}

static inline void parser_excp_symbol(ParseProcess* pproc, char c) {
    Token* next_token = pproc->next_token(pproc);
    if(next_token == NULL || next_token->type != TOKEN_TYPE_SYMBOL || next_token->cval != c) {
        parser_error("Expecting the symbol `%c` but `%c` was provided in %s:%d:%d", 
        c, next_token->cval, next_token->pos.filename, next_token->pos.line, next_token->pos.col);
    }
}

// ==================================================================================== //
//                               parser: Token Ident
// ==================================================================================== //


static inline bool parser_next_token_is_operator(ParseProcess* pproc, const char* op) {
    Token* tok = pproc->peek_token(pproc);
    return tok && tok->type == TOKEN_TYPE_OPERATOR && STR_EQ(tok->sval, op);
}

static inline bool parser_next_token_is_keyword(ParseProcess* pproc, const char* kw) {
    Token* tok = pproc->peek_token(pproc);
    return tok && tok->type == TOKEN_TYPE_KEYWORD && STR_EQ(tok->sval, kw);
}

static inline bool parser_next_token_is_ident(ParseProcess* pproc, const char* ident) {
    Token* tok = pproc->peek_token(pproc);
    return tok && tok->type == TOKEN_TYPE_IDENTIFIER && STR_EQ(tok->sval, ident);
}

static inline bool parser_next_token_is_datatype(ParseProcess* pproc, const char* dt) {
    Token* tok = pproc->peek_token(pproc);
    return tok && tok->type == TOKEN_TYPE_DATATYPE && STR_EQ(tok->sval, dt);
}

static inline bool parser_next_token_is_symbol(ParseProcess* pproc, char sym) {
    Token* tok = pproc->peek_token(pproc);
    return tok && tok->type == TOKEN_TYPE_SYMBOL && tok->cval == sym;
}

// ==================================================================================== //
//                             parser: AST Consturct
// ==================================================================================== //


static inline void parser_make_stmt(ParseProcess* pproc);
static inline void parser_make_expr_stmt(ParseProcess* pproc);
static inline void parser_make_compound_stmt(ParseProcess* pproc);

// stmt = "{" compound-stmt
//      | expr-stmt
static inline void parser_make_stmt(ParseProcess* pproc) {
    if(parser_next_token_is_symbol(pproc, '{')) {
        parser_make_compound_stmt(pproc);
    }
    parser_make_expr_stmt(pproc);
}
// compound-stmt = stmt* "}"
static inline void parser_make_compound_stmt(ParseProcess* pproc) {

}
// expr-stmt = expr? ";"
static inline void parser_make_expr_stmt(ParseProcess* pproc) {

}

static inline void parser_make_expr_node(ParseProcess* pproc, Node* nd_l, Node* nd_r, const char* op) {
    pproc->create_node(pproc, &(Node){
        .type = NODE_TYPE_EXPR,
        .expr.op = op
    });
}




// 处理 keyword：
// keyword = "mod" ident "{" (ident ":" newline)* "}"
//         | "use" 
//         | "type" --> handle type
//         | "fn"   --> handle fn
static inline void parser_handle_keyword(ParseProcess* pproc, const char* kw) {
    LOG_TAG
    Token* tok;
    if(STR_EQ(kw, "mod")) {
        pproc->next_token(pproc);
        tok = pproc->peek_token(pproc);
        if(tok->type == TOKEN_TYPE_IDENTIFIER) {
            pproc->root->prog.main_mod->mod.name = tok->sval;
            pproc->next_token(pproc);
        } else {
            pproc->root->prog.main_mod->mod.name = pproc->root->prog.name;
            pproc->next_token(pproc);
        }
    }else if(STR_EQ(kw, "use")) {
        pproc->next_token(pproc);
        if(parser_next_token_is_symbol(pproc, '{')) {
            pproc->next_token(pproc);
            while(pproc->next_token(pproc)->type = TOKEN_TYPE_IDENTIFIER) {

            }
        }
        pproc->push_node(pproc, &(Node){
            .type = NODE_TYPE_EXPR,
        });
    }else if(STR_EQ(kw, "type")) {
        pproc->next_token(pproc);
    }else if(STR_EQ(kw, "fn")) {
        pproc->next_token(pproc);
    }else {
        parser_error("TODO handle keyword `%s`", kw);
    }
}

// ==================================================================================== //
//                            parser: Token Operations
// ==================================================================================== //

Token* parse_process_next_token(ParseProcess* pproc) {
    return vector_peek(pproc->lex_proc->token_vec);
};

Token* parse_process_peek_token(ParseProcess* pproc) {
    Token* next_tok = vector_peek_no_increment(pproc->lex_proc->token_vec);
    if(next_tok->type == TOKEN_TYPE_NEWLINE) {
        vector_peek(pproc->lex_proc->token_vec);
    }
    return vector_peek_no_increment(pproc->lex_proc->token_vec);
};

Token* parse_process_excp_token(ParseProcess* pproc, NodeType type) {
    Token* token = pproc->next_token(pproc);
    if(token->type != type) {
        parser_error("Unexpected token type: %s", token_get_type_str(token));
    }
    return token;
};


// ==================================================================================== //
//                            parser: Node Operations
// ==================================================================================== //

Node* parse_process_peek_node(ParseProcess* pproc) {
    return vector_back_ptr_or_null(pproc->node_vec);
}

Node* parse_process_pop_node(ParseProcess* pproc) {
    Node* last_node = (Node*)vector_back_ptr_or_null(pproc->node_vec);
    Node* last_node_root = (Node*)vector_back_ptr_or_null(pproc->node_tree_vec);
    vector_pop(pproc->node_vec);
    if(last_node && last_node == last_node_root) {
        vector_pop(pproc->node_tree_vec);
    }
    return last_node;
}

void parse_process_push_node(ParseProcess* pproc, Node* node) {
    vector_push(pproc->node_vec, node);
}

Node* parse_process_create_node(ParseProcess* pproc, Node* _node) {
    Node* node = malloc(sizeof(Node));
    memcpy(node, _node, sizeof(Node));
    pproc->push_node(pproc, node);
    return node;
}


// ==================================================================================== //
//                              parser Pub API: 
// ==================================================================================== //

ParseProcess* parse_process_create(LexProcess* lproc) {
    LOG_TAG
    ParseProcess* pproc = malloc(sizeof(ParseProcess));
    *pproc = (ParseProcess) {
        .lex_proc = lproc,
        .node_vec = vector_create(sizeof(Node)),
        .node_tree_vec = vector_create(sizeof(Node)),
        .symbol_tbl  = hashmap_create(),
        .next_token  = parse_process_next_token,
        .peek_token  = parse_process_peek_token,
        .excp_token  = parse_process_excp_token,
        .peek_node   = parse_process_peek_node,
        .pop_node    = parse_process_pop_node,
        .push_node   = parse_process_push_node,
        .create_node = parse_process_create_node
    };

    
    Node* mprog = parse_process_create_node(pproc, &(Node){
        .type = NODE_TYPE_PROG,
        .depth = 0,
        .prog.name = fio_get_bare_filename(pproc->lex_proc->compile_proc->cfile),
    });

    Node* mmod = parse_process_create_node(pproc, &(Node){
        .type = NODE_TYPE_MOD,
        .depth = 1,
    });

    pproc->root = vector_at(pproc->node_vec, 0);
    pproc->root->prog.main_mod = vector_at(pproc->node_vec, 1);

    log_info("root: %p", pproc->root);

    return pproc;
}

void parse_process_free(ParseProcess* pproc) {
    LOG_TAG
    if(!pproc) {
        return;
    }
    vector_free(pproc->node_tree_vec);
    vector_free(pproc->node_vec);
    hashmap_destroy(pproc->symbol_tbl);
    if(pproc->next_token)   pproc->next_token = NULL;
    if(pproc->peek_token)   pproc->peek_token = NULL;
    if(pproc->excp_token)   pproc->excp_token = NULL;
    if(pproc->peek_node)    pproc->peek_node = NULL;
    if(pproc->pop_node)     pproc->pop_node = NULL;
    if(pproc->push_node)    pproc->push_node = NULL;
    if(pproc->create_node)  pproc->create_node = NULL;
    free(pproc);
}

int parse_process_next(ParseProcess* pproc) {
    LOG_TAG
    Token* tok = pproc->peek_token(pproc);
    if(!tok) {
        return -1;
    }
    int res = 0;
    switch(tok->type) {
        case TOKEN_TYPE_NUMBER:
            break;
        case TOKEN_TYPE_IDENTIFIER:
            break;
        case TOKEN_TYPE_OPERATOR:
            break;
        case TOKEN_TYPE_SYMBOL:
            break;
        case TOKEN_TYPE_KEYWORD:
            parser_handle_keyword(pproc, tok->sval);
            break;
        case TOKEN_TYPE_COMMENT:
            break;
        case TOKEN_TYPE_NEWLINE:
            break;
        case TOKEN_TYPE_EOF:
            res = -1;
            break;
        default:
            parser_error("Unexpected token type: %s", token_get_type_str(tok));
            // token_read(tok);
            break;
    }
    return res;
}
