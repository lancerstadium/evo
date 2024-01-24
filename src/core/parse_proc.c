
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

static OpGroup op_precedence[] = {
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


static const char* keyword_variable_modified[] = {
    "unsigned", "signed", "static"
};

static const char* keyword_variable_datatype[] = {
    [DATA_TYPE_ANY]   = "void",
    [DATA_TYPE_CHAR]  = "char",
    [DATA_TYPE_SHORT] = "short",
    [DATA_TYPE_INT]   = "int",
    [DATA_TYPE_LONG]  = "long",
    [DATA_TYPE_FLOAT] = "float",
    [DATA_TYPE_DOUBLE]= "double",
};

#define PARSE_KEYWORD_VAR_MODIFIED_NUM  GET_ARR_LEN(keyword_variable_modified)

#define PARSE_KEYWORD_VAR_DATATYPE_NUM  GET_ARR_LEN(keyword_variable_datatype)

static inline bool is_keyword_variable_modified(const char* str) {
    bool is_key_mod = false;
    int i;
    for(i = 0; i < PARSE_KEYWORD_VAR_MODIFIED_NUM; i++) {
        if(keyword_variable_modified[i] && STR_EQ(str, keyword_variable_modified[i])){
            is_key_mod = true;
            return is_key_mod;
        }
    }
    return is_key_mod;
}

static inline bool is_keyword_variable_datatype(const char* str) {
    bool is_key_type = false;
    int i;
    for(i = 0; i < PARSE_KEYWORD_VAR_DATATYPE_NUM; i++) {
        if(keyword_variable_datatype[i] && STR_EQ(str, keyword_variable_datatype[i])){
            is_key_type = true;
            return is_key_type;
        }
    }
    return is_key_type;
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
                .type = TOKEN_TYPE_NUMBER,
                .llnum = tok->llnum
            });
            break;
        case TOKEN_TYPE_IDENTIFIER:
            nd = pproc->create_node(pproc, &(Node){
                .type = TOKEN_TYPE_IDENTIFIER,
                .sval = tok->sval
            });
            break;
        default:
            parser_error("Problem converting token to node. No valid node exists for token of type %i\n", tok->type);
            break;
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

static inline bool parser_next_token_is_symbol(ParseProcess* pproc, char sym) {
    LOG_TAG
    Token* tok = pproc->peek_token(pproc);
    return tok && tok->type == TOKEN_TYPE_SYMBOL && tok->cval == sym;
}


// ==================================================================================== //
//                               parser: Make Node
// ==================================================================================== //

static inline void parser_make_expr_node(ParseProcess* pproc, Node* nd_l, Node* nd_r, const char* op) {
    pproc->create_node(pproc, &(Node){
        .type = NODE_TYPE_EXPRESSION,
        .expr.op = op,
        .expr.left = nd_l,
        .expr.right = nd_r
    });
}

static inline void parser_make_var_node(ParseProcess* pproc, DataType* dt, Token* name_nd, Node* val_nd) {
    LOG_TAG
    pproc->create_node(pproc, &(Node){
        .type = NODE_TYPE_VARIABLE,
        .var.type = dt,
        .var.name = name_nd->sval,
        .var.val = val_nd
    });
}

static inline void parser_make_func_node(ParseProcess* pproc, DataType* dt, const char* name, Vector* argv, Node* body) {
    LOG_TAG
    pproc->create_node(pproc, &(Node){
        .type = NODE_TYPE_FUNCTION,
        .func.rtype = dt,
        .func.name = name,
        .func.argv = argv,
        .func.body_nd = body
    });
}

static inline void parser_make_body_node(ParseProcess* pproc, Vector* body_vec) {
    pproc->create_node(pproc, &(Node){
        .type = NODE_TYPE_BODY,
        .body.statements = body_vec
    });
}

// ==================================================================================== //
//                               parser: AST Construct
// ==================================================================================== //


static inline void parser_expressionable(ParseProcess* pproc);
static inline void parser_for_parentheses(ParseProcess* pproc);
static inline void parser_stmt(ParseProcess* pproc);

static inline void parser_expr(ParseProcess* pproc) {
    Token* op_tok = pproc->next_token(pproc);
    const char *op = op_tok->sval;
    Node* nd_l = pproc->pop_node(pproc);
    parser_expressionable(pproc);
    Node* nd_r = pproc->pop_node(pproc);

    parser_make_expr_node(pproc, nd_l, nd_r, op);
}

static inline int parser_expressionable_single(ParseProcess* pproc) {
    Token* tok = pproc->peek_token(pproc);
    if(!tok) {
        return -1;
    }
    int res = -1;
    switch(tok->type) {
        case TOKEN_TYPE_NUMBER:
        case TOKEN_TYPE_IDENTIFIER:
            parser_single_token2node(pproc);
            res = 0;
            break;
        case TOKEN_TYPE_OPERATOR:
            parser_expr(pproc);
            res = 0;
            break;
        case TOKEN_TYPE_SYMBOL:
            if(tok->cval == '(') {
                parser_for_parentheses(pproc);
                res = 0;
            }
            break;
    }
    return res;
}


static inline void parser_expressionable(ParseProcess* pproc) {
    Node* last_node = NULL;
    while (parser_expressionable_single(pproc) == 0) {
        // We will loop until theirs nothing more of an expression
    }
}

static inline void parser_for_parentheses(ParseProcess* pproc) {
    parser_excp_symbol(pproc, '(');
    parser_expressionable(pproc);
    parser_excp_symbol(pproc, ')');
}


static inline void parser_for_symbol(ParseProcess* pproc) {
    Token* tok = pproc->peek_token(pproc);
    switch (tok->cval) {
        case '(':
            parser_for_parentheses(pproc);
            break;
        case ';':
            pproc->next_token(pproc);
            break;
        default:
            parser_error("Unexpected symbol: `%c`", tok->cval);
            break;
    }
}

// ==================================================================================== //
//                            parser: DataType Operations
// ==================================================================================== //

static inline void parser_datatype_modifiers(ParseProcess* pproc, DataType* dt) {
    memset(dt, 0, sizeof(DataType));
    Token* tok = pproc->peek_token(pproc);
    while(tok) {
        if(!is_keyword_variable_modified(tok->sval)) {
            break;
        }
        if(STR_EQ(tok->sval, "signed")) {
            dt->flags |= DATA_TYPE_FLAG_IS_SIGNED;
        }else if(STR_EQ(tok->sval, "static")) {
            dt->flags |= DATA_TYPE_FLAG_IS_STATIC;
        }else if(STR_EQ(tok->sval, "const")) {
            dt->flags |= DATA_TYPE_FLAG_IS_CONST;
        }
        pproc->next_token(pproc);
        tok = pproc->peek_token(pproc);
    }
    
}

// 解析数据类型的类型部分。即"int"、"long"
static inline void parser_datatype_type(ParseProcess* pproc, DataType* dt) {
    Token* dt_tok = pproc->next_token(pproc);
    int i;
    for(i = 0; i < PARSE_KEYWORD_VAR_DATATYPE_NUM; i++) {
        if(keyword_variable_datatype[i] && STR_EQ(dt_tok->sval, keyword_variable_datatype[i])){
            dt->type = (DataTypeEnum)i;
            switch(dt->type) {
                case DATA_TYPE_CHAR:    dt->size = 1; break;
                case DATA_TYPE_SHORT:   dt->size = 2; break;
                case DATA_TYPE_INT:     dt->size = 4; break;
                case DATA_TYPE_LONG:    dt->size = 8; break;
                case DATA_TYPE_FLOAT:   dt->size = 4; break;
                case DATA_TYPE_DOUBLE:  dt->size = 8; break;
                default: break;
            }
            break;
        }
    }
    dt->type_str = dt_tok->sval;
}

static inline void parser_variable(ParseProcess* pproc, DataType* dt, Token* name_tok) {
    LOG_TAG
    Node* val_nd = NULL;
    // 解析赋值变量
    if(parser_next_token_is_operator(pproc, "=")) {
        pproc->next_token(pproc);
        parser_expressionable(pproc);
        val_nd = pproc->pop_node(pproc);
    }
    parser_make_var_node(pproc, dt, name_tok, val_nd);
}

static inline void parser_variable_full(ParseProcess* pproc) {
    LOG_TAG
    DataType dt;
    parser_datatype_modifiers(pproc, &dt);
    parser_datatype_type(pproc, &dt);
    LOG_TAG
    Token* name_tok = pproc->next_token(pproc);
    parser_variable(pproc, &dt, name_tok);
}

static inline void parser_function_argument(ParseProcess* pproc) {
    parser_variable_full(pproc);
}

static inline Vector* parser_function_arguments(ParseProcess* pproc) {
    LOG_TAG
    Vector* argv = vector_create(sizeof(Node*));
    while(!parser_next_token_is_symbol(pproc, ')')) {
        parser_variable_full(pproc);
        Node* argument_node = pproc->pop_node(pproc);
        vector_push(argv, &argument_node);

        if(!parser_next_token_is_symbol(pproc, ',')) {
            break;
        }

        pproc->next_token(pproc);
    }
    return argv;
}

static inline void parser_body_single_stmt(ParseProcess* pproc, Vector* body_vec) {
    Node* stmt_nd = NULL;
    parser_stmt(pproc);
    stmt_nd = pproc->pop_node(pproc);
    vector_push(body_vec, &stmt_nd);
    parser_make_body_node(pproc, body_vec);
}

static inline void parser_body_multi_stmts(ParseProcess* pproc, Vector* body_vec) {
    LOG_TAG
    Node* stmt_nd = NULL;
    parser_excp_symbol(pproc, '{');
    while(!parser_next_token_is_symbol(pproc, '}')) {
        parser_stmt(pproc);
        stmt_nd = pproc->pop_node(pproc);
        vector_push(body_vec, &stmt_nd);
    }
    parser_excp_symbol(pproc, '}');
    parser_make_body_node(pproc, body_vec);
}

static inline void parser_body(ParseProcess* pproc) {
    LOG_TAG
    Vector* body_vec = vector_create(sizeof(Node*));
    if(!parser_next_token_is_symbol(pproc, '{')) {
        LOG_TAG
        parser_body_single_stmt(pproc, body_vec);
        return;
    }
    parser_body_multi_stmts(pproc, body_vec);
}

static inline void parser_function_body(ParseProcess* pproc) {
    parser_body(pproc);
}

static inline void parser_function(ParseProcess* pproc, DataType* dt, Token* name_tok) {
    LOG_TAG
    Vector* argv = NULL;
    parser_excp_symbol(pproc, '(');
    argv = parser_function_arguments(pproc);
    parser_excp_symbol(pproc, ')');
    LOG_TAG

    if(parser_next_token_is_symbol(pproc, '{')) {
        LOG_TAG
        parser_function_body(pproc);
        Node* body_nd = pproc->pop_node(pproc);
        parser_make_func_node(pproc, dt, name_tok->sval, argv, body_nd);
        LOG_TAG
        return;
    }

    LOG_TAG
    parser_excp_symbol(pproc, ';');
    parser_make_func_node(pproc, dt, name_tok->sval, argv, NULL);
}

static inline void parser_variable_or_function(ParseProcess* pproc) {
    LOG_TAG
    // 变量有数据类型，函数也有返回类型。
    // 让我们解析数据类型
    DataType dt;
    parser_datatype_modifiers(pproc, &dt);
    parser_datatype_type(pproc, &dt);
    Token* name_tok = pproc->next_token(pproc);
    if(parser_next_token_is_symbol(pproc, '(')) {
        LOG_TAG
        parser_function(pproc, &dt, name_tok);
        LOG_TAG
        return;
    }
    LOG_TAG
    parser_variable(pproc, &dt, name_tok);
    parser_excp_symbol(pproc, ';');
}

static inline void parser_keyword(ParseProcess* pproc) {
    LOG_TAG
    Token* tok = pproc->peek_token(pproc);
    if(is_keyword_variable_modified(tok->sval) || is_keyword_variable_datatype(tok->sval)) {
        parser_variable_or_function(pproc);
        LOG_TAG
        return;
    }
    parser_error("Unexpected keyword `%s`\n", tok->sval);
}


static inline void parser_stmt(ParseProcess* pproc) {
    LOG_TAG
    if(pproc->peek_token(pproc)->type == TOKEN_TYPE_KEYWORD) {
        parser_keyword(pproc);
        LOG_TAG
        return;
    }
    parser_expressionable(pproc);
    parser_excp_symbol(pproc, ';');
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

static inline void parser_swap_node(Node** nd1, Node** nd2) {
    Node* tmp_nd = *nd1;
    *nd1 = *nd2;
    *nd2 = tmp_nd;
}

Node* parse_process_peek_node(ParseProcess* pproc) {
    return vector_back_or_null(pproc->node_vec);
}

Node* parse_process_pop_node(ParseProcess* pproc) {
    Node* last_node = (Node*)vector_back_ptr_or_null(pproc->node_vec);
    Node* last_node_root = (Node*)vector_back_ptr_or_null(pproc->node_tree_vec);
    vector_pop(pproc->node_vec);
    if(last_node == last_node_root) {
        vector_pop(pproc->node_tree_vec);
    }
    return last_node;
}

void parse_process_push_node(ParseProcess* pproc, Node* node) {
    node_read(node);
    vector_push(pproc->node_vec, &node);
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
    ParseProcess* pproc = malloc(sizeof(ParseProcess));
    *pproc = (ParseProcess) {
        .lex_proc = lproc,
        .node_vec = vector_create(sizeof(Node)),
        .node_tree_vec = vector_create(sizeof(Node)),
        .next_token = parse_process_next_token,
        .peek_token = parse_process_peek_token,
        .excp_token = parse_process_excp_token,
        .peek_node  = parse_process_peek_node,
        .pop_node   = parse_process_pop_node,
        .push_node  = parse_process_push_node,
        .create_node = parse_process_create_node
    };

    return pproc;
}

void parse_process_free(ParseProcess* pproc) {
    LOG_TAG
    if(!pproc) {
        return;
    }
    vector_free(pproc->node_tree_vec);
    vector_free(pproc->node_vec);
    if(pproc->next_token) pproc->next_token = NULL;
    if(pproc->peek_token) pproc->peek_token = NULL;
    if(pproc->excp_token) pproc->excp_token = NULL;
    if(pproc->peek_node) pproc->peek_node = NULL;
    if(pproc->pop_node) pproc->pop_node = NULL;
    if(pproc->push_node) pproc->push_node = NULL;
    if(pproc->create_node) pproc->create_node = NULL;
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
        case TOKEN_TYPE_IDENTIFIER:
        case TOKEN_TYPE_OPERATOR:
            parser_expressionable(pproc);
            break;
        case TOKEN_TYPE_SYMBOL:
            parser_for_symbol(pproc);
            break;
        case TOKEN_TYPE_KEYWORD:
            parser_keyword(pproc);
            break;
        case TOKEN_TYPE_COMMENT:
        case TOKEN_TYPE_NEWLINE:
            pproc->next_token(pproc);
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
