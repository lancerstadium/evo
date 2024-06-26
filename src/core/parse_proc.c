
#include "parser.h"


// ==================================================================================== //
//                               parser Pri Data
// ==================================================================================== //

// #define PARSE_MAX_OPS_IN_GROUP 12

// typedef enum {
//     PARSE_OPERATOR_ASSOC_LEFT2RIGHT,
//     PARSE_OPERATOR_ASSOC_RIGHT2LEFT
// } OperatorAssociativity;

// typedef struct {
//     char* ops[PARSE_MAX_OPS_IN_GROUP];
//     OperatorAssociativity asc;
// } OpGroup;

// OpGroup op_level[] = {
//     {.ops = {"++", "--", "()", "(", "[", "]", ".", "->", NULL},                             .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
//     {.ops = {"*", "/", "%%", NULL},                                                         .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
//     {.ops = {"+", "-", NULL},                                                               .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
//     {.ops = {"<<", ">>", NULL},                                                             .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
//     {.ops = {"<", "<=", ">", ">=", NULL},                                                   .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
//     {.ops = {"==", "!=", NULL},                                                             .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
//     {.ops = {"&", NULL},                                                                    .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
//     {.ops = {"^", NULL},                                                                    .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
//     {.ops = {"|", NULL},                                                                    .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
//     {.ops = {"&&", NULL},                                                                   .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
//     {.ops = {"||", NULL},                                                                   .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
//     {.ops = {"?", ":", NULL},                                                               .asc = PARSE_OPERATOR_ASSOC_RIGHT2LEFT},
//     {.ops = {"=", "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "^=", "|=", NULL},      .asc = PARSE_OPERATOR_ASSOC_RIGHT2LEFT},
//     {.ops = {",", NULL},                                                                    .asc = PARSE_OPERATOR_ASSOC_LEFT2RIGHT},
// };

// #define PARSE_OP_LEVEL_NUM  GET_ARR_LEN(op_level)

// static inline int parser_get_op_level(const char* op, OpGroup** group_out) {
//     *group_out = NULL;
//     for(int i = 0; i < PARSE_OP_LEVEL_NUM; i++) {
//         for(int b = 0; op_level[i].ops[b]; b++) {
//             const char *_op = op_level[i].ops[b];
//             if(STR_EQ(op, _op)) {
//                 *group_out = &op_level[i];
//                 return i;
//             }
//         }
//     }
//     return -1;
// }

// static inline bool parser_left_op_has_priority(const char* op_left, const char* op_right) {
//     OpGroup* group_left = NULL;
//     OpGroup* group_right = NULL;

//     // 一样的运算符？
//     if(STR_EQ(op_left, op_right)){
//         return false;
//     }

//     int level_left = parser_get_op_level(op_left, &group_left);
//     int level_right = parser_get_op_level(op_right, &group_right);
//     if(group_left->asc == PARSE_OPERATOR_ASSOC_RIGHT2LEFT) {
//         return false;
//     }
//     return level_left <= level_right;
// }

// static inline bool is_unary_operator(const char *op) {
//     return STR_EQ(op, "-") || STR_EQ(op, "!") || STR_EQ(op, "~");
// }

// ==================================================================================== //
//                            parser: Token -> Node
// ==================================================================================== //

static inline void parser_excp_operator(ParseProcess* pproc, char* op) {
    Token* next_token = pproc->next_token(pproc);
    if(next_token == NULL) {
        parser_error("Expecting the operator `%s` but `None` was provided in %s:%d:%d", 
        op, next_token->pos.filename, next_token->pos.line, next_token->pos.col);
    }else if(next_token->type != TOKEN_TYPE_OPERATOR) {
        parser_error("Expecting the operator `%s` but type `%s` was provided in %s:%d:%d", 
        op, token_get_type_str(next_token), next_token->pos.filename, next_token->pos.line, next_token->pos.col);
    }else if(!STR_EQ(next_token->sval, op)) {
        parser_error("Expecting the operator `%s` but `%s` was provided in %s:%d:%d", 
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
    if(next_token == NULL) {
        parser_error("Expecting the symbol `%c` but `%c` was provided in %s:%d:%d", 
        c, next_token->cval, next_token->pos.filename, next_token->pos.line, next_token->pos.col);
    }else if(next_token->type != TOKEN_TYPE_SYMBOL) {
        parser_error("Expecting the symbol `%c` but type `%s` was provided in %s:%d:%d", 
        c, token_get_type_str(next_token), next_token->pos.filename, next_token->pos.line, next_token->pos.col);
    }else if(next_token->cval != c) {
        parser_error("Expecting the symbol `%c` but `%c` was provided in %s:%d:%d", 
        c, next_token->cval, next_token->pos.filename, next_token->pos.line, next_token->pos.col);
    }
}

static inline void parser_excp_newline(ParseProcess* pproc) {
    Token* next_token = pproc->next_token(pproc);
    if(next_token == NULL) {
        parser_error("Expecting the newline `\\n` but `None` was provided in %s:%d:%d", 
        next_token->pos.filename, next_token->pos.line, next_token->pos.col);
    }else if(next_token->type != TOKEN_TYPE_NEWLINE) {
        parser_error("Expecting the newline `\\n` but type `%s` was provided in %s:%d:%d", 
        token_get_type_str(next_token), next_token->pos.filename, next_token->pos.line, next_token->pos.col);
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

static inline bool parser_next_token_is_newline(ParseProcess* pproc) {
    Token* tok = pproc->peek_token(pproc);
    return tok && tok->type == TOKEN_TYPE_NEWLINE;
}

static inline bool parser_next_token_is_EOF(ParseProcess* pproc) {
    Token* tok = pproc->peek_token(pproc);
    return !tok || tok->type == TOKEN_TYPE_EOF;
}

// ==================================================================================== //
//                             parser: AST Consturct
// ==================================================================================== //

static inline Node* parser_make_stmt_node(ParseProcess* pproc);
static inline Node* parser_make_expr_node(ParseProcess* pproc);
static inline Node* parser_make_func_node(ParseProcess* pproc);
static inline Node* parser_make_enum_node(ParseProcess* pproc);
static inline Node* parser_make_struct_node(ParseProcess* pproc);

// num   <- number
// ident <- identifier (":" datatype)*
static inline Node* parser_single_token2node(ParseProcess* pproc) {
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
            const char* name = tok->sval;
            Pos* pos = &tok->pos;
            nd = pproc->create_node(pproc, &(Node){
                .type = NODE_TYPE_IDENT,
                .sval = name,
                .ident.pos = pos
            });
            // datatype ident
            int id_type = DATA_TYPE_I32;
            const char* id_type_str = datatype_str[DATA_TYPE_I32];
            
            if(parser_next_token_is_operator(pproc, ":")) {
                pproc->next_token(pproc);
                tok = pproc->peek_token(pproc);
                if(tok->type == TOKEN_TYPE_DATATYPE){
                    id_type = tok->inum;
                    id_type_str = datatype_str[tok->inum];
                    pproc->next_token(pproc);
                }
            }
            nd->ident.dtype.type = id_type;
            nd->ident.dtype.type_str = id_type_str;
            break;
        case TOKEN_TYPE_EOF:
            nd = pproc->create_node(pproc, &(Node){
                .type = NODE_TYPE_EOF,
            });
            break;
        default:
            parser_error("Problem converting token to node. No valid token type `%s` at %s:%d:%d", token_get_type_str(tok), tok->pos.filename, tok->pos.line, tok->pos.col);
            break;
    }
    return nd;
}

// prim = "(" expr ")" 
//      | num
//      | ident
static inline Node* parser_make_prim_node(ParseProcess* pproc) {
    Node* node;
    Token* tok = pproc->peek_token(pproc);
    // ( expr )
    if(parser_next_token_is_symbol(pproc, '(')) {
        tok = pproc->next_token(pproc);
        node = parser_make_expr_node(pproc);
        parser_excp_symbol(pproc, ')');
        return node;
    } else if(tok->type == TOKEN_TYPE_NUMBER) {
        node = parser_single_token2node(pproc);
        return node;
    } else if(tok->type == TOKEN_TYPE_IDENTIFIER) {
        node = parser_single_token2node(pproc);
        return node;
    }

    parser_error("Unexpect expression at %s:%d:%d", tok->pos.filename, tok->pos.line, tok->pos.col);
    return NULL;
}

// unary = ("+" | "-" | "*" | "&" )? unary 
//       | prim
static inline Node* parser_make_unary_node(ParseProcess* pproc) {
    Token* tok;
    if(parser_next_token_is_operator(pproc, "+")){
        tok = pproc->next_token(pproc);
        parser_make_unary_node(pproc);
    }else if( 
       parser_next_token_is_operator(pproc, "-")  || 
       parser_next_token_is_operator(pproc, "*")  || 
       parser_next_token_is_operator(pproc, "&")) {
        tok = pproc->next_token(pproc);
        return pproc->create_node(pproc, &(Node){
            .type = NODE_TYPE_EXPR,
            .expr.lnd = parser_make_unary_node(pproc),
            .expr.rnd = NULL,
            .expr.op  = tok ? tok->sval : ""
        });
    } else {
        return parser_make_prim_node(pproc);
    }
}

// mul = unary ("*" unary | "/" unary)*
static inline Node* parser_make_mul_node(ParseProcess* pproc) {
    Node* node = parser_make_unary_node(pproc);
    Token* tok;
    for(;;) {
        if(parser_next_token_is_operator(pproc, "*")  || 
           parser_next_token_is_operator(pproc, "/")) {
            tok = pproc->next_token(pproc);
            node = pproc->create_node(pproc, &(Node){
                .type = NODE_TYPE_EXPR,
                .expr.lnd = node,
                .expr.rnd = parser_make_unary_node(pproc),
                .expr.op  = tok ? tok->sval : ""
            });
            continue;
        }
        return node;
    }
}

// add = mul ("+" mul | "-" mul)*
static inline Node* parser_make_add_node(ParseProcess* pproc) {
    Node* node = parser_make_mul_node(pproc);
    Token* tok;
    for(;;) {
        if(parser_next_token_is_operator(pproc, "+")  || 
           parser_next_token_is_operator(pproc, "-")) {
            tok = pproc->next_token(pproc);
            node = pproc->create_node(pproc, &(Node){
                .type = NODE_TYPE_EXPR,
                .expr.lnd = node,
                .expr.rnd = parser_make_mul_node(pproc),
                .expr.op  = tok ? tok->sval : ""
            });
            continue;
        }
        return node;
    }
}

// relat = add ("<" add | "<=" add | ">" add | ">=" add)*
static inline Node* parser_make_relat_node(ParseProcess* pproc) {
    Node* node = parser_make_add_node(pproc);
    Token* tok;
    for(;;) {
        if(parser_next_token_is_operator(pproc, "<")  || 
           parser_next_token_is_operator(pproc, "<=") ||
           parser_next_token_is_operator(pproc, ">")  ||
           parser_next_token_is_operator(pproc, ">=")) {
            tok = pproc->next_token(pproc);
            node = pproc->create_node(pproc, &(Node){
                .type = NODE_TYPE_EXPR,
                .expr.lnd = node,
                .expr.rnd = parser_make_add_node(pproc),
                .expr.op  = tok ? tok->sval : ""
            });
            continue;
        }
        return node;
    }
}

// equal = relat ("==" relat | "!=" relat)*
static inline Node* parser_make_equal_node(ParseProcess* pproc) {
    Node* node = parser_make_relat_node(pproc);
    Token* tok;
    for(;;) {
        if(parser_next_token_is_operator(pproc, "==") || parser_next_token_is_operator(pproc, "!=")) {
            tok = pproc->next_token(pproc);
            node = pproc->create_node(pproc, &(Node){
                .type = NODE_TYPE_EXPR,
                .expr.lnd = node,
                .expr.rnd = parser_make_relat_node(pproc),
                .expr.op  = tok ? tok->sval : ""
            });
            continue;
        }
        return node;
    }
}

// assgin = equal ("=" assign | ":=" assign | "~=" assign)?
static inline Node* parser_make_assign_node(ParseProcess* pproc) {

    Node* node = parser_make_equal_node(pproc);
    Token* tok;
    if(parser_next_token_is_operator(pproc, "=") || parser_next_token_is_operator(pproc, ":=") || parser_next_token_is_operator(pproc, "~=")) {
        tok = pproc->next_token(pproc);
        node = pproc->create_node(pproc, &(Node){
            .type = NODE_TYPE_EXPR,
            .expr.lnd = node,
            .expr.rnd = parser_make_assign_node(pproc),
            .expr.op  = tok ? tok->sval : ""
        });
    }
    return node;
}

// expr = assign
static inline Node* parser_make_expr_node(ParseProcess* pproc) {

    return parser_make_assign_node(pproc);
}

// expr-stmt = expr? newline
static inline Node* parser_make_expr_stmt_node(ParseProcess* pproc) {
    if(parser_next_token_is_newline(pproc)) {
        pproc->next_token(pproc);
        return pproc->create_node(pproc, &(Node){
            .type = NODE_TYPE_STMT,
        });
    }

    Node* node = pproc->create_node(pproc, &(Node){
        .type = NODE_TYPE_STMT,
        .sval = "expr-stmt",
        .stmt.lnd = NULL,
        .stmt.rnd = NULL
    });

    node->stmt.lnd = parser_make_expr_node(pproc);

    // parser_excp_newline(pproc);

    return node;
}

// body-stmt = stmt* "}"
static inline Node* parser_make_body_stmt_node(ParseProcess* pproc) {
    Node* node = pproc->create_node(pproc, &(Node){
        .type = NODE_TYPE_BODY,
        .body.stmt = NULL
    });
    while(!parser_next_token_is_symbol(pproc, '}')) {
        node->body.stmt = parser_make_stmt_node(pproc);
    }

    parser_excp_symbol(pproc, '}');
    pproc->tmp_nd = pproc->tmp_nd->pnd;
    return node;
}

// stmt = struct
//      | enum
//      | func
//      | "return" (expr | newline)
//      | "if" expr "{" body-stmt ("else" "{" body-stmt)? 
//      | "for" expr* "{" body-stmt 
//      | "{" body-stmt
//      | expr-stmt
static inline Node* parser_make_stmt_node(ParseProcess* pproc) {
    Node* node = NULL;
    if(parser_next_token_is_keyword(pproc, "struct")){
        node = parser_make_struct_node(pproc);
    }else if(parser_next_token_is_keyword(pproc, "enum")){
        node = parser_make_enum_node(pproc);
    }else if(parser_next_token_is_keyword(pproc, "fn")){
        node = parser_make_func_node(pproc);
    }else if(parser_next_token_is_keyword(pproc, "return")) {
        pproc->next_token(pproc);
        node = pproc->create_node(pproc, &(Node){
            .type = NODE_TYPE_STMT,
            .sval = "return-stmt",
            .stmt.ret = NULL
        });
        if(!parser_next_token_is_newline(pproc)) {
            node->stmt.ret = parser_make_expr_node(pproc);
        }
    }else if(parser_next_token_is_keyword(pproc, "if")) {
        pproc->next_token(pproc);
        node = pproc->create_node(pproc, &(Node){
            .type = NODE_TYPE_STMT,
            .sval = "if-stmt",
            .stmt.cond = NULL,
            .stmt.then = NULL,
            .stmt.els  = NULL
        });
        node->stmt.cond = parser_make_expr_node(pproc);
        parser_excp_symbol(pproc, '{');
        node->stmt.then = parser_make_body_stmt_node(pproc);
        if(parser_next_token_is_keyword(pproc, "else")) {
            pproc->next_token(pproc);
            parser_excp_symbol(pproc, '{');
            node->stmt.els = parser_make_body_stmt_node(pproc);
        }

    }else if(parser_next_token_is_keyword(pproc, "for")){
        pproc->next_token(pproc);
        node = pproc->create_node(pproc, &(Node){
            .type = NODE_TYPE_STMT,
            .sval = "for-stmt",
            .stmt.cond = NULL,
            .stmt.then = NULL,
        });
        if(!parser_next_token_is_symbol(pproc, '{')) {
            node->stmt.cond = parser_make_expr_node(pproc);
        }
        parser_excp_symbol(pproc, '{');
        node->stmt.then = parser_make_body_stmt_node(pproc);

    }else if(parser_next_token_is_symbol(pproc, '{')) {
        pproc->next_token(pproc);
        node = parser_make_body_stmt_node(pproc);
    }else {
        node = parser_make_expr_stmt_node(pproc);
    }
    return node;
}

// param = ident
static inline Vector* parser_make_param_vector(ParseProcess* pproc) {
    Token* tok = pproc->peek_token(pproc);
    Vector* param_vec = vector_create(sizeof(Node));
    Node* id_nd = NULL;
    while(tok->type == TOKEN_TYPE_IDENTIFIER || parser_next_token_is_symbol(pproc, ',')) {
        if(param_vec->count > 0) {
            parser_excp_symbol(pproc, ',');
        }
        id_nd = parser_single_token2node(pproc);
        vector_push(param_vec, id_nd);
        tok = pproc->peek_token(pproc);
    }
    return param_vec;
}

// func = "fn" (ident ".")* ident "(" param ")" ":" Datatype body*
static inline Node* parser_make_func_node(ParseProcess* pproc) {
    pproc->next_token(pproc);
    // ident*
    Token* tok = pproc->peek_token(pproc);
    const char* func_name = NULL;
    if(tok->type == TOKEN_TYPE_IDENTIFIER) {
        func_name = tok->sval;
        pproc->next_token(pproc);
        log_info("func name: %s", func_name);
    }
    Node* node = pproc->create_node(pproc, &(Node){
        .type = NODE_TYPE_FUNC,
        .func.name = func_name,
        .func.param_vec = NULL,
        .func.fn_rtype = (DataType) {
            .type = DATA_TYPE_I32,
            .type_str = datatype_str[DATA_TYPE_I32]
        },
        .func.fn_body = NULL
    });

    // "(" param ")"
    parser_excp_symbol(pproc, '(');
    Vector* param_vec = NULL;
    if(!parser_next_token_is_symbol(pproc, ')')) {
        param_vec = parser_make_param_vector(pproc);
    }
    parser_excp_symbol(pproc, ')');
    node->func.param_vec = param_vec;
    
    // ":" Datatype
    if(parser_next_token_is_operator(pproc, ":")) {
        pproc->next_token(pproc);
        tok = pproc->peek_token(pproc);
        if(tok->type == TOKEN_TYPE_DATATYPE) {
            node->func.fn_rtype.type = tok->inum;
            node->func.fn_rtype.type_str = datatype_str[tok->inum];
            pproc->next_token(pproc);
        }
    }
    // body*
    if(parser_next_token_is_symbol(pproc, '{')) {
        pproc->next_token(pproc);
        node->func.fn_body = parser_make_body_stmt_node(pproc);
    }
    return node;
}

// body-enum = ident* "}"
static inline Node* parser_make_body_enum_node(ParseProcess* pproc) {
    Node* id_nd = NULL;
    Token* tok = pproc->peek_token(pproc);
    Node* node = pproc->create_node(pproc, &(Node){
        .type = NODE_TYPE_BODY,
        .body.stmt = NULL,
        .body.sym_tbl = hashmap_create()
    });
    while(!parser_next_token_is_symbol(pproc, '}')) {
        if(tok->type == TOKEN_TYPE_IDENTIFIER) {
            id_nd = parser_single_token2node(pproc);
            hashmap_set(node->body.sym_tbl, id_nd->sval, id_nd);
        }else {
            pproc->next_token(pproc);
        }
    }
    parser_excp_symbol(pproc, '}');
    pproc->tmp_nd = pproc->tmp_nd->pnd;
    return node;
}

// enum = "enum" ident* "{" ident* "}"
static inline Node* parser_make_enum_node(ParseProcess* pproc) {
    pproc->next_token(pproc);
    // ident*
    Token* tok = pproc->peek_token(pproc);
    const char* enum_name = NULL;
    if(tok->type == TOKEN_TYPE_IDENTIFIER) {
        enum_name = tok->sval;
        pproc->next_token(pproc);
        log_info("enum name: %s", enum_name);
    }

    Node* node = pproc->create_node(pproc, &(Node){
        .type = NODE_TYPE_ENUM,
        .enm.name = enum_name ? enum_name : "(None)",
        .enm.enm_body = NULL
    });

    // "{" enum-body
    parser_excp_symbol(pproc, '{');
    node->enm.enm_body = parser_make_body_enum_node(pproc);
    return node;
}


// body-struct = (param | func)* "}"
static inline Node* parser_make_body_struct_node(ParseProcess* pproc) {
    Token* tok = NULL;
    Node* node = pproc->create_node(pproc, &(Node){
        .type = NODE_TYPE_BODY,
        .body.stmt = NULL,
        .body.sym_tbl = hashmap_create()
    });
    while(!parser_next_token_is_symbol(pproc, '}')) {
        tok = pproc->peek_token(pproc);
        if(parser_next_token_is_keyword(pproc, "fn")) {
            node->body.stmt = parser_make_func_node(pproc);
        }else if(tok->type == TOKEN_TYPE_IDENTIFIER) {
            Node* id_nd = NULL;
            while(tok->type == TOKEN_TYPE_IDENTIFIER || parser_next_token_is_symbol(pproc, ',')) {
                if(parser_next_token_is_symbol(pproc, ',')){
                    parser_excp_symbol(pproc, ',');
                }
                id_nd = parser_single_token2node(pproc);
                hashmap_set(node->body.sym_tbl, id_nd->sval, id_nd);
                tok = pproc->peek_token(pproc);
            }
        }else {
            pproc->next_token(pproc);
        }
    }
    parser_excp_symbol(pproc, '}');
    pproc->tmp_nd = pproc->tmp_nd->pnd;
    return node;
}

// struct = "struct" ident* "{" body-struct
static inline Node* parser_make_struct_node(ParseProcess* pproc) {
    pproc->next_token(pproc);
    // ident*
    Token* tok = pproc->peek_token(pproc);
    const char* stc_name = NULL;
    if(tok->type == TOKEN_TYPE_IDENTIFIER) {
        stc_name = tok->sval;
        pproc->next_token(pproc);
        log_info("struct name: %s", stc_name);
    }

    Node* node = pproc->create_node(pproc, &(Node){
        .type = NODE_TYPE_STRUCT,
        .stc.name = stc_name ? stc_name : "(None)",
        .stc.stc_body = NULL
    });

    // "{" body-struct
    parser_excp_symbol(pproc, '{');
    node->stc.stc_body = parser_make_body_struct_node(pproc);
    return node;
}


// mod = ("mod" ident* "(" param ")" )*  ("use" "(" param ")")* stmt*
static inline Node* parser_make_mod_node(ParseProcess* pproc) {
    LOG_TAG
    Token* tok = pproc->peek_token(pproc);
    const char* kw = NULL;
    if(tok->type == TOKEN_TYPE_KEYWORD) {
        kw = tok->sval;
        if(STR_EQ(kw, "mod")) {
            pproc->next_token(pproc);
            // ident*
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
        }
    }
    pproc->root->prog.main_mod->mod.mod_body = pproc->create_node(pproc, &(Node){
        .type = NODE_TYPE_BODY,
        .body.stmt = NULL
    });
    while(!parser_next_token_is_EOF(pproc)) {
        pproc->root->prog.main_mod->mod.mod_body->body.stmt = parser_make_stmt_node(pproc);
    }

    pproc->tmp_nd = pproc->tmp_nd->pnd;
    return pproc->root->prog.main_mod;
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
    Node* back_nd = vector_back(pproc->node_vec);
    pproc->root = vector_at(pproc->node_vec, 0);

    // 更新 tmp_nd
    // | back_nd |  tmp_nd |
    // |  PROG   |   yes   |
    // |  MOD    |   yes   |
    // |  ENUM   |   yes   |
    // |  STRUCT |   yes   |
    // |  FUNC   |   yes   |
    if(!pproc->tmp_nd) 
        pproc->tmp_nd = pproc->root;

    switch (back_nd->type) {
        case NODE_TYPE_PROG: break;
        case NODE_TYPE_MOD:
        case NODE_TYPE_BODY:
            back_nd->pnd = pproc->tmp_nd;
            back_nd->depth = pproc->tmp_nd->depth + 1;
            pproc->tmp_nd = back_nd;
            break;
        case NODE_TYPE_STRUCT:
        case NODE_TYPE_ENUM:
        case NODE_TYPE_FUNC:
        case NODE_TYPE_NUM:
        case NODE_TYPE_IDENT:
        case NODE_TYPE_EXPR:
        default:
            back_nd->pnd = pproc->tmp_nd;
            back_nd->depth = pproc->tmp_nd->depth;
            break;
    }

    return back_nd;
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
        .config = &(ParseConfig) {
            .default_datatype = DATA_TYPE_I32,
            .default_datatype_str = datatype_str[DATA_TYPE_I32]
        },
        .symbol_tbl  = hashmap_create(),
        .next_token  = parse_process_next_token,
        .peek_token  = parse_process_peek_token,
        .excp_token  = parse_process_excp_token,
        .peek_node   = parse_process_peek_node,
        .pop_node    = parse_process_pop_node,
        .push_node   = parse_process_push_node,
        .create_node = parse_process_create_node
    };

    
    pproc->root = pproc->create_node(pproc, &(Node){
        .type = NODE_TYPE_PROG,
        .depth = 0,
        .pnd = NULL,
        .prog.name = fio_get_bare_filename(pproc->lex_proc->compile_proc->cfile),
    });

    pproc->root->prog.main_mod = pproc->create_node(pproc, &(Node){
        .type = NODE_TYPE_MOD,
        .mod.name = pproc->root->prog.name,
        .mod.sym_tbl = hashmap_create()
    });

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
    Token* tok = pproc->peek_token(pproc);
    if(!tok) {
        return -1;
    }
    int res = 0;
    switch(tok->type) {
        case TOKEN_TYPE_NUMBER:
        case TOKEN_TYPE_IDENTIFIER:;
        case TOKEN_TYPE_OPERATOR:
        case TOKEN_TYPE_STRING:
        case TOKEN_TYPE_SYMBOL:
        case TOKEN_TYPE_DATATYPE:
        case TOKEN_TYPE_KEYWORD:
            parser_make_mod_node(pproc);
            break;
        case TOKEN_TYPE_PRE_KEYWORD:
            pproc->next_token(pproc);
            break;
        case TOKEN_TYPE_COMMENT:
            pproc->next_token(pproc);
            break;
        case TOKEN_TYPE_NEWLINE:
            pproc->next_token(pproc);
            break;
        case TOKEN_TYPE_EOF:
            parser_single_token2node(pproc);
            res = -1;
            break;
        default:
            parser_error("Unexpected token type: %s at %s:%d:%d", token_get_type_str(tok), tok->pos.filename, tok->pos.line, tok->pos.col);
            // token_read(tok);
            break;
    }
    return res;
}
