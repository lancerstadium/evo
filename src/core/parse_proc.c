
#include "parser.h"


// ==================================================================================== //
//                               parser Pri Data
// ==================================================================================== //

static const char* op_precedence[] = {
    "*", "/", "%%", "+", "-"
};


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
        parser_error("Expecting the symbol %c but something else was provided", c);
    }
}


// ==================================================================================== //
//                               parser: Make Node
// ==================================================================================== //


static inline void parser_expressionable(ParseProcess* pproc);
static inline void parser_for_parentheses(ParseProcess* pproc);

static inline Node* parser_make_expr_node(ParseProcess* pproc, Node* nd_l, Node* nd_r, const char* op) {
    return pproc->create_node(pproc, &(Node){
        .type = NODE_TYPE_EXPRESSION,
        .expr.op = op,
        .expr.left = nd_l,
        .expr.right = nd_r
    });
}

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
        default:
            parser_error("Unexpected symbol: `%c`", tok->cval);
            break;
    }
}






// ==================================================================================== //
//                            parser: Token Operations
// ==================================================================================== //

Token* parse_process_next_token(ParseProcess* pproc) {
    return vector_peek(pproc->lex_proc->token_vec);
};

Token* parse_process_peek_token(ParseProcess* pproc) {
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
    return *((Node**)(vector_back(pproc->node_vec)));
}

Node* parse_process_pop_node(ParseProcess* pproc) {
    Node* last_node = *((Node**)(vector_back(pproc->node_vec)));
    Node* last_node_root = *((Node**)(vector_back(pproc->node_tree_vec)));

    vector_pop(pproc->node_vec);
    if(last_node == last_node_root) {
        vector_pop(pproc->node_tree_vec);
    }
    return last_node;
}

void parse_process_push_node(ParseProcess* pproc, Node* node) {
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
        case TOKEN_TYPE_IDENTIFIER:
        case TOKEN_TYPE_OPERATOR:
            parser_expressionable(pproc);
            break;
        case TOKEN_TYPE_SYMBOL:
            parser_for_symbol(pproc);
            break;
        default:
            parser_error("Unexpected token type: %s", token_get_type_str(tok));
            // token_read(tok);
            break;
    }
    return res;
}
