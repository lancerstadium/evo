
// ==================================================================================== //
//                                     evoc: parser
// ==================================================================================== //

#include "evoc.h"

// ==================================================================================== //
//                                    Pri API: parser
// ==================================================================================== //

// parser语法分析：创建新节点
static Node* node_new(NodeType type) {
    Node *node = (Node*)calloc(1, sizeof(Node));
    node->type = type;
    return node;
}
// parser语法分析：创建二元运算节点
static Node* node_new_binary(NodeType type, Node *lhs, Node *rhs) {
    Node* node = node_new(type);
    node->lhs = lhs;
    node->rhs = rhs;
    return node;
}
// parser语法分析：创建一元运算节点
static Node* node_new_unary(NodeType type, Node *expr) {
    Node* node = node_new(type);
    node->lhs = expr;
    return node;
}
// parser语法分析：创建数字节点
static Node* node_new_num(int val) {
    Node* node = node_new(ND_NUM);
    node->val = val;
    return node;
}

static Node* stmt(Token **rest, Token *tok);
static Node* expr_stmt(Token **rest, Token *tok);
static Node* expr(Token **rest, Token *tok);
// static Node* assign(Token **rest, Token *tok);
static Node* equality(Token **rest, Token *tok);
static Node* relational(Token **rest, Token *tok);
static Node* add(Token **rest, Token *tok);
static Node* mul(Token **rest, Token *tok);
static Node* unary(Token **rest, Token *tok);
static Node* prim(Token **rest, Token *tok);

// stmt = expr-stmt
static Node *stmt(Token **rest, Token *tok) {
    return expr_stmt(rest, tok);
}
// expr-stmt = expr ";"
static Node *expr_stmt(Token **rest, Token *tok) {
    Node *node = node_new_unary(ND_EXPR_STMT, expr(&tok, tok));
    *rest = token_skip(tok, ";");
    return node;
}
// expr = equality
static Node *expr(Token **rest, Token *tok) {
    return equality(rest, tok);
}
// equality = relational ("==" relational | "!=" relational)*
static Node *equality(Token **rest, Token *tok) {
    Node *node = relational(&tok, tok);
    for (;;) {
        if(token_equal(tok, "==")) {
            node = node_new_binary(ND_EQU, node, relational(&tok, tok->next));
            continue;
        }
        if(token_equal(tok, "!=")) {
            node = node_new_binary(ND_NEQ, node, relational(&tok, tok->next));
            continue;
        }
        *rest = tok;
        return node;
    }
}
// relational = add ("<" add | "<=" add | ">" add | ">=" add)*
static Node *relational(Token **rest, Token *tok) {
    Node *node = add(&tok, tok);
    for (;;) {
        if(token_equal(tok, "<")) {
            node = node_new_binary(ND_LSS, node, add(&tok, tok->next));
            continue;
        }
        if(token_equal(tok, "<=")) {
            node = node_new_binary(ND_LEQ, node, add(&tok, tok->next));
            continue;
        }
        if(token_equal(tok, ">")) {
            node = node_new_binary(ND_GTR, node, add(&tok, tok->next));
            continue;
        }
        if(token_equal(tok, ">=")) {
            node = node_new_binary(ND_GEQ, node, add(&tok, tok->next));
            continue;
        }
        *rest = tok;
        return node;
    }
}
// add = mul ("+" mul | "-" mul)*
static Node *add(Token **rest, Token *tok) {
    Node *node = mul(&tok, tok);
    for (;;) {
        if(token_equal(tok, "+")) {
            node = node_new_binary(ND_ADD, node, mul(&tok, tok->next));
            continue;
        }
        if(token_equal(tok, "-")) {
            node = node_new_binary(ND_SUB, node, mul(&tok, tok->next));
            continue;
        }
        *rest = tok;
        return node;
    }
}
// mul = unary ("*" unary | "/" unary)*
static Node *mul(Token **rest, Token *tok) {
    Node *node = unary(&tok, tok);
    for (;;) {
        if(token_equal(tok, "*")) {
            node = node_new_binary(ND_MUL, node, unary(&tok, tok->next));
            continue;
        }
        if(token_equal(tok, "/")) {
            node = node_new_binary(ND_DIV, node, unary(&tok, tok->next));
            continue;
        }
        *rest = tok;
        return node;
    }
}
// unary = ("+" | "-")? unary | prim
static Node *unary(Token **rest, Token *tok) {
    if(token_equal(tok, "+")) {             // + unary
        return unary(rest, tok->next);
    }
    if(token_equal(tok, "-")) {             // - unary
        return node_new_unary(ND_NEG, unary(rest, tok->next));
    }
    *rest = tok;
    return prim(rest, tok);                 // prim
}
// prim = "(" expr ")" | num
static Node *prim(Token **rest, Token *tok) {
    if(token_equal(tok, "(")) {              // "(" expr ")"
        Node *node = expr(&tok, tok->next);
        *rest = token_skip(tok, ")");
        return node;
    }
    if(tok->type == TK_NUM) {               // num
        Node *node = node_new_num(tok->val);
        *rest = tok->next;
        return node;
    }
    evoc_err_tok(tok, "expected expression");
    return NULL;
}

// ==================================================================================== //
//                                    Pub API: parser
// ==================================================================================== //


// parser语法分析：解析
Node* evoc_parse(Token *tok) {
    Node head = {};
    Node *cur = &head;
    while(tok->type != TK_EOF) {
        cur = cur->next = stmt(&tok, tok);
    }
    return head.next;
}


