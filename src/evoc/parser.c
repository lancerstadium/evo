
// ==================================================================================== //
//                                     evoc: parser
// ==================================================================================== //

#include "evoc.h"

// ==================================================================================== //
//                                    Pri Data: parser Var
// ==================================================================================== //

// 本地变量链表
Var* local_vars;
// var变量：新建变量
static Var* var_new(char* name) {
    Var* var = (Var*)calloc(1, sizeof(Var));
    var->name = name;
    var->next = local_vars;
    local_vars = var;
    return var;
}
// var变量：查找变量
static Var* var_find(Token *tok) {
    for(Var *var = local_vars; var; var = var->next) {
        // 如果长度相等，且名字相等，则返回变量指针，否则返回NULL
        if(strlen(var->name) == tok->len && !strncmp(var->name, tok->loc, tok->len)) {
            return var;
        }
    }
    return NULL;
}

// ==================================================================================== //
//                                    Pri API: parser Node
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
// parser语法分析：创建变量节点
static Node* node_new_var(Var* var) {
    Node* node = node_new(ND_VAR);
    node->var = var;
    return node;
}


// ==================================================================================== //
//                                    Pri API: parser AST
// ==================================================================================== //

static Node* stmt(Token **rest, Token *tok);
static Node* compound_stmt(Token **rest, Token *tok);
static Node* expr_stmt(Token **rest, Token *tok);
static Node* expr(Token **rest, Token *tok);
static Node* assign(Token **rest, Token *tok);
static Node* equality(Token **rest, Token *tok);
static Node* relational(Token **rest, Token *tok);
static Node* add(Token **rest, Token *tok);
static Node* mul(Token **rest, Token *tok);
static Node* unary(Token **rest, Token *tok);
static Node* prim(Token **rest, Token *tok);

// stmt = "return" expr ";"
//      | "if" "(" expr ")" stmt ("else" stmt)? 
//      | "for" "(" expr-stmt? ";" expr? ")" stmt
//      | "while" "(" expr ")" stmt
//      | "{" compound-stmt 
//      | expr-stmt
static Node *stmt(Token **rest, Token *tok) {
    if(token_equal(tok, "return")) {
        Node *node = node_new_unary(ND_RETURN, expr(&tok, tok->next));
        *rest = token_skip(tok, ";");
        return node;
    }
    if(token_equal(tok, "if")) {
        Node *node = node_new(ND_IF);
        *rest = token_skip(tok, "(");
        node->cond = expr(&tok, tok);
        *rest = token_skip(tok, ")");
        node->then = stmt(&tok, tok);
        if(token_equal(tok, "else")) {
            node->els = stmt(&tok, tok->next);
        }
        *rest = tok;
        return node;
    }
    if(token_equal(tok, "for")) {
        Node* node = node_new(ND_LOOP);
        tok = token_skip(tok->next, "(");
        node->init = expr_stmt(&tok, tok);
        if(!token_equal(tok, ";")) {
            node->cond = expr(&tok, tok);
        }
        tok = token_skip(tok, ";");
        
        if(!token_equal(tok, ")")) {
            node->inc = expr(&tok, tok);
        }
        tok = token_skip(tok, ")");

        node->then = stmt(rest, tok);
        return node;
    }
    if(token_equal(tok, "while")) {
        Node* node = node_new(ND_LOOP);
        tok = token_skip(tok->next, "(");
        node->cond = expr(&tok, tok);
        tok = token_skip(tok, ")");
        node->then = stmt(rest, tok);
        return node;
    }
    if(token_equal(tok, "{")) {
        return compound_stmt(rest, tok->next);
    }
    return expr_stmt(rest, tok);
}
// compound-stmt = stmt* "}"
static Node *compound_stmt(Token **rest, Token *tok) {
    Node head = {};
    Node *cur = &head;
    while(!token_equal(tok, "}")) {
        cur = cur->next = stmt(&tok, tok);
    }
    Node *node = node_new(ND_BLOCK);
    node->body = head.next;
    *rest = tok->next;
    return node;
}
// expr-stmt = expr? ";"
static Node *expr_stmt(Token **rest, Token *tok) {
    if (token_equal(tok, ";")) {
        *rest = tok->next;
        return node_new(ND_BLOCK);
    }
    Node *node = node_new_unary(ND_EXPR_STMT, expr(&tok, tok));
    *rest = token_skip(tok, ";");
    return node;
}
// expr = assign
static Node *expr(Token **rest, Token *tok) {
    return assign(rest, tok);
}
// assign = equality ("=" assign)?
static Node *assign(Token **rest, Token *tok) {
    Node *node = equality(&tok, tok);
    if(token_equal(tok, "=")) {
        node = node_new_binary(ND_ASSIGN, node, assign(&tok, tok->next));
    }
    *rest = tok;
    return node;
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
// unary = ("+" | "-")? unary 
//       | prim
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
// prim = "(" expr ")" 
//      | ident 
//      | num
static Node *prim(Token **rest, Token *tok) {
    if(token_equal(tok, "(")) {              // "(" expr ")"
        Node *node = expr(&tok, tok->next);
        *rest = token_skip(tok, ")");
        return node;
    }
    if(tok->type == TK_IDENT) {              // ident
        Var* var = var_find(tok);
        if(!var) {
            var = var_new(strndup(tok->loc, tok->len));
        }
        *rest = tok->next;
        return node_new_var(var);
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
Func* evoc_parse(Token *tok) {
    tok = token_skip(tok, "{");
    Func *prog = calloc(1, sizeof(Func));
    prog->body = compound_stmt(&tok, tok);
    prog->local_vars = local_vars;
    return prog;
}


