

#include "parser.h"



// ==================================================================================== //
//                           parser Pub API: Parser Tokens
// ==================================================================================== //

int parse(ParseProcess *pproc) {
    LOG_TAG
    vector_set_peek_pointer(pproc->lex_proc->token_vec, 0);
    Node* nd = NULL;
    parse_process_next(pproc);
    // while(parse_process_next(pproc) == 0) {
    //     // 弹出在堆栈上创建的节点，以便我们可以将其添加到树的根
    //     // 我们弹出的这个元素来自 parse_next
    //     Node* node = pproc->pop_node(pproc);
    //     // 将根元素推入树
    //     vector_push(pproc->node_tree_vec, &node);
    //     // 还将其推回我们刚刚弹出的主节点堆栈
    //     pproc->push_node(pproc, node);
    // }
    for(int i = 0; i < pproc->node_vec->count; i++) {
        nd = vector_at(pproc->node_vec, i);
        node_read(nd);
    }
    return PARSER_ANALYSIS_OK;
}

