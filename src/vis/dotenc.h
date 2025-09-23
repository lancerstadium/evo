#ifndef DOTENC_H
#define DOTENC_H

#ifdef __cplusplus
extern "C" {
#endif
#include <evo.h>

void dot_header(char* buffer);
void dot_node(char* buffer, node_t* nd);
void dot_link_in(char* buffer, int tdx, int ndx);
void dot_link_out(char* buffer, int ndx, int tdx);
void dot_subgraph(char* buffer, graph_t* sg);
void dot_graph(char* buffer, graph_t* g);
void dot_footer(char* buffer);


#ifdef __cplusplus
}
#endif
#endif // DOTENC_H