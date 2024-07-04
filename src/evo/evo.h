/**
 * =================================================================================== //
 * @file evo.h
 * @author lancerstadium (lancerstadium@163.com)
 * @brief evo main header file
 * @version 0.1
 * @date 2024-07-03
 * @copyright Copyright (c) 2024
 * =================================================================================== //
 */

// ==================================================================================== //
//                                       evo/evo.h
// ==================================================================================== //

#ifndef __EVO_EVO_H__
#define __EVO_EVO_H__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ==================================================================================== //
//                                       include
// ==================================================================================== //

#include "hashmap.h"

// ==================================================================================== //
//                                       define
// ==================================================================================== //

#if defined(__GNUC__) || defined(__clang__)
#define EVO_API     __attribute__((visibility("default")))
#elif defined(_MSC_VER)
#define EVO_API     __declspec(dllexport)
#elif  // __GNUC__ || __clang__
#define EVO_API
#endif  // __GNUC__ || __clang__


#define EVO_DIM_MAX     8
#define EVO_ALIGN_SIZE 16

// ==================================================================================== //
//                                       typedef
// ==================================================================================== //

typedef struct op op_t;
typedef struct node node_t;
typedef struct graph graph_t;
typedef struct tensor tensor_t;
typedef struct device device_t;
typedef enum op_type op_type_t;
typedef struct context context_t;
typedef enum node_type node_type_t;
typedef struct scheduler scheduler_t;
typedef struct optimizer optimizer_t;
typedef struct interface interface_t;
typedef struct allocator allocator_t;
typedef enum tensor_type tensor_type_t;

// ==================================================================================== //
//                                       evo: tensor type
// ==================================================================================== //

enum tensor_type {
    EVO_TENSOR_TYPE_UNDEFINED = 0,
    EVO_TENSOR_TYPE_BOOL = 9,
    EVO_TENSOR_TYPE_INT8 = 3,
    EVO_TENSOR_TYPE_INT16 = 5,
    EVO_TENSOR_TYPE_INT32 = 6,
    EVO_TENSOR_TYPE_INT64 = 7,
    EVO_TENSOR_TYPE_UINT8 = 2,
    EVO_TENSOR_TYPE_UINT16 = 4,
    EVO_TENSOR_TYPE_UINT32 = 12,
    EVO_TENSOR_TYPE_UINT64 = 13,
    EVO_TENSOR_TYPE_BFLOAT16 = 16,
    EVO_TENSOR_TYPE_FLOAT16 = 10,
    EVO_TENSOR_TYPE_FLOAT32 = 1,
    EVO_TENSOR_TYPE_FLOAT64 = 11,
    EVO_TENSOR_TYPE_COMPLEX64 = 14,
    EVO_TENSOR_TYPE_COMPLEX128 = 15,
    EVO_TENSOR_TYPE_STRING = 8,
};

EVO_API const char *tensor_type_tostring(tensor_type_t type);
EVO_API int tensor_type_sizeof(tensor_type_t type);

// ==================================================================================== //
//                                       evo: tensor
// ==================================================================================== //

struct tensor {
    tensor_type_t type;             /* Tensor data type             */
    char* name;                     /* Tensor name                  */
    int index;                      /* Tensor index                 */
    int dims[EVO_DIM_MAX];          /* Shape of dim array           */
    int ndim;                       /* Valid entry number           */
    void* datas;                    /* Tensor data addr             */
    uint8_t ndata;                  /* Tensor data size             */
    uint8_t  szelem;                /* Tensor element size          */
    uint32_t nelem;                 /* Tensor element number        */
    int16_t pnode;                  /* Tensor parent node index     */

    uint8_t is_reshaped : 1;        /* Tensor is reshaped           */
    uint8_t is_constant : 1;        /* Tensor is constant           */
    uint8_t is_input : 1;           /* Tensor is input              */
    uint8_t is_output : 1;          /* Tensor is output             */
    uint8_t is_iallocated: 1;       /* Tensor is iallocated         */
    int8_t layout : 1;              /* Tensor is layout 0NCHW1NHWC  */
};

EVO_API tensor_t * tensor_new(graph_t*, const char*, tensor_type_t);
EVO_API void tensor_free(tensor_t*, graph_t*);
EVO_API int tensor_set_shape(tensor_t *tensor, int ndim, int *dims);
EVO_API char* tensor_set_name_by_index(graph_t *graph, int index) ;

// ==================================================================================== //
//                                       evo: op type
// ==================================================================================== //

enum op_type {
    EVO_OP_TYPE_GENERIC,
    EVO_OP_TYPE_ABS,
    EVO_OP_TYPE_ADD,
};

// ==================================================================================== //
//                                       evo: op
// ==================================================================================== //

struct op {
    op_type_t type;                 /* Operator type        */
    const char* name;               /* Operator name        */
    uint8_t is_same_shape : 1;      /* Operator same shape  */
    uint16_t param_size;            /* Size of param mem buf*/
    void* param_mem;                /* Param mem buffer     */
    int (*infer_shape)(node_t*);    /* Operator shape fn    */
    void (*init)(op_t*);            /* Operator init fn     */
    void (*release)(op_t*);         /* Operator release fn  */
};

void op_init(op_t*);

// ==================================================================================== //
//                                       evo: node type
// ==================================================================================== //

enum node_type { 
    EVO_NODE_TYPE_GENERIC, 
    EVO_NODE_TYPE_INPUT, 
    EVO_NODE_TYPE_OUTPUT, 
    EVO_NODE_TYPE_INTERMEDIATE, 
};

// ==================================================================================== //
//                                       evo: node
// ==================================================================================== //

struct node {
    char *name;                     /* Node name            */
    uint16_t index;                 /* Index of Node Graph  */
    node_type_t type;               /* Type of Node         */
    uint8_t ninput;                 /* Number of Input      */
    uint8_t noutput;                /* Number of Output     */

    uint16_t * input_tensors;       /* Input Tensor Indexes */
    uint16_t * output_tensors;      /* Output Tensor Indexes*/

    op_t op;                        /* Operator             */
    graph_t *graph;                 /* Owner Graph          */
};

EVO_API node_t * node_new(graph_t*, const char*, op_type_t);
EVO_API void node_free(node_t*, graph_t*);

// ==================================================================================== //
//                                       evo: graph
// ==================================================================================== //

struct graph {
    tensor_t **tensors;             /* Graph tensors list       */
    node_t **nodes;                 /* Graph nodes list         */
    uint16_t ntensor;               /* Count of all tensor      */
    uint16_t nnode;                 /* Count of all note        */

    int8_t data_layout : 1;         /* Data layout: 0NCHW/1NHWC */
};

EVO_API graph_t * graph_new(context_t*);
EVO_API void graph_free(graph_t*, context_t*);

// ==================================================================================== //
//                                       evo: context
// ==================================================================================== //

struct context {
    char *name;                     /* Context name             */
    scheduler_t *scd;               
    device_t *dev;
};

// ==================================================================================== //
//                                       evo: scheduler
// ==================================================================================== //

struct scheduler {
    const char* name;                       /* Scheduler name       */
    void (*prerun)(scheduler_t*, graph_t*); /* Scheduler pre run fn */
    void (*run)(scheduler_t*, graph_t*);
    void (*wait)(scheduler_t*, graph_t*);
    void (*posrun)(scheduler_t*, graph_t*);
};

// ==================================================================================== //
//                                       evo: optimizer
// ==================================================================================== //

struct optimizer {
    void (*run)(graph_t*);
};

// ==================================================================================== //
//                                       evo: interface
// ==================================================================================== //

struct interface {
};

// ==================================================================================== //
//                                       evo: allocator
// ==================================================================================== //

struct allocator {
    void (*desc)(device_t*);
    void (*eval)(device_t*, graph_t*);
    void (*alloc)(device_t*, graph_t*);   /* Alloc resource       */
    void (*release)(device_t*, graph_t*); /* Release all allocated*/
};

// ==================================================================================== //
//                                       evo: device
// ==================================================================================== //

struct device {
    const char* name;
    interface_t* itf;
    allocator_t* alc;
    optimizer_t* opt;
    scheduler_t* scd;
};

void device_init(device_t*, const char*);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __EVO_EVO_H__