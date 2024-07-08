/**
 * =================================================================================== //
 * @file evo.h
 * @author lancerstadium (lancerstadium@163.com)
 * @brief evo main header file
 * @version 0.1
 * @date 2024-07-03
 * @copyright Copyright (c) 2024
 * @attention EVO use linux-c style code:
 *      1. function: use `int` return bool (0: true, -1: false)
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

#include "map.h"
#include "vec.h"

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

#define EVO_UNUSED      __attribute__((unused))
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
typedef struct resolver resolver_t;
typedef struct scheduler scheduler_t;
typedef struct optimizer optimizer_t;
typedef struct interface interface_t;
typedef struct allocator allocator_t;
typedef enum tensor_type tensor_type_t;
typedef struct serializer serializer_t;

// ==================================================================================== //
//                                       evo: internal
// ==================================================================================== //

EVO_UNUSED static device_t* internal_device_registry = NULL;

EVO_API int device_registry_init(const char*);
EVO_API device_t* device_registry_find(const char*);
EVO_API device_t* device_registry_get(int);
EVO_API void device_registry_release();

// ==================================================================================== //
//                                       evo: tensor type
// ==================================================================================== //

enum tensor_type {
    TENSOR_TYPE_UNDEFINED = 0,
    TENSOR_TYPE_BOOL = 9,
    TENSOR_TYPE_INT8 = 3,
    TENSOR_TYPE_INT16 = 5,
    TENSOR_TYPE_INT32 = 6,
    TENSOR_TYPE_INT64 = 7,
    TENSOR_TYPE_UINT8 = 2,
    TENSOR_TYPE_UINT16 = 4,
    TENSOR_TYPE_UINT32 = 12,
    TENSOR_TYPE_UINT64 = 13,
    TENSOR_TYPE_BFLOAT16 = 16,
    TENSOR_TYPE_FLOAT16 = 10,
    TENSOR_TYPE_FLOAT32 = 1,
    TENSOR_TYPE_FLOAT64 = 11,
    TENSOR_TYPE_COMPLEX64 = 14,
    TENSOR_TYPE_COMPLEX128 = 15,
    TENSOR_TYPE_STRING = 8,
};

EVO_API const char *tensor_type_tostring(tensor_type_t);
EVO_API int tensor_type_sizeof(tensor_type_t);

// ==================================================================================== //
//                                       evo: tensor
// ==================================================================================== //

struct tensor {
    tensor_type_t type;                                 /* Tensor data type             */
    char* name;                                         /* Tensor name                  */
    int index;                                          /* Tensor index                 */
    int dims[EVO_DIM_MAX];                              /* Shape of dim array           */
    int strides[EVO_DIM_MAX];                           /* Offset of dim array          */
    int ndim;                                           /* Valid entry number           */
    void* datas;                                        /* Tensor data addr             */
    uint8_t ndata;                                      /* Tensor data size             */
    uint8_t  szelem;                                    /* Tensor element size          */
    uint32_t nelem;                                     /* Tensor element number        */
    int16_t pnode;                                      /* Tensor parent node index     */

    uint8_t is_reshaped : 1;                            /* Tensor is reshaped           */
    uint8_t is_constant : 1;                            /* Tensor is constant           */
    uint8_t is_input : 1;                               /* Tensor is input              */
    uint8_t is_output : 1;                              /* Tensor is output             */
    uint8_t is_iallocated: 1;                           /* Tensor is iallocated         */
    uint8_t layout : 1;                                 /* Tensor is layout 0NCHW1NHWC  */
};

EVO_API tensor_t * tensor_new(const char*, tensor_type_t);
EVO_API tensor_t * tensor_reinit(tensor_t*, tensor_type_t, int, int*);
EVO_API void tensor_free(tensor_t*);
EVO_API void tensor_dump(tensor_t*);
EVO_API int tensor_set_shape(tensor_t*, int, int*);
EVO_API char* tensor_set_name_by_index(graph_t*, int) ;
EVO_API int tensor_get_index_by_name(graph_t *, const char *);

// ==================================================================================== //
//                                       evo: op type
// ==================================================================================== //

enum op_type {
    OP_TYPE_GENERIC,
    OP_TYPE_ABS,
    OP_TYPE_ADD,
};

// ==================================================================================== //
//                                       evo: op
// ==================================================================================== //

struct op {
    op_type_t type;                                     /* Operator type                */
    const char* name;                                   /* Operator name                */
    void (*run)(node_t*);                               /* Operator run fn              */

    uint8_t is_same_shape : 1;                          /* Operator same shape          */

    uint16_t param_size;                                /* Size of param mem buf        */
    void* param_mem;                                    /* Param mem buffer             */
};

EVO_API void op_copy(op_t *s, op_t *t);

// ==================================================================================== //
//                                       evo: resolver
// ==================================================================================== //

struct resolver {
    const char* name;                                   /* Resolver Name                */
    void* (*init)();                                    /* Operator init fn             */
    void (*release)(void*);                             /* Operator release fn          */

    op_t* op_tbl;                                       /* Operator table               */
};

EVO_API resolver_t* resolver_get_default();

// ==================================================================================== //
//                                       evo: node type
// ==================================================================================== //

enum node_type { 
    NODE_TYPE_GENERIC,
    NODE_TYPE_INPUT, 
    NODE_TYPE_OUTPUT, 
    NODE_TYPE_INTER, 
};

// ==================================================================================== //
//                                       evo: node
// ==================================================================================== //

struct node {
    char *name;                                         /* Node name                    */
    uint16_t index;                                     /* Index of Node Graph          */
    node_type_t type;                                   /* Type of Node                 */
    uint8_t ninput;                                     /* Number of Input              */
    uint8_t noutput;                                    /* Number of Output             */

    tensor_t ** input_tensors;                          /* Input Tensor List            */
    tensor_t ** output_tensors;                         /* Output Tensor List           */

    op_t* op;                                           /* Operator                     */
    int opset;                                          /* Operator set                 */
    graph_t *graph;                                     /* Owner Graph                  */
    context_t *ctx;                                     /* Owner Context                */

    void* node_proto;

    int (*reshape)(struct node *);
    void (*operator)(struct node *);
};

EVO_API node_t * node_new(graph_t*, const char*, op_type_t);
EVO_API void node_dump(node_t*);
EVO_API void node_free(node_t*, graph_t*);

// ==================================================================================== //
//                                       evo: graph status
// ==================================================================================== //

enum graph_status {
    GRAPH_STATUS_INIT,
    GRAPH_STATUS_READY,
    GRAPH_STATUS_RUN,
    GRAPH_STATUS_SUSPEND,
    GRAPH_STATUS_RESUME,
    GRAPH_STATUS_ABORT,
    GRAPH_STATUS_DONE,
    GRAPH_STATUS_ERROR
};

// ==================================================================================== //
//                                       evo: graph
// ==================================================================================== //

struct graph {

    tensor_t **tensors;                                 /* Graph tensors list           */
    node_t **nodes;                                     /* Graph nodes list             */
    uint16_t ntensor;                                   /* Count of all tensor          */
    uint16_t nnode;                                     /* Count of all note            */

    serializer_t *sez;                                  /* Serializer of graph          */
    device_t *dev;                                      /* Device of graph              */
    context_t *ctx;                                     /* Owner Context                */

    uint8_t data_layout : 1;                            /* Data layout: 0NCHW/1NHWC     */
    uint8_t is_sub : 1;                                 /* Graph is sub graph           */
    uint8_t status : 4;                                 /* Status of Graph              */

    union {
        struct {                                        /* When is_sub = 0              */
            struct graph * sub_vec;                     /* P|Vector of sub graphs       */
            uint16_t *input_inodes_vec;                 /* P|Input nodes index Vector   */
            uint16_t *output_inodes_vec;                /* P|Output nodes index Vector  */
            uint16_t ninput_node;                       /* P|Input nodes number         */
            uint16_t noutput_node;                      /* P|Output nodes umber         */
        };
        struct {                                        /* When is_sub = 1              */
            uint16_t *nodes_vec;                        /* S|Node index Vec from parents*/
            uint16_t *input_itensors_vec;               /* S|Input tensors index Vector */
            uint16_t *output_itensors_vec;              /* S|Output tensors index Vector*/
            struct graph * pgraph;                      /* S|Parent graph of this sub   */
            void *info;                                 /* S|RunTime Info on device     */
        };
    };
};

EVO_API graph_t * graph_new(context_t*);
EVO_API graph_t * graph_sub(graph_t*);
EVO_API void graph_push_tenser(graph_t*, tensor_t*);
EVO_API void graph_push_node(graph_t*, node_t*);
EVO_API node_t* graph_get_node(graph_t*, int);
EVO_API void graph_prerun(graph_t*);
EVO_API void graph_run(graph_t*);
EVO_API void graph_wait(graph_t*);
EVO_API void graph_posrun(graph_t*);
EVO_API void graph_free(graph_t*);

// ==================================================================================== //
//                                       evo: context
// ==================================================================================== //

struct context {
    char *name;                                         /* Context name                 */
    graph_t  *graph;                                    /* Context graph entry          */

    scheduler_t *scd;                                   /* Context scheduler            */
    serializer_t *sez;                                  /* Serializer of contex         */
    device_t *dev;                                      /* Context device               */

    void* model;                                        /* Context model proto          */
    uint32_t model_size;                                /* Context model size           */

    hashmap_t *tensor_map;                              /* Context tensor map           */
};

EVO_API context_t * context_new(const char*);
EVO_API tensor_t* context_get_tensor(context_t*, const char*);
EVO_API void context_free(context_t*);

// ==================================================================================== //
//                                       evo: serializer
// ==================================================================================== //

struct serializer {
    const char* fmt;                                    /* Serializer format name       */

    context_t * (*load) (struct serializer*, const void *, int);
    context_t * (*load_file) (struct serializer*, const char*);
    graph_t* (*load_graph) (context_t*);                /* Serializer load ctx to graph */
    void (*unload) (context_t*);                        /* Serializer unload model      */
};

EVO_API serializer_t *serializer_new(const char*);
EVO_API void serializer_free(serializer_t *);


// ==================================================================================== //
//                                       evo: scheduler
// ==================================================================================== //

struct scheduler {
    const char* name;                                   /* Scheduler name               */
    void (*prerun)(scheduler_t*, graph_t*);             /* Scheduler pre run fn         */
    void (*run)(scheduler_t*, graph_t*);                /* Scheduler run fn             */
    void (*wait)(scheduler_t*, graph_t*);               /* Scheduler wait fn            */
    void (*posrun)(scheduler_t*, graph_t*);             /* Scheduler pos run fn         */
};

EVO_API scheduler_t* scheduler_get_default();

// ==================================================================================== //
//                                       evo: device
// ==================================================================================== //

struct device {
    const char* name;                                   /* Device Name                  */
    resolver_t * rsv;                                   /* Device ops' Resolver         */
    interface_t* itf;                                   /* Device Interface: Main       */
    allocator_t* alc;                                   /* Device Mem Allocator         */
    optimizer_t* opt;                                   /* Device Graph Optimizer       */
    scheduler_t* scd;                                   /* Device Own Scheduler         */
};

EVO_API device_t * device_new(const char*);
EVO_API op_t * device_find_op(device_t*, op_type_t);
EVO_API device_t* device_reg(const char*);
EVO_API int device_reg_dev(device_t*);
EVO_API int device_unreg_dev(device_t*);

// ==================================================================================== //
//                                  evo: interface (device)
// ==================================================================================== //

struct interface {
    int (*init)(device_t*);                             /* Device Init: rsv ...         */
    int (*prerun)(device_t*, graph_t*);                 /* Pre Run Graph: True Entry    */
    int (*run)(device_t*, graph_t*);                    /* Run Graph: True Entry        */
    int (*posrun)(device_t*, graph_t*);                 /* Post Run Graph: True Entry   */
    int (*release)(device_t*);                          /* Device Release: rsv ...      */
};

// ==================================================================================== //
//                                  evo: allocator (device)
// ==================================================================================== //

struct allocator {
    void (*desc)(device_t*);
    void (*eval)(device_t*, graph_t*);
    void (*alloc)(device_t*, graph_t*);                 /* Alloc resource               */
    void (*release)(device_t*, graph_t*);               /* Release all allocated        */
};


// ==================================================================================== //
//                                  evo: optimizer (device)
// ==================================================================================== //

struct optimizer {
    void (*run)(graph_t*);
};




#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __EVO_EVO_H__