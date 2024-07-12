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

// ==================================================================================== //
//                                       include
// ==================================================================================== //

#include "util/map.h"
#include "util/vec.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

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
typedef struct context context_t;
typedef struct resolver resolver_t;
typedef struct scheduler scheduler_t;
typedef struct optimizer optimizer_t;
typedef struct interface interface_t;
typedef struct allocator allocator_t;
typedef struct serializer serializer_t;

typedef device_t** device_vec_t;
typedef tensor_t** tensor_vec_t;
typedef graph_t** graph_vec_t;
typedef node_t** node_vec_t;

// ==================================================================================== //
//                                       evo: internal
// ==================================================================================== //

EVO_UNUSED static device_vec_t internal_device_registry = NULL;

EVO_API int device_registry_init(const char*);
EVO_API device_t* device_registry_find(const char*);
EVO_API device_t* device_registry_get(int);
EVO_API void device_registry_release();

// ==================================================================================== //
//                                       evo: tensor type
// ==================================================================================== //

typedef enum tensor_type {
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
} tensor_type_t;

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
    size_t ndata;                                       /* Tensor data size             */
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
EVO_API int tensor_index2offset(tensor_t*, int*);
EVO_API void tensor_offset2index(tensor_t*, int, int*);
EVO_API int tensor_reshape(tensor_t*, int, int*);
EVO_API int tensor_reshape_ident(tensor_t*, tensor_t*, tensor_type_t);
EVO_API int tensor_reshape_multi_broadcast(tensor_t*, tensor_t*, tensor_t*, tensor_type_t);
EVO_API void* tensor_broadcast_map_address(tensor_t*, tensor_t*, int);
EVO_API void tensor_apply(tensor_t*, void*, size_t);
EVO_API char* tensor_set_name_by_index(graph_t*, int) ;
EVO_API int tensor_get_index_by_name(graph_t *, const char *);

// ==================================================================================== //
//                                       evo: op type
// ==================================================================================== //

typedef enum op_type {
    // ==== OP: No Operation
    OP_TYPE_NOP,
    // ==== OP: A head
    OP_TYPE_ABS,
    OP_TYPE_ACOS,
    OP_TYPE_ACOSH,
    OP_TYPE_ADD,
    OP_TYPE_AND,
    OP_TYPE_ARG_MAX,
    OP_TYPE_ARG_MIN,
    OP_TYPE_ASIN,
    OP_TYPE_ASINH,
    OP_TYPE_ATAN,
    OP_TYPE_ATANH,
    OP_TYPE_AVERAGE_POOL,
    // ==== OP: B head
    OP_TYPE_BATCH_NORMALIZATION,
    OP_TYPE_BITSHIFT,
    // ==== OP: C head
    OP_TYPE_CAST,
    OP_TYPE_CEIL,
    OP_TYPE_CELU,
    OP_TYPE_CLIP,
    OP_TYPE_COMPRESS,
    OP_TYPE_CONCAT,
    OP_TYPE_CONCAT_FROM_SEQUENCE,
    OP_TYPE_CONSTANT,
    OP_TYPE_CONSTANT_OF_SHAPE,
    OP_TYPE_CONV,
    OP_TYPE_CONV_INTERGER,
    OP_TYPE_CONV_TRANSPOSE,
    OP_TYPE_COS,
    OP_TYPE_COSH,
    OP_TYPE_CUM_SUM,
    // ==== OP: D head
    OP_TYPE_DEPTH_TO_SPACE,
    OP_TYPE_DEQUANTIZE_LINEAR,
    OP_TYPE_DET,
    OP_TYPE_DIV,
    OP_TYPE_DROPOUT,
    OP_TYPE_DYNAMIC_QUANTIZE_LINEAR,
    // ==== OP: E head
    OP_TYPE_EINSUM,
    OP_TYPE_ELU,
    OP_TYPE_EQUAL,
    OP_TYPE_ERF,
    OP_TYPE_EXP,
    OP_TYPE_EXPAND,
    OP_TYPE_EYELIKE,
    // ==== OP: F head
    OP_TYPE_FLATTEN,
    OP_TYPE_FLOOR,
    // ==== OP: G head
    OP_TYPE_GRU,
    OP_TYPE_GATHER,
    OP_TYPE_GATHER_ELEMENTS,
    OP_TYPE_GATHER_ND,
    OP_TYPE_GEMM,
    OP_TYPE_GLOBAL_AVERAGEPOOL,
    OP_TYPE_GLOBAL_LP_POOL,
    OP_TYPE_GLOBAL_MAX_POOL,
    OP_TYPE_GREATER,
    OP_TYPE_GREATER_OR_EQUAL,
    // ==== OP: H head
    OP_TYPE_HARD_SIGMOID,
    OP_TYPE_HARDMAX,
    OP_TYPE_HARD_SWISH,
    // ==== OP: I head
    OP_TYPE_IDENTITY,
    OP_TYPE_IF,
    OP_TYPE_INSTANCE_NORMALIZATION,
    OP_TYPE_IS_INF,
    OP_TYPE_IS_NAN,
    // ==== OP: L head
    OP_TYPE_LRN,
    OP_TYPE_LSTM,
    OP_TYPE_LEAKY_RELU,
    OP_TYPE_LESS,
    OP_TYPE_LESS_OR_EQUAL,
    OP_TYPE_LOG,
    OP_TYPE_LOG_SOFTMAX,
    OP_TYPE_LOOP,
    OP_TYPE_LP_NORMALIZATION,
    OP_TYPE_LP_POOL,
    // ==== OP: M head
    OP_TYPE_MAT_MUL,
    OP_TYPE_MAT_MUL_INTERGER,
    OP_TYPE_MAX,
    OP_TYPE_MAX_POOL,
    OP_TYPE_MAX_ROI_POOL,
    OP_TYPE_MAX_UNPOOL,
    OP_TYPE_MEAN,
    OP_TYPE_MEAN_VARIANCE_NORMALIZATION,
    OP_TYPE_MIN,
    OP_TYPE_MOD,
    OP_TYPE_MUL,
    OP_TYPE_MULTINOMIAL,
    // ==== OP: N head
    OP_TYPE_NEG,
    OP_TYPE_NEGATIVE_LOG_LIKELIHOOD_LOSS,
    OP_TYPE_NON_MAX_SUPPRESSION,
    OP_TYPE_NON_ZERO,
    OP_TYPE_NOT,
    OP_TYPE_ONE_HOT,
    OP_TYPE_OR,
    // ==== OP: P head
    OP_TYPE_PRELU,
    OP_TYPE_PAD,
    OP_TYPE_POW,
    // ==== OP: Q head
    OP_TYPE_QLINEAR_CONV,
    OP_TYPE_QLINEAR_MAT_MUL,
    OP_TYPE_QUANTIZE_LINEAR,
    // ==== OP: R head
    OP_TYPE_RNN,
    OP_TYPE_RANDOM_NORMAL,
    OP_TYPE_RANDOM_NORMAL_LIKE,
    OP_TYPE_RANDOM_UNIFORM,
    OP_TYPE_RANDOM_UNIFORM_LIKE,
    OP_TYPE_RANGE,
    OP_TYPE_RECIPROCAL,
    OP_TYPE_REDUCE_L1,
    OP_TYPE_REDUCE_L2,
    OP_TYPE_REDUCE_LOG_SUM,
    OP_TYPE_REDUCE_LOG_SUM_EXP,
    OP_TYPE_REDUCE_MAX,
    OP_TYPE_REDUCE_MEAN,
    OP_TYPE_REDUCE_MIN,
    OP_TYPE_REDUCE_PROD,
    OP_TYPE_REDUCE_SUM,
    OP_TYPE_REDUCE_SUM_SQUARE,
    OP_TYPE_RELU,
    OP_TYPE_RESHAPE,
    OP_TYPE_RESIZE,
    OP_TYPE_REVERSE_SEQUENCE,
    OP_TYPE_ROI_ALIGN,
    OP_TYPE_ROUND,
    // ==== OP: S head
    OP_TYPE_SCAN,
    OP_TYPE_SCATTER,
    OP_TYPE_SCATTER_ELEMENTS,
    OP_TYPE_SCATTER_ND,
    OP_TYPE_SELU,
    OP_TYPE_SEQUENCE_AT,
    OP_TYPE_SEQUENCE_CONSTRUCT,
    OP_TYPE_SEQUENCE_EMPTY,
    OP_TYPE_SEQUENCE_ERASE,
    OP_TYPE_SEQUENCE_INSERT,
    OP_TYPE_SEQUENCE_LENGTH,
    OP_TYPE_SHAPE,
    OP_TYPE_SHRINK,
    OP_TYPE_SIGMOID,
    OP_TYPE_SIGN,
    OP_TYPE_SIN,
    OP_TYPE_SINH,
    OP_TYPE_SIZE,
    OP_TYPE_SLICE,
    OP_TYPE_SOFTMAX,
    OP_TYPE_SOFTMAX_CROSS_ENTROPY_LOSS,
    OP_TYPE_SOFTPLUS,
    OP_TYPE_SOFTSIGN,
    OP_TYPE_SPACE_TO_DEPTH,
    OP_TYPE_SPLIT,
    OP_TYPE_SPLIT_TO_SEQUENCE,
    OP_TYPE_SQRT,
    OP_TYPE_SQUEEZE,
    OP_TYPE_STRING_NORMALIZER,
    OP_TYPE_SUB,
    OP_TYPE_SUM,
    // ==== OP: T head
    OP_TYPE_TAN,
    OP_TYPE_TANH,
    OP_TYPE_TF_IDF_VECTORIZER,
    OP_TYPE_THRESHOLDED_RELU,
    OP_TYPE_TILE,
    OP_TYPE_TOP_K,
    OP_TYPE_TRANSPOSE,
    OP_TYPE_TRILU,
    // ==== OP: U head
    OP_TYPE_UNIQUE,
    OP_TYPE_UNSQUEEZE,
    OP_TYPE_UPSAMPLE,
    // ==== OP: W head
    OP_TYPE_WHERE,
    // ==== OP: X head
    OP_TYPE_XOR,
    // ==== OP: last
    OP_TYPE_LAST
} op_type_t;

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

typedef enum node_type { 
    NODE_TYPE_GENERIC,
    NODE_TYPE_INPUT, 
    NODE_TYPE_OUTPUT, 
    NODE_TYPE_INTER, 
} node_type_t;

// ==================================================================================== //
//                                       evo: node
// ==================================================================================== //

struct node {
    char *name;                                         /* Node name                    */
    uint16_t index;                                     /* Index of Node Graph          */
    node_type_t type;                                   /* Type of Node                 */
    
    uint8_t nin;                                        /* Number of Input              */
    uint8_t nout;                                       /* Number of Output             */
    tensor_t ** in;                                     /* Input Tensor List            */
    tensor_t ** out;                                    /* Output Tensor List           */

    op_t* op;                                           /* Operator                     */
    int opset;                                          /* Operator set                 */
    graph_t *graph;                                     /* Owner Graph                  */
    context_t *ctx;                                     /* Owner Context                */

    void* node_proto;
    void* priv;
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
            graph_vec_t sub_vec;                        /* P|Vector of sub graphs       */
            uint16_t *input_inodes_vec;                 /* P|Input nodes index Vector   */
            uint16_t *output_inodes_vec;                /* P|Output nodes index Vector  */
            uint16_t ninput_node;                       /* P|Input nodes number         */
            uint16_t noutput_node;                      /* P|Output nodes umber         */
        };
        struct {                                        /* When is_sub = 1              */
            int idx;                                    /* S|Index in sub vector        */
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
EVO_API void graph_step(graph_t*, int);
EVO_API void graph_run(graph_t*);
EVO_API void graph_wait(graph_t*);
EVO_API void graph_posrun(graph_t*);
EVO_API void graph_dump(graph_t*);
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
    context_t * (*load_model) (struct serializer*, const char*);
    tensor_t * (*load_tensor) (const char*);
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
    void (*step)(scheduler_t*, graph_t*, int);          /* Scheduler step fn            */
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
EVO_API int device_unreg(const char*);
EVO_API int device_reg_dev(device_t*);
EVO_API int device_unreg_dev(device_t*);

// ==================================================================================== //
//                                  evo: interface (device)
// ==================================================================================== //

struct interface {
    int (*init)(device_t*);                             /* Device Init: rsv ...         */
    int (*prerun)(device_t*, graph_t*);                 /* Pre Run Graph: True Entry    */
    int (*step)(device_t*, graph_t*, int);              /* Step Run Graph: True Entry   */
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