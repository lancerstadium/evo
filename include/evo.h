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

#include "evo/util/map.h"
#include "evo/util/vec.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ==================================================================================== //
//                                       define
// ==================================================================================== //

#if defined(__GNUC__) || defined(__clang__)
#define EVO_API         __attribute__((visibility("default")))
#define EVO_UNUSED      __attribute__((unused))
#define EVO_PACKED(D)   D __attribute__((packed))
#elif defined(_MSC_VER)
#define EVO_API         __declspec(dllexport)
#define EVO_UNUSED      __pragma(warning(suppress:4100))
#define EVO_PACKED(D)   D __attribute__((packed))
#elif  // __GNUC__ || __clang__
#define EVO_API
#define EVO_UNUSED
#endif  // __GNUC__ || __clang__

#define EVO_DFT_DEV         "cpu"
#define EVO_DIM_MAX         8
#define EVO_ALIGN_SIZE      16

#include "evo/config.h"

// ==================================================================================== //
//                                       typedef
// ==================================================================================== //

typedef struct op op_t;
typedef struct node node_t;
typedef struct graph graph_t;
typedef struct model model_t;
typedef struct tensor tensor_t;
typedef struct device device_t;
typedef struct runtime runtime_t;
typedef struct resolver resolver_t;
typedef struct profiler profiler_t;
typedef struct scheduler scheduler_t;
typedef struct optimizer optimizer_t;
typedef struct interface interface_t;
typedef struct allocator allocator_t;
typedef struct attribute attribute_t;
typedef struct predictor predictor_t;
typedef struct serializer serializer_t;
typedef struct internal_context internal_context_t;

typedef attribute_t** attribute_vec_t;
typedef device_t** device_vec_t;
typedef tensor_t** tensor_vec_t;
typedef graph_t** graph_vec_t;
typedef node_t** node_vec_t;

// ==================================================================================== //
//                                       evo: internal
// ==================================================================================== //

struct internal_context {
    model_t * mdl;
    device_vec_t dev_vec;
};

EVO_UNUSED static internal_context_t internal_context_info = {
    NULL, 
    NULL
};

EVO_API void internal_model_set(model_t*);
EVO_API model_t* internal_model_init();
EVO_API model_t* internal_model_get();
EVO_API int internal_device_init(const char*);
EVO_API device_t* internal_device_find(const char*);
EVO_API device_t* internal_device_find_nowarn(const char*);
EVO_API device_t* internal_device_get(int);
EVO_API device_t* internal_device_get_default();
EVO_API void internal_device_release();

// ==================================================================================== //
//                                       evo: runtime
// ==================================================================================== //

struct runtime {
    device_vec_t dev_reg_vec;                           /* Runtime Devices' Registry    */
    serializer_t * sez;                                 /* Runtime Serializer           */
    model_t * mdl;                                      /* Runtime Model                */
};

EVO_API runtime_t * runtime_new(const char*);
EVO_API model_t * runtime_load(runtime_t*, const char*);
EVO_API model_t* runtime_load_raw(runtime_t*, const void*, size_t);
EVO_API tensor_t* runtime_load_tensor(runtime_t*, const char*);
EVO_API void runtime_set_tensor(runtime_t*, const char*, tensor_t*);
EVO_API tensor_t * runtime_get_tensor(runtime_t*, const char*);
EVO_API void runtime_unload(runtime_t*);
EVO_API void runtime_run(runtime_t*);
EVO_API void runtime_graph_dump(runtime_t*);
EVO_API void runtime_free(runtime_t*);
EVO_API device_t* runtime_reg_dev(runtime_t*, const char*);
EVO_API void runtime_unreg_dev(runtime_t*, const char*);

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
    int16_t pnode;                                      /* Tensor parent node index     */

    uint8_t is_reshaped : 1;                            /* Tensor is reshaped           */
    uint8_t is_constant : 1;                            /* Tensor is constant           */
    uint8_t is_input : 1;                               /* Tensor is input              */
    uint8_t is_output : 1;                              /* Tensor is output             */
    uint8_t is_iallocated: 1;                           /* Tensor is iallocated         */
    uint8_t layout : 1;                                 /* Tensor is layout 0NCHW/1NHWC */
};

EVO_API tensor_t * tensor_new(const char*, tensor_type_t);
EVO_API tensor_t * tensor_new_int64(const char*, int*, int, int64_t*, size_t);
EVO_API tensor_t * tensor_new_float32(const char*, int*, int, float*, size_t);
EVO_API tensor_t * tensor_reinit(tensor_t*, tensor_type_t, int, int*);
EVO_API tensor_t * tensor_permute(tensor_t*, int, int*);
EVO_API tensor_t * tensor_argmax(tensor_t*, int, int, int);
EVO_API tensor_t * tensor_softmax(tensor_t*, int);
EVO_API tensor_t * tensor_squeeze(tensor_t*, int*, int);
EVO_API tensor_t * tensor_nhwc2nchw(tensor_t*);
EVO_API tensor_t * tensor_nchw2nhwc(tensor_t*);
EVO_API void tensor_free(tensor_t*);
EVO_API void tensor_copy(tensor_t*, tensor_t*);
EVO_API bool tensor_equal(tensor_t*, tensor_t*);
EVO_API char* tensor_dump_shape(tensor_t*);
EVO_API void tensor_dump(tensor_t*);
EVO_API void tensor_dump2(tensor_t*);
EVO_API char* tensor_to_string(tensor_t*);
EVO_API int tensor_index2offset(tensor_t*, int*);
EVO_API void tensor_offset2index(tensor_t*, int, int*);
EVO_API int tensor_reshape(tensor_t*, int, int*);
EVO_API int tensor_reshape_ident(tensor_t*, tensor_t*, tensor_type_t);
EVO_API int tensor_reshape_multi_broadcast(tensor_t*, tensor_t*, tensor_t*, tensor_type_t);
EVO_API void* tensor_broadcast_map_address(tensor_t*, tensor_t*, int);
EVO_API int tensor_broadcast_is_valid(tensor_t*, int*, int);
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
    void (*run)(node_t*);                               /* Operator run fn              */

    uint8_t is_same_shape : 1;                          /* Operator same shape          */
};

EVO_API const char* op_name(op_type_t);

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
//                                       evo: attribute type
// ==================================================================================== //

typedef enum attribute_type {
    ATTRIBUTE_TYPE_UNDEFINED,
    ATTRIBUTE_TYPE_FLOAT,
    ATTRIBUTE_TYPE_INT,
    ATTRIBUTE_TYPE_STRING,
    ATTRIBUTE_TYPE_TENSOR,
    ATTRIBUTE_TYPE_GRAPH,
    ATTRIBUTE_TYPE_FLOATS,
    ATTRIBUTE_TYPE_INTS,
    ATTRIBUTE_TYPE_BYTES,
    ATTRIBUTE_TYPE_TENSORS,
    ATTRIBUTE_TYPE_GRAPHS,
} attribute_type_t;

// ==================================================================================== //
//                                       evo: attribute
// ==================================================================================== //

struct attribute {
    char * name;
    attribute_type_t  type;
    union {
        float            f;
        int64_t          i;
        struct {
            size_t      ns;
            char       *ss;
        };
        tensor_t       * t;
        graph_t        * g;
        struct {
            size_t      nf;
            float      *fs;
        };
        struct {
            size_t      ni;
            int64_t    *is;
        };
        struct {
            size_t      nb;
            uint8_t    *bs;
        };
        struct {
            size_t      nt;
            tensor_t ** ts;
        };
        struct {
            size_t      ng;
            graph_t  ** gs;
        };
    };
};

EVO_API attribute_t* attribute_undefined(char *);
EVO_API attribute_t* attribute_float(char*, float);
EVO_API attribute_t* attribute_int(char*, int);
EVO_API attribute_t* attribute_string(char*, char*, size_t);
EVO_API attribute_t* attribute_floats(char*, float*, size_t);
EVO_API attribute_t* attribute_ints(char*, int64_t*, size_t);
EVO_API attribute_t* attribute_bytes(char*, uint8_t*, size_t);
EVO_API attribute_t* attribute_tensor(char*, tensor_t*);
EVO_API void attribute_free(attribute_t*);

// ==================================================================================== //
//                                       evo: node type
// ==================================================================================== //

typedef enum node_type { 
    NODE_TYPE_GENERIC,
    NODE_TYPE_INPUT, 
    NODE_TYPE_OUTPUT, 
    NODE_TYPE_HIDDEN, 
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
    model_t *mdl;                                       /* Owner model                  */

    attribute_vec_t attr_vec;                           /* Attribute Vec of node        */

    void* node_proto;
    void* priv;
};

EVO_API node_t * node_temp(const char*, op_type_t);
EVO_API node_t* node_new(graph_t*, const char*, op_type_t);
EVO_API attribute_t* node_get_attr(node_t*, const char*);
EVO_API float node_get_attr_float(node_t*, const char*, float);
EVO_API int64_t node_get_attr_int(node_t*, const char*, int64_t);
EVO_API char * node_get_attr_string(node_t*, const char*, char*); 
EVO_API int node_get_attr_floats(node_t*, const char*, float**);
EVO_API int node_get_attr_ints(node_t*, const char*, int64_t**);
EVO_API tensor_t* node_get_attr_tensor(node_t*, const char*, tensor_t*);
EVO_API void node_dump(node_t*);
EVO_API void node_free(node_t*);

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
    char *name;                                         /* Graph name                   */
    tensor_t **tensors;                                 /* Graph tensors list           */
    node_t **nodes;                                     /* Graph nodes list             */
    uint16_t ntensor;                                   /* Count of all tensor          */
    uint16_t nnode;                                     /* Count of all note            */

    serializer_t *sez;                                  /* Serializer of graph          */
    device_t *dev;                                      /* Device of graph              */
    model_t *mdl;                                       /* Owner model                  */

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
            profiler_t *prof;                           /* S|RunTime Profile on device  */
        };
    };
};

EVO_API graph_t * graph_new(model_t*);
EVO_API graph_t * graph_sub_new(graph_t*);
EVO_API graph_t * graph_as_sub(graph_t*);
EVO_API void graph_push_tenser(graph_t*, tensor_t*);
EVO_API void graph_push_node(graph_t*, node_t*);
EVO_API node_t* graph_get_node(graph_t*, int);
EVO_API void graph_add_layer(graph_t*, node_type_t, tensor_t**, int, int, attribute_t**, int);
EVO_API void graph_add_input(graph_t*, int, int*);
EVO_API void graph_add_dense(graph_t*, int, const char*);
EVO_API void graph_add_flatten(graph_t*);
EVO_API void graph_add_conv2d(graph_t*, int, int);
EVO_API void graph_add_maxpool2d(graph_t*, int, int);
EVO_API void graph_prerun(graph_t*);
EVO_API void graph_step(graph_t*, int);
EVO_API void graph_run(graph_t*);
EVO_API void graph_wait(graph_t*);
EVO_API void graph_posrun(graph_t*);
EVO_API void graph_dump(graph_t*);
EVO_API void graph_dump2(graph_t*);
EVO_API void graph_exec_report(graph_t*);
EVO_API void graph_exec_report_level(graph_t*, int);
EVO_API void graph_free(graph_t*);

// ==================================================================================== //
//                                       evo: model
// ==================================================================================== //

struct model {
    char *name;                                         /* model name                   */
    graph_t  *graph;                                    /* model graph entry            */
    scheduler_t *scd;                                   /* model scheduler              */
    serializer_t *sez;                                  /* Serializer of Model          */
    device_t *dev;                                      /* model device                 */
    union {
        void* model_proto;                              /* model model proto            */
        const void* cmodel;                             /* model const model proto      */
    };
    uint32_t model_size;                                /* model model size             */
    hashmap_t *tensor_map;                              /* model tensor map             */
};

EVO_API model_t * model_new(const char*);
EVO_API tensor_t* model_get_tensor(model_t*, const char*);
EVO_API void model_dump_tensor(model_t*);
EVO_API void model_free(model_t*);

// ==================================================================================== //
//                                       evo: predictor
// ==================================================================================== //

struct predictor {
    int n_pes;                                          /* Number of Process Engine     */
    int pe_fp32s;                                       /* FP32 Operator per PE         */
    int fp32_cycles;                                    /* Cycles per FP32 Operator     */
    int batch_size;                                     /* Size of Batch                */
    double frequency;                                   /* Frequency per PE(Hz)         */
    double mem_bandwidth;                               /* Memory Bandwidth(B/s)        */
    double mem_fp32_bandwidth;                          /* Memory Bandwidth(FP32/s)     */
    double l2_fp32_bandwidth;                           /* L2 Cache Bandwidth(FP32/s)   */
    double memory_efficiency;                           /* Memory Efficiency            */
    double mem_concurrent_fp32;                         /* Parallel Memory Bandwidth    */
    double launch_time;                                 /* Launch Time(ms)              */
    uint8_t is_max_mode : 1;                            /* Latency Mode: 0ADD/1MAX      */
};

// ==================================================================================== //
//                                       evo: profiler type
// ==================================================================================== //

typedef enum profiler_type {
    PROFILER_TYPE_CUSTOM = 0,
    PROFILER_TYPE_EXEC = 1,
} profiler_type_t;

// ==================================================================================== //
//                                       evo: profiler
// ==================================================================================== //

struct profiler {
    profiler_type_t type;                               /* Profiler Type                */
    void (*report)(profiler_t*, int);                   /* Profiler Report func         */
    union {
        struct {                                        /* when type == EXEC            */
            int exec_node_idx;                          /* EXEC|Node Index of exec      */
            int exec_nnode;                             /* EXEC|Node Number of exec     */
            node_vec_t exec_node_vec;                   /* EXEC|Node vector of exec     */
            double * exec_time_vec;                     /* EXEC|0..nnode-1 + nnode:sum  */
        };
        void * custom;                                  /* when type == CUSTOM          */
    };
};

EVO_API profiler_t * profiler_new(profiler_type_t);
EVO_API void profiler_report(profiler_t*, int);
EVO_API void profiler_free(profiler_t*);

// ==================================================================================== //
//                                       evo: serializer
// ==================================================================================== //

struct serializer {
    const char* fmt;                                    /* Serializer format name       */

    model_t * (*load) (struct serializer*, const void *, size_t);
    model_t * (*load_model) (struct serializer*, const char*);
    tensor_t * (*load_tensor) (const char*);
    graph_t* (*load_graph) (model_t*);                  /* Serializer load mdl to graph */
    void (*unload) (model_t*);                          /* Serializer unload model      */
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
    void (*alloc)(device_t*, graph_t*);                 /* Alloc resource               */
    void (*release)(device_t*, graph_t*);               /* Release all allocated        */
};


// ==================================================================================== //
//                                  evo: optimizer (device)
// ==================================================================================== //

struct optimizer {
    void (*graph_spilte)(graph_t*);
    void (*graph_optimize)(graph_t*);
};

// ==================================================================================== //
//                                  evo: Vision (vis)
// ==================================================================================== //

// ==================================================================================== //
//                                  evo: typedef (vis)
// ==================================================================================== //

typedef struct font font_t;
typedef struct image image_t;
typedef struct canvas canvas_t;
typedef struct renderer renderer_t;
typedef struct rectangle rectangle_t;
typedef canvas_t* (*render_fn_t)(canvas_t*, float);

// ==================================================================================== //
//                                  evo: image type (vis)
// ==================================================================================== //

typedef enum image_type {
    IMAGE_TYPE_UNKNOWN,
    IMAGE_TYPE_BMP,
    IMAGE_TYPE_PNG,
    IMAGE_TYPE_JPG,
    IMAGE_TYPE_TGA,
    IMAGE_TYPE_HDR,
    IMAGE_TYPE_GIF,
} image_type_t;

// ==================================================================================== //
//                                  evo: image (vis)
// ==================================================================================== //

struct image {
    char* name;
    image_type_t type;
    tensor_t *raw;
    attribute_vec_t attr_vec;
};

EVO_API image_t* image_from_tensor(tensor_t*);
EVO_API image_t* image_heatmap(tensor_t*,int);
EVO_API image_t* image_blank(const char*, size_t, size_t);
EVO_API image_t* image_load(const char*);
EVO_API image_t* image_load_mnist(const char*, const char*);
EVO_API image_t* image_load_cifar10(const char*, int);
EVO_API image_t* image_channel(image_t*, int);
EVO_API image_t* image_extract_channel(image_t*, int);
EVO_API int image_width(image_t*);
EVO_API int image_height(image_t*);
EVO_API void image_save_grey(image_t*, const char*, int);
EVO_API void image_save(image_t*, const char*);
EVO_API char* image_dump_shape(image_t*);
EVO_API void image_dump_raw(image_t*, int);
EVO_API image_t* image_get(image_t*, int);
EVO_API image_t* image_get_batch(image_t*, int, int*);
EVO_API tensor_t* image_get_raw(image_t*, int);
EVO_API tensor_t* image_get_raw_batch(image_t*, int, int*);
EVO_API void image_crop_center(image_t*, int, int);
EVO_API image_t* image_copy(image_t*);
EVO_API void image_resize(image_t*, int, int);
EVO_API image_t* image_merge(image_t*, image_t*, float);
EVO_API void image_push(image_t*, image_t*);
EVO_API attribute_t* image_get_attr(image_t*, const char*);
EVO_API void image_set_deloys(image_t*, int64_t*, int);
EVO_API void image_free(image_t*);

// ==================================================================================== //
//                                  evo: canvas (vis)
// ==================================================================================== //

struct canvas {
    image_t* background;
    uint32_t* pixels;
    size_t width;
    size_t height;
};

#define pixel_red(color) (((color) & 0x000000FF) >> (8 * 0))
#define pixel_green(color) (((color) & 0x0000FF00) >> (8 * 1))
#define pixel_blue(color) (((color) & 0x00FF0000) >> (8 * 2))
#define pixel_alpha(color) (((color) & 0xFF000000) >> (8 * 3))
#define pixel_rgba(r, g, b, a) ((((r) & 0xFF) << (8 * 0)) | (((g) & 0xFF) << (8 * 1)) | (((b) & 0xFF) << (8 * 2)) | (((a) & 0xFF) << (8 * 3)))
#define canvas_pixel(cav, x, y) ((cav)->pixels)[(x) + (cav->width)*(y)]

EVO_API canvas_t* canvas_new(size_t, size_t);
EVO_API canvas_t* canvas_sub_new(canvas_t*, int, int, int, int);
EVO_API canvas_t* canvas_from_image(image_t*);
EVO_API void canvas_export(canvas_t*, const char*);
EVO_API uint32_t* canvas_get(canvas_t*, size_t, size_t);
EVO_API void canvas_fill(canvas_t*, uint32_t);
EVO_API void canvas_blend(uint32_t*, uint32_t);
EVO_API uint32_t color_mix_heat(float);
EVO_API uint32_t color_mix(uint32_t, uint32_t, float);
EVO_API uint32_t color_mix2(uint32_t, uint32_t, int, int);
EVO_API uint32_t color_mix3(uint32_t, uint32_t, uint32_t, int, int, int);
EVO_API bool canvas_is_in_bound(canvas_t*, int, int);
EVO_API bool canvas_barycentric(int, int, int, int, int, int, int, int, int*, int*, int*);
EVO_API void canvas_line(canvas_t*, int, int, int, int, uint32_t);
EVO_API bool canvas_normalize_rectangle(canvas_t*, int, int, int, int, rectangle_t*);
EVO_API bool canvas_normalize_triangle(canvas_t*, int, int, int, int, int, int, int*, int*, int*, int*);
EVO_API void canvas_draw(canvas_t*, int , int, int, int, uint32_t*);
EVO_API void canvas_rectangle(canvas_t*, int, int, int, int, uint32_t);
EVO_API void canvas_rectangle_c2(canvas_t*, int, int, int, int, uint32_t, uint32_t);
EVO_API void canvas_frame(canvas_t*, int, int, int, int, size_t, uint32_t);
EVO_API void canvas_triangle_3c(canvas_t*, int, int, int, int, int, int, uint32_t, uint32_t, uint32_t);
EVO_API void canvas_triangle_3z(canvas_t*, int, int, int, int, int, int, float, float, float);
EVO_API void canvas_triangle_3uv(canvas_t*, int, int, int, int, int, int, float, float, float, float, float, float, float, float, float, canvas_t*);
EVO_API void canvas_triangle(canvas_t*, int, int, int, int, int, int, uint32_t);
EVO_API void canvas_ellipse(canvas_t*, int, int, int, int, uint32_t);
EVO_API void canvas_circle(canvas_t*, int, int, int, uint32_t);
EVO_API void canvas_text(canvas_t*, const char*, int, int, font_t*, size_t, uint32_t);
EVO_API void canvas_free(canvas_t*);

// ==================================================================================== //
//                                  evo: font (vis)
// ==================================================================================== //

struct font {
    const char *glyphs;
    size_t height;
    size_t width;
};

extern font_t default_font;                             /* FONT|Default font for evo    */

// ==================================================================================== //
//                                  evo: rectangle (vis)
// ==================================================================================== //

struct rectangle {
    int x1, y1;
    int x2, y2;
};

// ==================================================================================== //
//                                  evo: renderer type (vis)
// ==================================================================================== //

typedef enum renderer_type {
    RENDERER_TYPE_GIF,
#if defined(EVO_GUI_ENB)
#if defined(__linux__) && !defined(__ANDROID__)
    RENDERER_TYPE_LINUX,
#elif defined(__ANDROID__)
    RENDERER_TYPE_ANDROID,
#elif defined(__IOS__)
    RENDERER_TYPE_IOS,
#elif defined(__MACOS__)
    RENDERER_TYPE_OSX,
#elif defined(_WIN32)
    RENDERER_TYPE_WIN,
#endif  // gui platform
#endif  // EVO_GUI_ENB
} renderer_type_t;

// ==================================================================================== //
//                                  evo: renderer (vis)
// ==================================================================================== //

struct renderer {
    int width, height;
    renderer_type_t type;
    void *priv;
    void (*render)(struct renderer*, render_fn_t);
};

EVO_API renderer_t* renderer_new(int, int, renderer_type_t);
EVO_API void renderer_run(renderer_t*, render_fn_t);
EVO_API void renderer_free(renderer_t*);
#if defined(EVO_GUI_ENB)
EVO_API int renderer_should_close(renderer_t*);
#endif  // EVO_GUI_ENB


#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __EVO_EVO_H__