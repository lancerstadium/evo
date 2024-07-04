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



// ==================================================================================== //
//                                       typedef
// ==================================================================================== //


typedef struct op op_t;
typedef struct node node_t;
typedef struct graph graph_t;
typedef struct tensor tensor_t;
typedef struct device device_t;
typedef struct context context_t;
typedef struct scheduler scheduler_t;
typedef struct optimizer optimizer_t;
typedef struct interface interface_t;
typedef struct allocator allocator_t;
typedef enum tensor_type tensor_type_t;

// ==================================================================================== //
//                                       evo: tensor type
// ==================================================================================== //

enum tensor_type {
    EVO_TENSOR_TYPE_UNDEFINED,

    EVO_TENSOR_TYPE_INT8,
    EVO_TENSOR_TYPE_INT16,
    EVO_TENSOR_TYPE_INT32,
    EVO_TENSOR_TYPE_INT64,

    EVO_TENSOR_TYPE_UINT8,
    EVO_TENSOR_TYPE_UINT16,
    EVO_TENSOR_TYPE_UINT32,
    EVO_TENSOR_TYPE_UINT64,

    EVO_TENSOR_TYPE_FLOAT16,
    EVO_TENSOR_TYPE_FLOAT32,
    EVO_TENSOR_TYPE_FLOAT64,
};

// ==================================================================================== //
//                                       evo: tensor
// ==================================================================================== //

struct tensor {
    tensor_type_t type;                                     /* Tensor data type     */
    unsigned int * dims;                                    /* Shape of dim array   */
    unsigned int ndim;                                      /* Valid entry number   */
    void * datas;                                           /* Tensor data addr     */
    unsigned int ndata;                                     /* Tensor data size     */
};

// ==================================================================================== //
//                                       evo: op
// ==================================================================================== //

struct op {
    unsigned int type;                                      /* Operator type        */
    const char* name;                                       /* Operator name        */
    void (*run)(node_t*);                                   /* Operator run fn      */
};

// ==================================================================================== //
//                                       evo: node
// ==================================================================================== //

struct node {

};

// ==================================================================================== //
//                                       evo: graph
// ==================================================================================== //

struct graph {

};

// ==================================================================================== //
//                                       evo: context
// ==================================================================================== //

struct context {
    
};


// ==================================================================================== //
//                                       evo: scheduler
// ==================================================================================== //

struct scheduler {
    const char* name;                                       /* Scheduler name       */
    void (*prerun)(scheduler_t*, graph_t*);                 /* Scheduler pre run fn */
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
    void (*alloc)(device_t*, graph_t*);                     /* Alloc resource       */
    void (*release)(device_t*, graph_t*);                   /* release all allocated*/
};


// ==================================================================================== //
//                                       evo: device
// ==================================================================================== //

struct device {
    const char* name;
    interface_t* itf;
    allocator_t* act;
    optimizer_t* opt;
    scheduler_t* scd;
};

void device_init(device_t*, const char* name);



#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __EVO_EVO_H__