#ifndef ETM_BUILDER_H
#define ETM_BUILDER_H

/* Generated by flatcc 0.6.2 FlatBuffers schema compiler for C by dvide.com */

#ifndef ETM_READER_H
#include "etm_reader.h"
#endif
#ifndef FLATBUFFERS_COMMON_BUILDER_H
#include "flatbuffers_common_builder.h"
#endif
#include "flatcc/flatcc_prologue.h"
#ifndef flatbuffers_identifier
#define flatbuffers_identifier 0
#endif
#ifndef flatbuffers_extension
#define flatbuffers_extension "bin"
#endif

#define __etm_TensorType_formal_args , etm_TensorType_enum_t v0
#define __etm_TensorType_call_args , v0
__flatbuffers_build_scalar(flatbuffers_, etm_TensorType, etm_TensorType_enum_t)
#define __etm_AttributeType_formal_args , etm_AttributeType_enum_t v0
#define __etm_AttributeType_call_args , v0
__flatbuffers_build_scalar(flatbuffers_, etm_AttributeType, etm_AttributeType_enum_t)
#define __etm_NodeType_formal_args , etm_NodeType_enum_t v0
#define __etm_NodeType_call_args , v0
__flatbuffers_build_scalar(flatbuffers_, etm_NodeType, etm_NodeType_enum_t)
#define __etm_OpType_formal_args , etm_OpType_enum_t v0
#define __etm_OpType_call_args , v0
__flatbuffers_build_scalar(flatbuffers_, etm_OpType, etm_OpType_enum_t)

typedef flatbuffers_union_ref_t etm_GraphInfo_union_ref_t;
typedef flatbuffers_union_vec_ref_t etm_GraphInfo_union_vec_ref_t;
static etm_GraphInfo_union_ref_t etm_GraphInfo_clone(flatbuffers_builder_t *B, etm_GraphInfo_union_t t);

static const flatbuffers_voffset_t __etm_Tensor_required[] = { 0 };
typedef flatbuffers_ref_t etm_Tensor_ref_t;
static etm_Tensor_ref_t etm_Tensor_clone(flatbuffers_builder_t *B, etm_Tensor_table_t t);
__flatbuffers_build_table(flatbuffers_, etm_Tensor, 15)

static const flatbuffers_voffset_t __etm_Attribute_required[] = { 0 };
typedef flatbuffers_ref_t etm_Attribute_ref_t;
static etm_Attribute_ref_t etm_Attribute_clone(flatbuffers_builder_t *B, etm_Attribute_table_t t);
__flatbuffers_build_table(flatbuffers_, etm_Attribute, 11)

static const flatbuffers_voffset_t __etm_Node_required[] = { 0 };
typedef flatbuffers_ref_t etm_Node_ref_t;
static etm_Node_ref_t etm_Node_clone(flatbuffers_builder_t *B, etm_Node_table_t t);
__flatbuffers_build_table(flatbuffers_, etm_Node, 8)

static const flatbuffers_voffset_t __etm_PGraphInfo_required[] = { 0 };
typedef flatbuffers_ref_t etm_PGraphInfo_ref_t;
static etm_PGraphInfo_ref_t etm_PGraphInfo_clone(flatbuffers_builder_t *B, etm_PGraphInfo_table_t t);
__flatbuffers_build_table(flatbuffers_, etm_PGraphInfo, 5)

static const flatbuffers_voffset_t __etm_SGraphInfo_required[] = { 0 };
typedef flatbuffers_ref_t etm_SGraphInfo_ref_t;
static etm_SGraphInfo_ref_t etm_SGraphInfo_clone(flatbuffers_builder_t *B, etm_SGraphInfo_table_t t);
__flatbuffers_build_table(flatbuffers_, etm_SGraphInfo, 5)

static const flatbuffers_voffset_t __etm_Graph_required[] = { 0 };
typedef flatbuffers_ref_t etm_Graph_ref_t;
static etm_Graph_ref_t etm_Graph_clone(flatbuffers_builder_t *B, etm_Graph_table_t t);
__flatbuffers_build_table(flatbuffers_, etm_Graph, 6)

static const flatbuffers_voffset_t __etm_Op_required[] = { 0 };
typedef flatbuffers_ref_t etm_Op_ref_t;
static etm_Op_ref_t etm_Op_clone(flatbuffers_builder_t *B, etm_Op_table_t t);
__flatbuffers_build_table(flatbuffers_, etm_Op, 4)

static const flatbuffers_voffset_t __etm_Model_required[] = { 0 };
typedef flatbuffers_ref_t etm_Model_ref_t;
static etm_Model_ref_t etm_Model_clone(flatbuffers_builder_t *B, etm_Model_table_t t);
__flatbuffers_build_table(flatbuffers_, etm_Model, 3)

#define __etm_Tensor_formal_args ,\
  etm_TensorType_enum_t v0, flatbuffers_string_ref_t v1, int32_t v2, flatbuffers_int32_vec_ref_t v3,\
  flatbuffers_uint8_vec_ref_t v4, uint64_t v5, uint8_t v6, uint32_t v7,\
  int16_t v8, flatbuffers_bool_t v9, flatbuffers_bool_t v10, flatbuffers_bool_t v11,\
  flatbuffers_bool_t v12, flatbuffers_bool_t v13, uint16_t v14
#define __etm_Tensor_call_args ,\
  v0, v1, v2, v3,\
  v4, v5, v6, v7,\
  v8, v9, v10, v11,\
  v12, v13, v14
static inline etm_Tensor_ref_t etm_Tensor_create(flatbuffers_builder_t *B __etm_Tensor_formal_args);
__flatbuffers_build_table_prolog(flatbuffers_, etm_Tensor, etm_Tensor_file_identifier, etm_Tensor_type_identifier)

#define __etm_Attribute_formal_args ,\
  flatbuffers_string_ref_t v0, etm_AttributeType_enum_t v1, float v2, int64_t v3,\
  flatbuffers_string_ref_t v4, flatbuffers_float_vec_ref_t v7, flatbuffers_int64_vec_ref_t v8
#define __etm_Attribute_call_args ,\
  v0, v1, v2, v3,\
  v4, v7, v8
static inline etm_Attribute_ref_t etm_Attribute_create(flatbuffers_builder_t *B __etm_Attribute_formal_args);
__flatbuffers_build_table_prolog(flatbuffers_, etm_Attribute, etm_Attribute_file_identifier, etm_Attribute_type_identifier)

#define __etm_Node_formal_args , flatbuffers_string_ref_t v0, uint16_t v1, etm_NodeType_enum_t v2, etm_Attribute_vec_ref_t v7
#define __etm_Node_call_args , v0, v1, v2, v7
static inline etm_Node_ref_t etm_Node_create(flatbuffers_builder_t *B __etm_Node_formal_args);
__flatbuffers_build_table_prolog(flatbuffers_, etm_Node, etm_Node_file_identifier, etm_Node_type_identifier)

#define __etm_PGraphInfo_formal_args ,\
  etm_Graph_vec_ref_t v0, flatbuffers_uint16_vec_ref_t v1, flatbuffers_uint16_vec_ref_t v2, uint16_t v3, uint16_t v4
#define __etm_PGraphInfo_call_args ,\
  v0, v1, v2, v3, v4
static inline etm_PGraphInfo_ref_t etm_PGraphInfo_create(flatbuffers_builder_t *B __etm_PGraphInfo_formal_args);
__flatbuffers_build_table_prolog(flatbuffers_, etm_PGraphInfo, etm_PGraphInfo_file_identifier, etm_PGraphInfo_type_identifier)

#define __etm_SGraphInfo_formal_args , int32_t v0, flatbuffers_uint16_vec_ref_t v1, flatbuffers_uint16_vec_ref_t v2, flatbuffers_uint16_vec_ref_t v3
#define __etm_SGraphInfo_call_args , v0, v1, v2, v3
static inline etm_SGraphInfo_ref_t etm_SGraphInfo_create(flatbuffers_builder_t *B __etm_SGraphInfo_formal_args);
__flatbuffers_build_table_prolog(flatbuffers_, etm_SGraphInfo, etm_SGraphInfo_file_identifier, etm_SGraphInfo_type_identifier)

#define __etm_Graph_formal_args ,\
  etm_Tensor_vec_ref_t v0, etm_Node_vec_ref_t v1, uint16_t v2, flatbuffers_bool_t v3, etm_GraphInfo_union_ref_t v5
#define __etm_Graph_call_args ,\
  v0, v1, v2, v3, v5
static inline etm_Graph_ref_t etm_Graph_create(flatbuffers_builder_t *B __etm_Graph_formal_args);
__flatbuffers_build_table_prolog(flatbuffers_, etm_Graph, etm_Graph_file_identifier, etm_Graph_type_identifier)

#define __etm_Op_formal_args , etm_OpType_enum_t v0, flatbuffers_string_ref_t v1, flatbuffers_int32_vec_ref_t v2, flatbuffers_int32_vec_ref_t v3
#define __etm_Op_call_args , v0, v1, v2, v3
static inline etm_Op_ref_t etm_Op_create(flatbuffers_builder_t *B __etm_Op_formal_args);
__flatbuffers_build_table_prolog(flatbuffers_, etm_Op, etm_Op_file_identifier, etm_Op_type_identifier)

#define __etm_Model_formal_args , flatbuffers_string_ref_t v0, etm_Op_vec_ref_t v1, flatbuffers_string_vec_ref_t v2
#define __etm_Model_call_args , v0, v1, v2
static inline etm_Model_ref_t etm_Model_create(flatbuffers_builder_t *B __etm_Model_formal_args);
__flatbuffers_build_table_prolog(flatbuffers_, etm_Model, etm_Model_file_identifier, etm_Model_type_identifier)

static inline etm_GraphInfo_union_ref_t etm_GraphInfo_as_NONE(void)
{ etm_GraphInfo_union_ref_t uref; uref.type = etm_GraphInfo_NONE; uref.value = 0; return uref; }
static inline etm_GraphInfo_union_ref_t etm_GraphInfo_as_PGraphInfo(etm_PGraphInfo_ref_t ref)
{ etm_GraphInfo_union_ref_t uref; uref.type = etm_GraphInfo_PGraphInfo; uref.value = ref; return uref; }
static inline etm_GraphInfo_union_ref_t etm_GraphInfo_as_SGraphInfo(etm_SGraphInfo_ref_t ref)
{ etm_GraphInfo_union_ref_t uref; uref.type = etm_GraphInfo_SGraphInfo; uref.value = ref; return uref; }
__flatbuffers_build_union_vector(flatbuffers_, etm_GraphInfo)

static etm_GraphInfo_union_ref_t etm_GraphInfo_clone(flatbuffers_builder_t *B, etm_GraphInfo_union_t u)
{
    switch (u.type) {
    case 1: return etm_GraphInfo_as_PGraphInfo(etm_PGraphInfo_clone(B, (etm_PGraphInfo_table_t)u.value));
    case 2: return etm_GraphInfo_as_SGraphInfo(etm_SGraphInfo_clone(B, (etm_SGraphInfo_table_t)u.value));
    default: return etm_GraphInfo_as_NONE();
    }
}

__flatbuffers_build_scalar_field(0, flatbuffers_, etm_Tensor_type, etm_TensorType, etm_TensorType_enum_t, 4, 4, INT32_C(0), etm_Tensor)
__flatbuffers_build_string_field(1, flatbuffers_, etm_Tensor_name, etm_Tensor)
__flatbuffers_build_scalar_field(2, flatbuffers_, etm_Tensor_index, flatbuffers_int32, int32_t, 4, 4, INT32_C(0), etm_Tensor)
__flatbuffers_build_vector_field(3, flatbuffers_, etm_Tensor_dims, flatbuffers_int32, int32_t, etm_Tensor)
__flatbuffers_build_vector_field(4, flatbuffers_, etm_Tensor_datas, flatbuffers_uint8, uint8_t, etm_Tensor)
__flatbuffers_build_scalar_field(5, flatbuffers_, etm_Tensor_ndata, flatbuffers_uint64, uint64_t, 8, 8, UINT64_C(0), etm_Tensor)
__flatbuffers_build_scalar_field(6, flatbuffers_, etm_Tensor_szElem, flatbuffers_uint8, uint8_t, 1, 1, UINT8_C(0), etm_Tensor)
__flatbuffers_build_scalar_field(7, flatbuffers_, etm_Tensor_nElem, flatbuffers_uint32, uint32_t, 4, 4, UINT32_C(0), etm_Tensor)
__flatbuffers_build_scalar_field(8, flatbuffers_, etm_Tensor_pNode, flatbuffers_int16, int16_t, 2, 2, INT16_C(0), etm_Tensor)
__flatbuffers_build_scalar_field(9, flatbuffers_, etm_Tensor_isReshaped, flatbuffers_bool, flatbuffers_bool_t, 1, 1, UINT8_C(0), etm_Tensor)
__flatbuffers_build_scalar_field(10, flatbuffers_, etm_Tensor_isConstant, flatbuffers_bool, flatbuffers_bool_t, 1, 1, UINT8_C(0), etm_Tensor)
__flatbuffers_build_scalar_field(11, flatbuffers_, etm_Tensor_isInput, flatbuffers_bool, flatbuffers_bool_t, 1, 1, UINT8_C(0), etm_Tensor)
__flatbuffers_build_scalar_field(12, flatbuffers_, etm_Tensor_isOutput, flatbuffers_bool, flatbuffers_bool_t, 1, 1, UINT8_C(0), etm_Tensor)
__flatbuffers_build_scalar_field(13, flatbuffers_, etm_Tensor_isIallocated, flatbuffers_bool, flatbuffers_bool_t, 1, 1, UINT8_C(0), etm_Tensor)
__flatbuffers_build_scalar_field(14, flatbuffers_, etm_Tensor_layout, flatbuffers_uint16, uint16_t, 2, 2, UINT16_C(0), etm_Tensor)

static inline etm_Tensor_ref_t etm_Tensor_create(flatbuffers_builder_t *B __etm_Tensor_formal_args)
{
    if (etm_Tensor_start(B)
        || etm_Tensor_ndata_add(B, v5)
        || etm_Tensor_type_add(B, v0)
        || etm_Tensor_name_add(B, v1)
        || etm_Tensor_index_add(B, v2)
        || etm_Tensor_dims_add(B, v3)
        || etm_Tensor_datas_add(B, v4)
        || etm_Tensor_nElem_add(B, v7)
        || etm_Tensor_pNode_add(B, v8)
        || etm_Tensor_layout_add(B, v14)
        || etm_Tensor_szElem_add(B, v6)
        || etm_Tensor_isReshaped_add(B, v9)
        || etm_Tensor_isConstant_add(B, v10)
        || etm_Tensor_isInput_add(B, v11)
        || etm_Tensor_isOutput_add(B, v12)
        || etm_Tensor_isIallocated_add(B, v13)) {
        return 0;
    }
    return etm_Tensor_end(B);
}

static etm_Tensor_ref_t etm_Tensor_clone(flatbuffers_builder_t *B, etm_Tensor_table_t t)
{
    __flatbuffers_memoize_begin(B, t);
    if (etm_Tensor_start(B)
        || etm_Tensor_ndata_pick(B, t)
        || etm_Tensor_type_pick(B, t)
        || etm_Tensor_name_pick(B, t)
        || etm_Tensor_index_pick(B, t)
        || etm_Tensor_dims_pick(B, t)
        || etm_Tensor_datas_pick(B, t)
        || etm_Tensor_nElem_pick(B, t)
        || etm_Tensor_pNode_pick(B, t)
        || etm_Tensor_layout_pick(B, t)
        || etm_Tensor_szElem_pick(B, t)
        || etm_Tensor_isReshaped_pick(B, t)
        || etm_Tensor_isConstant_pick(B, t)
        || etm_Tensor_isInput_pick(B, t)
        || etm_Tensor_isOutput_pick(B, t)
        || etm_Tensor_isIallocated_pick(B, t)) {
        return 0;
    }
    __flatbuffers_memoize_end(B, t, etm_Tensor_end(B));
}

__flatbuffers_build_string_field(0, flatbuffers_, etm_Attribute_name, etm_Attribute)
__flatbuffers_build_scalar_field(1, flatbuffers_, etm_Attribute_type, etm_AttributeType, etm_AttributeType_enum_t, 4, 4, INT32_C(0), etm_Attribute)
__flatbuffers_build_scalar_field(2, flatbuffers_, etm_Attribute_f, flatbuffers_float, float, 4, 4, 0.00000000f, etm_Attribute)
__flatbuffers_build_scalar_field(3, flatbuffers_, etm_Attribute_i, flatbuffers_int64, int64_t, 8, 8, INT64_C(0), etm_Attribute)
__flatbuffers_build_string_field(4, flatbuffers_, etm_Attribute_s, etm_Attribute)
/* Skipping build of deprecated field: 'etm_Attribute_t' */

/* Skipping build of deprecated field: 'etm_Attribute_g' */

__flatbuffers_build_vector_field(7, flatbuffers_, etm_Attribute_fs, flatbuffers_float, float, etm_Attribute)
__flatbuffers_build_vector_field(8, flatbuffers_, etm_Attribute_is, flatbuffers_int64, int64_t, etm_Attribute)
/* Skipping build of deprecated field: 'etm_Attribute_ts' */

/* Skipping build of deprecated field: 'etm_Attribute_gs' */


static inline etm_Attribute_ref_t etm_Attribute_create(flatbuffers_builder_t *B __etm_Attribute_formal_args)
{
    if (etm_Attribute_start(B)
        || etm_Attribute_i_add(B, v3)
        || etm_Attribute_name_add(B, v0)
        || etm_Attribute_type_add(B, v1)
        || etm_Attribute_f_add(B, v2)
        || etm_Attribute_s_add(B, v4)
        || etm_Attribute_fs_add(B, v7)
        || etm_Attribute_is_add(B, v8)) {
        return 0;
    }
    return etm_Attribute_end(B);
}

static etm_Attribute_ref_t etm_Attribute_clone(flatbuffers_builder_t *B, etm_Attribute_table_t t)
{
    __flatbuffers_memoize_begin(B, t);
    if (etm_Attribute_start(B)
        || etm_Attribute_i_pick(B, t)
        || etm_Attribute_name_pick(B, t)
        || etm_Attribute_type_pick(B, t)
        || etm_Attribute_f_pick(B, t)
        || etm_Attribute_s_pick(B, t)
        || etm_Attribute_fs_pick(B, t)
        || etm_Attribute_is_pick(B, t)) {
        return 0;
    }
    __flatbuffers_memoize_end(B, t, etm_Attribute_end(B));
}

__flatbuffers_build_string_field(0, flatbuffers_, etm_Node_name, etm_Node)
__flatbuffers_build_scalar_field(1, flatbuffers_, etm_Node_index, flatbuffers_uint16, uint16_t, 2, 2, UINT16_C(0), etm_Node)
__flatbuffers_build_scalar_field(2, flatbuffers_, etm_Node_type, etm_NodeType, etm_NodeType_enum_t, 4, 4, INT32_C(0), etm_Node)
/* Skipping build of deprecated field: 'etm_Node_in' */

/* Skipping build of deprecated field: 'etm_Node_out' */

/* Skipping build of deprecated field: 'etm_Node_op' */

/* Skipping build of deprecated field: 'etm_Node_graph' */

__flatbuffers_build_table_vector_field(7, flatbuffers_, etm_Node_attrVec, etm_Attribute, etm_Node)

static inline etm_Node_ref_t etm_Node_create(flatbuffers_builder_t *B __etm_Node_formal_args)
{
    if (etm_Node_start(B)
        || etm_Node_name_add(B, v0)
        || etm_Node_type_add(B, v2)
        || etm_Node_attrVec_add(B, v7)
        || etm_Node_index_add(B, v1)) {
        return 0;
    }
    return etm_Node_end(B);
}

static etm_Node_ref_t etm_Node_clone(flatbuffers_builder_t *B, etm_Node_table_t t)
{
    __flatbuffers_memoize_begin(B, t);
    if (etm_Node_start(B)
        || etm_Node_name_pick(B, t)
        || etm_Node_type_pick(B, t)
        || etm_Node_attrVec_pick(B, t)
        || etm_Node_index_pick(B, t)) {
        return 0;
    }
    __flatbuffers_memoize_end(B, t, etm_Node_end(B));
}

__flatbuffers_build_table_vector_field(0, flatbuffers_, etm_PGraphInfo_subVec, etm_Graph, etm_PGraphInfo)
__flatbuffers_build_vector_field(1, flatbuffers_, etm_PGraphInfo_inputInodesVec, flatbuffers_uint16, uint16_t, etm_PGraphInfo)
__flatbuffers_build_vector_field(2, flatbuffers_, etm_PGraphInfo_outputInodesVec, flatbuffers_uint16, uint16_t, etm_PGraphInfo)
__flatbuffers_build_scalar_field(3, flatbuffers_, etm_PGraphInfo_ninputNode, flatbuffers_uint16, uint16_t, 2, 2, UINT16_C(0), etm_PGraphInfo)
__flatbuffers_build_scalar_field(4, flatbuffers_, etm_PGraphInfo_noutputNode, flatbuffers_uint16, uint16_t, 2, 2, UINT16_C(0), etm_PGraphInfo)

static inline etm_PGraphInfo_ref_t etm_PGraphInfo_create(flatbuffers_builder_t *B __etm_PGraphInfo_formal_args)
{
    if (etm_PGraphInfo_start(B)
        || etm_PGraphInfo_subVec_add(B, v0)
        || etm_PGraphInfo_inputInodesVec_add(B, v1)
        || etm_PGraphInfo_outputInodesVec_add(B, v2)
        || etm_PGraphInfo_ninputNode_add(B, v3)
        || etm_PGraphInfo_noutputNode_add(B, v4)) {
        return 0;
    }
    return etm_PGraphInfo_end(B);
}

static etm_PGraphInfo_ref_t etm_PGraphInfo_clone(flatbuffers_builder_t *B, etm_PGraphInfo_table_t t)
{
    __flatbuffers_memoize_begin(B, t);
    if (etm_PGraphInfo_start(B)
        || etm_PGraphInfo_subVec_pick(B, t)
        || etm_PGraphInfo_inputInodesVec_pick(B, t)
        || etm_PGraphInfo_outputInodesVec_pick(B, t)
        || etm_PGraphInfo_ninputNode_pick(B, t)
        || etm_PGraphInfo_noutputNode_pick(B, t)) {
        return 0;
    }
    __flatbuffers_memoize_end(B, t, etm_PGraphInfo_end(B));
}

__flatbuffers_build_scalar_field(0, flatbuffers_, etm_SGraphInfo_idx, flatbuffers_int32, int32_t, 4, 4, INT32_C(0), etm_SGraphInfo)
__flatbuffers_build_vector_field(1, flatbuffers_, etm_SGraphInfo_nodesVec, flatbuffers_uint16, uint16_t, etm_SGraphInfo)
__flatbuffers_build_vector_field(2, flatbuffers_, etm_SGraphInfo_inputItensorsVec, flatbuffers_uint16, uint16_t, etm_SGraphInfo)
__flatbuffers_build_vector_field(3, flatbuffers_, etm_SGraphInfo_outputItensorsVec, flatbuffers_uint16, uint16_t, etm_SGraphInfo)
/* Skipping build of deprecated field: 'etm_SGraphInfo_pgraph' */


static inline etm_SGraphInfo_ref_t etm_SGraphInfo_create(flatbuffers_builder_t *B __etm_SGraphInfo_formal_args)
{
    if (etm_SGraphInfo_start(B)
        || etm_SGraphInfo_idx_add(B, v0)
        || etm_SGraphInfo_nodesVec_add(B, v1)
        || etm_SGraphInfo_inputItensorsVec_add(B, v2)
        || etm_SGraphInfo_outputItensorsVec_add(B, v3)) {
        return 0;
    }
    return etm_SGraphInfo_end(B);
}

static etm_SGraphInfo_ref_t etm_SGraphInfo_clone(flatbuffers_builder_t *B, etm_SGraphInfo_table_t t)
{
    __flatbuffers_memoize_begin(B, t);
    if (etm_SGraphInfo_start(B)
        || etm_SGraphInfo_idx_pick(B, t)
        || etm_SGraphInfo_nodesVec_pick(B, t)
        || etm_SGraphInfo_inputItensorsVec_pick(B, t)
        || etm_SGraphInfo_outputItensorsVec_pick(B, t)) {
        return 0;
    }
    __flatbuffers_memoize_end(B, t, etm_SGraphInfo_end(B));
}

__flatbuffers_build_table_vector_field(0, flatbuffers_, etm_Graph_tensors, etm_Tensor, etm_Graph)
__flatbuffers_build_table_vector_field(1, flatbuffers_, etm_Graph_nodes, etm_Node, etm_Graph)
__flatbuffers_build_scalar_field(2, flatbuffers_, etm_Graph_dataLayout, flatbuffers_uint16, uint16_t, 2, 2, UINT16_C(0), etm_Graph)
__flatbuffers_build_scalar_field(3, flatbuffers_, etm_Graph_isSub, flatbuffers_bool, flatbuffers_bool_t, 1, 1, UINT8_C(0), etm_Graph)
__flatbuffers_build_union_field(5, flatbuffers_, etm_Graph_more, etm_GraphInfo, etm_Graph)
__flatbuffers_build_union_table_value_field(flatbuffers_, etm_Graph_more, etm_GraphInfo, PGraphInfo, etm_PGraphInfo)
__flatbuffers_build_union_table_value_field(flatbuffers_, etm_Graph_more, etm_GraphInfo, SGraphInfo, etm_SGraphInfo)

static inline etm_Graph_ref_t etm_Graph_create(flatbuffers_builder_t *B __etm_Graph_formal_args)
{
    if (etm_Graph_start(B)
        || etm_Graph_tensors_add(B, v0)
        || etm_Graph_nodes_add(B, v1)
        || etm_Graph_more_add_value(B, v5)
        || etm_Graph_dataLayout_add(B, v2)
        || etm_Graph_isSub_add(B, v3)
        || etm_Graph_more_add_type(B, v5.type)) {
        return 0;
    }
    return etm_Graph_end(B);
}

static etm_Graph_ref_t etm_Graph_clone(flatbuffers_builder_t *B, etm_Graph_table_t t)
{
    __flatbuffers_memoize_begin(B, t);
    if (etm_Graph_start(B)
        || etm_Graph_tensors_pick(B, t)
        || etm_Graph_nodes_pick(B, t)
        || etm_Graph_more_pick(B, t)
        || etm_Graph_dataLayout_pick(B, t)
        || etm_Graph_isSub_pick(B, t)) {
        return 0;
    }
    __flatbuffers_memoize_end(B, t, etm_Graph_end(B));
}

__flatbuffers_build_scalar_field(0, flatbuffers_, etm_Op_type, etm_OpType, etm_OpType_enum_t, 4, 4, INT32_C(0), etm_Op)
__flatbuffers_build_string_field(1, flatbuffers_, etm_Op_name, etm_Op)
__flatbuffers_build_vector_field(2, flatbuffers_, etm_Op_inputIndexes, flatbuffers_int32, int32_t, etm_Op)
__flatbuffers_build_vector_field(3, flatbuffers_, etm_Op_outputIndexs, flatbuffers_int32, int32_t, etm_Op)

static inline etm_Op_ref_t etm_Op_create(flatbuffers_builder_t *B __etm_Op_formal_args)
{
    if (etm_Op_start(B)
        || etm_Op_type_add(B, v0)
        || etm_Op_name_add(B, v1)
        || etm_Op_inputIndexes_add(B, v2)
        || etm_Op_outputIndexs_add(B, v3)) {
        return 0;
    }
    return etm_Op_end(B);
}

static etm_Op_ref_t etm_Op_clone(flatbuffers_builder_t *B, etm_Op_table_t t)
{
    __flatbuffers_memoize_begin(B, t);
    if (etm_Op_start(B)
        || etm_Op_type_pick(B, t)
        || etm_Op_name_pick(B, t)
        || etm_Op_inputIndexes_pick(B, t)
        || etm_Op_outputIndexs_pick(B, t)) {
        return 0;
    }
    __flatbuffers_memoize_end(B, t, etm_Op_end(B));
}

__flatbuffers_build_string_field(0, flatbuffers_, etm_Model_name, etm_Model)
__flatbuffers_build_table_vector_field(1, flatbuffers_, etm_Model_oplists, etm_Op, etm_Model)
__flatbuffers_build_string_vector_field(2, flatbuffers_, etm_Model_tensorName, etm_Model)

static inline etm_Model_ref_t etm_Model_create(flatbuffers_builder_t *B __etm_Model_formal_args)
{
    if (etm_Model_start(B)
        || etm_Model_name_add(B, v0)
        || etm_Model_oplists_add(B, v1)
        || etm_Model_tensorName_add(B, v2)) {
        return 0;
    }
    return etm_Model_end(B);
}

static etm_Model_ref_t etm_Model_clone(flatbuffers_builder_t *B, etm_Model_table_t t)
{
    __flatbuffers_memoize_begin(B, t);
    if (etm_Model_start(B)
        || etm_Model_name_pick(B, t)
        || etm_Model_oplists_pick(B, t)
        || etm_Model_tensorName_pick(B, t)) {
        return 0;
    }
    __flatbuffers_memoize_end(B, t, etm_Model_end(B));
}

#include "flatcc/flatcc_epilogue.h"
#endif /* ETM_BUILDER_H */