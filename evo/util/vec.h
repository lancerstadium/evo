/**
 * =================================================================================== //
 * @file vec.h
 * @author lancerstadium (lancerstadium@163.com)
 * @brief vector_t header file
 * @version 0.1
 * @date 2024-07-06
 * @copyright Copyright (c) 2024
 * =================================================================================== //
 */

// ==================================================================================== //
//                                    evo/util/vec.h
// ==================================================================================== //

#ifndef __EVO_UTIL_VEC_H__
#define __EVO_UTIL_VEC_H__

#ifdef __cpp_decltype
#include <type_traits>
#define typeof(T) std::remove_reference<std::add_lvalue_reference<decltype(T)>::type>::type
#endif

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stdbool.h>
#include <stdlib.h>

// generic type for internal use
typedef void* vector_t;
// number of elements in a vector_t
typedef size_t vec_size_t;
// number of bytes for a type
typedef size_t vec_type_t;

// TODO: more rigorous check for typeof support with different compilers
#if _MSC_VER == 0 || __STDC_VERSION__ >= 202311L || defined __cpp_decltype

// shortcut defines

// vec_addr is a vector_t* (aka type**)
#define vector_add_dst(vec_addr)\
	((typeof(*vec_addr))(\
	    _vector_add_dst((vector_t*)vec_addr, sizeof(**vec_addr))\
	))
#define vector_insert_dst(vec_addr, pos)\
	((typeof(*vec_addr))(\
	    _vector_insert_dst((vector_t*)vec_addr, sizeof(**vec_addr), pos)))

#define vector_add(vec_addr, value)\
	(*vector_add_dst(vec_addr) = value)
#define vector_insert(vec_addr, pos, value)\
	(*vector_insert_dst(vec_addr, pos) = value)

#else

#define vector_add_dst(vec_addr, type)\
	((type*)_vector_add_dst((vector_t*)vec_addr, sizeof(type)))
#define vector_insert_dst(vec_addr, type, pos)\
	((type*)_vector_insert_dst((vector_t*)vec_addr, sizeof(type), pos))

#define vector_add(vec_addr, type, value)\
	(*vector_add_dst(vec_addr, type) = value)
#define vector_insert(vec_addr, type, pos, value)\
	(*vector_insert_dst(vec_addr, type, pos) = value)

#endif

// vec is a vector_t (aka type*)
#define vector_erase(vec, pos, len)\
	(_vector_erase((vector_t)vec, sizeof(*vec), pos, len))
#define vector_remove(vec, pos)\
	(_vector_remove((vector_t)vec, sizeof(*vec), pos))

#define vector_reserve(vec_addr, capacity)\
	(_vector_reserve((vector_t*)vec_addr, sizeof(**vec_addr), capacity))

#define vector_copy(vec)\
	(_vector_copy((vector_t)vec, sizeof(*vec)))

vector_t vector_create(void);

void vector_free(vector_t vec);

void* _vector_add_dst(vector_t* vec_addr, vec_type_t type_size);

void* _vector_insert_dst(vector_t* vec_addr, vec_type_t type_size, vec_size_t pos);

void _vector_erase(vector_t vec_addr, vec_type_t type_size, vec_size_t pos, vec_size_t len);

void _vector_remove(vector_t vec_addr, vec_type_t type_size, vec_size_t pos);

void vector_pop(vector_t vec);

void _vector_reserve(vector_t* vec_addr, vec_type_t type_size, vec_size_t capacity);

vector_t _vector_copy(vector_t vec, vec_type_t type_size);

vec_size_t vector_size(vector_t vec);

vec_size_t vector_capacity(vector_t vec);


#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __EVO_UTIL_VEC_H__