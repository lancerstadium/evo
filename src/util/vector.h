#ifndef UTIL_VECTOR_H
#define UTIL_VECTOR_H

#include "gtype.h"
#include <stdio.h>

// We want at least 20 vector element spaces in reserve before having
// to reallocate memory again
#define VECTOR_ELEMENT_INCREMENT 20

enum
{
    VECTOR_FLAG_PEEK_DECREMENT = 0b00000001
};

typedef struct vector
{
    void* data;
    // The pointer index is the index that will be read next upon calling "vector_peek".
    // This index will then be incremented
    int pindex;
    int rindex;
    int mindex;
    int count;
    int flags;
    size_t esize;


    // Vector of struct vector, holds saves of this vector. YOu can save the internal state
    // at all times with vector_save
    // Data is not restored and is permenant, save does not respect data, only pointers
    // and variables are saved. Useful to temporarily push the vector state
    // and restore it later.
    struct vector* saves;
} Vector;


Vector* vector_create(size_t esize);
void vector_free(Vector* vector);
void* vector_at(Vector* vector, int index);
void* vector_peek_ptr_at(Vector* vector, int index);
void* vector_peek_no_increment(Vector* vector);
void* vector_peek(Vector* vector);
void *vector_peek_at(Vector* vector, int index);
void vector_set_flag(Vector* vector, int flag);
void vector_unset_flag(Vector* vector, int flag);

/**
 * Pops off the last peeked element
 */
void vector_pop_last_peek(Vector* vector);

/**
 * Peeks into the vector of pointers, returning the pointer value its self
 * 
 * Use this function instead of vector_peek if this is a vector of pointers
 */
void* vector_peek_ptr(Vector* vector);
void vector_set_peek_pointer(Vector* vector, int index);
void vector_set_peek_pointer_end(Vector* vector);
void vector_push(Vector* vector, void* elem);
void vector_push_at(Vector* vector, int index, void *ptr);
void vector_pop(Vector* vector);
void vector_peek_pop(Vector* vector);

void* vector_back(Vector* vector);
void *vector_back_or_null(Vector*vector);

void* vector_back_ptr(Vector* vector);
void* vector_back_ptr_or_null(Vector* vector);

/**
 * Returns the vector data as a char pointer 
 */
const char* vector_string(Vector* vec);

/**
 * Returns true if this vector is empty
 */
bool vector_empty(Vector* vector);
void vector_clear(Vector* vector);

int vector_count(Vector* vector);
/**
 * freads from the file directly into the vector
 */
int vector_fread(Vector* vector, int amount, FILE* fp);
/**
 * Returns a void pointer pointing to the data of this vector
 */
void* vector_data_ptr(Vector* vector);

int vector_insert(Vector* vector_dst, Vector*vector_src, int dst_index);

/**
 * Pops the element at the given data address.
 * \param vector The vector to pop an element on
 * \param address The address that is part of the vector->data range to pop off.
 * \return Returns the index that we popped off.
 */
int vector_pop_at_data_address(Vector* vector, void* address);

/**
 * Pops the given value from the vector. Only the first value found is popped
 */
int vector_pop_value(Vector* vector, void* val);

void vector_pop_at(Vector* vector, int index);

/**
 * Decrements the peek pointer so that the next peek
 * will point at the last peeked token
 */
void vector_peek_back(Vector* vector);

/**
 * Returns the current index that a vector_push would push too
 */
int vector_current_index(Vector* vector);

/**
 * Saves the state of the vector
 */
void vector_save(Vector* vector);
/**
 * Restores the state of the vector
 */
void vector_restore(Vector* vector);

/**
 * Purges the last save state as if it never happend
 */
void vector_save_purge(Vector* vector);



/**
 * Returns the element size per element in this vector
 */
size_t vector_element_size(Vector* vector);


/**
 * Clones the given vector including all vector data, saves are ignored
 */
Vector* vector_clone(Vector* vector);


#endif  // UTIL_VECTOR_H