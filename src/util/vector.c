
#include "vector.h"
#include <stddef.h>
#include <memory.h>
#include <stdlib.h>
#include <assert.h>

static bool vector_in_bounds_for_at(Vector* vector, int index)
{
    return (index >= 0 && index < vector->rindex);
}

static bool vector_in_bounds_for_pop(Vector* vector, int index)
{
    return (index >= 0 && index < vector->mindex);
}

static void vector_assert_bounds_for_pop(Vector* vector, int index)
{
    assert(vector_in_bounds_for_pop(vector, index));
}

Vector*vector_create_no_saves(size_t esize)
{
    Vector* vector = calloc(sizeof(Vector), 1);
    vector->data = malloc(esize * VECTOR_ELEMENT_INCREMENT);
    vector->mindex = VECTOR_ELEMENT_INCREMENT;
    vector->rindex = 0;
    vector->pindex = 0;
    vector->esize = esize;
    vector->count = 0;
    return vector;
}

size_t vector_total_size(Vector* vector)
{
    return vector->count * vector->esize;
}

size_t vector_element_size(Vector* vector)
{
    return vector->esize;
}

Vector*vector_clone(Vector* vector)
{
    void *new_data_address = calloc(vector->esize, vector->count + VECTOR_ELEMENT_INCREMENT);
    memcpy(new_data_address, vector->data, vector_total_size(vector));
    Vector*new_vec = calloc(sizeof(Vector), 1);
    memcpy(new_vec, vector, sizeof(Vector));
    new_vec->data = new_data_address;

    // Saves are not cloned with vector_clone yet.
    // assert(vector->saves == NULL);
    return new_vec;
}

Vector*vector_create(size_t esize)
{
    Vector*vec = vector_create_no_saves(esize);
    vec->saves = vector_create_no_saves(sizeof(Vector));
    return vec;
}

void vector_free(Vector* vector)
{
    if(!vector) return;
    free(vector->data);
    free(vector);
}

int vector_current_index(Vector* vector)
{
    return vector->rindex;
}

void vector_resize_for_index(Vector* vector, int start_index, int total_elements)
{
    if (start_index + total_elements < vector->mindex)
    {
        // Nothing to resize
        return;
    }

    vector->data = realloc(vector->data, ((start_index + total_elements + VECTOR_ELEMENT_INCREMENT) * vector->esize));
    assert(vector->data);
    vector->mindex = start_index + total_elements;
}

void vector_resize_for(Vector* vector, int total_elements)
{
    vector_resize_for_index(vector, vector->rindex, total_elements);
}

void vector_resize(Vector* vector)
{
    // Zero elements as we want to see if we have already overflowed the index
    vector_resize_for(vector, 0);
}

void *vector_at(Vector* vector, int index)
{
    return vector->data + (index * vector->esize);
}

void vector_set_peek_pointer(Vector* vector, int index)
{
    vector->pindex = index;
}

void vector_set_peek_pointer_end(Vector* vector)
{
    vector_set_peek_pointer(vector, vector->rindex - 1);
}

void *vector_peek_at(Vector* vector, int index)
{
    if (!vector_in_bounds_for_at(vector, index))
    {
        return NULL;
    }

    void* ptr = vector_at(vector, index);
    return ptr;
}

void *vector_peek_no_increment(Vector* vector)
{
    if (!vector_in_bounds_for_at(vector, vector->pindex))
    {
        return NULL;
    }

    void *ptr = vector_at(vector, vector->pindex);
    return ptr;
}

void vector_peek_back(Vector* vector)
{
    vector->pindex--;
}

void *vector_peek(Vector* vector)
{
    void *ptr = vector_peek_no_increment(vector);
    if (!ptr)
    {
        return NULL;
    }

    if (vector->flags & VECTOR_FLAG_PEEK_DECREMENT)
        vector->pindex--;
    else
        vector->pindex++;

    return ptr;
}

void vector_set_flag(Vector* vector, int flag)
{
    vector->flags |= flag;
}

void vector_unset_flag(Vector* vector, int flag)
{
    vector->flags &= ~flag;
}

void *vector_peek_ptr(Vector* vector)
{
    void **ptr = vector_peek(vector);
    if (!ptr)
    {
        return NULL;
    }

    return *ptr;
}

void *vector_peek_ptr_at(Vector* vector, int index)
{
    if (index < 0 || index > vector->count)
    {
        return NULL;
    }

    void **ptr = vector_at(vector, index);
    if (!ptr)
    {
        return NULL;
    }

    return *ptr;
}

void *vector_back_ptr(Vector* vector)
{
    void **ptr = vector_back(vector);
    if (!ptr)
    {
        return NULL;
    }

    return *ptr;
}

void vector_save(Vector* vector)
{
    // Let's save the state of this vector to its self
    Vector tmp_vec = *vector;
    // We not allowed to modify the saves so set it to NULL
    // when we push it to the save stack.
    tmp_vec.saves = NULL;
    vector_push(vector->saves, &tmp_vec);
}

void vector_restore(Vector* vector)
{
    Vector save_vec = *((Vector*)(vector_back(vector->saves)));
    save_vec.saves = vector->saves;
    *vector = save_vec;
    vector_pop(vector->saves);
}

void vector_save_purge(Vector* vector)
{
    vector_pop(vector->saves);
}

void vector_pop_last_peek(Vector* vector)
{
    assert(vector->pindex >= 1);
    vector_pop_at(vector, vector->pindex-1);
}

void vector_push(Vector* vector, void *elem)
{
    void *ptr = vector_at(vector, vector->rindex);
    memcpy(ptr, elem, vector->esize);

    vector->rindex++;
    vector->count++;

    if (vector->rindex >= vector->mindex)
    {
        vector_resize(vector);
    }
}

int vector_fread(Vector* vector, int amount, FILE *fp)
{
    size_t read_amount = fread(vector->data, 1, 1, fp);
    while (read_amount)
    {
        vector_push(vector, &read_amount);
        read_amount = fread(vector->data, 1, 1, fp);
    }

    return 0;
}

const char *vector_string(Vector*vec)
{
    return vec->data;
}

void *vector_data_end(Vector* vector)
{
    return vector->data + vector->rindex * vector->esize;
}

size_t vector_elements_left(Vector* vector, int index)
{
    return vector->count - index;
}

int vector_elements_until_end(Vector* vector, int index)
{
    return vector->count - index;
}

void vector_shift_right_in_bounds_no_increment(Vector* vector, int index, int amount)
{
    vector_resize_for_index(vector, index, amount);
    int eindex = (index + amount);
    size_t bytes_to_move = vector_elements_until_end(vector, index) * vector->esize;
    memcpy(vector_at(vector, eindex), vector_at(vector, index), bytes_to_move);
    memset(vector_at(vector, index), 0x00, amount * vector->esize);
}

void vector_shift_right_in_bounds(Vector* vector, int index, int amount)
{
    vector_shift_right_in_bounds_no_increment(vector, index, amount);
    vector->rindex += amount;
    vector->count += amount;
}

void vector_stretch(Vector* vector, int index)
{
    if (index < vector->rindex)
        return;

    vector_resize_for_index(vector, index, 0);
    vector->count = index;
    vector->rindex = index;
}

int vector_pop_value(Vector* vector, void* val)
{
    int old_pp = vector->pindex;
    vector_set_peek_pointer(vector, 0);
    void* ptr = vector_peek_ptr(vector);
    int index = 0;
    while(ptr)
    {
        if (ptr == val)
        {
            vector_pop_at(vector, index);
            break;
        }
        ptr = vector_peek_ptr(vector);
        index++;
    }

    vector_set_peek_pointer(vector, old_pp);
}

int vector_pop_at_data_address(Vector* vector, void *address)
{
    int index = (address - vector->data) / vector->esize;
    vector_pop_at(vector, index);
    return index;
}

void vector_shift_right(Vector* vector, int index, int amount)
{
    if (index < vector->rindex)
    {
        vector_shift_right_in_bounds(vector, index, amount);
        return;
    }

    // We don't need to shift anything because we are out of bounds
    // lets stretch the vector up to index+amount
    vector_stretch(vector, index + amount);
    vector_shift_right_in_bounds_no_increment(vector, index, amount);
}

void vector_pop_at(Vector* vector, int index)
{
    void *dst_pos = vector_at(vector, index);
    void *next_element_pos = dst_pos + vector->esize;
    void *end_pos = vector_data_end(vector);
    size_t total = (size_t)end_pos - (size_t)next_element_pos;
    memcpy(dst_pos, next_element_pos, total);
    vector->count -= 1;
    vector->rindex -= 1;
}

void vector_peek_pop(Vector* vector)
{
    // Popping at a peek is an akward one
    // we will need to shift all the elements to the left, annoying...
    // This will also invalidate any pointers pointing directly to the vector data
    vector_pop_at(vector, vector->pindex);
}

void vector_push_multiple_at(Vector* vector, int dst_index, void *ptr, int total)
{
    vector_shift_right(vector, dst_index, total);
    void *dst_ptr = vector_at(vector, dst_index);
    size_t total_bytes = total * vector->esize;
    memcpy(dst_ptr, ptr, total_bytes);
}

void vector_push_at(Vector* vector, int index, void *ptr)
{
    vector_shift_right(vector, index, 1);

    void *data_ptr = vector_at(vector, index);
    memcpy(data_ptr, ptr, vector->esize);
}

int vector_insert(Vector*vector_dst, Vector*vector_src, int dst_index)
{
    if (vector_dst->esize != vector_src->esize)
    {
        return -1;
    }

    vector_push_multiple_at(vector_dst, dst_index, vector_at(vector_src, 0), vector_count(vector_src));

    return 0;
}

void vector_pop(Vector* vector)
{
    // here
    if(vector_empty(vector)) {
        fprintf(stderr, "Vector empty!\n");
        return;
    }

    // Popping from the vector will just decrement the index, no need to free memory
    // the next push will overwrite it.
    vector->rindex -= 1;
    vector->count -= 1;

    vector_assert_bounds_for_pop(vector, vector->rindex);
}

void *vector_data_ptr(Vector* vector)
{
    return vector->data;
}

bool vector_empty(Vector* vector)
{
    return vector_count(vector) == 0;
}

void vector_clear(Vector* vector)
{
    while (vector_count(vector))
    {
        vector_pop(vector);
    }
}

void *vector_back_or_null(Vector* vector)
{
    // We can't go back or we will access an invalid element
    // out of bounds...
    if (!vector_in_bounds_for_at(vector, vector->rindex - 1))
    {
        return NULL;
    }

    return vector_at(vector, vector->rindex - 1);
}

void *vector_back_ptr_or_null(Vector* vector)
{
    void **ptr = vector_back_or_null(vector);
    if (ptr)
    {
        return *ptr;
    }

    return NULL;
}

void *vector_back(Vector* vector)
{
    vector_assert_bounds_for_pop(vector, vector->rindex - 1);

    return vector_at(vector, vector->rindex - 1);
}

int vector_count(Vector* vector)
{
    return vector->count;
}