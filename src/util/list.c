/******************************************/
/*                                        */
/*        Alexander Agdgomlishvili        */
/*                                        */
/*         cdevelopment@mail.com          */
/*                                        */
/******************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include "list.h"

typedef struct
{
  int count;          /* Number of items in the list. */
  int alloc_size;     /* Allocated size in quantity of items */
  int lastSearchPos;  /* Position of last search - first_match or LastMatch */
  size_t item_size;   /* Size of each item in bytes. */
  void *items;        /* Pointer to the list */
} List_priv_;  

int List_Realloc_(List *l, int n)
{
  List_priv_ *p = (List_priv_*) l->priv;
  if (n < p->count)
  {
    fprintf(stderr, "List: ERROR! Can not realloc to '%i' size - count is '%i'\n", n, p->count);
    assert(n >= p->count);
    return 0;
  }

  if (n == 0 && p->alloc_size == 0)
    n = 2;

  void *ptr = realloc(p->items, p->item_size * n);
  if (ptr == NULL)
  {
    fprintf(stderr, "List: ERROR! Can not reallocate memory!\n");
    return 0;
  }
  p->items = ptr;
  p->alloc_size = n;
  return 1;
}

void *List_Add_(List *l, void *o)
{
  List_priv_ *p = (List_priv_*) l->priv;
  if (p->count == p->alloc_size && 
        List_Realloc_(l, p->alloc_size * 2) == 0)
    return NULL;
  
  char *data = (char*) p->items;
  data = data + p->count * p->item_size;
  memcpy(data, o, p->item_size);
  p->count++;
  return data;
}

void *List_Insert_(List *l, void *o, int n)
{
  List_priv_ *p = (List_priv_*) l->priv;
  if (n < 0 || n > p->count)
  {
    fprintf(stderr, "List: ERROR! Insert position outside range - %d; n - %d.\n", 
                        p->count, n);
    assert(n >= 0 && n <= p->count);
    return NULL;
  }

  if (p->count == p->alloc_size && 
        List_Realloc_(l, p->alloc_size * 2) == 0)
    return NULL;

  size_t step = p->item_size;
  char *data = (char*) p->items + n * step;
  memmove(data + step, data, (p->count - n) * step);
  memcpy(data, o, step);
  p->count++;
  return data;
}

void *List_Replace_(List *l, void *o, int n)
{
  List_priv_ *p = (List_priv_*) l->priv;
  if (n < 0 || n >= p->count)
  {
    fprintf(stderr, "List: ERROR! Replace position outside range - %d; n - %d.\n", 
                        p->count, n);
    assert(n >= 0 && n < p->count);
    return NULL;
  }

  char *data = (char*) p->items;
  data = data + n * p->item_size;
  memcpy(data, o, p->item_size);
  return data;
}

void List_Remove_(List *l, int n)
{
  List_priv_ *p = (List_priv_*) l->priv;
  if (n < 0 || n >= p->count)
  {
    fprintf(stderr, "List: ERROR! Remove position outside range - %d; n - %d.\n",
                        p->count, n);
    assert(n >= 0 && n < p->count);
    return;
  }

  size_t step = p->item_size;
  char *data = (char*)p->items + n * step;
  memmove(data, data + step, (p->count - n - 1) * step);
  p->count--;

  if (p->alloc_size > 3 * p->count && p->alloc_size >= 4) /* Dont hold much memory */
    List_Realloc_(l, p->alloc_size / 2);
}

void *List_At_(List *l, int n)
{
  List_priv_ *p = (List_priv_*) l->priv;
  if (n < 0 || n >= p->count)
  {
    fprintf(stderr, "List: ERROR! Get position outside range - %d; n - %d.\n", 
                      p->count, n);
    assert(n >= 0 && n < p->count);
    return NULL;
  }

  char *data = (char*) p->items;
  data = data + n * p->item_size;
  return data;
}

void *List_first_match_(List *l, const void *o, size_t shift, size_t size, int string)
{
  List_priv_ *p = (List_priv_*) l->priv;    
  char *data = (char*) p->items;
  size_t step = p->item_size;
  p->lastSearchPos = -1;  

  if (shift + size > p->item_size)
  {
    fprintf(stderr, "List: ERROR! Wrong ranges for first_match - "
                "shift '%zu', size '%zu', item_size '%zu'\n", shift, size, p->item_size);
    assert(shift + size <= p->item_size);
    return NULL;    
  }

  if (shift == 0 && size == 0)
    size = p->item_size;

  size_t i = shift;
  size_t end = p->count * step;
  int index = 0;

  if (string)
  {
    for (; i < end; i += step, index++)
    {
      if (strncmp(data + i, o, size) == 0)
      {
        p->lastSearchPos = index;  
        return (data + i - shift);
      }
    }
  }
  else
  {
    for (; i < end; i += step, index++)
    {
      if (memcmp(data + i, o, size) == 0)
      {
        p->lastSearchPos = index;
        return (data + i - shift);
      }
    }
  }

  return NULL;
}

void *List_last_match_(struct List *l, const void *o, size_t shift, size_t size, int string)
{
  List_priv_ *p = (List_priv_*) l->priv;    
  char *data = (char*) p->items;
  size_t step = p->item_size;
  p->lastSearchPos = -1;

  if (shift + size > p->item_size)
  {
    fprintf(stderr, "List: ERROR! Wrong ranges for last_match - "
                "shift '%zu', size '%zu', item_size '%zu'\n", shift, size, p->item_size);
     assert(shift + size <= p->item_size);
    return NULL;
  }

  if (shift == 0 && size == 0)
    size = p->item_size;

  int index = p->count - 1;
  long i = index * step + shift;
  if (string)
  {  
    for (; i >= 0; i -= step, index--)
    {
      if (strncmp(data + i, o, size) == 0)
      {
        p->lastSearchPos = index;
        return (data + i - shift);
      }
    }
  }  
  else
  {  
    for (; i >= 0; i -= step, index--)
    {
      if (memcmp(data + i, o, size) == 0)
      {
        p->lastSearchPos = index;
        return (data + i - shift);
      }
    }
  }

  return NULL;  
}

int List_index_(List *l)
{
  List_priv_ *p = (List_priv_*) l->priv;
  return p->lastSearchPos;
}

int List_swap_(List *l, int a, int b)
{
  List_priv_ *p = (List_priv_*) l->priv;

  if (a < 0 || a >= p->count || b < 0 || b >= p->count)
  {
    fprintf(stderr, "List: ERROR! Swap position outside range - %i, %i; count - %d.\n", 
                      a, b, p->count);
    assert(a >= 0 && a < p->count && b >= 0 && b < p->count);
    return 0;
  }

  if (a == b) return 1; /* ? Good ? :D */

  char *data = (char*) p->items;
  size_t step = p->item_size;

  if (p->count == p->alloc_size && 
        List_Realloc_(l, p->alloc_size + 1) == 0)
    return 0;

  memcpy(data + p->count * step, data + a * step, step);
  memcpy(data + a * step, data + b * step, step);
  memcpy(data + b * step, data + p->count * step, step);
  return 1;
}

int List_Count_(List *l)
{
  List_priv_ *p = (List_priv_*) l->priv;
  return p->count;
}

int List_AllocSize_(List *l)
{
  List_priv_ *p = (List_priv_*) l->priv;
  return p->alloc_size;
}

size_t List_ItemSize_(List *l)
{
  List_priv_ *p = (List_priv_*) l->priv;
  return p->item_size;
}

void List_Clear_(List *l)
{
  List_priv_ *p = (List_priv_*) l->priv;
  free(p->items);
  p->items = NULL;
  p->alloc_size = 0;
  p->count = 0;
}

void List_Free_(List *l)
{
  List_priv_ *p = (List_priv_*) l->priv;
  free(p->items);
  free(p);
  free(l);
  l = NULL;
}

void List_print_(List *l, size_t shift, int n, const char *type)
{
  List_priv_ *p = (List_priv_*) l->priv;

  if (shift >= p->item_size)
  {
    fprintf(stderr, "List: ERROR! Wrong shift value for list print - "
                "shift '%zu', item_size '%zu'\n", shift, p->item_size);
     assert(shift < p->item_size);
    return;
  }

  printf("\nList:  count = %i  item_size = %zu   alloc_size = %i   type = %s\n",
                      p->count, p->item_size, p->alloc_size, type);

  if (n > 0)
  {
    int tp = -1;
    if (type == NULL) tp = 0;  /* Print out pointers */
    else if (strcmp(type, "char") == 0) tp = 1;
    else if (strcmp(type, "short") == 0) tp = 2;
    else if (strcmp(type, "int") == 0) tp = 3;
    else if (strcmp(type, "long") == 0) tp = 4;
    else if (strcmp(type, "uintptr_t") == 0) tp = 5;
    else if (strcmp(type, "size_t") == 0) tp = 6;
    else if (strcmp(type, "double") == 0) tp = 7;
    else if (strcmp(type, "string") == 0) tp = 8;

    if (tp == -1)
    {  
      fprintf(stderr, "List: Can not print - not supported type - %s\n\n", type);
      return;
    }  

    n = (n > p->count) ? p->count : n;
    char *data = (char*) p->items + shift;
    size_t step = p->item_size;
    int i = 0;
    for (; i < n; i++)
    {
      switch (tp)
      {
        case 0: printf("%p  ", data); break;
        case 1: printf("%c ", *(char*) data); break;
        case 2: printf("%hi  ", *(short*) data); break;
        case 3: printf("%i  ", *(int*) data); break;
        case 4: printf("%li  ", *(long*) data); break;
        case 5: printf("0x%lx  ", *(uintptr_t*) data); break;
        case 6: printf("%zu  ", *(size_t*) data); break;
        case 7: printf("%f  ", *(double*) data); break;
        case 8: printf("%s\n", data); break;
        default: return;
      }  

      data += step;
    }
    printf("\n\n");
  }
}

List *List_init(size_t objSize)
{
  List *lst = malloc(sizeof(List));
  List_priv_ *p = malloc(sizeof(List_priv_));
  if (!lst || !p)
  {
    fprintf(stderr, "List: ERROR! Can not allocate List!\n");
    return NULL;
  }
  p->count = 0;
  p->alloc_size = 0;
  p->lastSearchPos = -1;
  p->item_size = objSize;
  p->items = NULL;
  lst->add = &List_Add_;
  lst->insert = &List_Insert_;
  lst->replace = &List_Replace_;
  lst->remove = &List_Remove_;
  lst->at = &List_At_;
  lst->realloc = &List_Realloc_;
  lst->count = &List_Count_;
  lst->first_match = &List_first_match_;
  lst->last_match = &List_last_match_;
  lst->index = &List_index_;
  lst->swap = &List_swap_;
  lst->alloc_size = &List_AllocSize_;
  lst->item_size = &List_ItemSize_;
  lst->print = &List_print_;
  lst->clear = &List_Clear_;
  lst->free = &List_Free_;
  lst->priv = p;
  return lst;
}
