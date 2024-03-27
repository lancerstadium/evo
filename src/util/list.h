/******************************************/
/*                                        */
/*        Alexander Agdgomlishvili        */
/*                                        */
/*         cdevelopment@mail.com          */
/*                                        */
/******************************************/

#ifndef UTIL_LIST_H
#define UTIL_LIST_H

#include "gtype.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct List
{
  void * (* add)         (struct List *l, void *o);            /* Add object to the end of a list */
  void * (* insert)      (struct List *l, void *o, int n);     /* Insert object at position 'n' */
  void * (* replace)     (struct List *l, void *o, int n);     /* Replace object at position 'n' */
  void   (* remove)      (struct List *l, int n);              /* Remove object at position 'n' */
  void * (* at)          (struct List *l, int n);              /* Get object at position 'n' */
  int    (* realloc)     (struct List *l, int n);              /* Reallocate list to 'size' items */
  int    (* count)       (struct List *l);                     /* Get list size in items */
  void * (* first_match) (struct List *l, const void *o, size_t shift, size_t size, int string);
                                                                /* Returns object with first match of string or byte compare */
  void * (* last_match)  (struct List *l, const void *o, size_t shift, size_t size, int string);
                                                                /* Returns object with last match of string or byte compare */
  int    (* index)       (struct List *l);                     /* Get index of previos search match */
  int    (* swap)        (struct List *l, int a, int b);       /* Swap, replace two items with index a b */
  int    (* alloc_size)  (struct List *l);                     /* Get allocated size in items */
  size_t (* item_size)   (struct List *l);                     /* Get item size in bytes */
  void   (* print)       (struct List *l, size_t shift, int n, const char *type);   /* Print list data */
  void   (* clear)       (struct List *l);                     /* Clear list */
  void   (* free)        (struct List *l);                     /* Destroy struct List and all data */
  void  *priv;           /* NOT FOR USE, private data */
} List;

List *List_init(size_t objSize); /* Set list object size in bytes */

/*  void *add(struct List *l, void *o);
        Returns pointer to added object; Returns NULL if failed.

    void *insert(struct List *l, void *o, int n);
        Returns pointer to inserted object; Returns NULL if failed.

    void *replace(struct List *l, void *o, int n);
        Returns pointer to replaced object; Returns NULL if failed.

    void *at(struct List *l, int n);
        Returns pointer to object at index n;

    int realloc(struct List *l, int n);
        Return 1 when success. Returns 0 if failed.

    void *first_match(struct List *l, const void *o, size_t shift, size_t size, int string);
        Returns pointer to list item when first match found. Straight scanning, from 0 to list end.
        Returns NULL if search failed.  

    void *last_match(struct List *l, const void *o, size_t shift, size_t size, int string);
        Returns pointer to list item when first match found. Reverse scanning, from list end to 0.
        Returns NULL if search failed.

    int index(struct List *l);
        Returns index of last search first_match or last_match. Returns -1 if search failed.
    
    void print(struct List *l, size_t shift, int n, const char *type);
        Prints data of "int n" list items with offset "size_t shift" and type "const char *type".
        Supported types: char, short, int, long, uintptr_t, size_t, double, string.
        If type is NULL just pointers data will be printed. 
*/

#ifdef __cplusplus
}
#endif

#endif /* UTIL_LIST_H */