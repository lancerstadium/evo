

#ifndef UTIL_HASHMAP_H
#define UTIL_HASHMAP_H

#include "gtype.h"

/**
 * Hash HashMap* implementation for C.
 * 
 * This hash HashMap* uses strings as keys, and allows association of any arbitrary
 * value type through the use of `void *` pointers. Additionally, this
 * implementation leaves memory management of stored values to the client (for 
 * example, destroying a HashMap* using `hashmap_destroy` will free the memory for the
 * HashMap* itself, but it will not free memory that was used to store its values).
 */
typedef struct hashmap HashMap;

/**
 * Create a new, empty HashMap*.
 * 
 * The returned HashMap* has dynamically allocated memory associated with it, and
 * this memory must be reclaimed after use with `hashmap_destroy`.
 */
HashMap* hashmap_create();

/**
 * Free the memory used for a HashMap* after use.
 * 
 * Note that this routine does not free any memory that was allocated for the
 * values stored in the HashMap*. That memory must be freed by the client as
 * appropriate.
 */
void hashmap_destroy(HashMap* m);

/**
 * Get the size of a HashMap*.
 */
int hashmap_size(const HashMap* m);

/**
 * Determine whether a HashMap* contains a given key.
 * 
 * Keys are case-sensitive.
 */
bool hashmap_contains(const HashMap* m, const char *key);

/**
 * Set the value for a given key within a HashMap*.
 * 
 * This will add a new key if it does not exist. If the key already exists, the
 * new value will replace the old one.
 */
void hashmap_set(HashMap* m, const char *key, void *value);

/**
 * Retrieve the value for a given key in a HashMap*.
 * 
 * Crashes if the HashMap* does not contain the given key.
 */
void *hashmap_get(const HashMap* m, const char *key);

/**
 * Remove a key and return its value from a HashMap*.
 * 
 * Crashes if the HashMap* does not already contain the key.
 */
void *hashmap_remove(HashMap* m, const char *key);

/**
 * Iterate over a HashMap*'s keys.
 * 
 * Usage:
 * 
 * for (char *key = hashmap_first(m); key != NULL; key = hashmap_next(m, key)) {
 *   ...
 * }
 * 
 * Note that the `key` passed to `hashmap_next` must have come from a previous call
 * to `hashmap_first` or `hashmap_next`. Passing strings from other sources produces
 * undefined behavior.
 */
const char *hashmap_first(HashMap* m);
const char *hashmap_next(HashMap* m, const char *key);



#endif // UTIL_HASHMAP_H