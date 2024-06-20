

#ifndef _UTIL_SLIST_H_
#define _UTIL_SLIST_H_

#define Slist_foreach(l) \
    for (; (l); (l) = (l)->next)

typedef struct Slist {
    struct Slist *next;
} Slist;

typedef int (*Slist_cmp)(const Slist *p1, const Slist *p2);

/* Add element sorted based on the Slist_cmp function */
static inline void Slist_add(Slist *head, Slist *n, Slist_cmp fn) {
    Slist **t = &head;
    while (*t) {
        if (fn(n, *t) < 0) {
            n->next = *t;
            break;
        }
        t = &(*t)->next;
    }
    *t = n;
}

static inline void Slist_append(Slist *head, Slist *n) {
    Slist **t = &head;
    while (*t) {
        t = &(*t)->next;
    }
    *t = n;
}

static inline void Slist_remove(Slist *head, Slist *n) {
    Slist **t = &head;
    while (*t) {
        if (*t == n) {
            *t = (*t)->next;
        }
        t = &(*t)->next;
    }
}

static inline unsigned int Slist_size(Slist *head) {
    unsigned int size = 0;
    while (head) {
        size++;
        head = head->next;
    }
    return size;
}

#endif // _UTIL_SLIST_H_