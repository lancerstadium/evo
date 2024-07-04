/**
 * =================================================================================== //
 * @file util/list.h
 * @author lancerstadium (lancerstadium@163.com)
 * @brief util: list data struct
 * @version 0.1
 * @date 2024-07-03
 * @copyright Copyright (c) 2024
 * =================================================================================== //
 */

// ==================================================================================== //
//                                  evo/util/list.h
// ==================================================================================== //


#ifndef __EVO_UTIL_LIST_H__
#define __EVO_UTIL_LIST_H__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ==================================================================================== //
//                                       include
// ==================================================================================== //

#include <stddef.h>

// ==================================================================================== //
//                                       macros
// ==================================================================================== //

#ifndef container_of
#define container_of(ptr, type, member) \
	({const typeof(((type *)0)->member) *__mptr = (ptr); (type *)((char *)__mptr - offsetof(type,member));})
#endif

// ==================================================================================== //
//                                       typedef
// ==================================================================================== //

typedef struct list_node list_node_t;
typedef struct hlist_node hlist_node_t;
typedef struct hlist_head hlist_head_t;

// ==================================================================================== //
//                                    list: list_node
// ==================================================================================== //

// Doubly Linked List Node
struct list_node {
    struct list_node *next;
    struct list_node *prev;
};

// List Node Connct2: p->m->n
static inline void __list_cnc(list_node_t *p, list_node_t *n) {
    n->prev = p;
    p->next = n;
}
// List Node Connct3: p->m->n
static inline void __list_cnc3(list_node_t *p, list_node_t *m, list_node_t *n) {
    n->prev = m;
    m->next = n;
    m->prev = p;
    p->next = m;
}
// List Node Extract from list
static inline void __list_extract(list_node_t *entry) {
    __list_cnc(entry->prev, entry->next);
}
// List Cut by Position util end node
static inline void __list_cut_pos(list_node_t *entry, list_node_t *end, list_node_t *rest) {
    list_node_t* new_first = end->next;
	rest->next = entry->next;
	rest->next->prev = rest;
	rest->prev = end;
	end->next = rest;
	entry->next = new_first;
	new_first->prev = entry;
}
static inline void __list_splice(const list_node_t *entry, list_node_t *prev, list_node_t *next) {
    list_node_t * first = entry->next;
    list_node_t * last = entry->prev;
    first->prev = prev;
    prev->next = first;
    last->next = next;
    next->prev = last;
}

// ==================================================================================== //
//                                    list: list
// ==================================================================================== //

// List Init from single node
static inline void list_init(list_node_t *l) {
    l->next = l;
    l->prev = l;
}
// List Insert n node behind head
static inline void list_add(list_node_t *l, list_node_t *n) {
    __list_cnc3(l, n, l->next);
}
// List Insert n node behind tail
static inline void list_push(list_node_t *l, list_node_t *n) {
    __list_cnc3(l->prev, n, l);
}
// List Delete Entry
static inline void list_del(list_node_t *l) {
    __list_extract(l);
    l->next = NULL;
    l->prev = NULL;
}
// List Replace old entry to n
static inline void list_replace(list_node_t *l, list_node_t *n) {
    n->next = l->next;
    n->next->prev = n;
    n->prev = l->prev;
    n->prev->next = n;
}
// List replace old entry to n, and reinit old entry
static inline void list_replace_init(list_node_t *l, list_node_t *n) {
    list_replace(l, n);
    list_init(l);
}
// List Delete Node l from one and Insert as another's head
static inline void list_move(list_node_t *l, list_node_t *entry) {
    __list_extract(l);
    list_add(entry, l);
}
// List Delete Node l from one and Insert as another's tail
static inline void list_move_tail(list_node_t *l, list_node_t *entry) {
    __list_extract(l);
    list_push(entry, l);
}
// List part Move to tail
static inline void list_part_tail(list_node_t *l, list_node_t *start, list_node_t *end) {
    start->prev->next = end->next;
    end->next->prev = start->prev;
    l->prev->next = start;
    start->prev = l->prev;
    end->next = l;
    l->prev = end;
}
// List Check if is first node in @entry list
static inline int list_is_first(const list_node_t *l, const list_node_t *entry) {
    return l->prev == entry;
}
// List Check if is last node in @entry list
static inline int list_is_last(const list_node_t *l, const list_node_t *entry) {
    return l->next == entry;
}
// List Check if is empty
static inline int list_is_empty(const list_node_t *l) {
    return l->next == l;
}
// List Check if is empty strict
static inline int list_is_empty_strict(const list_node_t *l) {
    return (l->next == l) && (l->next == l->prev);
}
// List Check if is single ode
static inline int list_is_single(const list_node_t *l) {
    return !list_is_empty(l) && (l->next == l->prev);
}
// List Rotate Left one node
static inline void list_rol(list_node_t *l) {
    if(!list_is_empty(l))
        list_move_tail(l->next, l);
}
// List Cut util node end, l got new position and rest got rest node last is end
static inline void list_cut_pos1(list_node_t *l, list_node_t *end, list_node_t *rest) {
    if(list_is_empty(l))
        return;
    if(list_is_single(l) && ((l->next != end) && (l != end)))
        return;
    if(l == end)
        list_init(rest);
    else
        __list_cut_pos(l, end, rest);
}
// List Cut util node end, l got new position next is start and rest got rest node
static inline void list_cut_pos2(list_node_t *l, list_node_t *start, list_node_t *rest) {
    if(l->next == start) {
        list_init(rest);
        return;
    }
    rest->next = l->next;
    rest->next->prev = l;
    rest->prev = start->prev;
    rest->prev->next = rest;
    l->next = start;
    start->prev = l;
}
// List Join l2 in l1 head
static inline void list_splice(list_node_t *l1, const list_node_t *l2) {
    if(!list_is_empty(l2))
        __list_splice(l2, l1, l1->next);
}
// List Join l2 in l1 head and reinit l2
static inline void list_splice_init(list_node_t *l1, list_node_t *l2) {
    if(!list_is_empty(l2)) {
        __list_splice(l2, l1, l1->next);
        list_init(l2);
    }
}
// List Join l2 in l1 tail
static inline void list_splice_tail(list_node_t *l1, const list_node_t *l2) {
    if(!list_is_empty(l2))
        __list_splice(l2, l1->prev, l1);
}
// List Join l2 in l1 tail and reinit l2
static inline void list_splice_tail_init(list_node_t *l1, list_node_t *l2) {
    if(!list_is_empty(l2)) {
        __list_splice(l2, l1->prev, l1);
        list_init(l2);
    }
}

// ==================================================================================== //
//                                    list: Macro function
// ==================================================================================== //

// List new
#define list_new(N)    list_node_t N = { &(N), &(N) }
// Get the struct for this entry
#define list_entry(ptr, type, member) \
    container_of(ptr, type, member)
// Get the first element from a list
#define list_first_entry(ptr, type, member) \
	list_entry((ptr)->next, type, member)
// Get the last element from a list
#define list_last_entry(ptr, type, member) \
	list_entry((ptr)->prev, type, member)
// Get the next element in list
#define list_next_entry(pos, member) \
	list_entry((pos)->member.next, typeof(*(pos)), member)
// Get the prev element in list
#define list_prev_entry(pos, member) \
	list_entry((pos)->member.prev, typeof(*(pos)), member)
// Get the first element from a list
#define list_first_entry_or_null(ptr, type, member) ({        \
    struct list_head *head__ = (ptr);                         \
    struct list_head *pos__ = head__->next;                   \
    pos__ != head__ ? list_entry(pos__, type, member) : NULL; \
})
// Iterate over a list
#define list_foreach(pos, head) \
	for (pos = (head)->next; pos != (head); pos = pos->next)
// iterate over a list backwards
#define list_foreach_prev(pos, head) \
	for (pos = (head)->prev; pos != (head); pos = pos->prev)
// Iterate over a list safe against removal of list entry
#define list_foreach_safe(pos, n, head) \
	for (pos = (head)->next, n = pos->next; pos != (head); \
		pos = n, n = pos->next)
// Iterate over a list backwards safe against removal of list entry
#define list_foreach_prev_safe(pos, n, head) \
    for (pos = (head)->prev, n = pos->prev;  \
         pos != (head);                      \
         pos = n, n = pos->prev)
// Iterate over list of given type
#define list_foreach_entry(pos, head, member) \
	for (pos = list_first_entry(head, typeof(*pos), member); \
	     &pos->member != (head); \
	     pos = list_next_entry(pos, member))



// ==================================================================================== //
//                                    list: hlist
// ==================================================================================== //

struct hlist_node {
    struct hlist_node *next;
    struct hlist_node **pprev;
};

struct hlist_head {
    struct hlist_node *first;
};




#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __EVO_UTIL_LIST_H__