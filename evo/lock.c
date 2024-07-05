

// ==================================================================================== //
//                                        include
// ==================================================================================== //

#include "lock.h"
#include "sys.h"

// ==================================================================================== //
//                                        include
// ==================================================================================== //


static inline void mutex_init_bare_metal(mutex_t* mutex) {
    mutex->locker = sys_malloc(sizeof(mutex->locker));
    *((int*)(mutex->locker)) = 0;
}

static inline void mutex_lock_bare_metal(mutex_t* mutex) {
    *((int*)(mutex->locker)) = 1;
}

static inline void mutex_unlock_bare_metal(mutex_t* mutex) {
    *((int*)(mutex->locker)) = 0;
}

static inline void mutex_free_bare_metal(mutex_t* mutex) {
    if (mutex->locker != 0) {
        sys_free(mutex->locker);
    }

    mutex->locker = 0;
}

// ==================================================================================== //
//                                        lock API
// ==================================================================================== //

void mutex_init(mutex_t* mutex) {
    mutex->init = mutex_init_bare_metal;
    mutex->lock = mutex_lock_bare_metal;
    mutex->unlock = mutex_unlock_bare_metal;
    mutex->free = mutex_free_bare_metal;
    return mutex->init(mutex);
}

void mutex_lock(mutex_t* mutex) {
    return mutex->lock(mutex);
}

void mutex_unlock(mutex_t* mutex) {
    return mutex->unlock(mutex);
}

void mutex_free(mutex_t* mutex) {
    return mutex->free(mutex);
}
