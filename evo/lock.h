/**
 * =================================================================================== //
 * @file lock.h
 * @author lancerstadium (lancerstadium@163.com)
 * @brief lock header file
 * @version 0.1
 * @date 2024-07-03
 * @copyright Copyright (c) 2024
 * =================================================================================== //
 */

// ==================================================================================== //
//                                       evo/lock.h
// ==================================================================================== //

#ifndef __EVO_LOCK_H__
#define __EVO_LOCK_H__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ==================================================================================== //
//                                        typedef
// ==================================================================================== //

typedef struct abstract_mutex mutex_t;

// ==================================================================================== //
//                                         lock
// ==================================================================================== //

struct abstract_mutex {
    void* locker;                                           /* platform dependence  */
    void (*init)(struct abstract_mutex* mutex);             /* init this mutex      */
    void (*lock)(struct abstract_mutex* mutex);             /* lock this mutex      */
    void (*unlock)(struct abstract_mutex* mutex);           /* unlock this mutex    */
    void (*free)(struct abstract_mutex* mutex);             /* destroy this mutex   */
};
void mutex_init(mutex_t* mutex);
void mutex_lock(mutex_t* mutex);
void mutex_unlock(mutex_t* mutex);
void mutex_free(mutex_t* mutex);


#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __EVO_LOCK_H__