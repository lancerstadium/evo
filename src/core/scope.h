
#ifndef CORE_SCOPE_H
#define CORE_SCOPE_H

#include "data.h"


typedef enum {
    SCOPE_TYPE_PROC,            // 进程域
    SCOPE_TYPE_MOD,             // 模块域
    SCOPE_TYPE_macro,           // 宏域
    SCOPE_TYPE_DEF,             // 定义域
    SCOPE_TYPE_ENUM,            // 枚举域
    SCOPE_TYPE_IMPL,            // 应用域
    SCOPE_TYPE_FN,              // 函数域
    SCOPE_TYPE_NAME,            // 命名域
    SCOPE_TYPE_BASE,            // 基本域
} ScopeType;


typedef enum {
    SCOPE_FLAG_PUB  = 0b00000001,             // 公共域段
    SCOPE_FLAG_PRI  = 0b00000010,             // 私有域段
    SCOPE_FLAG_PROC = 0b00000100,             // 程序域段
    SCOPE_FLAG_FEAT = 0b00001000,             // 域特征
    SCOPE_FLAG_DESC = 0b00010000,             // 域描述信息
} ScopeFlag;


typedef struct {
    int flags;                  // 标志
    ScopeType type;             // 域类型
    const char* name;           // 域名
    Pos pos;                    // 域位置
    Buffer* desc;               // 域描述信息
    int depth;                  // 位于几层域内：proc为第0层
} ScopeInfo;


// 域
typedef struct scope {
    ScopeInfo info;             // 域信息
    struct scope* sup;          // 父域
    Vector* subs;               // 子域
    Vector* refs;               // 依赖域

    Vector* scope_tok_vec;      // 域修饰 token 令牌
    Vector* pub_tok_vec;        // pub token 令牌
    Vector* pri_tok_vec;        // pri token 令牌
    Vector* proc_tok_vec;       // proc token 令牌
    Vector* feat_tok_vec;       // 域特征 token 令牌

    union {
        struct {
            Vector* scope_tok_vec;      // 域修饰 token 令牌
            Vector* pub_tok_vec;        // pub token 令牌
            Vector* pri_tok_vec;        // pri token 令牌
            Vector* proc_tok_vec;       // proc token 令牌
        } proc;
        struct {
            Vector* scope_tok_vec;      // 域修饰 token 令牌
            Vector* pub_tok_vec;        // pub token 令牌
            Vector* pri_tok_vec;        // pri token 令牌
            Vector* proc_tok_vec;       // proc token 令牌
        } mod;
        struct {
            Vector* scope_tok_vec;      // 域修饰 token 令牌
            Vector* pub_tok_vec;        // pub token 令牌
        } macro;
        struct {
            Vector* scope_tok_vec;      // 域修饰 token 令牌
            Vector* pub_tok_vec;        // pub token 令牌
            Vector* pri_tok_vec;        // pri token 令牌
            Vector* feat_tok_vec;       // 域特征 token 令牌
        } def;
        struct {
            Vector* scope_tok_vec;      // 域修饰 token 令牌
            Vector* pub_tok_vec;        // pub token 令牌
        } enm;
        struct {
            Vector* scope_tok_vec;      // 域修饰 token 令牌
            Vector* pub_tok_vec;        // pub token 令牌
            Vector* pri_tok_vec;        // pri token 令牌
            Vector* feat_tok_vec;       // 域特征 token 令牌
        } impl;
        struct {
            Vector* scope_tok_vec;      // 域修饰 token 令牌
            Vector* type_tok_vec;       // 函数类型 token 令牌
            Vector* pub_tok_vec;        // pub token 令牌
            Vector* feat_tok_vec;       // 域特征 token 令牌
        } fn;
        struct {
            Vector* proc_tok_vec;       // proc token 令牌
        } name;
        struct {
            Vector* proc_tok_vec;       // proc token 令牌
        } base;
    };
} Scope;

#endif