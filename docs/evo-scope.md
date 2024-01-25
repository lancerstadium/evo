
# evo scope: 域设计



## 0 基础域

1. 在`evo`中，每个基础域（base scope）用`{ ... }`来表示。
2. 基础域中可以抽象出两种域：定义域（def scope）和过程域（proc scope）。
3. 定义域存储一些数据和函数（属性和方法）的声明和实现，而过程域存储着数据和函数（属性和方法）的具体操作。
4. 其中每个基础域都有公共（pub）和私密（pri）的子域：使用域访问符号`:`访问子域，数据和函数默认存放于`pub`。
5. 其中`scope:pub`中的声明的数据和函数可以被任何其他域调用，而`scope:pri`中的声明只能被本身或其子域调用。
6. `scope:pub`内声明的函数必须被实现于`scope:pub`，同理`scope:pri`也是如此。
7. `scope:pri`中存储了用户不可见的域私密属性：`__scope_name`（域名：默认none或string）和`__ret_val`（返回值：默认i32），以及虚函数表`__vfn_tbl（def scope 类型）`
8. `base scope`被其他过程域`proc scope`包含时，默认返回`__ret_val`值。
9. 当有传入`scope_name`时，会有一系列函数在编译前（预处理时）被定义。
10. 当有非

```
[scope_name] {
    // base scope

    pri:

        // ==== User Unknown ====
        *__scope_name : addr;       // scope name string: default none
        __scope_size  : i32;        // scope size number: default 0
        *__ret_val    : addr;       // return value: default 0 (i32)
        *__vfn_tbl    : addr;       // addr of functions: default none
        *__proc_entry : addr;       // proc scope entry addr: default none
        *__proc_exit  : addr;       // proc scope exit addr: default none


        // Before Compile: auto gen init func

        fn __[scope_name]_vfn_tbl_init#() : none {
            // init the __vfn_tbl by define functions
            // preprocessor can ident scope's func size
            all_def_funcs_size : i32 = #preprocess_get_scope_size([scope_name]);
            __vfn_tbl = malloc(sizeof([all_def_funcs_size]));
            *__vfn_tbl = {
                .fn1 = addr1,
                .fn2 = addr2,
                ...
            };
        }

        fn __[scope_name]_proc_scope_init#() : none {
            // init the proc scope
            *__proc_entry = malloc(sizeof(1024)); // ?
        }

        fn __[scope_name]_scope_init#() : none {
            // init the scope by define vars and funcs
            *__scope_name = [scope_name];
            __scope_size = sizeof([addr * 5 + i32 * 1]);
            *self = malloc(__scope_size);
            __[scope_name]_vfn_tbl_init();
            __[scope_name]_proc_scope_init();
            *__ret_val = 0;
        }

        fn __[scope_name]_scope_free#() : none {
            // free the scope
            free(__vfn_tbl);
            free(__scope_name);
            free(__ret_val);
            free(__proc_entry);
            free(self);
        }

    pub:
        *self        : addr;        // head addr of this scope: default none
        fn [scope_name]() : scope {
            return malloc(sizeof(__scope_size));
        }

        // Proc scope
        __proc_entry#:
            // ==== User Unknown ====
            __[scope_name]_scope_init();
        
        entry:
            // **** User Code ****



        __proc_exit#:
            // ==== User Unknown ====
            __[scope_name]_scope_free();

        exit:
            // **** User Code ****
}

```


## 1 定义域


```
def {




}

```


