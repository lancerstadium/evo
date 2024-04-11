# evo ir 设计



解析出如下的 AST：

```json
CompUnit : {
    Mod : [
        name : "std",
        FnDef : [
            {
                name : "main",
                type : "i32",
                params : [],
                body : {
                    params : [],
                    stmts : [
                        Return {
                            value: 0
                        }
                    ]
                }
            },
        ]
    ]
}

```