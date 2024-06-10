# evo ir 设计

> [text](https://4ch12dy.github.io/2017/10/11/x86%E6%8C%87%E4%BB%A4%E7%BC%96%E7%A0%81%E7%AC%94%E8%AE%B0/X86%E6%8C%87%E4%BB%A4%E7%BC%96%E7%A0%81%E7%AC%94%E8%AE%B0/)

解析出如下的 AST：

```
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