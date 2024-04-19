# evo


```
   $$$$$$\  $$\    $$\  $$$$$$\  
  $$  __$$\ \$$\  $$  |$$  __$$\ 
  $$$$$$$$ | \$$\$$  / $$ /  $$ |
  $$   ____|  \$$$  /  $$ |  $$ |
  \$$$$$$$\    \$  /   \$$$$$$  |
   \_______|    \_/     \______/ 

```


- `evo` is designed to be an evolvable programming language. 

- `.ec` is the extension of evo code file. 

- The tool chain of evo contains modules as follows:
    - `evoc` ==> evo's compiler
    - `evor` ==> evo's runtime



## 目录

```
src
 ┣━ app             // 程序
 ┣━ arch            // 架构
 ┣━ back            // 后端：IR -> Asm
 ┣━ emu             // 模拟器：CPU, MEM
 ┣━ front           // 前端：C -> IR
 ┣━ ir              // 中间表示：
 ┣━ lift            // 提升：Asm -> IR
 ┣━ opt             // 优化
 ┣━ util            // 工具
 ┗━ main.rs

```


## 设计参考

- [chibicc](https://www.sigbus.info/compilerbook)
- [lua](https://github.com/lua/lua.git)
- [scf](http://baseworks.info/gitweb/scf.git)
- [dlang](https://github.com/dlang/dlang.org.git)
- [koopa](https://github.com/pku-minic/koopa)
- [khemu](https://github.com/KireinaHoro/khemu)