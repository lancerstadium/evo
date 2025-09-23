
## EDB Usage

```
edb
 ┣━ build               # Build Dir
 ┣━ npc                 # Hardware Design  (Client)
 ┃ ┣━ csrc              # Verilator Module cpp
 ┃ ┃ ┗━ top.cpp         # SW Entry | Client Entry <-- C
 ┃ ┣━ vsrc              # RTL code
 ┃ ┃ ┗━ top.v           # HW Entry
 ┃ ┗━ Makefile
 ┣━ src                 # EDB core (Server)
 ┃ ┣━ edb.c             # Server Entry <-- S
 ┃ ┣━ edb.h             # Client Socket Define
 ┃ ┣━ linenoise.c
 ┃ ┣━ linenoise.h
 ┃ ┗━ sob.h
 ┣━ Makefile
 ┗━ README.md
```

More detail: [evo-tool](../../docs/evo-tool.md)

Attention: 
1. Make sure OS support `POSIX` interface.
2. Make sure you've download `verilator`.

### 1 Build

- Make `edb-server` and `edb-client`:

```shell
# make evo first
make
# make sure enter edb dir
cd tools/edb
# build
make
# Or you can singly build edb-server
# make server
```

- If you want to build `edb-client` singly, needn't build `evo`, just:
```shell
# make sure enter edb dir
cd tools/edb
# singly build client
make client
```


### 2 Connect

1. Server should Open First:
```shell
# make sure enter edb dir
cd tools/edb
# open server
make run
```


2. Connect client to server:
```shell
# make sure enter edb dir
cd tools/edb
# open client
make sim
```