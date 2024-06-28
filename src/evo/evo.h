
#ifndef _EVO_EVO_H_
#define _EVO_EVO_H_

#include <evo/cfg.h>

#ifdef __cplusplus
extern "C" {
#endif


// ==================================================================================== //
//                                    evo: Addr & Width
// ==================================================================================== //

typedef struct {
    union {
        u8 as_u8;
        i8 as_i8;
    };
} Byte;
typedef struct {
    union {
        u16 as_u16;
        i16 as_i16;
    };
} Half;
typedef struct {
    union {
        u32 as_u32;
        i32 as_i32;
        f32 as_f32;
    };
} Word;
typedef struct {
    union {
        u64 as_u64;
        i64 as_i64;
        f64 as_f64;
        ptr as_ptr;
    };
} Dword;

#define Width_OP(W, T, OP)          CONCAT3(W##_, T##_, OP)
#define Width_OP_def(W, T, OP)      UNUSED Width_OP(W, T, OP)

#define Width_def(W, T)              \
    W Width_OP_def(W, T, new)(T b) { \
        return (W){                  \
            .as_##T = b};            \
    }

// ==================================================================================== //
//                                    evo: Type
// ==================================================================================== //


/**
 * @brief 
 * 
 * ### Insn Format Constant Length
 * - riscv format:
 * ```txt
 * (32-bits):  <-- Big Endian View.
 *  ┌────────┬─────┬─────┬────┬──────┬────┬─────┬────────────────┐
 *  │31    25│24 20│19 15│  12│11   7│6  0│ Typ │      Arch      │
 *  ├────────┼─────┼─────┼────┼──────┼────┼─────┼────────────────┤
 *  │   f7   │ rs2 │ rs1 │ f3 │  rd  │ op │  R  │ RISC-V, EIR    │
 *  ├────────┴─────┼─────┼────┼──────┼────┼─────┼────────────────┤
 *  │    imm[12]   │ rs1 │ f3 │  rd  │ op │  I  │ RISC-V, EIR    │
 *  ├────────┬─────┼─────┼────┼──────┼────┼─────┼────────────────┤
 *  │  imm1  │ rs2 │ rs1 │ f3 │ imm0 │ op │  S  │ RISC-V         │
 *  ├─┬──────┼─────┼─────┼────┼────┬─┼────┼─────┼────────────────┤
 *  │3│  i1  │ rs2 │ rs1 │ f3 │ i0 │2│ op │  B  │ RISC-V         │
 *  ├─┴──────┴─────┴─────┴────┼────┴─┼────┼─────┼────────────────┤
 *  │          imm[20]        │  rd  │ op │  U  │ RISC-V         │
 *  ├─┬──────────┬─┬──────────┼──────┼────┼─────┼────────────────┤
 *  │3│   imm0   │1│   imm2   │  rd  │ op │  J  │ RISC-V         │
 *  ├─┴──────────┴─┴──────────┴──────┴────┼─────┼────────────────┤
 *  │   ?????                             │  ?  │ ?????          │
 *  └─────────────────────────────────────┴─────┴────────────────┘
 * ```
 * 
 * - arm format:
 * ```txt
 *  <Opcode>{<Cond>}<S>  <Rd>, <Rn> {,<Opcode2>}
 * ```
 * 
 * ### Insn Format Variable Length
 * - evo format:
 * ```txt
 *  Max reg nums: 2^8 = 256
 *  Max general opcode nums: 2^8 = 256 (Without bits/sign mode)
 *  Max extend 1 opcode nums: 2^8 * 2^8 = 65536 (Without bits/sign mode)
 *  --> Little Endian View.
 *  ┌────────────────┬───────────────────────────────────────────┐
 *  │    Type: E     │             Arch: EIR                     │
 *  ├────────────────┼────────────────────────┬──────────────────┤
 *  │   flag & Op    │        A field         │     B field      │
 *  ├────────────┬───┼────────────────────────┼──────────────────┤
 *  │     1      │ 1 │          ???           │       ???        │
 *  ├────────────┼───┼────────────────────────┼──────────────────┤
 *  │ 000 000 00 │ o │ 000.x: off(0)          │ 000.x: off(0)    │
 *  │ ─── ─── ── │ p │ 001.x: reg(1)          │ 001.0: imm(4)    │
 *  │ BBB AAA sb │ c │ 010.x: reg(1,1)        │ 001.1: imm(8)    │
 *  │         ^^ │ o │ 011.x: reg(1,1,1)      │ 010.0: imm(4,4)  │
 *  │       flag │ d │ 100.x: reg(1,1,1,1)    │ 010.1: imm(8,8)  │
 *  │            │ e │ 101.0: reg(1,1)imm(4)  │ 011.0: imm(4,4,4)│
 *  │            │   │ 101.1: reg(1,1)imm(8)  │ 011.1: imm(8,8,8)│
 *  │            │   │ 110.0: reg(1,1)imm(4,4)│ 100.0: mem(7)    │
 *  │            │   │ 110.1: reg(1,1)imm(8,8)│ 100.1: mem(11)   │
 *  │            │   │ 111.x: opcode(1)       │ 101.0: mem(7,7)  │
 *  │            │   │                        │ 101.1: mem(11,11)│
 *  │            │   │                        │ 110.x: off(0)ExtC│
 *  │            │   │                        │ 111.x: off(0)ExtV│
 *  └────────────┴───┴────────────────────────┴──────────────────┘
 * 
 *  flag:
 *    0. bits mode: 0: 32-bits, 1: 64-bits
 *    1. sign mode: 0: signed,  1: unsigned
 *    You can see such as: (`_i32`, `_u32`, `_i64`, `_u64`) in insn name.
 * 
 *  decode:
 *    -> check opcode: if AAA is 111, append two bytes else append one byte
 *    -> check flag: get A/B field length and parse as operands
 *    -> if BBB/VVV is 110/111: read one more byte and check ExtC/ExtV flag, 
 *         extend length and repeat (if AAA is 111, append to opcode)
 *    -> if BBB/VVV is not 110/111, read end
 *    -> match opcode and operands
 *    
 * 
 *  encode:
 *    -> you should encode opcode and all flags.
 *    -> then you can fill the operands to blank bytes.
 * 
 *  ┌────────────────┬───────────┬──────────────────────────────────┐
 *  │   ExtC flag    │  A field  │             B field              │
 *  ├────────────────┼───────────┼──────────────────────────────────┤
 *  │       1        │    ???    │               ???                │
 *  ├────────────────┼───────────┼──────────────────────────────────┤
 *  │  000 000 00    │    ...    │ 000.x: (110->000) off(0)         │
 *  │  ─── ─── ──    │   Same    │ 001.z: imm(4)   / imm(8)         │
 *  │  BBB AAA MM    │    as     │ 010.z: imm(4,4) / imm(8,8)       │
 *  │                │  A field  │ ... (Same as B field)            │
 *  │                │           │ 110.x: (110->110) off(0)ExtC     │
 *  │                │           │ 111.x: (110->111) off(0)ExtV     │
 *  └────────────────┴───────────┴──────────────────────────────────┘
 * 
 *  Extension Constant(Same as A&B field):
 *    -> find in first Byte 0b110 in forward flag.
 *    -> Read 1 more Byte and check flag for length of fields.
 *    -> MM : Mem accessing enhance mode, 00: 8-byte, 01: 16-byte, 10: 32-byte, 11: 64-byte.
 * 
 *  ┌────────────────┬───────────────────────────────────────────┐
 *  │   ExtV flag    │               ExtV Field                  │
 *  ├────────────────┼───────────────────────────────────────────┤
 *  │       1        │                   ???                     │
 *  ├────────────────┼───────────────────────────────────────────┤
 *  │   000  00000   │   000.x: (111->000) off(0)                │
 *  │   ───  ─────   │   001.x: vec,len                          │
 *  │   VVV  index   │   010.x: vec,vec,len                      │
 *  │                │   ... (User define Operand Pattern)       │
 *  │        00002   │   110.x: (111->110) off(0)ExtC            │
 *  │  (ExtV `VEC`)  │   111.x: (111->111) off(0)ExtV            │
 *  └────────────────┴───────────────────────────────────────────┘
 * 
 *  Extension Variable(User define field):
 *    -> find in first Byte 0b111 in forward flag.
 *    -> Read 1 more Byte and check ExtV table index.
 *    -> According to index deal with operands.
 * 
 * ```
 * 
 * - i386/x86_64 format:
 * ```txt
 * (Variable-length/Byte): MAX 15 Bytes. --> Little Endian View.
 *  ┌──────────────────┬─────────────────────────────────────────┐
 *  │     Type: X      │          Arch: i386, x86_64             │
 *  ├──────┬───────────┼─────┬──────────────┬────────────┬───┬───┤
 *  │ Pref │    Rex    │ Opc │    ModR/M    │    SIB     │ D │ I │
 *  ├──────┼───────────┼─────┼──────────────┼────────────┼───┼───┤
 *  │ 0,1  │    0,1    │ 1~3 │     0,1      │    0,1     │ 0 │ 0 │
 *  ├──────┼───────────┼─────┼──────────────┼────────────┤ 1 │ 1 │
 *  │ insn │ 0100 1101 │ ??? │  ?? ??? ???  │ ?? ??? ??? │ 2 │ 2 │
 *  │ addr │ ──── ──── │ ─── │  ── ─── ───  │ ── ─── ─── │ 4 │ 4 │
 *  │ .... │ patt WRXB │ po  │ mod r/op r/m │ ss idx bas │   │ 8'│
 *  └──────┴───────────┴─────┴──────────────┴────────────┴───┴───┘
 *  Default Env:        64bit    32bit   16bit
 *    - address-size:    64       32      16
 *    - operand-size:    32       32      16
 *  Opcode:
 *    0. po: 1~3 Byte of Opcode such as: ADD eax, i32 (po=0x05).
 *    1. trans2: 2 Byte po, if first Byte is 0x0f.
 *    2. trans3: 3 Byte po, if fisrt and second Bytes are 0x0f 38.
 *    3. field extention: po need to concat ModR/M.op(3-bits) field.
 *  Imm(I):
 *    0. imm: (0,1,2,4,8) Byte of Imm such as: ADD eax 0X4351FF23 (imm=0x23 ff 51 43).
 *  Hidden Reg:
 *    0. eax: when po=0x05, auto use eax as target reg. (Insn=0x05 23 ff 51 43)
 *  ModR/M :
 *    0~2. r/m - As Direct/Indirect operand(E): Reg/Mem.
 *    3~5. r/op - As Reg ref(G), or as 3-bit opcode extension.
 *    6~7. mod - 0b00: [base], 0b01: [base + disp8], 0b10: [base + disp32], 0b11: Reg.
 *    Such as: ADD ecx, esi (po=0x01), set ModR/M : 0b11 110(esi) 001(ecx)=0xf1.
 *    Get (Insn=0x01 f1)
 *  Prefixs(Legacy):
 *    - instruction prefix
 *    - address-size override prefix: 0x67(Default: 32 -> 16)
 *    - operand-size override prefix: 0x66(Default: 32 -> 16)
 *    - segment override prefix: 0x2e(CS) 0x3e(DS) 0x26(ES) 0x36(SS) 0x64(FS) 0x65(GS)
 *    - repne/repnz prefix: 0xf2 0xf3
 *    - lock prefix: 0xf0
 *    Such as: MOV r/m32, r32 (po=0x89), set opr-prefix: 0x66
 *    Get MOV r/m16, r16 (Insn=0x66 89 ..)
 *  SIB :
 *    0~2. base
 *    3~5. index
 *    6~7. scale - 0b00: [idx], 0b01: [idx*2], 0b10: [idx*4], 0b11: [idx*8].
 *  Rex Prefix(Only x86_64):
 *    0. B - Extension of SIB.base field.
 *    1. X - Extension of SIB.idx field.
 *    2. R - Extension of ModR/M.reg field (Reg Num: 8 -> 16).
 *    3. W - 0: 64-bits operand, 1: default(32-bits) operand.
 *    5~8. 0100 - Fixed bit patten.
 *    Such as: ADD rcx, rsi (po=0x01, 64-bits), set Rex: 0b0100 1000=0x48.
 *    Get (Insn=0x48 01 f1)
 *    Such as: ADD rcx, r9(0b1001) (po=0x01, 64-bits), set Rex: 0b0100 1100=0x4c.
 *    Set ModR/M : 0b11 001 001=0xc9, Get (Insn=0x4c 01 c9)
 *  Disp(D):
 *    0. imm: (0,1,2,4) Byte of Imm as addr disp.
 * ```
 * 
 * ### Pattern
 * 
 * ``` txt
 * 
 * ----------------------------------------------------------------------------------------
 * 
 * Pattern (Ctrl)
 *  - Insn      :   $name
 *  - Insn Sep  :   ;
 *  - Insn Bit  :   I(scl)(<numb>)([hi:lo|...])
 *  - 
 * 
 * Pattern (Oprs)
 *  - Off       :   x
 *  - All       :   o
 *  - Reg       :   r(idx)(<name>)([hi:lo|...])
 *  - Imm       :   i(scl)(<numb>)([hi:lo|...])
 *  - Mem       :   m(scl)(<flag>)([hi:lo|...])
 *  - Lab       :   l([hi:lo|...])
 *  - Cond      :
 *  - Extend    : 
 *  - Shift     :   s()
 *  - Option    :   t(000: UXTB, 001: UXTH, 010: UXTW(LSL 32-bits), 011: UXTX(LSL 64-bits), 100: SXTB, 101: SXTH, 110: SXTW, 111: SXTX)
 * 
 * ----------------------------------------------------------------------------------------
 * 
 * Pattern (RV)
 *  - Opcode    :   rvop    = I[ 6: 0]
 *  - Funct3    :   rvf3    = I[14:12]
 *  - Funct7    :   rvf7    = I[31:25]
 *  - Reg Dest  :   rvrd    = r[11: 7]
 *  - Reg Src1  :   rvr1    = r[19:15]
 *  - Reg Src2  :   rvr2    = r[24:20]
 *  - Imm I     :   rvii    = i[31:20]                  - (12-bits)
 *  - Imm S     :   rvis    = i[11:7|31:25]             - (12-bits)
 *  - Imm B     :   rvib    = i[11:8|30:25|7|31]        - (12-bits)
 *  - Imm U     :   rviu    = i[31:12]                  - (20-bits)
 *  - Imm J     :   rvij    = i[30:21|20|19:12|31]      - (20-bits)
 * 
 * Pattern (RV-CSR)
 *  - Reg Csr1  :   rvrt    = r[ 6: 2]                  - Csr (16-bits Insn)
 *  - Reg Csr2  :   rvru    = r[ 9: 7]                  - Csr (16-bits Insn)
 *  - Reg Csr3  :   rvrv    = r[ 4: 2]                  - Csr (16-bits Insn)
 *  - Imm K     :   rvik    = i[12|6:2]                 - Csr (16-bits Insn) (6-bits)
 * 
 * ----------------------------------------------------------------------------------------
 * 
 * Pattern (ARM64)
 *  - Opcode    :   a64op   = I[31:21]
 *  - Reg Dest  :   a64rd   = r[ 4: 0]
 *  - Reg Src1  :   a64rn   = r[ 9: 5]
 *  - Reg Src2  :   a64rm   = r[20:16]
 *  - Reg Src3  :   a64ra   = r[14:10]
 *  - Imm 26b   :   a64i26  = i[25: 0]
 *  - Imm 5b    :   a64i5   = i[20:16]
 *  - Imm 16b   :   a64i16  = i[20: 5]
 *  - Imm 19b   :   a64i19  = i[23: 5]
 *  - Imm 9b    :   a64i9   = i[20:12]
 *  - Imm 12b   :   a64i12  = i[21:10]
 *  - Imm s     :   a64is   = i[15:10]
 *  - Imm r     :   a64ir   = i[21:16]
 *  - Imm 7b    :   a64i7   = i[21:15]
 * 
 * Pattern (ARM64-SME)
 *  - Imm 6b    :   a64i6   = i[10: 5]
 * 
 * ----------------------------------------------------------------------------------------
 * 
 * Note:
 *  - num       :   [0-9A-F]+                           - Include Hex/Dec/Bin
 *  - dec       :   [0-9]+                              - Dec Integer Number
 *  - hex       :   0x[0-9A-F]+                         - Hex Number
 *  - bin       :   0b[01]+                             - Bin Number
 *  - idx       :   [0..32/64]                          - (Dec) Reg ID Index
 *  - scl       :   [0..3]                              - (Dec) Scale 1 / 2 / 4 / 8 Byte
 *  - flag      :   [...|mm|c|f|s]                      - (Bin) U8 Flag: Signed, Float, Compressed, Reg/Imm Addr Mode and so on ...
 * 
 * ----------------------------------------------------------------------------------------
 * 
 * ```
 */
typedef enum {
    /* ctrl */
    TY_I    =   1 << 0,
    TY_S    =   1 << 1,
    TY_N    =   1 << 2,
    TY_ctrl =   TY_I | TY_S | TY_N,
    /* oprs */
    TY_x    =   1 << 3,
    TY_o    =   1 << 4,
    TY_r    =   1 << 5,
    TY_i    =   1 << 6,
    TY_m    =   1 << 7,
    TY_l    =   1 << 8,
    TY_c    =   1 << 9,
    TY_e    =   1 << 10,
    TY_t    =   1 << 11,
    TY_oprs =   TY_x | TY_o | TY_r | TY_i | TY_m | TY_l | TY_c | TY_e | TY_t ,                     
} TyKd;

typedef struct {
    size_t h;
    size_t l;
} BitMap;

#define BitMap_new(H, L)        { .h = (H), .l = (L) }
#define BitMap_arr(...)         (BitMap[]){ __VA_ARGS__ }
#define BitMap_chk(M, I)        (((M) != NULL) && ((M) + (I)) != NULL)

typedef struct Ty {
    TyKd k;
    char* sym;
    BitMap* map;
    size_t  len;
    u32 flag;
    struct Ty* or;
} Ty;

#define Ty_new(K, V, ...)           { .or = NULL   , .flag = (V) , .k = CONCAT(TY_,K), .map = (BitMap[]){__VA_ARGS__}, .len = (sizeof((BitMap[]){__VA_ARGS__}) / sizeof(BitMap)), .sym = STR(K) }
#define Ty_or(T, K, V, ...)         { .or = &(Ty)T , .flag = (V) , .k = CONCAT(TY_,K), .map = (BitMap[]){__VA_ARGS__}, .len = (sizeof((BitMap[]){__VA_ARGS__}) / sizeof(BitMap)), .sym = STR(K) }
#define Ty_I(V, ...)                Ty_new(I, V, __VA_ARGS__)
#define Ty_S(V, ...)                Ty_new(S, V, __VA_ARGS__)
#define Ty_N(V, ...)                Ty_new(N, V, __VA_ARGS__)
#define Ty_x(V, ...)                Ty_new(x, V, __VA_ARGS__)
#define Ty_o(V, ...)                Ty_new(o, V, __VA_ARGS__)
#define Ty_r(V, ...)                Ty_new(r, V, __VA_ARGS__)
#define Ty_i(V, ...)                Ty_new(i, V, __VA_ARGS__)
#define Ty_m(V, ...)                Ty_new(m, V, __VA_ARGS__)
#define Ty_l(V, ...)                Ty_new(l, V, __VA_ARGS__)
#define Ty_c(V, ...)                Ty_new(c, V, __VA_ARGS__)
#define Ty_e(V, ...)                Ty_new(e, V, __VA_ARGS__)
#define Ty_t(V, ...)                Ty_new(t, V, __VA_ARGS__)
char* Ty_sym(Ty t);

typedef struct {
    Ty* t;
    size_t len;
} Tys;

#define Tys_new(...) { .t = (Ty[]){__VA_ARGS__}, .len = (sizeof((Ty[]){__VA_ARGS__}) / sizeof(Ty)) }
char* Tys_sym(Tys v);

// ==================================================================================== //
//                                    evo: Val
// ==================================================================================== //

typedef struct {
    u8* b;
    size_t len;
} Val;

#define Val_new(...)  { .b = (u8[]){__VA_ARGS__}, .len = (sizeof((u8[]){__VA_ARGS__}) / sizeof(u8)) }
#define Val_zero(N)   { .b = (u8[]){0}, .len = (N) }

#define Val_u8(V) \
    {   .b = (u8[]){   \
        (u8)((V) >>  0), \
        }, .len = 1 }

#define Val_u16(V) \
    {   .b = (u8[]){   \
        (u8)((V) >>  0), \
        (u8)((V) >>  8), \
        }, .len = 2 }

#define Val_u32(V) \
    {   .b = (u8[]){   \
        (u8)((V) >>  0), \
        (u8)((V) >>  8), \
        (u8)((V) >> 16), \
        (u8)((V) >> 24), \
        }, .len = 4 }

#define Val_u64(V) \
    {   .b = (u8[]){  \
        (u8)((V) >>  0), \
        (u8)((V) >>  8), \
        (u8)((V) >> 16), \
        (u8)((V) >> 24), \
        (u8)((V) >> 32), \
        (u8)((V) >> 38), \
        (u8)((V) >> 46), \
        (u8)((V) >> 54), \
        }, len = 8}

Val* Val_str(char* str);
void Val_copy(Val* v, Val* other);
void Val_free(Val* v);
Val* Val_from(Val* val);
Val* Val_from_file(char* path);
Val* Val_inc(Val* v, size_t l);
Val* Val_from_u32(u32* val, size_t len);
Val* Val_new_u8(u8 val);
Val* Val_new_u16(u16 val);
Val* Val_new_u32(u32 val);
Val* Val_new_u64(u64 val);
Val* Val_new_i8(i8 val);
Val* Val_new_i16(i16 val);
Val* Val_new_i32(i32 val);
Val* Val_new_i64(i64 val);
Val* Val_to_i8(Val* v);
Val* Val_to_i16(Val* v);
Val* Val_to_i32(Val* v);
Val* Val_to_i64(Val* v);
Val* Val_to_u8(Val* v);
Val* Val_to_u16(Val* v);
Val* Val_to_u32(Val* v);
Val* Val_to_u64(Val* v);
char* Val_as_hex(Val *v, bool with_tag);
char* Val_as_bin(Val* v);
char* Val_as_str(Val* v);
u8 Val_as_u8(Val *v, size_t i);
u16 Val_as_u16(Val *v, size_t i);
u32 Val_as_u32(Val *v, size_t i);
u64 Val_as_u64(Val *v, size_t i);
i8 Val_as_i8(Val *v, size_t i);
i16 Val_as_i16(Val *v, size_t i);
i32 Val_as_i32(Val *v, size_t i);
i64 Val_as_i64(Val *v, size_t i);
u8 Val_get_u8(Val *v, size_t i);
u16 Val_get_u16(Val *v, size_t i);
u32 Val_get_u32(Val *v, size_t i);
u64 Val_get_u64(Val *v, size_t i);
void Val_set_u8(Val *v, size_t i, u8 val);
void Val_set_u16(Val *v, size_t i, u16 val);
void Val_set_u32(Val *v, size_t i, u32 val);
void Val_set_u64(Val *v, size_t i, u64 val);
Val* Val_set_val(Val* v, size_t idx, Val* val, size_t len);
u8* Val_get_ref(Val* v, size_t idx);
void Val_set_ref(Val* v, size_t idx, u8* val, size_t len);
Val* Val_as_val(Val* v, size_t i, size_t len);
Val* Val_alloc(size_t len);
Val* Val_get_bit(Val *v, size_t hi, size_t lo);
Val* Val_set_bit(Val *v, size_t hi, size_t lo, Val *val);
bool Val_eq_bit(Val *v, size_t hi, size_t lo, Val *val);
u64 Val_get_map(Val *v, BitMap* map, size_t len);
Val* Val_ext_map(Val *v, BitMap* map, size_t len);
Val* Val_set_map(Val *v, BitMap* map, size_t len, u64 val);
Val* Val_wrt_map(Val *v, BitMap* map, size_t len, Val *val);
bool Val_eq_map(Val *v, BitMap* map, size_t len, Val *val);
bool Val_cmp_map(Val *v, BitMap* map, size_t len, Val *val);
Val* Val_ext_map(Val *v, BitMap* map, size_t len);

#define ValHex(V)    Val_as_hex(V, true)
#define ValBin(V)    Val_as_bin(V)

// ==================================================================================== //
//                                    evo: Reg
// ==================================================================================== //

#define RegID(T)      CONCAT(RegID_, T)
#define RegID_T(T, ...) \
    typedef enum {      \
        __VA_ARGS__     \
        T##_REGID_SIZE  \
    } RegID(T)

#define RegID_def(T, ...) \
    RegID_T(T, __VA_ARGS__)

#define RegMax(T)               T##_REGID_SIZE
#define RegTbl(T)               T##_reg_tbl
#define REG(T, ID)              (((ID) < RegMax(T)) ? &RegTbl(T)[ID] : NULL)
#define RegName(T, ID)          REG(T, ID)->name
#define RegAlias(T, ID)         REG(T, ID)->alias
#define RegMap(T, ID)           REG(T, ID)->map
#define RegDef(T)               CONCAT(RegDef_, T)
#define RegDef_OP(T, OP)        CONCAT3(RegDef_, T ## _, OP)
#define RegDef_OP_def(T, OP)    UNUSED RegDef_OP(T, OP)
#define RegDef_T(T)        \
    typedef struct {       \
        RegID(T) id;       \
        const char* name;  \
        const char* alias; \
        BitMap  map;       \
    } RegDef(T)

#define RegDef_def(T, ...)                                  \
    RegDef_T(T);                                            \
    UNUSED static RegDef(T) RegTbl(T)[RegMax(T)] = {__VA_ARGS__};  \
    void RegDef_OP_def(T, displayone)(char* res, size_t i); \
    void RegDef_OP_def(T, display)(char* res);

#define RegDef_fn_def(T) \
    void RegDef_OP_def(T, displayone)(char* res, size_t i) {                                          \
        if (i < RegMax(T)) {                                                                          \
            sprintf(res, "%2d: %-3s   [%3lu:%3lu] (%s)", RegTbl(T)[i].id, RegTbl(T)[i].name, RegTbl(T)[i].map.h , RegTbl(T)[i].map.l , RegTbl(T)[i].alias);   \
        }                                                                                             \
    }                                                                                                 \
    void RegDef_OP_def(T, display)(char* res) {                                                       \
        for (size_t i = 0; i < RegMax(T); i++) {                                                      \
            sprintf(res, "%2d: %-3s (%s)\n", RegTbl(T)[i].id, RegTbl(T)[i].name, RegTbl(T)[i].alias); \
        }                                                                                             \
    }

#define RegDef_display(T, res)  RegDef_OP(T, display)(res)
#define RegDef_displayone(T, res, i)  RegDef_OP(T, displayone)(res, i)


// ==================================================================================== //
//                                    evo: Insn
// ==================================================================================== //

#define InsnID(T)               CONCAT(InsnID_, T)
#define InsnID_T(T, ...) \
    typedef enum {       \
        T##_NOP = 0,     \
        __VA_ARGS__      \
        T##_INSNID_SIZE  \
    } InsnID(T)

#define InsnID_def(T, ...) \
    InsnID_T(T, __VA_ARGS__)

#define InsnMax(T)              T##_INSNID_SIZE
#define InsnTbl(T)              T##_insn_tbl
#define INSN(T, ID)             ((ID < InsnMax(T)) ? &InsnTbl(T)[ID] : NULL)
#define InsnMnem(T, ID)         INSN(T, ID)->mnem
#define InsnFlag(T, ID)         INSN(T, ID)->flag
#define InsnBC(T, ID)           INSN(T, ID)->bc
#define InsnTC(T, ID)           INSN(T, ID)->tc
#define InsnDef(T)              CONCAT(InsnDef_, T)
#define InsnDef_OP(T, OP)       CONCAT3(InsnDef_, T ## _, OP)
#define InsnDef_OP_def(T, OP)   UNUSED InsnDef_OP(T, OP)
#define InsnDef_T(T)      \
    typedef struct {      \
        InsnID(T) id;     \
        const char* mnem; \
        u32 flag;         \
        Val bc;           \
        Tys tc;           \
        Tys tr;           \
    } InsnDef(T)

#define InsnDef_def(T, ...)                                          \
    InsnDef_T(T);                                                    \
    UNUSED static InsnDef(T) InsnTbl(T)[InsnMax(T)] = { [T##_NOP]    = { .id = T##_NOP    , .mnem = "nop"     , .bc = Val_u32(0x00) }, __VA_ARGS__}; \
    bool InsnDef_OP_def(T, match)(Val * bc, size_t i);               \
    void InsnDef_OP_def(T, displayone)(char* res, size_t i);         \
    void InsnDef_OP_def(T, display)(char* res);

#define InsnDef_fn_def(T)                                                                                          \
    bool InsnDef_OP_def(T, match)(Val * bc, size_t i) {                                                            \
        InsnDef(T)* insn = &InsnTbl(T)[i];                                                                         \
        Log_ast(bc->len >= insn->bc.len, "InsnDef: bc len mismatch %s <= %s", ValHex(bc), ValHex(&(insn->bc)));    \
        bool is_match = false;                                                                                     \
        Val* v = &(insn->bc);                                                                                      \
        for (size_t j = 0; j < insn->tc.len; j++) {                                                                \
            BitMap* bm = (insn->tc.t[j]).map;                                                                      \
            size_t bml = (insn->tc.t[j]).len;                                                                      \
            is_match = Val_cmp_map(v, bm, bml, bc);                                                                \
            if (!is_match) {                                                                                       \
                Log_warn("InsnDef: bc %s mismatch insn %s[%lu:%lu]", ValHex(bc), insn->mnem, bm->h, bm->l);        \
                break;                                                                                             \
            }                                                                                                      \
        }                                                                                                          \
        return is_match;                                                                                           \
    }                                                                                                              \
    void InsnDef_OP_def(T, displayone)(char* res, size_t i) {                                                      \
        if (i < InsnMax(T)) {                                                                                      \
            sprintf(res, "%-14s %s %s", ValHex(&InsnTbl(T)[i].bc), InsnTbl(T)[i].mnem, Tys_sym(InsnTbl(T)[i].tr)); \
        }                                                                                                          \
    }                                                                                                              \
    void InsnDef_OP_def(T, display)(char* res) {                                                                   \
        for (size_t i = 0; i < InsnMax(T); i++) {                                                                  \
            sprintf(res, "%s\n", InsnTbl(T)[i].mnem);                                                              \
        }                                                                                                          \
    }

#define InsnDef_match(T, bc, i) InsnDef_OP(T, match)(bc, i)
#define InsnDef_display(T, res) InsnDef_OP(T, display)(res)
#define InsnDef_displayone(T, res, i)  InsnDef_OP(T, displayone)(res, i)

#define Insn(T) CONCAT(Insn_, T)
#define Insn_OP(T, OP) CONCAT3(Insn_, T##_, OP)
#define Insn_OP_def(T, OP) UNUSED Insn_OP(T, OP)
#define Insn_T(T, S)      \
    typedef struct {      \
        InsnID(T) id;     \
        Val bc;           \
        Val** oprs;       \
        size_t len;       \
        u32 flag;         \
        S                 \
    } Insn(T)

#define Insn_def(T, S, ...)                                    \
    Insn_T(T, S);                                              \
    void Insn_OP_def(T, display)(Insn(T) * insn, char* res);   \
    Insn(T) * Insn_OP_def(T, new)(size_t id);                  \
    Insn(T) * Insn_OP_def(T, match)(Val * bc);                 \
    void Insn_OP_def(T, encode)(Insn(T) * insn, Val * args[]); \
    Insn(T) * Insn_OP_def(T, decode)(Val * bc);                \
    __VA_ARGS__

#define Insn_fn_def(T)                                                                                                \
    void Insn_OP_def(T, display)(Insn(T) * insn, char* res) {                                                         \
        char res_buf[32];                                                                                             \
        res[0] = '\0';                                                                                                \
        sprintf(res_buf, "%-14s %s ", ValHex(&insn->bc), InsnTbl(T)[insn->id].mnem);                                  \
        strcat(res, res_buf);                                                                                         \
        for (size_t i = 0; i < insn->len; i++) {                                                                      \
            if (insn->oprs[i] != NULL) {                                                                              \
                if (InsnTbl(T)[insn->id].tr.t[i].k == TY_r) {                                                         \
                    sprintf(res_buf, "%s", RegName(RV, Val_as_u8(insn->oprs[i], 0)));                                 \
                } else {                                                                                              \
                    sprintf(res_buf, "%s%s", Val_as_hex(insn->oprs[i], false), Ty_sym(InsnTbl(T)[insn->id].tr.t[i])); \
                }                                                                                                     \
                strcat(res, res_buf);                                                                                 \
                if (i != insn->len - 1) {                                                                             \
                    strcat(res, " ");                                                                                 \
                }                                                                                                     \
            }                                                                                                         \
        }                                                                                                             \
    }                                                                                                                 \
    Insn(T) * Insn_OP_def(T, new)(size_t id) {                                                                        \
        Log_ast(id < InsnMax(T), "Invalid instruction id: %lu", id);                                                  \
        Insn(T)* res = malloc(sizeof(Insn(T)));                                                                       \
        res->id = id;                                                                                                 \
        res->bc = InsnTbl(T)[id].bc;                                                                                  \
        res->len = InsnTbl(T)[id].tr.len;                                                                             \
        res->oprs = malloc(res->len * sizeof(Val));                                                                   \
        memset(res->oprs, 0, res->len * sizeof(Val));                                                                 \
        res->flag = InsnTbl(T)[id].flag;                                                                              \
        return res;                                                                                                   \
    }                                                                                                                 \
    Insn(T) * Insn_OP_def(T, match)(Val * bc) {                                                                       \
        for (size_t i = 0; i < InsnMax(T); i++) {                                                                     \
            if (InsnDef_match(T, bc, i)) {                                                                            \
                Insn(T)* res = Insn_OP(T, new)(i);                                                                    \
                return res;                                                                                           \
            }                                                                                                         \
        }                                                                                                             \
        return Insn_OP(T, new)(0);                                                                                    \
    }                                                                                                                 \
    void Insn_OP_def(T, encode)(Insn(T) * insn, Val * args[]) {                                                       \
        Log_ast(insn != NULL, "Insn: insn is null");                                                                  \
        Log_ast(args != NULL, "Insn: args are null");                                                                 \
        InsnDef(T)* df = INSN(T, insn->id);                                                                           \
        for (size_t i = 0; i < insn->len; i++) {                                                                      \
            if (args[i] != NULL) {                                                                                    \
                BitMap* bm = (df->tr.t[i]).map;                                                                       \
                size_t bml = (df->tr.t[i]).len;                                                                       \
                insn->oprs[i] = args[i];                                                                              \
                Val_wrt_map(&insn->bc, bm, bml, args[i]);                                                             \
            }                                                                                                         \
        }                                                                                                             \
    }                                                                                                                 \
    Insn(T) * Insn_OP_def(T, decode)(Val * bc) {                                                                      \
        Insn(T)* insn = Insn_OP(T, match)(bc);                                                                        \
        if (insn != NULL) {                                                                                           \
            InsnDef(T)* df = INSN(T, insn->id);                                                                       \
            for (size_t i = 0; i < insn->len; i++) {                                                                  \
                BitMap* bm = (df->tr.t[i]).map;                                                                       \
                size_t bml = (df->tr.t[i]).len;                                                                       \
                insn->oprs[i] = Val_ext_map(bc, bm, bml);                                                             \
                Val_copy(&insn->bc, bc);                                                                              \
            }                                                                                                         \
        }                                                                                                             \
        return insn;                                                                                                  \
    }

#define Insn_display(T, insn, res)  Insn_OP(T, display)(insn, res)
#define Insn_new(T, id) Insn_OP(T, new)(id)
#define Insn_size(T, insn) ((insn)->bc.len)
#define Insn_match(T, bc) Insn_OP(T, match)(bc)
#define Insn_encode(T, insn, args) Insn_OP(T, encode)(insn, args)
#define Insn_decode(T, bc) Insn_OP(T, decode)(bc)

// ==================================================================================== //
//                                    evo: Block
// ==================================================================================== //




// ==================================================================================== //
//                                    evo: CPU
// ==================================================================================== //

typedef enum {
    CPU_IDLE,                   /* CPU During Init/Reset Status */
    CPU_RUN,                    /* CPU Fetch/Decode/Execute Status */
    CPU_STOP,                   /* CPU Stop Running Status */
    CPU_ABORT,                  /* CPU Abort and deal with cause Status */
    CPU_QUIT,                   /* CPU Quit Running Status */
} CPUStatus;
UNUSED static char* cpustatus_tbl1 [] = {
    "IDLE",
    "RUNN",
    "STOP",
    "ABOT",
    "QUIT",
};
UNUSED static char* cpustatus_tbl2 [] = {
    _BLUE("IDLE"),
    _GREEN("RUNN"),
    _YELLOW("STOP"),
    _RED("ABOT"),
    _MAGENTA("QUIT"),
};

#define CPUState(T)              CONCAT(CPUState_, T)
#define CPUState_OP(T, OP)       CONCAT3(CPUState_, T##_, OP)
#define CPUState_OP_def(T, OP)   UNUSED CPUState_OP(T, OP)
#define CPUState_T(T, S)      \
    typedef struct {          \
        CPUStatus status;     \
        Val* pc;              \
        Val* snpc;            \
        Val* dnpc;            \
        Val* reg[RegMax(T)];  \
        Val* mem;             \
        S                     \
    } CPUState(T)

#define CPUState_def(T, S, ...)                                                             \
    CPUState_T(T, S);                                                                       \
    CPUState(T) * CPUState_OP_def(T, init)(size_t mem_size);                                \
    void CPUState_OP_def(T, reset)(CPUState(T) * cpu);                                      \
    void CPUState_OP_def(T, stop)(CPUState(T) * cpu);                                       \
    void CPUState_OP_def(T, abort)(CPUState(T) * cpu);                                      \
    void CPUState_OP_def(T, quit)(CPUState(T) * cpu);                                       \
    void CPUState_OP_def(T, set_mem)(CPUState(T) * cpu, Val * addr, Val * val, size_t len); \
    Val* CPUState_OP_def(T, get_mem)(CPUState(T) * cpu, Val * addr, size_t len);            \
    void CPUState_OP_def(T, set_reg)(CPUState(T) * cpu, size_t id, Val * val);              \
    Val* CPUState_OP_def(T, get_reg)(CPUState(T) * cpu, size_t id);                         \
    void CPUState_OP_def(T, displayreg)(CPUState(T) * cpu, char* res, size_t id);           \
    void CPUState_OP_def(T, display)(CPUState(T) * cpu, char* res);                         \
    Val* CPUState_OP_def(T, fetch)(CPUState(T) * cpu);                                      \
    Insn(T) * CPUState_OP_def(T, decode)(CPUState(T) * cpu, Val * val);                     \
    void CPUState_OP_def(T, execute)(CPUState(T) * cpu, Insn(T) * insn);                    \
    __VA_ARGS__

#define CPUState_fn_def(T)                                                                   \
    CPUState(T) * CPUState_OP_def(T, init)(size_t mem_size) {                                \
        CPUState(T)* cpu = malloc(sizeof(CPUState(T)));                                      \
        cpu->status = CPU_IDLE;                                                              \
        cpu->pc = Val_alloc(8);                                                              \
        cpu->snpc = Val_alloc(8);                                                            \
        cpu->dnpc = Val_alloc(8);                                                            \
        for (size_t i = 0; i < RegMax(T); i++) {                                             \
            cpu->reg[i] = Val_alloc(8);                                                      \
        }                                                                                    \
        cpu->mem = Val_alloc(mem_size / 8);                                                  \
        return cpu;                                                                          \
    }                                                                                        \
    void CPUState_OP_def(T, stop)(CPUState(T) * cpu) {                                       \
        Log_ast(cpu, "CPUState_stop: cpu is null");                                          \
        cpu->status = CPU_STOP;                                                              \
    }                                                                                        \
    void CPUState_OP_def(T, abort)(CPUState(T) * cpu) {                                      \
        Log_ast(cpu, "CPUState_abort: cpu is null");                                         \
        cpu->status = CPU_ABORT;                                                             \
    }                                                                                        \
    void CPUState_OP_def(T, quit)(CPUState(T) * cpu) {                                       \
        Log_ast(cpu, "CPUState_quit: cpu is null");                                          \
        cpu->status = CPU_QUIT;                                                              \
    }                                                                                        \
    void CPUState_OP_def(T, set_mem)(CPUState(T) * cpu, Val * addr, Val * val, size_t len) { \
        Log_ast(cpu, "CPUState_set_mem: cpu is null");                                       \
        Val_set_val(cpu->mem, Val_as_u64(addr, 0), val, len);                                \
    }                                                                                        \
    Val* CPUState_OP_def(T, get_mem)(CPUState(T) * cpu, Val * addr, size_t len) {            \
        Log_ast(cpu, "CPUState_get_mem: cpu is null");                                       \
        Val* val = Val_as_val(cpu->mem, Val_as_u64(addr, 0), len);                           \
        return val;                                                                          \
    }                                                                                        \
    void CPUState_OP_def(T, set_reg)(CPUState(T) * cpu, size_t id, Val * val) {              \
        Log_ast(cpu, "CPUState_set_reg: cpu is null");                                       \
        RegDef(T)* df = REG(T, id);                                                          \
        if (df) {                                                                            \
            size_t idx = df->id;                                                             \
            Val_wrt_map(cpu->reg[idx], &df->map, 1, val);                                    \
        }                                                                                    \
    }                                                                                        \
    Val* CPUState_OP_def(T, get_reg)(CPUState(T) * cpu, size_t id) {                         \
        Log_ast(cpu, "CPUState_get_reg: cpu is null");                                       \
        RegDef(T)* df = REG(T, id);                                                          \
        if (df) {                                                                            \
            size_t idx = df->id;                                                             \
            return Val_ext_map(cpu->reg[idx], &df->map, 1);                                  \
        }                                                                                    \
        return NULL;                                                                         \
    }                                                                                        \
    void CPUState_OP_def(T, reset)(CPUState(T) * cpu) {                                      \
        Log_ast(cpu, "CPUState_reset: cpu is null");                                         \
        cpu->status = CPU_IDLE;                                                              \
        cpu->pc = Val_alloc(8);                                                              \
        for (size_t i = 0; i < RegMax(T); i++) {                                             \
            cpu->reg[i] = Val_alloc(8);                                                      \
        }                                                                                    \
        cpu->mem = Val_alloc(cpu->mem->len);                                                 \
    }                                                                                        \
    void CPUState_OP_def(T, displayreg)(CPUState(T) * cpu, char* res, size_t id) {           \
        Log_ast(cpu, "CPUState_displayreg: cpu is null");                                    \
        RegDef(T)* df = REG(T, id);                                                          \
        if (df) {                                                                            \
            size_t idx = df->id;                                                             \
            sprintf(res, "%3s: %s", RegName(T, id), ValHex(cpu->reg[idx]));                  \
        }                                                                                    \
    }                                                                                        \
    void CPUState_OP_def(T, display)(CPUState(T) * cpu, char* res) {                         \
        Log_ast(cpu, "CPUState_display: cpu is null");                                       \
        sprintf(res, "CPU<%s>: %4s", #T, cpustatus_tbl1[cpu->status]);                       \
    }

#define CPUState_init(T, S)  CPUState_OP(T, init)(S)
#define CPUState_stop(T, C)   CPUState_OP(T, stop)(C)
#define CPUState_abort(T, C)  CPUState_OP(T, abort)(C)
#define CPUState_quit(T, C)   CPUState_OP(T, quit)(C)
#define CPUState_reset(T, C)   CPUState_OP(T, reset)(C)
#define CPUState_set_mem(T, C, A, V, L) CPUState_OP(T, set_mem)(C, A, V, L)
#define CPUState_get_mem(T, C, A, L) CPUState_OP(T, get_mem)(C, A, L)
#define CPUState_set_reg(T, C, ID, V) CPUState_OP(T, set_reg)(C, ID, V)
#define CPUState_get_reg(T, C, ID)  CPUState_OP(T, get_reg)(C, ID)
#define CPUState_displayreg(T, C, S, ID) CPUState_OP(T, displayreg)(C, S, ID)
#define CPUState_display(T, C, S) CPUState_OP(T, display)(C, S)
// Need to Impl
#define CPUState_fetch(T, C) CPUState_OP(T, fetch)(C)
#define CPUState_decode(T, C, V) CPUState_OP(T, decode)(C, V)
#define CPUState_execute(T, C, I) CPUState_OP(T, execute)(C, I)

// ==================================================================================== //
//                                    evo: Task
// ==================================================================================== //


#define TaskCtx(T) CONCAT(TaskCtx_, T)
#define TaskCtx_OP(T, OP) CONCAT3(TaskCtx_, T##_, OP)
#define TaskCtx_OP_def(T, OP) UNUSED TaskCtx_OP(T, OP)
#define TaskCtx_OP_ISA(T, OP, I) CONCAT4(TaskCtx_, T##_, OP##_, I)
#define TaskCtx_OP_ISA_def(T, OP, I) UNUSED TaskCtx_OP_ISA(T, OP, I)
#define TaskCtx_T(T, S) \
    typedef struct {    \
        S               \
    } TaskCtx(T)

#define TaskCtx_def(T, S) \
    TaskCtx_T(T, S);

#define Task(T) CONCAT(Task_, T)
#define Task_OP(T, OP) CONCAT3(Task_, T##_, OP)
#define Task_OP_def(T, OP) UNUSED Task_OP(T, OP)
#define Task_OP_ISA(T, OP, I) CONCAT4(Task_, T##_, OP##_, I)
#define Task_OP_ISA_def(T, OP, I) UNUSED Task_OP_ISA(T, OP, I)
#define Task_dbg(T, ...)  Log_dbg(_MAGENTA("[" #T "] ") __VA_ARGS__)
#define Task_err(T, ...)  Log_err(_MAGENTA("[" #T "] ") __VA_ARGS__)
#define Task_warn(T, ...) Log_warn(_MAGENTA("[" #T "] ") __VA_ARGS__)
#define Task_info(T, ...) Log_info(_MAGENTA("[" #T "] ") __VA_ARGS__)
#define Task_ast(T, expr, ...)  Log_ast(expr, _MAGENTA("[" #T "] ") __VA_ARGS__)
#define Task_str(T) STR(T)
#define Task_T(T)            \
    typedef struct Task(T) { \
        const char* name;    \
        TaskCtx(T) ctx;      \
    } Task(T)

#define Task_def(T, S, ...)                                     \
    TaskCtx_def(T, S);                                          \
    Task_T(T);                                                  \
    __VA_ARGS__                                                 \
    Task(T) * Task_OP_def(T, init)(const char* name, Val* val); \
    void Task_OP_def(T, run)(Task(T) * t);                      \
    void Task_OP_def(T, rundbg)(Task(T) * t, Val * val);        \
    void TaskCtx_OP_def(T, init)(TaskCtx(T) * ctx, Val * val);  \
    void TaskCtx_OP_def(T, run)(TaskCtx(T) * ctx);              \
    void TaskCtx_OP_def(T, rundbg)(TaskCtx(T) * ctx, Val * val);

#define Task_fn_def(T)                                           \
    Task(T) * Task_OP_def(T, init)(const char* name, Val* val) { \
        Task(T)* t = malloc(sizeof(Task(T)));                    \
        t->name = name;                                          \
        TaskCtx_OP(T, init)(&t->ctx, val);                       \
        Task_info(T, "Task `"_YELLOW("%s") "` init", name);      \
        return t;                                                \
    }                                                            \
    void Task_OP_def(T, run)(Task(T) * t) {                      \
        TaskCtx_OP(T, run)(&t->ctx);                             \
    }                                                            \
    void Task_OP_def(T, rundbg)(Task(T) * t, Val * val) {        \
        TaskCtx_OP(T, rundbg)(&t->ctx, val);                     \
    }

#define Task_init(T, name, V) Task_OP(T, init)(name, V)
#define Task_run(T, t) Task_OP(T, run)(t)
#define Task_rundbg(T, t, V) Task_OP(T, rundbg)(t, V)


// ==================================================================================== //
//                                    evo: ISA (Must Last)
// ==================================================================================== //

#if defined(CFG_SISA_EIR) || defined(CFG_IISA_EIR) || defined(CFG_TISA_EIR)
#include <isa/eir/def.h>
#endif
#if defined(CFG_SISA_X86) || defined(CFG_IISA_X86) || defined(CFG_TISA_X86)
#include <isa/x86/def.h>
#endif
#if defined(CFG_SISA_ARM) || defined(CFG_IISA_ARM) || defined(CFG_TISA_ARM)
#include <isa/arm/def.h>
#endif
#if defined(CFG_SISA_RV)  || defined(CFG_IISA_RV)  || defined(CFG_TISA_RV)
#include <isa/rv/def.h>
#endif


#ifdef __cplusplus
}
#endif  // __cplusplus

#endif // _EVO_EVO_H_