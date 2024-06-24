#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>
#include <stdarg.h>

typedef unsigned char u8;

/**
 * Instruction encoding
 *
 *   Each instruction is 1-15 bytes long:
 *   - Legacy prefixes (1-4 bytes, optional) (we do not consider these for now)
 *   - Opcode (1-3 bytes, required)
 *   - ModR/M (1 byte, if required)
 *   - SIB (1 byte, if required)
 *   - Displacement (1, 2, 4, or 8 bytes, if required)
 *   - Immediate (1, 2, 4, or 8 bytes, if required)
 */

enum
{
    ENC_NONE,
    ENC_PR,
    ENC_M,
    ENC_M_DISP,
    ENC_IMM,
    ENC_REL,
    ENC_RM,
    ENC_RM_DISP,
    ENC_RM_IMM,
    ENC_MR,
    ENC_MR_DISP,
    ENC_PR_IMM,

    ENC_COUNT
};

struct AsmIns
{
    int enc;
    int size;
    int opcsz;
    u8 opc[4];
    u8 reg;
    char mnem[8];
    int numop;
    int op1;
    int op2;
};

struct AsmIns isa[200];
int isasz;

void
addins0(
    char *mnem,
    int size,
    int opcsz,
    u8 opc0, u8 opc1, u8 opc2, u8 opc3,
    int enc)
{
    int mnemlen;
    struct AsmIns *ins;

    assert(isasz < 100);
    ins = &isa[isasz++];
    ins->size = size;
    ins->opcsz = opcsz;
    ins->opc[0] = opc0;
    ins->opc[1] = opc1;
    ins->opc[2] = opc2;
    ins->opc[3] = opc3;

    mnemlen = strlen(mnem);
    assert(mnemlen < 8);
    strncpy(ins->mnem, mnem, 7);
    ins->mnem[mnemlen] = 0;
    ins->enc = enc;

    ins->numop = 0;
}

void
addins1(
    char *mnem,
    int size,
    int opcsz,
    u8 opc0, u8 opc1, u8 opc2, u8 opc3,
    int enc,
    u8 reg,
    int op1)
{
    int mnemlen;
    struct AsmIns *ins;

    assert(isasz < 100);
    ins = &isa[isasz++];
    ins->size = size;
    ins->opcsz = opcsz;
    ins->opc[0] = opc0;
    ins->opc[1] = opc1;
    ins->opc[2] = opc2;
    ins->opc[3] = opc3;
    ins->reg = reg;

    mnemlen = strlen(mnem);
    assert(mnemlen < 8);
    strncpy(ins->mnem, mnem, 7);
    ins->mnem[mnemlen] = 0;
    ins->enc = enc;

    ins->numop = 1;
    ins->op1 = op1;
}

void
addins2(
    char *mnem,
    int size,
    int opcsz,
    u8 opc0, u8 opc1, u8 opc2, u8 opc3,
    int enc,
    u8 reg,
    int op1,
    int op2)
{
    int mnemlen;
    struct AsmIns *ins;

    assert(isasz < 100);
    ins = &isa[isasz++];
    ins->size = size;
    ins->opcsz = opcsz;
    ins->opc[0] = opc0;
    ins->opc[1] = opc1;
    ins->opc[2] = opc2;
    ins->opc[3] = opc3;
    ins->reg = reg;

    mnemlen = strlen(mnem);
    assert(mnemlen < 8);
    strncpy(ins->mnem, mnem, 7);
    ins->mnem[mnemlen] = 0;
    ins->enc = enc;

    ins->numop = 2;
    ins->op1 = op1;
    ins->op2 = op2;
}

struct AsmIns*
getins0(char *mnem)
{
    int i;

    for(i = 0;
        i < isasz;
        ++i)
    {
        if(isa[i].numop == 0 &&
           !strcmp(isa[i].mnem, mnem))
        {
            return(&isa[i]);
        }
    }

    return(0);
}

struct AsmIns *
getins1(char *mnem, int optype)
{
    int i;

    for(i = 0;
        i < isasz;
        ++i)
    {
        if(isa[i].numop == 1 &&
           !strcmp(isa[i].mnem, mnem) &&
           isa[i].op1 == optype)
        {
            return(&isa[i]);
        }
    }

    return(0);
}

struct AsmIns *
getins2(char *mnem, int op1type, int op2type)
{
    int i;

    for(i = 0;
        i < isasz;
        ++i)
    {
        if(isa[i].numop == 2 &&
           !strcmp(isa[i].mnem, mnem) &&
           isa[i].op1 == op1type &&
           isa[i].op2 == op2type)
        {
            return(&isa[i]);
        }
    }

    return(0);
}

struct Lbl
{
    char name[33];
    unsigned int addr;
    int def;
};

struct Lbl ltbl[400];
int ltblsz;

struct Lbl *
addlbl(char *name, unsigned int addr)
{
    struct Lbl *lbl;
    int len;

    assert(ltblsz < 100);

    lbl = &ltbl[ltblsz++];

    len = strlen(name);
    assert(len < 33);
    strncpy(lbl->name, name, 33);
    lbl->name[len] = 0;
    lbl->addr = addr;
    lbl->def = 0;

    return(lbl);
}

struct Lbl*
getlbl(char *name)
{
    int i;

    for(i = 0;
        i < ltblsz;
        ++i)
    {
        if(!strcmp(name, ltbl[i].name))
        {
            return(&ltbl[i]);
        }
    }
    return(0);
}

char *srcf;
char *src;
int srcl;

void
asmfatal(char *fmt, ...)
{
    va_list ap;

    printf("[!] ERROR: Line %d: ", srcl);
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");

    exit(1);
}

char buff[256];
unsigned int caddr;
unsigned int csize;

enum
{
    OP_INV,
    OP_REG,
    OP_IMM,
    OP_IND,
    OP_IND_DISP,

    /* sub types */
    OP_LBL,

    OP_COUNT
};

struct Op
{
    int type;
    int subtype;
    int val;
    int disp;
    struct Lbl *lbl;
};

enum
{
    DIR_ZERO,
    DIR_LONG,

    DIR_COUNT
};


struct Line
{
    int num;
    unsigned int addr;
    struct AsmIns *ins;
    struct Op op1;
    struct Op op2;
    struct Op op3;
    struct Line *next;
    int dir;
    int val;
};

struct Line*
mkline(
    int num,
    unsigned int addr,
    struct AsmIns *ins,
    struct Op op1,
    struct Op op2)
{
    struct Line *l;

    l = (struct Line*)malloc(sizeof(struct Line));
    if(l)
    {
        l->num = num;
        l->addr = addr;
        l->ins = ins;
        l->op1 = op1;
        l->op2 = op2;
        l->next = 0;
    }

    return(l);
}

struct Line *firstline;
struct Line *lastline;

#define hexval(d) (((d) >= '0' && (d) <= '9') ? ((d) - '0') : ((d) - 'a' + 10))

int
getreg(char *rbuff)
{
    if(!strcmp(rbuff, "eax"))
    {
        return(0);
    }
    else if(!strcmp(rbuff, "ecx"))
    {
        return(1);
    }
    else if(!strcmp(rbuff, "edx"))
    {
        return(2);
    }
    else if(!strcmp(rbuff, "ebx"))
    {
        return(3);
    }
    else if(!strcmp(rbuff, "esp"))
    {
        return(4);
    }
    else if(!strcmp(rbuff, "ebp"))
    {
        return(5);
    }
    else if(!strcmp(rbuff, "esi"))
    {
        return(6);
    }
    else if(!strcmp(rbuff, "edi"))
    {
        return(7);
    }
    return(-1);
}

int
getop(struct Op *op)
{
    char rbuff[8];
    int i;

    while(*src && isspace(*src) && *src != '\n')
    {
        ++src;
    }

    if(!*src || *src == '\n')
    {
        return(0);
    }

    switch(*src)
    {
        case '$':
        {
            int isneg;
            int base;

            ++src;
            isneg = 0;
            base = 10;
            op->val = 0;
            if(*src == '-')
            {
                ++src;
                isneg = 1;
            }
            if(*src == '0' && *(src+1) == 'x')
            {
                src += 2;
                base = 16;
            }
            if(base == 10)
            {
                if(!isdigit(*src))
                {
                    asmfatal("Invalid immediate");
                }
                while(isdigit(*src))
                {
                    op->val = op->val*base + (*src - '0');
                    ++src;
                }
            }
            else if(base == 16)
            {
                if(!isdigit(*src) && !isxdigit(*src))
                {
                    asmfatal("Invalid immediate");
                }
                while(isxdigit(*src))
                {
                    char d;
                    d = tolower(*src);
                    op->val = op->val*base + hexval(d);
                    ++src;
                }
            }
            else
            {
                asmfatal("Invalid base for immediate");
            }

            if(isneg)
            {
                op->val = -op->val;
            }
            op->type = OP_IMM;
            op->subtype = OP_IMM;
        } break;

        case '%':
        {
            ++src;
            rbuff[0] = 0;
            i = 0;
            while(*src != ',' && !isspace(*src))
            {
                rbuff[i++] = *src;
                ++src;
            }
            rbuff[i] = 0;

            if(i == 0)
            {
                asmfatal("Invalid register");
            }

            op->type = OP_REG;
            op->val = getreg(rbuff);
            if(op->val < 0)
            {
                asmfatal("Invalid register '%s'", rbuff);
            }
        } break;

        case '.':case '_':
        case 'a':case 'b':case 'c':case 'd':case 'e':case 'f':case 'g':case 'h':
        case 'i':case 'j':case 'k':case 'l':case 'm':case 'n':case 'o':case 'p':
        case 'q':case 'r':case 's':case 't':case 'u':case 'v':case 'w':case 'x':
        case 'y':case 'z':
        case 'A':case 'B':case 'C':case 'D':case 'E':case 'F':case 'G':case 'H':
        case 'I':case 'J':case 'K':case 'L':case 'M':case 'N':case 'O':case 'P':
        case 'Q':case 'R':case 'S':case 'T':case 'U':case 'V':case 'W':case 'X':
        case 'Y':case 'Z':
        {
            struct Lbl *lbl;
            char lbuff[33];

            lbuff[0] = 0;
            i = 0;
            while(*src == '.' || *src == '_' || isalnum(*src))
            {
                lbuff[i++] = *src++;
            }
            lbuff[i] = 0;

            lbl = getlbl(lbuff);
            if(!lbl)
            {
                lbl = addlbl(lbuff, 0);
            }

            op->type = OP_IMM;
            op->subtype = OP_LBL;
            op->lbl = lbl;
        } break;

        default:
        {
            int disp;
            int isneg;

            isneg = 0;
            disp = 0;
            if(*src == '-')
            {
                isneg = 1;
                ++src;
            }

            if(isneg && !isdigit(*src))
            {
                asmfatal("Invalid displacement");
            }

            /* TODO: hexadecimal displacement */
            while(isdigit(*src))
            {
                disp = disp*10 + (*src - '0');
                ++src;
            }

            if(isneg)
            {
                disp = -disp;
            }

            if(*src == '(')
            {
                ++src;

                while(*src && *src != '%')
                {
                    ++src;
                }

                if(*src != '%')
                {
                    asmfatal("Invalid indirect operand (expected register)");
                }
                ++src;

                rbuff[0] = 0;
                i = 0;
                while(*src && *src != ',' && !isspace(*src) && *src != ')')
                {
                    rbuff[i++] = *src;
                    ++src;
                }
                rbuff[i] = 0;

                if(i == 0)
                {
                    asmfatal("Invalid register");
                }

                while(*src && isspace(*src) && *src != ')')
                {
                    ++src;
                }

                if(*src != ')')
                {
                    asmfatal("Invalid indirect operand");
                }
                ++src;

                op->val = getreg(rbuff);
                if(op->val < 0)
                {
                    asmfatal("Invalid register '%s'", rbuff);
                }
                op->type = OP_IND;
                if(disp)
                {
                    op->type = OP_IND_DISP;
                    op->disp = disp;
                }
            }
            else
            {
                asmfatal("Invalid operand");
            }
        } break;
    }

    return(1);
}

struct Line *
addline(struct AsmIns *ins, struct Op op1, struct Op op2)
{
    struct Line *l;

    l = mkline(srcl, caddr, ins, op1, op2);
    if(!lastline)
    {
        firstline = l;
        lastline = firstline;
    }
    else
    {
        lastline->next = l;
        lastline = lastline->next;
    }

    return(l);
}

int
asmline()
{
    int i;
    struct AsmIns *ins;
    struct Lbl *lbl;
    struct Line *l;
    char mnem[33];
    int numop;
    struct Op op1 = {0};
    struct Op op2 = {0};

    if(!*src)
    {
        return(0);
    }

    while(isspace(*src))
    {
        if(*src == '\n')
        {
            ++srcl;
        }
        ++src;
    }

    mnem[0] = 0;
    numop = 0;
    op1.type = OP_INV;
    op2.type = OP_INV;
    i = 0;
    while(*src && !isspace(*src))
    {
        mnem[i++] = *src++;
    }
    if(i == 0)
    {
        return(0);
    }
    mnem[i] = 0;

    if(mnem[i-1] == ':')
    {
        mnem[i-1] = 0;
        lbl = getlbl(mnem);
        if(lbl && lbl->def)
        {
            asmfatal("Label '%s' already defined", mnem);
        }
        if(!lbl)
        {
            lbl = addlbl(mnem, caddr);
        }
        else
        {
            lbl->addr = caddr;
        }
        lbl->def = 1;
    }
    else
    {
        if(getop(&op1))
        {
            ++numop;
        }
        while(*src && isspace(*src) && *src != '\n')
        {
            ++src;
        }
        if(*src == ',')
        {
            ++src;
            if(getop(&op2))
            {
                ++numop;
            }
        }

        ins = 0;
        if(!strcmp(mnem, ".zero"))
        {
            if(numop != 1 || op1.type != OP_IMM || op1.val <= 0)
            {
                asmfatal("Invalid operand for directive '.zero'");
            }

            l = addline(0, op1, op2);
            l->dir = DIR_ZERO;
            l->val = op1.val;
            caddr += op1.val;
            csize += op1.val;
        }
        else if(!strcmp(mnem, ".long"))
        {
            if(numop != 1 || op1.type != OP_IMM || op1.val <= 0)
            {
                asmfatal("Invalid operand for directive '.long'");
            }

            l = addline(0, op1, op2);
            l->dir = DIR_LONG;
            l->val = op1.val;
            caddr += 4;
            csize += 4;
        }
        else
        {
            if(numop == 0)
            {
                ins = getins0(mnem);
            }
            else if(numop == 1)
            {
                ins = getins1(mnem, op1.type);
            }
            else if(numop == 2)
            {
                ins = getins2(mnem, op1.type, op2.type);
            }
            else
            {
                asmfatal("Invalid number of operands");
            }

            if(!ins)
            {
                asmfatal("Invalid instruction '%s'", mnem);
            }
            addline(ins, op1, op2);
            caddr += ins->size;
            csize += ins->size;
        }
    }

    return(1);
}

void
emit(FILE *f, u8 b)
{
    fputc((int)b, f);
}

void
emitd(FILE *f, int dw)
{
    emit(f, (u8)((dw & 0x000000ff) >>  0));
    emit(f, (u8)((dw & 0x0000ff00) >>  8));
    emit(f, (u8)((dw & 0x00ff0000) >> 16));
    emit(f, (u8)((dw & 0xff000000) >> 24));
}

int
immval(struct Op *op)
{
    if(op->type != OP_IMM)
    {
        asmfatal("Invalid immediate operand");
    }

    if(op->subtype == OP_IMM)
    {
        return(op->val);
    }
    else if(op->subtype == OP_LBL)
    {
        return(op->lbl->addr);
    }
    else
    {
        asmfatal("Invalid immediate operand subtype");
    }

    return(op->val);
}

#define packmodregrm(mod, reg, rm)\
    ((u8)((((mod) & 0x03) << 6) |\
    (((reg) & 0x07) << 3) |\
    (((rm)  & 0x07) << 0)))

u8
getmodregrm(struct Line *l)
{
    u8 mod;
    u8 reg;
    u8 rm;
    u8 modregrm;

    if(l->op1.type == OP_REG && l->op2.type == OP_REG)
    {
        mod = 3;
        reg = l->op1.val;
        rm = l->op2.val;
    }
    else if((l->op1.type == OP_IND && l->op2.type == OP_REG) ||
            (l->op1.type == OP_REG && l->op2.type == OP_IND))
    {
        mod = 0;
        if(l->ins->enc == ENC_RM)
        {
            reg = l->op2.val;
            rm = l->op1.val;
        }
        else if(l->ins->enc == ENC_MR)
        {
            reg = l->op1.val;
            rm = l->op2.val;
        }
    }
    else if((l->op1.type == OP_IND_DISP && l->op2.type == OP_REG) ||
            (l->op1.type == OP_REG && l->op2.type == OP_IND_DISP))
    {
        mod = 2;
        if(l->ins->enc == ENC_RM_DISP)
        {
            reg = l->op2.val;
            rm = l->op1.val;
        }
        else if(l->ins->enc == ENC_MR_DISP)
        {
            reg = l->op1.val;
            rm = l->op2.val;
        }
    }
    else if(l->op1.type == OP_IMM && l->op2.type == OP_REG)
    {
        mod = 3;
        reg = l->ins->reg;
        rm = l->op2.val;
    }
    else if(l->op1.type == OP_IMM && l->op2.type == OP_IND)
    {
        mod = 0;
        reg = l->ins->reg;
        rm = l->op2.val;
    }
    else
    {
        asmfatal("Unhandled Mod Reg R/M encryption");
    }
        
    modregrm = packmodregrm(mod, reg, rm);
    return(modregrm);
}

void
codegen(FILE *f)
{
    struct Line *l;
    int i;
    struct AsmIns *ins;
    u8 modregrm;

    l = firstline;
    while(l)
    {
        srcl = l->num;
        ins = l->ins;
        if(ins)
        {
            switch(ins->enc)
            {
                case ENC_NONE:
                {
                    for(i = 0;
                        i < ins->opcsz;
                        ++i)
                    {
                        emit(f, ins->opc[i]);
                    }
                } break;

                case ENC_PR:
                {
                    for(i = 0;
                        i < ins->opcsz - 1;
                        ++i)
                    {
                        emit(f, ins->opc[i]);
                    }
                    emit(f, ins->opc[ins->opcsz-1] + l->op1.val);
                } break;

                case ENC_M:
                {
                    u8 mod;

                    for(i = 0;
                        i < ins->opcsz;
                        ++i)
                    {
                        emit(f, ins->opc[i]);
                    }

                    if(l->op1.type == OP_REG)
                    {
                        mod = 3;
                    }
                    else if(l->op1.type == OP_IND)
                    {
                        mod = 0;
                    }
                    else if(l->op1.type == OP_IND_DISP)
                    {
                        mod = 2;
                    }

                    modregrm = packmodregrm(mod, l->ins->reg, l->op1.val);
                    emit(f, modregrm);
                } break;

                case ENC_M_DISP:
                {
                    u8 mod;

                    for(i = 0;
                        i < ins->opcsz;
                        ++i)
                    {
                        emit(f, ins->opc[i]);
                    }

                    if(l->op1.type == OP_REG)
                    {
                        mod = 3;
                    }
                    else if(l->op1.type == OP_IND)
                    {
                        mod = 0;
                    }
                    else if(l->op1.type == OP_IND_DISP)
                    {
                        mod = 2;
                    }

                    modregrm = packmodregrm(mod, l->ins->reg, l->op1.val);
                    emit(f, modregrm);

                    emitd(f, l->op1.disp);
                } break;

                case ENC_IMM:
                {
                    for(i = 0;
                        i < ins->opcsz;
                        ++i)
                    {
                        emit(f, ins->opc[i]);
                    }
                    emitd(f, immval(&l->op1));
                } break;

                case ENC_REL:
                {
                    for(i = 0;
                        i < ins->opcsz;
                        ++i)
                    {
                        emit(f, ins->opc[i]);
                    }
                    emitd(f, immval(&l->op1) - (l->addr + ins->size));
                } break;

                case ENC_RM:
                case ENC_MR:
                {
                    for(i = 0;
                        i < ins->opcsz;
                        ++i)
                    {
                        emit(f, ins->opc[i]);
                    }

                    modregrm = getmodregrm(l);
                    emit(f, modregrm);
                } break;

                case ENC_RM_DISP:
                case ENC_MR_DISP:
                {
                    for(i = 0;
                        i < ins->opcsz;
                        ++i)
                    {
                        emit(f, ins->opc[i]);
                    }

                    modregrm = getmodregrm(l);
                    emit(f, modregrm);

                    if(l->op1.type == OP_IND_DISP)
                    {
                        emitd(f, l->op1.disp);
                    }
                    else if(l->op2.type == OP_IND_DISP)
                    {
                        emitd(f, l->op2.disp);
                    }
                } break;

                case ENC_RM_IMM:
                {
                    for(i = 0;
                        i < ins->opcsz;
                        ++i)
                    {
                        emit(f, ins->opc[i]);
                    }

                    modregrm = getmodregrm(l);
                    emit(f, modregrm);

                    if(l->op1.type == OP_IMM)
                    {
                        emitd(f, immval(&l->op1));
                    }
                    else if(l->op2.type == OP_IMM)
                    {
                        emitd(f, immval(&l->op2));
                    }
                } break;

                case ENC_PR_IMM:
                {
                    for(i = 0;
                        i < ins->opcsz - 1;
                        ++i)
                    {
                        emit(f, ins->opc[i]);
                    }
                    emit(f, ins->opc[ins->opcsz-1] + l->op2.val);
                    emitd(f, immval(&l->op1));
                } break;

                default:
                {
                    asmfatal("Unhandled encryption");
                } break;
            }
        }
        else
        {
            switch(l->dir)
            {
                case DIR_ZERO:
                {
                    for(i = 0;
                        i < l->val;
                        ++i)
                    {
                        emit(f, 0);
                    }
                } break;

                case DIR_LONG:
                {
                    emitd(f, l->val);
                } break;

                default:
                {
                    asmfatal("Unhandled directive");
                } break;
            }
        }
        l = l->next;
    }
}

u8 elfhdr[0x54] = {
    /* 00 */ 0x7f,0x45,0x4c,0x46,
    /* 04 */ 0x01,0x01,0x01,0x00,
    /* 08 */ 0x00,0x00,0x00,0x00,
    /* 0c */ 0x00,0x00,0x00,0x00,
    /* 10 */ 0x02,0x00,0x03,0x00,
    /* 14 */ 0x01,0x00,0x00,0x00,
    /* 18 */ 0x54,0x80,0x04,0x08,
    /* 1c */ 0x34,0x00,0x00,0x00,
    /* 20 */ 0x00,0x00,0x00,0x00,
    /* 24 */ 0x00,0x00,0x00,0x00,
    /* 28 */ 0x34,0x00,0x20,0x00,
    /* 2c */ 0x01,0x00,0x28,0x00,
    /* 30 */ 0x00,0x00,0x00,0x00,
    /* 34 */ 0x01,0x00,0x00,0x00,
    /* 38 */ 0x54,0x00,0x00,0x00,
    /* 3c */ 0x54,0x80,0x04,0x08,
    /* 40 */ 0x00,0x00,0x00,0x00,
    /* 44 */ 0x00,0x00,0x00,0x00, /* Size of segment in the file */
    /* 48 */ 0x00,0x00,0x00,0x00, /* Size of segment in memory */
    /* 4c */ 0x07,0x00,0x00,0x00,
    /* 50 */ 0x00,0x10,0x00,0x00
};

int
assemble(char *fnamein, char *fnameout)
{
    FILE *fin;
    FILE *fout;
    char *_src;
    int i;

    srcf = fnamein;
    fin = fopen(srcf, "r");
    fseek(fin, 0, SEEK_END);
    i = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    _src = (char*)malloc(i+1);
    if(!_src)
    {
        printf("[!] ERROR: Cannot read from file '%s'", srcf);
        exit(1);
    }
    i = fread(_src, 1, i, fin);
    _src[i] = 0;
    fclose(fin);

    /* Bootstraping opcode table */
    addins0("syscall", 2, 2, 0xcd, 0x80, 0x00, 0x00, ENC_NONE);
    addins0("leave",   1, 1, 0xc9, 0x00, 0x00, 0x00, ENC_NONE);
    addins0("ret",     1, 1, 0xc3, 0x00, 0x00, 0x00, ENC_NONE);
    addins0("nop",     1, 1, 0x90, 0x00, 0x00, 0x00, ENC_NONE);
    addins0("hlt",     1, 1, 0xf4, 0x00, 0x00, 0x00, ENC_NONE);
    addins0("cdq",     1, 1, 0x99, 0x00, 0x00, 0x00, ENC_NONE);

    addins1("incl",    1, 1, 0x40, 0x00, 0x00, 0x00, ENC_PR,     0x00, OP_REG);
    addins1("incl",    2, 1, 0xff, 0x00, 0x00, 0x00, ENC_M,      0x00, OP_IND);
    addins1("incl",    6, 1, 0xff, 0x00, 0x00, 0x00, ENC_M_DISP, 0x00, OP_IND_DISP);

    addins1("decl",    1, 1, 0x48, 0x00, 0x00, 0x00, ENC_PR,     0x00, OP_REG);
    addins1("decl",    2, 1, 0xff, 0x00, 0x00, 0x00, ENC_M,      0x01, OP_IND);
    addins1("decl",    6, 1, 0xff, 0x00, 0x00, 0x00, ENC_M_DISP, 0x01, OP_IND_DISP);

    addins1("pushl",   1, 1, 0x50, 0x00, 0x00, 0x00, ENC_PR,     0x00, OP_REG);
    addins1("pushl",   2, 1, 0xff, 0x00, 0x00, 0x00, ENC_M,      0x06, OP_IND);
    addins1("pushl",   6, 1, 0xff, 0x00, 0x00, 0x00, ENC_M_DISP, 0x06, OP_IND_DISP);
    addins1("pushl",   5, 1, 0x68, 0x00, 0x00, 0x00, ENC_IMM,    0x00, OP_IMM);

    addins1("popl",    1, 1, 0x58, 0x00, 0x00, 0x00, ENC_PR,     0x00, OP_REG);
    addins1("popl",    2, 1, 0x8f, 0x00, 0x00, 0x00, ENC_M,      0x00, OP_IND);
    addins1("popl",    6, 1, 0x8f, 0x00, 0x00, 0x00, ENC_M_DISP, 0x00, OP_IND_DISP);

    addins1("notl",    2, 1, 0xf7, 0x00, 0x00, 0x00, ENC_M,      0x02, OP_REG);
    addins1("negl",    2, 1, 0xf7, 0x00, 0x00, 0x00, ENC_M,      0x03, OP_REG);
    addins1("imull",   2, 1, 0xf7, 0x00, 0x00, 0x00, ENC_M,      0x05, OP_REG);
    addins1("idivl",   2, 1, 0xf7, 0x00, 0x00, 0x00, ENC_M,      0x07, OP_REG);
    addins1("sall",    2, 1, 0xd1, 0x00, 0x00, 0x00, ENC_M,      0x04, OP_REG);
    addins1("sarl",    2, 1, 0xd1, 0x00, 0x00, 0x00, ENC_M,      0x07, OP_REG);

    addins1("call",    5, 1, 0xe8, 0x00, 0x00, 0x00, ENC_REL,    0x00, OP_IMM);

    addins1("jmp",     5, 1, 0xe9, 0x00, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("ja",      6, 2, 0x0f, 0x87, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jae",     6, 2, 0x0f, 0x83, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jb",      6, 2, 0x0f, 0x82, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jbe",     6, 2, 0x0f, 0x86, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jc",      6, 2, 0x0f, 0x82, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("je",      6, 2, 0x0f, 0x84, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jz",      6, 2, 0x0f, 0x84, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jg",      6, 2, 0x0f, 0x8f, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jge",     6, 2, 0x0f, 0x8d, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jl",      6, 2, 0x0f, 0x8c, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jle",     6, 2, 0x0f, 0x8e, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jna",     6, 2, 0x0f, 0x86, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jnae",    6, 2, 0x0f, 0x82, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jnb",     6, 2, 0x0f, 0x83, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jnbe",    6, 2, 0x0f, 0x87, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jnc",     6, 2, 0x0f, 0x83, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jne",     6, 2, 0x0f, 0x85, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jng",     6, 2, 0x0f, 0x8e, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jnge",    6, 2, 0x0f, 0x8c, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jnl",     6, 2, 0x0f, 0x8d, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jnle",    6, 2, 0x0f, 0x8f, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jno",     6, 2, 0x0f, 0x81, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jnp",     6, 2, 0x0f, 0x8b, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jns",     6, 2, 0x0f, 0x89, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jnz",     6, 2, 0x0f, 0x85, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jo",      6, 2, 0x0f, 0x80, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jp",      6, 2, 0x0f, 0x8a, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jpe",     6, 2, 0x0f, 0x8a, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jpo",     6, 2, 0x0f, 0x8b, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("js",      6, 2, 0x0f, 0x88, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);
    addins1("jz",      6, 2, 0x0f, 0x84, 0x00, 0x00, ENC_REL, 0x00, OP_IMM);

    addins2("movl",    2, 1, 0x89, 0x00, 0x00, 0x00, ENC_RM,      0x00, OP_REG, OP_REG);
    addins2("movl",    2, 1, 0x8b, 0x00, 0x00, 0x00, ENC_RM,      0x00, OP_IND, OP_REG);
    addins2("movl",    6, 1, 0x8b, 0x00, 0x00, 0x00, ENC_RM_DISP, 0x00, OP_IND_DISP, OP_REG);
    addins2("movl",    5, 1, 0xb8, 0x00, 0x00, 0x00, ENC_PR_IMM,  0x00, OP_IMM, OP_REG);
    addins2("movl",    2, 1, 0x89, 0x00, 0x00, 0x00, ENC_MR,      0x00, OP_REG, OP_IND);
    addins2("movl",    6, 1, 0x89, 0x00, 0x00, 0x00, ENC_MR_DISP, 0x00, OP_REG, OP_IND_DISP);
    addins2("movl",    6, 1, 0xc7, 0x00, 0x00, 0x00, ENC_RM_IMM,  0x00, OP_IMM, OP_IND);

    addins2("addl",    2, 1, 0x01, 0x00, 0x00, 0x00, ENC_RM,      0x00, OP_REG, OP_REG);
    addins2("addl",    2, 1, 0x01, 0x00, 0x00, 0x00, ENC_RM,      0x00, OP_IND, OP_REG);
    addins2("addl",    6, 1, 0x01, 0x00, 0x00, 0x00, ENC_RM_DISP, 0x00, OP_IND_DISP, OP_REG);
    addins2("addl",    6, 1, 0x81, 0x00, 0x00, 0x00, ENC_RM_IMM,  0x00, OP_IMM, OP_REG);

    addins2("subl",    2, 1, 0x29, 0x00, 0x00, 0x00, ENC_RM,      0x00, OP_REG, OP_REG);
    addins2("subl",    2, 1, 0x29, 0x00, 0x00, 0x00, ENC_RM,      0x00, OP_IND, OP_REG);
    addins2("subl",    6, 1, 0x29, 0x00, 0x00, 0x00, ENC_RM_DISP, 0x00, OP_IND_DISP, OP_REG);
    addins2("subl",    6, 1, 0x81, 0x00, 0x00, 0x00, ENC_RM_IMM,  0x05, OP_IMM, OP_REG);

    addins2("cmpl",    2, 1, 0x39, 0x00, 0x00, 0x00, ENC_RM,      0x00, OP_REG, OP_REG);
    addins2("cmpl",    2, 1, 0x39, 0x00, 0x00, 0x00, ENC_RM,      0x00, OP_IND, OP_REG);
    addins2("cmpl",    6, 1, 0x39, 0x00, 0x00, 0x00, ENC_RM_DISP, 0x00, OP_IND_DISP, OP_REG);
    addins2("cmpl",    6, 1, 0x81, 0x00, 0x00, 0x00, ENC_RM_IMM,  0x07, OP_IMM, OP_REG);

    caddr = 0x08048054;

    src = _src;
    srcl = 1;
    while(asmline());

#if 0
    for(i = 0;
        i < ltblsz;
        ++i)
    {
        printf("%s:\t\t0x%.8x\n", ltbl[i].name, ltbl[i].addr);
    }
#endif

    fout = fopen(fnameout, "w");
    for(i = 0x00;
        i < 0x44;
        ++i)
    {
        emit(fout, elfhdr[i]);
    }
    emitd(fout, csize + 0x54);
    emitd(fout, csize + 0x54);
    for(i = 0x4c;
        i < 0x54;
        ++i)
    {
        emit(fout, elfhdr[i]);
    }
    codegen(fout);
    fclose(fout);

    return(0);
}

#ifndef ASMORG_API

int main(int argc, char *argv[])
{
    char *fin = 0;
    char *fout;

    fout = "a.out";
    if(argc > 1)
    {
        fin = argv[1];
    }
    else
    {
        printf("Usage:\n %s <input_file> [<output_file>]\n", argv[0]);
        return(1);
    }

    if(argc > 2)
    {
        fout = argv[2];
    }

    assemble(fin, fout);
    return(0);
}

#endif
