

#ifndef _DEF_ELF_H_
#define _DEF_ELF_H_

#include <evo/evo.h>
#include <elf.h>

#ifdef __cplusplus
extern "C" {
#endif


#define ELF_WORD_SIZE CFG_WORD_SIZE
#define ElfN(X) CONCAT3(Elf, ELF_WORD_SIZE, _ ## X)
#define ELFN(X) CONCAT3(ELF, ELF_WORD_SIZE, _ ## X)
#define ELF_R_SYM ELFN(R_SYM)
#define ELF_ST_BIND ELFN(ST_BIND)
#define ELF_ST_TYPE ELFN(ST_TYPE)

typedef ElfN(Addr)      Elf_addr;
typedef ElfN(Ehdr)      Elf_ehdr;
typedef ElfN(Phdr)      Elf_phdr;
typedef ElfN(Shdr)      Elf_shdr;
typedef ElfN(Dyn)       Elf_dyn;
typedef ElfN(Sym)       Elf_sym;
typedef ElfN(Rela)      Elf_rela;
typedef ElfN(Rel)       Elf_rel;
typedef ElfN(Word)      Elf_word;
typedef ElfN(Xword)     Elf_xword;
typedef ElfN(Sxword)    Elf_sxword;
typedef ElfN(Off)       Elf_off;
typedef ElfN(Verdef)    Elf_verdef;
typedef ElfN(Verneed)   Elf_verneed;
typedef ElfN(Versym)    Elf_versym;
typedef ElfN(Half)      Elf_half;


#define ELF_MACHINE 0xf3
#define ELF_START   0x10000
#define ELF_FLAGS   0
#define ELF_MAX_HEADER 1024
#define ELF_MAX_CODE 262144
#define ELF_MAX_DATA 262144
#define ELF_MAX_SYMTAB 65536
#define ELF_MAX_STRTAB 65536
#define ELF_MAX_SECTION 1024

/**
 * @brief Section Type Identifier
 * 
 * @note
 * ## Normal Section Type
 * 
 */
typedef enum {
    SH_NULL,
    SH_INTERP,
    SH_TEXT,                    /** code section */
    SH_DYNSTR,                  
    SH_DYNAMIC,
    SH_RELA_DYN,
    SH_REL_DYN,
    SH_RELA_PLT,
    SH_REL_PLT,
    SH_INIT,                    /** executable initial code section */
    SH_GOT_PLT,
    SH_DATA,                    /** writable data section */
    SH_DYNSYM,
    SH_HASH,
    SH_GNU_HASH,
    SH_VERNEED,
    SH_VERSYM,
    SH_PLT,
    SH_PLT_GOT,
    SH_FINI,
    SH_SHSTRTAB,
    SH_NOTE,
    SH_EH_FRAME_HDR,
    SH_EH_FRAME,
    SH_RODATA,                  /** read-only data section */
    SH_INIT_ARRAY,
    SH_FINI_ARRAY,
    SH_GOT,
    SH_BSS,                     /** uninitialized data section */
    SECTIONID_SIZE
} SectionId;


typedef struct {
    Slist head;
    Elf_shdr *shdr;
    int id;
} SectionList;

typedef struct {
    unsigned char *buf;
    unsigned int size;
} Section;


/**
 * @brief ELF-File Format
 * - Use `man 5 elf` for more information.
 */
typedef struct {
    Elf_ehdr *ehdr;
    Elf_phdr *phdr;
    Elf_shdr *shdr;
    Elf_dyn *dyn;
    Elf_sym *sym;
    Elf_addr base;
    union {
        Elf_rela *rela_plt;
        Elf_rel *rel_plt;
    };
    Elf_sxword relplt_type;
    union {
        Elf_rela *rela_dyn;
        Elf_rel *rel_dyn;
    };
    Elf_sxword reldyn_type;
    unsigned char *dynstr;
    unsigned char *buf;
    unsigned int size;
    struct {
        Elf_word nbucket;
        Elf_word nchain;
        Elf_word *bucket;   /* hash table buckets array */
        Elf_word *chain;    /* hash table chain array */
    } hash;
    struct {
        Elf_word nbucket;
        Elf_word symidx;    /* first accessible symbol in dynsym table */
        Elf_word maskwords; /* bloom filter words */
        Elf_word shift2;    /* bloom filter shift words */
        Elf_addr *bloom;    /* bloom filter */
        Elf_word *buckets;  /* hash table buckets array */
        Elf_word *chain;    /* hash table value array */
    } gnu_hash;

    /* section headers sorted by offset */
    SectionList *section_list;

    /* Sections not associated with a segment, e.g. .symtab, .strtab etc., and
     * the section header table. */
    Section sections;
} Elf;

typedef struct {
    char* header;
    char* section;
    char* code;
    char* data;
    char* strtab;
    char* symtab;
    int header_len;
    int code_start;
    int code_idx;
    int data_idx;
    int header_idx;
    int symbol_idx;
    int symtab_idx;
    int strtab_idx;
    int section_idx;
} ElfCtx;


ElfCtx* ElfCtx_init() {
    ElfCtx* t;
    t = (ElfCtx*)malloc(sizeof(ElfCtx));
    t->header_len = 0x54;
    t->code_start = ELF_START + t->header_len;
    t->code = malloc(ELF_MAX_CODE);
    t->data = malloc(ELF_MAX_DATA);
    t->header = malloc(ELF_MAX_HEADER);
    t->symtab = malloc(ELF_MAX_SYMTAB);
    t->strtab = malloc(ELF_MAX_STRTAB);
    t->section = malloc(ELF_MAX_SECTION);
    return t;
}

void ElfCtx_free(ElfCtx* t) {

    free(t->code);
    free(t->data);
    free(t->header);
    free(t->symtab);
    free(t->strtab);
    free(t->section);
}

void ElfCtx_section_wstr(ElfCtx *t, char *vals, int len) {
    for (int i = 0; i < len; i++) {
        t->section[t->section_idx++] = vals[i];
    }
}
void ElfCtx_data_wstr(ElfCtx *t, char *vals, int len) {
    for (int i = 0; i < len; i++) {
        t->data[t->data_idx++] = vals[i];
    }
}
void ElfCtx_header_wbyte(ElfCtx *t, int val) {
    t->header[t->header_idx++] = val;
}
void ElfCtx_section_wbyte(ElfCtx *t, char val) {
    t->section[t->section_idx++] = val;
}
int ElfCtx_wint(char *buf, int index, int val) {
    for (int i = 0; i < 4; i++)
        buf[index++] = EBYTE(val, i);
    return index;
}
void ElfCtx_header_wint(ElfCtx *t, int val) {
    t->header_idx = ElfCtx_wint(t->header, t->header_idx, val);
}
void ElfCtx_section_wint(ElfCtx *t, int val) {
    t->section_idx = ElfCtx_wint(t->section, t->section_idx, val);
}
void ElfCtx_symbol_wint(ElfCtx *t, int val) {
    t->symtab_idx = ElfCtx_wint(t->symtab, t->symtab_idx, val);
}
void ElfCtx_code_wint(ElfCtx *t, int val) {
    t->code_idx = ElfCtx_wint(t->code, t->code_idx, val);
}


void ElfCtx_align(ElfCtx *t) {
    int remainder = t->data_idx & 3;
    if (remainder)
        t->data_idx += (4 - remainder);

    remainder = t->symtab_idx & 3;
    if (remainder)
        t->symtab_idx += (4 - remainder);

    remainder = t->strtab_idx & 3;
    if (remainder)
        t->strtab_idx += (4 - remainder);
}

void ElfCtx_gen_sections(ElfCtx *t) {
    /* symtab section */
    for (int b = 0; b < t->symtab_idx; b++)
        ElfCtx_section_wbyte(t, t->symtab[b]);

    /* strtab section */
    for (int b = 0; b < t->strtab_idx; b++)
        ElfCtx_section_wbyte(t, t->strtab[b]);

    /* shstr section; len = 39 */
    ElfCtx_section_wbyte(t, 0);
    ElfCtx_section_wstr(t, ".shstrtab", 9);
    ElfCtx_section_wbyte(t, 0);
    ElfCtx_section_wstr(t, ".text", 5);
    ElfCtx_section_wbyte(t, 0);
    ElfCtx_section_wstr(t, ".data", 5);
    ElfCtx_section_wbyte(t, 0);
    ElfCtx_section_wstr(t, ".symtab", 7);
    ElfCtx_section_wbyte(t, 0);
    ElfCtx_section_wstr(t, ".strtab", 7);
    ElfCtx_section_wbyte(t, 0);

    /* section header table */

    /* NULL section */
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, 0);

    /* .text */
    ElfCtx_section_wint(t, 0xb);
    ElfCtx_section_wint(t, 1);
    ElfCtx_section_wint(t, 7);
    ElfCtx_section_wint(t, ELF_START + t->header_len);
    ElfCtx_section_wint(t, t->header_len);
    ElfCtx_section_wint(t, t->code_idx);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, 4);
    ElfCtx_section_wint(t, 0);

    /* .data */
    ElfCtx_section_wint(t, 0x11);
    ElfCtx_section_wint(t, 1);
    ElfCtx_section_wint(t, 3);
    ElfCtx_section_wint(t, t->code_start + t->code_idx);
    ElfCtx_section_wint(t, t->header_len + t->code_idx);
    ElfCtx_section_wint(t, t->data_idx);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, 4);
    ElfCtx_section_wint(t, 0);

    /* .symtab */
    ElfCtx_section_wint(t, 0x17);
    ElfCtx_section_wint(t, 2);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, t->header_len + t->code_idx + t->data_idx);
    ElfCtx_section_wint(t, t->symtab_idx); /* size */
    ElfCtx_section_wint(t, 4);
    ElfCtx_section_wint(t, t->symbol_idx);
    ElfCtx_section_wint(t, 4);
    ElfCtx_section_wint(t, 16);

    /* .strtab */
    ElfCtx_section_wint(t, 0x1f);
    ElfCtx_section_wint(t, 3);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, t->header_len + t->code_idx + t->data_idx +
                          t->symtab_idx);
    ElfCtx_section_wint(t, t->strtab_idx); /* size */
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, 1);
    ElfCtx_section_wint(t, 0);

    /* .shstr */
    ElfCtx_section_wint(t, 1);
    ElfCtx_section_wint(t, 3);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, t->header_len + t->code_idx + t->data_idx +
                          t->symtab_idx + t->strtab_idx);
    ElfCtx_section_wint(t, 39);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, 0);
    ElfCtx_section_wint(t, 1);
    ElfCtx_section_wint(t, 0);
}


void ElfCtx_gen_header(ElfCtx *t) {
    /* ELF header */
    ElfCtx_header_wint(t, 0x464c457f);                  /* Magic: 0x7F followed by ELF */
    ElfCtx_header_wbyte(t, 1);                          /* 32-bit */
    ElfCtx_header_wbyte(t, 1);                          /* little-endian */
    ElfCtx_header_wbyte(t, 1);                          /* EI_VERSION */
    ElfCtx_header_wbyte(t, 0);                          /* System V */
    ElfCtx_header_wint(t, 0);                           /* EI_ABIVERSION */
    ElfCtx_header_wint(t, 0);                           /* EI_PAD: unused */
    ElfCtx_header_wbyte(t, 2);                          /* ET_EXEC */
    ElfCtx_header_wbyte(t, 0);
    ElfCtx_header_wbyte(t, ELF_MACHINE);
    ElfCtx_header_wbyte(t, 0);
    ElfCtx_header_wint(t, 1);                           /* ELF version */
    ElfCtx_header_wint(t, ELF_START + t->header_len);   /* entry point */
    ElfCtx_header_wint(t, 0x34); /* program header offset */
    ElfCtx_header_wint(t, t->header_len + t->code_idx + t->data_idx + 39 +
                         t->symtab_idx +
                         t->strtab_idx); /* section header offset */
    /* flags */
    ElfCtx_header_wint(t, ELF_FLAGS);
    ElfCtx_header_wbyte(t, 0x34);                       /* header size */
    ElfCtx_header_wbyte(t, 0);
    ElfCtx_header_wbyte(t, 0x20);                       /* program header size */
    ElfCtx_header_wbyte(t, 0);
    ElfCtx_header_wbyte(t, 1);                          /* number of program headers */
    ElfCtx_header_wbyte(t, 0);
    ElfCtx_header_wbyte(t, 0x28);                       /* section header size */
    ElfCtx_header_wbyte(t, 0);
    ElfCtx_header_wbyte(t, 6);                          /* number of sections */
    ElfCtx_header_wbyte(t, 0);
    ElfCtx_header_wbyte(t, 5);                          /* section index with names */
    ElfCtx_header_wbyte(t, 0);

    /* program header - code and data combined */
    ElfCtx_header_wint(t, 1);                           /* PT_LOAD */
    ElfCtx_header_wint(t, t->header_len);               /* offset of segment */
    ElfCtx_header_wint(t, ELF_START + t->header_len);   /* virtual address */
    ElfCtx_header_wint(t, ELF_START + t->header_len);   /* physical address */
    ElfCtx_header_wint(t, t->code_idx + t->data_idx);   /* size in file */
    ElfCtx_header_wint(t, t->code_idx + t->data_idx);   /* size in memory */
    ElfCtx_header_wint(t, 7);                           /* flags */
    ElfCtx_header_wint(t, 4);                           /* alignment */
}

void ElfCtx_add_symbol(ElfCtx *t, char *symbol, int len, int pc) {
    ElfCtx_symbol_wint(t, t->strtab_idx);
    ElfCtx_symbol_wint(t, pc);
    ElfCtx_symbol_wint(t, 0);
    ElfCtx_symbol_wint(t, pc == 0 ? 0 : 1 << 16);

    strncpy(t->strtab + t->strtab_idx, symbol, len);
    t->strtab_idx += len;
    t->strtab[t->strtab_idx++] = 0;
    t->symbol_idx++;
}

void ElfCtx_gen(ElfCtx *t, char* outfile) {
    t->symbol_idx = 0;
    t->symtab_idx = 0;
    t->strtab_idx = 0;
    t->section_idx = 0;

    ElfCtx_align(t);
    ElfCtx_gen_header(t);
    ElfCtx_gen_sections(t);

    if(!outfile) {
        outfile = CFG_GEN_ELF;
    }

    FILE *fp = fopen(outfile, "wb");
    for (int i = 0; i < t->header_idx; i++)
        fputc(t->header[i], fp);
    for (int i = 0; i < t->code_idx; i++)
        fputc(t->code[i], fp);
    for (int i = 0; i < t->data_idx; i++)
        fputc(t->data[i], fp);
    for (int i = 0; i < t->section_idx; i++)
        fputc(t->section[i], fp);
    fclose(fp);
}

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif // _DEF_ELF_H_