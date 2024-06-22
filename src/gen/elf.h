

#ifndef _GEN_ELF_H_
#define _GEN_ELF_H_

#include <elf.h>

#ifdef __cplusplus
extern "C" {
#endif


// typedef ElfN(Addr)      Elf_addr;
// typedef ElfN(Ehdr)      Elf_ehdr;
// typedef ElfN(Phdr)      Elf_phdr;
// typedef ElfN(Shdr)      Elf_shdr;
// typedef ElfN(Dyn)       Elf_dyn;
// typedef ElfN(Sym)       Elf_sym;
// typedef ElfN(Rela)      Elf_rela;
// typedef ElfN(Rel)       Elf_rel;
// typedef ElfN(Word)      Elf_word;
// typedef ElfN(Xword)     Elf_xword;
// typedef ElfN(Sxword)    Elf_sxword;
// typedef ElfN(Off)       Elf_off;
// typedef ElfN(Verdef)    Elf_verdef;
// typedef ElfN(Verneed)   Elf_verneed;
// typedef ElfN(Versym)    Elf_versym;
// typedef ElfN(Half)      Elf_half;


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


// typedef struct {
//     Slist head;
//     Elf_shdr *shdr;
//     int id;
// } SectionList;

// typedef struct {
//     unsigned char *buf;
//     unsigned int size;
// } Section;


// /**
//  * @brief ELF-File Format
//  * - Use `man 5 elf` for more information.
//  */
// typedef struct {
//     Elf_ehdr *ehdr;
//     Elf_phdr *phdr;
//     Elf_shdr *shdr;
//     Elf_dyn *dyn;
//     Elf_sym *sym;
//     Elf_addr base;
//     union {
//         Elf_rela *rela_plt;
//         Elf_rel *rel_plt;
//     };
//     Elf_sxword relplt_type;
//     union {
//         Elf_rela *rela_dyn;
//         Elf_rel *rel_dyn;
//     };
//     Elf_sxword reldyn_type;
//     unsigned char *dynstr;
//     unsigned char *buf;
//     unsigned int size;
//     struct {
//         Elf_word nbucket;
//         Elf_word nchain;
//         Elf_word *bucket;   /* hash table buckets array */
//         Elf_word *chain;    /* hash table chain array */
//     } hash;
//     struct {
//         Elf_word nbucket;
//         Elf_word symidx;    /* first accessible symbol in dynsym table */
//         Elf_word maskwords; /* bloom filter words */
//         Elf_word shift2;    /* bloom filter shift words */
//         Elf_addr *bloom;    /* bloom filter */
//         Elf_word *buckets;  /* hash table buckets array */
//         Elf_word *chain;    /* hash table value array */
//     } gnu_hash;

//     /* section headers sorted by offset */
//     SectionList *section_list;

//     /* Sections not associated with a segment, e.g. .symtab, .strtab etc., and
//      * the section header table. */
//     Section sections;
// } Elf;

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

ElfCtx* ElfCtx_init();
void ElfCtx_free(ElfCtx* t);
void ElfCtx_gen(ElfCtx *t, char* outfile);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif // _GEN_ELF_H_