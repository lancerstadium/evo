

#ifndef _DEF_ELF_H_
#define _DEF_ELF_H_

#include <evo/evo.h>
#include <elf.h>

#ifdef __cplusplus
extern "C" {
#endif


#define ELF_WORD_SIZE EVO_WORD_SIZE
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

#if ELF_WORD_SIZE == 64
#define XFMT "0x%lx"
#define AFMT "%lx"
#define UFMT "%lu"
#else
#define XFMT "0x%x"
#define AFMT "%x"
#define UFMT "%u"
#endif

typedef enum {
    SH_NULL,
    SH_INTERP,
    SH_TEXT,
    SH_DYNSTR,
    SH_DYNAMIC,
    SH_RELA_DYN,
    SH_REL_DYN,
    SH_RELA_PLT,
    SH_REL_PLT,
    SH_INIT,
    SH_GOT_PLT,
    SH_DATA,
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
    SH_RODATA,
    SH_INIT_ARRAY,
    SH_FINI_ARRAY,
    SH_GOT,
    SH_BSS,
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
     * the section header table.
     */
    Section sections;
} Elf;



#ifdef __cplusplus
}
#endif  // __cplusplus

#endif // _DEF_ELF_H_