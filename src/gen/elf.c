
#include <gen/elf.h>
#include <sob/sob.h>

/**
 * @brief String table section `.strtab`. 
 * Record strings in new section's locations.
 */
typedef struct {
    SectionId id;
    char* name;
    unsigned int len;
} Strtbl;


UNUSED static Strtbl shstrtab[] = {
    { SH_NULL           , ""                ,  0 },
    { SH_INTERP         , ".interp"         ,  7 },
    { SH_TEXT           , ".text"           ,  5 },
    { SH_DYNSTR         , ".dynstr"         ,  7 },
    { SH_DYNAMIC        , ".dynamic"        ,  8 },
    { SH_RELA_DYN       , ".rela.dyn"       ,  9 },
    { SH_REL_DYN        , ".rel.dyn"        ,  8 },
    { SH_RELA_PLT       , ".rela.plt"       ,  9 },
    { SH_REL_PLT        , ".rel.plt"        ,  8 },
    { SH_INIT           , ".init"           ,  5 },
    { SH_GOT_PLT        , ".got.plt"        ,  8 },
    { SH_DATA           , ".data"           ,  5 },
    { SH_DYNSYM         , ".dynsym"         ,  7 },
    { SH_HASH           , ".hash"           ,  5 },
    { SH_GNU_HASH       , ".gnu.hash"       ,  9 },
    { SH_VERNEED        , ".gnu.version_r"  , 14 },
    { SH_VERSYM         , ".gnu.version"    , 12 },
    { SH_FINI           , ".fini"           ,  5 },
    { SH_SHSTRTAB       , ".shstrtab"       ,  9 },
    { SH_PLT_GOT        , ".plt.got"        ,  8 },
    { SH_NOTE           , ".note"           ,  5 },
    { SH_EH_FRAME_HDR   , ".eh_frame_hdr"   , 13 },
    { SH_EH_FRAME       , ".eh_frame"       ,  9 },
    { SH_RODATA         , ".rodata"         ,  7 },
    { SH_INIT_ARRAY     , ".init_array"     , 11 },
    { SH_FINI_ARRAY     , ".fini_array"     , 11 },
    { SH_BSS            , ".bss"            ,  4 },
};


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
        outfile = "a.out";
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