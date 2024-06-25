
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


ELFDump* ELFDump_init() {
    ELFDump* t;
    t = (ELFDump*)malloc(sizeof(ELFDump));
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

void ELFDump_free(ELFDump* t) {

    free(t->code);
    free(t->data);
    free(t->header);
    free(t->symtab);
    free(t->strtab);
    free(t->section);
}

void ELFDump_section_wstr(ELFDump *t, char *vals, int len) {
    for (int i = 0; i < len; i++) {
        t->section[t->section_idx++] = vals[i];
    }
}
void ELFDump_data_wstr(ELFDump *t, char *vals, int len) {
    for (int i = 0; i < len; i++) {
        t->data[t->data_idx++] = vals[i];
    }
}
void ELFDump_header_wbyte(ELFDump *t, int val) {
    t->header[t->header_idx++] = val;
}
void ELFDump_section_wbyte(ELFDump *t, char val) {
    t->section[t->section_idx++] = val;
}
int ELFDump_wint(char *buf, int index, int val) {
    for (int i = 0; i < 4; i++)
        buf[index++] = EBYTE(val, i);
    return index;
}
void ELFDump_header_wint(ELFDump *t, int val) {
    t->header_idx = ELFDump_wint(t->header, t->header_idx, val);
}
void ELFDump_section_wint(ELFDump *t, int val) {
    t->section_idx = ELFDump_wint(t->section, t->section_idx, val);
}
void ELFDump_symbol_wint(ELFDump *t, int val) {
    t->symtab_idx = ELFDump_wint(t->symtab, t->symtab_idx, val);
}
void ELFDump_code_wint(ELFDump *t, int val) {
    t->code_idx = ELFDump_wint(t->code, t->code_idx, val);
}


void ELFDump_align(ELFDump *t) {
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

void ELFDump_gen_sections(ELFDump *t) {
    /* symtab section */
    for (int b = 0; b < t->symtab_idx; b++)
        ELFDump_section_wbyte(t, t->symtab[b]);

    /* strtab section */
    for (int b = 0; b < t->strtab_idx; b++)
        ELFDump_section_wbyte(t, t->strtab[b]);

    /* shstr section; len = 39 */
    ELFDump_section_wbyte(t, 0);
    ELFDump_section_wstr(t, ".shstrtab", 9);
    ELFDump_section_wbyte(t, 0);
    ELFDump_section_wstr(t, ".text", 5);
    ELFDump_section_wbyte(t, 0);
    ELFDump_section_wstr(t, ".data", 5);
    ELFDump_section_wbyte(t, 0);
    ELFDump_section_wstr(t, ".symtab", 7);
    ELFDump_section_wbyte(t, 0);
    ELFDump_section_wstr(t, ".strtab", 7);
    ELFDump_section_wbyte(t, 0);

    /* section header table */

    /* NULL section */
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, 0);

    /* .text */
    ELFDump_section_wint(t, 0xb);
    ELFDump_section_wint(t, 1);
    ELFDump_section_wint(t, 7);
    ELFDump_section_wint(t, ELF_START + t->header_len);
    ELFDump_section_wint(t, t->header_len);
    ELFDump_section_wint(t, t->code_idx);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, 4);
    ELFDump_section_wint(t, 0);

    /* .data */
    ELFDump_section_wint(t, 0x11);
    ELFDump_section_wint(t, 1);
    ELFDump_section_wint(t, 3);
    ELFDump_section_wint(t, t->code_start + t->code_idx);
    ELFDump_section_wint(t, t->header_len + t->code_idx);
    ELFDump_section_wint(t, t->data_idx);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, 4);
    ELFDump_section_wint(t, 0);

    /* .symtab */
    ELFDump_section_wint(t, 0x17);
    ELFDump_section_wint(t, 2);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, t->header_len + t->code_idx + t->data_idx);
    ELFDump_section_wint(t, t->symtab_idx); /* size */
    ELFDump_section_wint(t, 4);
    ELFDump_section_wint(t, t->symbol_idx);
    ELFDump_section_wint(t, 4);
    ELFDump_section_wint(t, 16);

    /* .strtab */
    ELFDump_section_wint(t, 0x1f);
    ELFDump_section_wint(t, 3);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, t->header_len + t->code_idx + t->data_idx +
                          t->symtab_idx);
    ELFDump_section_wint(t, t->strtab_idx); /* size */
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, 1);
    ELFDump_section_wint(t, 0);

    /* .shstr */
    ELFDump_section_wint(t, 1);
    ELFDump_section_wint(t, 3);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, t->header_len + t->code_idx + t->data_idx +
                          t->symtab_idx + t->strtab_idx);
    ELFDump_section_wint(t, 39);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, 0);
    ELFDump_section_wint(t, 1);
    ELFDump_section_wint(t, 0);
}


void ELFDump_gen_header(ELFDump *t) {
    /* ELF header */
    ELFDump_header_wint(t, 0x464c457f);                  /* Magic: 0x7F followed by ELF */
    ELFDump_header_wbyte(t, 1);                          /* 32-bit */
    ELFDump_header_wbyte(t, 1);                          /* little-endian */
    ELFDump_header_wbyte(t, 1);                          /* EI_VERSION */
    ELFDump_header_wbyte(t, 0);                          /* System V */
    ELFDump_header_wint(t, 0);                           /* EI_ABIVERSION */
    ELFDump_header_wint(t, 0);                           /* EI_PAD: unused */
    ELFDump_header_wbyte(t, 2);                          /* ET_EXEC */
    ELFDump_header_wbyte(t, 0);
    ELFDump_header_wbyte(t, ELF_MACHINE);
    ELFDump_header_wbyte(t, 0);
    ELFDump_header_wint(t, 1);                           /* ELF version */
    ELFDump_header_wint(t, ELF_START + t->header_len);   /* entry point */
    ELFDump_header_wint(t, 0x34); /* program header offset */
    ELFDump_header_wint(t, t->header_len + t->code_idx + t->data_idx + 39 +
                         t->symtab_idx +
                         t->strtab_idx); /* section header offset */
    /* flags */
    ELFDump_header_wint(t, ELF_FLAGS);
    ELFDump_header_wbyte(t, 0x34);                       /* header size */
    ELFDump_header_wbyte(t, 0);
    ELFDump_header_wbyte(t, 0x20);                       /* program header size */
    ELFDump_header_wbyte(t, 0);
    ELFDump_header_wbyte(t, 1);                          /* number of program headers */
    ELFDump_header_wbyte(t, 0);
    ELFDump_header_wbyte(t, 0x28);                       /* section header size */
    ELFDump_header_wbyte(t, 0);
    ELFDump_header_wbyte(t, 6);                          /* number of sections */
    ELFDump_header_wbyte(t, 0);
    ELFDump_header_wbyte(t, 5);                          /* section index with names */
    ELFDump_header_wbyte(t, 0);

    /* program header - code and data combined */
    ELFDump_header_wint(t, 1);                           /* PT_LOAD */
    ELFDump_header_wint(t, t->header_len);               /* offset of segment */
    ELFDump_header_wint(t, ELF_START + t->header_len);   /* virtual address */
    ELFDump_header_wint(t, ELF_START + t->header_len);   /* physical address */
    ELFDump_header_wint(t, t->code_idx + t->data_idx);   /* size in file */
    ELFDump_header_wint(t, t->code_idx + t->data_idx);   /* size in memory */
    ELFDump_header_wint(t, 7);                           /* flags */
    ELFDump_header_wint(t, 4);                           /* alignment */
}

void ELFDump_add_symbol(ELFDump *t, char *symbol, int len, int pc) {
    ELFDump_symbol_wint(t, t->strtab_idx);
    ELFDump_symbol_wint(t, pc);
    ELFDump_symbol_wint(t, 0);
    ELFDump_symbol_wint(t, pc == 0 ? 0 : 1 << 16);

    strncpy(t->strtab + t->strtab_idx, symbol, len);
    t->strtab_idx += len;
    t->strtab[t->strtab_idx++] = 0;
    t->symbol_idx++;
}

void ELFDump_gen(ELFDump *t, char* outfile) {
    t->symbol_idx = 0;
    t->symtab_idx = 0;
    t->strtab_idx = 0;
    t->section_idx = 0;

    ELFDump_align(t);
    ELFDump_gen_header(t);
    ELFDump_gen_sections(t);

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