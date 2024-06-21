


#include <def/elf.h>


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

