#ifndef _GEN_EBC_H_
#define _GEN_EBC_H_

#include <evo/cfg.h>

#ifdef __cplusplus
extern "C" {
#endif


#define EBC_FILE_MAGIC      0xebc00ebc
#define EBC_FILE_VERSION    1


PACKED(struct EBCFileMeta{
    u32 magic;
    u16 version;
    u64 code_size;
    u64 entry;
    u64 mem_size;
    u64 mem_cap;
    u64 ext_size;
});

typedef struct EBCMeta EBCMeta;




#ifdef __cplusplus
}
#endif  // __cplusplus

#endif // _GEN_EBC_H_