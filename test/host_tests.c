
#define _DEFAULT_SOURCE

#include <sob/sob.h>
#include <evo/evo.h>
#include <unistd.h>
#include <sys/mman.h>


typedef int(*AddFunc)(int, int);
UnitTest_fn_def(test_native_exec){
    // x86 func: c = a + b
    u8 BinaryCode[] = {
		0x55,
		0x48, 0x89, 0xe5,
		0x89, 0x7d, 0xfc,
		0x89, 0x75, 0xf8,
		0x8b, 0x55, 0xfc,
		0x8b, 0x45, 0xf8,
		0x01, 0xd0,
		0x5d,
		0xc3
    };
    void* exec_buf = mmap(NULL, sizeof(BinaryCode), PROT_WRITE | PROT_EXEC,  MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    memcpy(exec_buf, BinaryCode, sizeof(BinaryCode));
    AddFunc func = (AddFunc)exec_buf;
    int ret = func(1, 2);
    UnitTest_msg("result: %d", ret);
    munmap(exec_buf, sizeof(BinaryCode));
    return NULL;
}


UnitTest_fn_def(all_tests) {
    UnitTest_add(test_native_exec);
    return NULL;
}

UnitTest_run(all_tests);