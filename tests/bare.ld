ENTRY(main)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 1M
}

SECTIONS
{
    . = ORIGIN(RAM); /* 内存起始地址 */

    .text : {
        *(.text*)
    } > RAM

    .rodata : {
        *(.rodata*)
    } > RAM

    .data : {
        _sdata = .;           /* 设置 .data 段的起始地址 */
        *(.data*)
        _edata = .;           /* 设置 .data 段的结束地址 */
    } > RAM

    .bss : {
        _sbss = .;            /* 设置 .bss 段的起始地址 */
        *(.bss*)
        *(COMMON)
        _ebss = .;            /* 设置 .bss 段的结束地址 */
    } > RAM

    . = ALIGN(8);
    _end = .;

    /* 栈段，假设栈大小为16KB */
    .stack (NOLOAD) : {
        . = ALIGN(16);
        _stack_start = .;
        . += 16K;
        _stack_top = .;
    } > RAM

    . = ALIGN(16);
    PROVIDE(_heap_start = _end);

    /* 手动定义符号 */
    PROVIDE(_sidata = LOADADDR(.data));  /* .data 段在 ROM 中的起始地址 */
    PROVIDE(__init_array_start = .);
    PROVIDE(__init_array_end = .);
    PROVIDE(__fini_array_start = .);
    PROVIDE(__fini_array_end = .);
    PROVIDE(__preinit_array_start = .);
    PROVIDE(__preinit_array_end = .);
    PROVIDE(__ehdr_start = .);

    .debug_frame : {
        *(.debug_frame)
    } > RAM
}