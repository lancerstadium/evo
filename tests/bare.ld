ENTRY(main)

SECTIONS
{
    . = 0x80000000; /* 内存起始地址 */

    .text : {
        *(.text.boot)
        *(.text*)
        KEEP(*(.init))
        KEEP(*(.fini))
        . = ALIGN(4);
    }

    .rodata : {
        *(.rodata)
        *(.rodata*)
        . = ALIGN(4);
    }

    .data : {
        _gp = . + 0x800; /* 设置全局指针 */
        *(.data)
        *(.data*)
        . = ALIGN(4);
    }

    .bss : {
        *(.bss)
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _end = .;
    }

    . = ALIGN(16);
    PROVIDE(_stack_top = 0x80010000); /* 栈顶 */

    PROVIDE(_heap_start = _end);

    /* 手动定义符号 */
    PROVIDE(__init_array_start = .);
    PROVIDE(__init_array_end = .);
    PROVIDE(__fini_array_start = .);
    PROVIDE(__fini_array_end = .);
    PROVIDE(__preinit_array_start = .);
    PROVIDE(__preinit_array_end = .);
    PROVIDE(__ehdr_start = .);

    .debug_frame : {
        *(.debug_frame)
    }
}
