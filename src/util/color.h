/**
 * @file color.h
 * @author lancer (you@domain.com)
 * @brief ANSI颜色宏头文件
 * @version 0.1
 * @date 2023-12-25
 * 
 * @copyright Copyright (c) 2023
 * @note ANSI 颜色宏
 * 
 */

#ifndef UTIL_COLOR_H
#define UTIL_COLOR_H

#define UTIL_COLOR_ENABLE

#ifdef UTIL_COLOR_ENABLE

#define ANSI_RESET "\x1b[0m"
#define ANSI_BOLD "\x1b[1m"
#define ANSI_DIM "\x1b[2m"
#define ANSI_ITALIC "\x1b[3m"
#define ANSI_UNDERLINE "\x1b[4m"
#define ANSI_BLINK "\x1b[5m"
#define ANSI_INVERT "\x1b[6m"
#define ANSI_REVERSE "\x1b[7m"
#define ANSI_HIDDEN "\x1b[8m"
#define ANSI_STRIKETHROUGH "\x1b[9m"

#define ANSI_BLACK "\x1b[30m"
#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_YELLOW "\x1b[33m"
#define ANSI_BLUE "\x1b[34m"
#define ANSI_MAGENTA "\x1b[35m"
#define ANSI_CYAN "\x1b[36m"
#define ANSI_WHITE "\x1b[37m"
#define ANSI_DEFAULT "\x1b[39m"

#define ANSI_BRIGHT_BLACK "\x1b[90m"
#define ANSI_BRIGHT_RED "\x1b[91m"
#define ANSI_BRIGHT_GREEN "\x1b[92m"
#define ANSI_BRIGHT_YELLOW "\x1b[93m"
#define ANSI_BRIGHT_BLUE "\x1b[94m"
#define ANSI_BRIGHT_MAGENTA "\x1b[95m"
#define ANSI_BRIGHT_CYAN "\x1b[96m"
#define ANSI_BRIGHT_WHITE "\x1b[97m"
#define ANSI_BRIGHT_DEFAULT "\x1b[99m"

#define ANSI_BG_BLACK "\x1b[40m"
#define ANSI_BG_RED "\x1b[41m"
#define ANSI_BG_GREEN "\x1b[42m"
#define ANSI_BG_YELLOW "\x1b[43m"
#define ANSI_BG_BLUE "\x1b[44m"
#define ANSI_BG_MAGENTA "\x1b[45m"
#define ANSI_BG_CYAN "\x1b[46m"
#define ANSI_BG_WHITE "\x1b[47m"
#define ANSI_BG_DEFAULT "\x1b[49m"

#define ANSI_FMT(msg, ...) ANSI_RESET __VA_ARGS__ msg  ANSI_RESET

#else

#define ANSI_RESET ""
#define ANSI_BOLD ""
#define ANSI_DIM ""
#define ANSI_ITALIC ""
#define ANSI_UNDERLINE ""
#define ANSI_BLINK ""
#define ANSI_INVERT ""
#define ANSI_REVERSE ""
#define ANSI_HIDDEN ""
#define ANSI_STRIKETHROUGH ""

#define ANSI_BLACK ""
#define ANSI_RED ""
#define ANSI_GREEN ""
#define ANSI_YELLOW ""
#define ANSI_BLUE ""
#define ANSI_MAGENTA ""
#define ANSI_CYAN ""
#define ANSI_WHITE ""
#define ANSI_DEFAULT ""

#define ANSI_BRIGHT_BLACK ""
#define ANSI_BRIGHT_RED ""
#define ANSI_BRIGHT_GREEN ""
#define ANSI_BRIGHT_YELLOW ""
#define ANSI_BRIGHT_BLUE ""
#define ANSI_BRIGHT_MAGENTA ""
#define ANSI_BRIGHT_CYAN ""
#define ANSI_BRIGHT_WHITE ""
#define ANSI_BRIGHT_DEFAULT ""

#define ANSI_BG_BLACK ""
#define ANSI_BG_RED ""
#define ANSI_BG_GREEN ""
#define ANSI_BG_YELLOW ""
#define ANSI_BG_BLUE ""
#define ANSI_BG_MAGENTA ""
#define ANSI_BG_CYAN ""
#define ANSI_BG_WHITE ""
#define ANSI_BG_DEFAULT ""

#define ANSI_FMT(msg, ...) msg

#endif



#define _white(s) ANSI_FMT(s, ANSI_WHITE)
#define _black(s) ANSI_FMT(s, ANSI_BLACK)
#define _green(s) ANSI_FMT(s, ANSI_GREEN)
#define _blue(s) ANSI_FMT(s, ANSI_BLUE)
#define _yellow(s) ANSI_FMT(s, ANSI_YELLOW)
#define _cyan(s) ANSI_FMT(s, ANSI_CYAN)
#define _red(s) ANSI_FMT(s, ANSI_RED)
#define _mag(s) ANSI_FMT(s, ANSI_MAGENTA)
#define _bblue(s) ANSI_FMT(s, ANSI_BRIGHT_BLUE)
#define _bmag(s) ANSI_FMT(s, ANSI_BRIGHT_MAGENTA)
#define _bred(s) ANSI_FMT(s, ANSI_BRIGHT_RED)

#define _bold(s) ANSI_FMT(s, ANSI_BOLD ANSI_WHITE)
#define _boldu(s) ANSI_FMT(s, ANSI_UNDERLINE ANSI_BOLD ANSI_WHITE)
#define _redb(s) ANSI_FMT(s, ANSI_BOLD ANSI_RED)
#define _greenb(s) ANSI_FMT(s, ANSI_BOLD ANSI_GREEN)
#define _yellowb(s) ANSI_FMT(s, ANSI_BOLD ANSI_YELLOW)
#define _blackb(s) ANSI_FMT(s, ANSI_BOLD ANSI_BLACK)
#define _magb(s) ANSI_FMT(s, ANSI_BOLD ANSI_MAGENTA)
#define _blacki(s) ANSI_FMT(s, ANSI_ITALIC ANSI_BLACK)



#endif // UTIL_COLOR_H