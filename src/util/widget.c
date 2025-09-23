#include <evo.h>
#include <evo/util/log.h>
#include <evo/util/sys.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

/* Function to get terminal width */
int get_terminal_width() {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    int width = 80;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
        width = csbi.srWindow.Right - csbi.srWindow.Left + 1;
    }
    return width;
#else
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0) {
        return w.ws_col;
    }
    return 80;
#endif
}

/* Helper function to format time as [MM:SS.ms] */
void format_time(char* buffer, double milliseconds) {
    int minutes = (int)(milliseconds / 60000);
    int seconds = (int)(milliseconds / 1000) % 60;
    int ms = (int)milliseconds % 1000;
    snprintf(buffer, 12, "%02d:%02d.%03d", minutes, seconds, ms);
}

/* Function to create a new progress bar */
progressbar_t* progressbar_new(const char *label, size_t max) {
    progressbar_t* bar = (progressbar_t*) malloc(sizeof(progressbar_t));
    if (bar == NULL) {
        fprintf(stderr, "Failed to allocate memory for progressbar\n");
        return NULL;
    }
    bar->max = max;
    bar->value = 0;
    bar->start = sys_time();
    bar->last_update_time = bar->start;
    bar->avg_step_time = 0.0;
    strncpy(bar->label, label, EVO_BAR_LABEL_LEN - 1);
    bar->label[EVO_BAR_LABEL_LEN - 1] = '\0';

    bar->format.begin = '|';
    bar->format.fill = '#';
    bar->format.empty = ' ';
    bar->format.end = '|';

    return bar;
}

/* 创建带格式的进度条 */
progressbar_t* progressbar_new_format(const char *label, size_t max, const char *format) {
    progressbar_t* bar = progressbar_new(label, max);

    if (bar != NULL && format != NULL) {
        size_t len = strlen(format);
        switch (len) {
            case 1: // 只修改填充字符
                bar->format.fill = format[0];
                break;
            case 2: // 修改填充、开始和结束字符
                bar->format.begin = format[0];
                bar->format.fill = format[1];
                bar->format.end = format[0];
                break;
            case 3: // 修改开始、填充、结束字符
                bar->format.begin = format[0];
                bar->format.fill = format[1];
                bar->format.end = format[2];
                break;
            case 4: // 修改开始、填充、空白、结束字符
                bar->format.begin = format[0];
                bar->format.fill = format[1];
                bar->format.empty = format[2];
                bar->format.end = format[3];
                break;
            default:
                break;
        }
    }
    return bar;
}

/* 更新进度条标签 */
void progressbar_update_label(progressbar_t *bar, const char *label) {
    if (bar != NULL && label != NULL) {
        strncpy(bar->label, label, EVO_BAR_LABEL_LEN - 1);
        bar->label[EVO_BAR_LABEL_LEN - 1] = '\0';  // 确保字符串以 NULL 结尾
    }
}

/* Function to update the progress bar to a specific value */
void progressbar_update(progressbar_t *bar, size_t value) {
    double current_time = sys_time();
    double step_time = current_time - bar->last_update_time;

    // Update last update time
    bar->last_update_time = current_time;

    // Calculate average step time
    if (bar->value > 0) {
        bar->avg_step_time = (bar->avg_step_time * (bar->value - 1) + step_time) / bar->value;
    }

    // Update the progress bar value
    bar->value = value > bar->max ? bar->max : value;

    // Get terminal width and adjust progress bar size
    int term_width = get_terminal_width();
    int label_length = strlen(bar->label);              
    int info_length = 40;
    int available_width = term_width - label_length - info_length;

    // Calculate progress bar fill
    int num_hashes = (bar->value * available_width) / bar->max;

    // Calculate elapsed time and estimated time remaining
    double elapsed_time = current_time - bar->start;
    double estimated_total_time = bar->avg_step_time * bar->max;
    double time_remaining = estimated_total_time - elapsed_time;

    // Format time strings
    char elapsed_str[10], remaining_str[10];
    format_time(elapsed_str, elapsed_time);
    format_time(remaining_str, time_remaining > 0 ? time_remaining : 0);

    // Calculate speed
    double speed = bar->value / (elapsed_time > 0 ? elapsed_time : 1);

    // Display progress bar
    printf("\r%s %c", bar->label, bar->format.begin);
    for (int i = 0; i < available_width; ++i) {
        if (i < num_hashes) {
            printf("%c", bar->format.fill);
        } else {
            printf("%c", bar->format.empty);
        }
    }
    printf("%c %zu/%zu [%s<%s, %.2fit/s]", bar->format.end, bar->value, bar->max, elapsed_str, remaining_str, speed);
    fflush(stdout);
}

/* Function to increment the progress bar by one */
void progressbar_inc(progressbar_t *bar) {
    if (bar->value < bar->max) {
        progressbar_update(bar, bar->value + 1);
    }
}

/* Function to free the memory used by the progress bar */
void progressbar_free(progressbar_t *bar) {
    free(bar);
    bar = NULL;
}

/* Function to finish the progress bar display */
void progressbar_finish(progressbar_t *bar) {
    progressbar_update(bar, bar->max);
    printf("\n");
}