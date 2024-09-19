#include <evo.h>
#include <evo/util/log.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* Function to initialize a new progress bar */
progressbar_t *progressbar_new(const char *label, size_t max) {
    return progressbar_new_format(label, max, "|#|");
}

/* Function to initialize a new progress bar with custom format */
progressbar_t *progressbar_new_format(const char *label, size_t max, const char *format) {
    progressbar_t *bar = (progressbar_t *)malloc(sizeof(progressbar_t));
    if (!bar) {
        LOG_ERR("Failed to allocate memory for progress bar\n");
        exit(1);
    }

    bar->max = max;
    bar->value = 0;
    bar->start = time(NULL);  // Record the start time
    strncpy(bar->label, label ? label : "", EVO_BAR_LABEL_LEN - 1);
    bar->label[EVO_BAR_LABEL_LEN - 1] = '\0';  // Ensure null-terminated

    if (format && strlen(format) == 3) {
        bar->format.begin = format[0];
        bar->format.fill = format[1];
        bar->format.end = format[2];
    } else {
        bar->format.begin = '|';
        bar->format.fill = '#';
        bar->format.end = '|';
    }

    return bar;
}

/* Function to free the memory used by the progress bar */
void progressbar_free(progressbar_t *bar) {
    free(bar);
}

/* Function to increment the progress bar by one */
void progressbar_inc(progressbar_t *bar) {
    if (bar->value < bar->max) {
        bar->value++;
        progressbar_update(bar, bar->value);
    }
}

/* Function to update the progress bar to a specific value */
void progressbar_update(progressbar_t *bar, size_t value) {
    bar->value = value > bar->max ? bar->max : value;
    unsigned int percent = (unsigned int)((bar->value * 100) / bar->max);
    int num_hashes = (int)(percent / 2);  // Display 50 characters

    printf("\r%s %c", bar->label, bar->format.begin);
    for (int i = 0; i < 50; ++i) {
        if (i < num_hashes) {
            printf("%c", bar->format.fill);
        } else {
            printf(" ");
        }
    }
    printf("%c %3d%%", bar->format.end, percent);
    fflush(stdout);
}

/* Function to update the label of the progress bar */
void progressbar_update_label(progressbar_t *bar, const char *label) {
    strncpy(bar->label, label, EVO_BAR_LABEL_LEN - 1);
    bar->label[EVO_BAR_LABEL_LEN - 1] = '\0';  // Ensure null-terminated
}

/* Function to finish the progress bar display */
void progressbar_finish(progressbar_t *bar) {
    progressbar_update(bar, bar->max);
    printf("\n");  // Move to the next line after completion
}