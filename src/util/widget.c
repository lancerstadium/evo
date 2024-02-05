
#include "widget.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct progressbar *progressbar_new(const char *title, int max, const char *(*fill)())
{
	struct progressbar *bar;
	int len;

	bar = (struct progressbar *) malloc(sizeof(*bar));
	if (!bar)
		return NULL;

	len = strlen(title);
	bar->title = (char *) malloc(len+1);
	if (!bar->title) {
		free(bar);
		return NULL;
	}

	strncpy(bar->title, title, len);
	bar->title[len] = '\0';
	
	bar->filler_len = strlen(fill());
	len = bar->filler_len * PROGRESSBAR_WIDTH;

	bar->filler = (char *) malloc(len + 1);
	if (!bar->filler) {
		free(bar->title);
		free(bar);
		return NULL;
	}
	
	memset(bar->filler, 0, len);
	bar->filler[len] = '\0';

	bar->cur = 0;
	bar->max = max - 1;

	bar->fill = fill;

	bar->sym_count = 1;

        return bar;
}

void progressbar_inc(struct progressbar *bar)
{
	double percentage;
	int symbols_to_display;
	int pos;
        static int cur_width = 0;

	if (!bar)
		return;

	percentage = (double) bar->cur / bar->max;

        if ((int) (PROGRESSBAR_WIDTH * percentage) > cur_width) {
                symbols_to_display = percentage * PROGRESSBAR_WIDTH;
                
                for (pos = cur_width; pos < symbols_to_display; ++pos)
                        memcpy(bar->filler + pos * bar->filler_len, bar->fill(), bar->filler_len);

                printf("%s [%s%s%*s] %3d%%\r", bar->title, bar->filler, (percentage < 0.99)? ">" : "#", PROGRESSBAR_WIDTH - symbols_to_display, "", (int) (percentage * 100));
                fflush(stdout);
                
                cur_width = PROGRESSBAR_WIDTH * percentage;

                if (bar->cur == bar->max)
                        cur_width = 0;
        }

	++bar->cur;
}

void progressbar_finish(struct progressbar *bar)
{
        if (!bar)
                return;

	free(bar->filler);
	free(bar->title);
	free(bar);

	printf("\n");
}