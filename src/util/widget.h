
#ifndef UTIL_WIDGET_H
#define UTIL_WIDGET_H


#ifdef __cplusplus
extern "C" {
#endif

struct progressbar {
	int cur;
	int max;
	char *title;
	char *filler;
	int filler_len;
	const char *(*fill)();
	int sym_count;
#define PROGRESSBAR_WIDTH 60
};

struct progressbar *progressbar_new(const char *title, int max, const char *(*fill)());
void progressbar_inc(struct progressbar *bar);
void progressbar_finish(struct progressbar *bar);

#ifdef __cplusplus
}
#endif


#endif  // UTIL_WIDGET_H