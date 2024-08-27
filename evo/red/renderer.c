#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "../evo.h"
#include "../util/log.h"
#include "../util/math.h"
#include "../util/sys.h"


// ==================================================================================== //
//                                  renderer: typedef
// ==================================================================================== //

typedef struct window window_t;
typedef enum {KEY_A, KEY_D, KEY_S, KEY_W, KEY_SPACE, KEY_NUM} keycode_t;
typedef enum {BUTTON_L, BUTTON_R, BUTTON_NUM} button_t;

// ==================================================================================== //
//                                  renderer: gif
// ==================================================================================== //

typedef struct {
    char* name;
} renderer_gif_t;

void renderer_render_gif(renderer_t* rd, render_fn_t rd_fn) {
    if (!rd || !rd->priv || !rd_fn) return;
    renderer_gif_t* priv = rd->priv;
    canvas_t* cav = rd_fn(0);
    int64_t ndelay = 60;       // 0 < ndelay <= 60
    int64_t delays[ndelay];
    int64_t delay = 5;         // 5 ms
    delays[0] = delay;
    for (size_t i = 1; i < ndelay; i++) {
        delays[i] = delay;
        canvas_t* cav_tmp = rd_fn((1.f * i) / 60.f);
        image_push(cav->background, cav_tmp->background);
        // canvas_free(cav_tmp);
    }
    image_set_deloys(cav->background, delays, ndelay);
    canvas_export(cav, priv->name);
    canvas_free(cav);
}

#if defined(EVO_GUI_MODE)

#define EVO_GUI_TITLE   "renderer"
#define EVO_GUI_WIDTH   960
#define EVO_GUI_HEIGHT  720

#if defined(__linux__) && !defined(__ANDROID__)

// ==================================================================================== //
//                                  renderer: linux
// ==================================================================================== //

#include <X11/X.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

typedef struct {
    /* platform */
    Window handle;
    XImage *ximage;
    Display *g_display;
    XContext g_context;
    /* common data */
    image_t *surface;
    int should_close;
    char keys[KEY_NUM];
    char buttons[BUTTON_NUM];
    // callbacks_t callbacks;
    void *userdata;
} renderer_linux_t;

static void open_display_linux(renderer_linux_t *priv) {
    if(!priv) return;
    priv->g_display = XOpenDisplay(NULL);
    if (!priv->g_display) {
        LOG_ERR("Error: cannot open display\n");
        return;
    }
    priv->g_context = XUniqueContext();
}

static void close_display_linux(renderer_linux_t *priv) {
    if(!priv) return;
    XCloseDisplay(priv->g_display);
    priv->g_display = NULL;
}

static Window create_window_linux(Display *g_display, const char *title, int width, int height) {
    int screen = XDefaultScreen(g_display);
    unsigned long border = XWhitePixel(g_display, screen);
    unsigned long background = XBlackPixel(g_display, screen);
    Window root = XRootWindow(g_display, screen);
    Window handle;
    XSizeHints *size_hints;
    XClassHint *class_hint;
    Atom delete_window;
    long mask;

    handle = XCreateSimpleWindow(g_display, root, 0, 0, width, height, 0,
                                 border, background);

    /* not resizable */
    size_hints = XAllocSizeHints();
    size_hints->flags = PMinSize | PMaxSize;
    size_hints->min_width = width;
    size_hints->max_width = width;
    size_hints->min_height = height;
    size_hints->max_height = height;
    XSetWMNormalHints(g_display, handle, size_hints);
    XFree(size_hints);

    /* application name */
    class_hint = XAllocClassHint();
    class_hint->res_name = (char*)title;
    class_hint->res_class = (char*)title;
    XSetClassHint(g_display, handle, class_hint);
    XFree(class_hint);

    /* event subscription */
    mask = KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask;
    XSelectInput(g_display, handle, mask);
    delete_window = XInternAtom(g_display, "WM_DELETE_WINDOW", True);
    XSetWMProtocols(g_display, handle, &delete_window, 1);

    return handle;
}

static void create_surface_linux(renderer_linux_t *priv, int width, int height, image_t **out_surface, XImage **out_ximage) {
    int screen = XDefaultScreen(priv->g_display);
    int depth = XDefaultDepth(priv->g_display, screen);
    Visual *visual = XDefaultVisual(priv->g_display, screen);
    image_t *surface;
    XImage *ximage;

    if(depth != 24 && depth != 32) {
        LOG_ERR("Error: depth %d is not supported\n", depth);
        return;
    }
    surface = image_blank("surface", width, height);
    ximage = XCreateImage(priv->g_display, visual, depth, ZPixmap, 0, (char*)surface->raw->datas, width, height, 32, 0);

    *out_surface = surface;
    *out_ximage = ximage;
}

#endif  // gui platform

#endif  // EVO_GUI_MODE

// ==================================================================================== //
//                                  renderer: API
// ==================================================================================== //

renderer_t* renderer_new(renderer_type_t type) {
    renderer_t* rd = malloc(sizeof(renderer_t));
    rd->type = type;
    switch (type) {
        case RENDERER_TYPE_GIF: {
            renderer_gif_t* priv = malloc(sizeof(renderer_gif_t));
            if (priv) {
                memset(priv, 0, sizeof(renderer_gif_t));
                priv->name = strdup("renderer.gif");
            }
            rd->priv = priv;
            rd->render = renderer_render_gif;
            break;
        }
#if defined(EVO_GUI_MODE)
#if defined(__linux__) && !defined(__ANDROID__)
        case RENDERER_TYPE_LINUX: {
            renderer_linux_t* priv = malloc(sizeof(renderer_linux_t));
            if (priv) {
                memset(priv, 0, sizeof(renderer_linux_t));
                // 1. Platform initialize
                open_display_linux(priv);
                // 2. Create window & surface
                priv->handle = create_window_linux(priv->g_display, EVO_GUI_TITLE, EVO_GUI_WIDTH, EVO_GUI_HEIGHT);
                create_surface_linux(priv, EVO_GUI_WIDTH, EVO_GUI_HEIGHT, &priv->surface, &priv->ximage);
                // 3. Save context & map window
                XSaveContext(priv->g_display, priv->handle, priv->g_context, (XPointer)priv);
                XMapWindow(priv->g_display, priv->handle);
                XFlush(priv->g_display);
            }
            rd->priv = priv;
            rd->render = NULL;
        }
#endif // gui platform
#endif // EVO_GUI_MODE
        default:
            rd->priv = NULL;
            rd->render = NULL;
            break;
    }
    return rd;
}

void renderer_run(renderer_t* rd, render_fn_t rd_fn) {
    if (rd && rd->render && rd_fn) {
        rd->render(rd, rd_fn);
    }
}

void renderer_free(renderer_t* rd) {
    if (rd) {
        if (rd->priv) {
            switch(rd->type) {
                case RENDERER_TYPE_GIF: {
                    renderer_gif_t* priv = rd->priv;
                    if(priv->name) free(priv->name);
                }
#if defined(EVO_GUI_MODE)
#if defined(__linux__) && !defined(__ANDROID__)
                case RENDERER_TYPE_LINUX: {
                    renderer_linux_t* priv = rd->priv;
                    XUnmapWindow(priv->g_display, priv->handle);
                    XDeleteContext(priv->g_display, priv->handle, priv->g_context);
                    if (priv->ximage) { priv->ximage->data = NULL; XDestroyImage(priv->ximage); }
                    close_display_linux(priv);
                    XFlush(priv->g_display);
                    if (priv->surface) image_free(priv->surface);
                }
#endif // gui platform
#endif // EVO_GUI_MODE
                default: break;
            }
            free(rd->priv);
        }
        free(rd);
        rd = NULL;
    }
}
