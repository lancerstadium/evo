#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <evo.h>
#include <evo/util/log.h>
#include <evo/util/math.h>
#include <evo/util/sys.h>


// ==================================================================================== //
//                                  renderer: typedef
// ==================================================================================== //

#define EVO_GUI_TITLE   "renderer"
#define EVO_GUI_WIDTH   960
#define EVO_GUI_HEIGHT  720

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
    int64_t ndelay = 60;       // 0 < ndelay <= 60
    int64_t delays[ndelay];
    canvas_t* cav = canvas_new(rd->width, rd->height);
    canvas_t* cav_all = canvas_new(cav->width, cav->height);
    int64_t delay = 5;         // 5 ms
    delays[0] = delay;
    for (size_t i = 0; i < ndelay; i++) {
        delays[i] = delay;
        cav = rd_fn(cav, (1.f * i) / 60.f);
        image_push(cav_all->background, cav->background);
    }
    image_set_deloys(cav_all->background, delays, ndelay);
    canvas_export(cav_all, priv->name);
    canvas_free(cav_all);
    canvas_free(cav);
}

#if defined(EVO_GUI_ENB)

#if defined(__linux__) && !defined(__ANDROID__)

// ==================================================================================== //
//                                  renderer: linux
// ==================================================================================== //

#include <X11/X.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

typedef struct renderer_linux renderer_linux_t;

typedef struct {
    void (*key_callback)(renderer_linux_t* priv, keycode_t key, int pressed);
    void (*button_callback)(renderer_linux_t* priv, button_t button, int pressed);
    void (*scroll_callback)(renderer_linux_t* priv, float offset);
} callbacks_t;

struct renderer_linux {
    /* platform */
    Window handle;
    XImage *ximage;
    Display *g_display;
    XContext g_context;
    /* common data */
    canvas_t* surface;
    char* pixels_buffer;
    int should_close;
    char keys[KEY_NUM];
    char buttons[BUTTON_NUM];
    callbacks_t callbacks;
    void *userdata;
};

static void initialize_path_linux() {
    char path[256];
    ssize_t bytes;
    int error;

    bytes = readlink("/proc/self/exe", path, 256 - 1);
    if(bytes == -1) return;
    path[bytes] = '\0';
    *strrchr(path, '/') = '\0';

    error = chdir(path);
    if(error != 0) return;
    error = chdir("assets");
    if(error != 0) return;
}

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
    XDestroyWindow(priv->g_display, priv->handle);
    XCloseDisplay(priv->g_display);
    priv->g_display = NULL;
}

static Window create_window_linux(renderer_linux_t *priv, const char *title, int width, int height) {
    // int screen = XDefaultScreen(priv->g_display);
    // unsigned long border = XWhitePixel(priv->g_display, screen);
    // unsigned long background = XBlackPixel(priv->g_display, screen);
    // Window root = XRootWindow(priv->g_display, screen);
    // Window handle;
    // XSizeHints *size_hints;
    // XClassHint *class_hint;
    // Atom delete_window;
    // long mask;

    // handle = XCreateSimpleWindow(priv->g_display, root, 0, 0, width, height, 0,
    //                              border, background);

    int screen = DefaultScreen(priv->g_display);
    priv->handle = XCreateSimpleWindow(priv->g_display, RootWindow(priv->g_display, screen), 0, 0, width, height, 0, BlackPixel(priv->g_display, screen), WhitePixel(priv->g_display, screen));
    XStoreName(priv->g_display, priv->handle, EVO_GUI_TITLE);
    XSelectInput(priv->g_display, priv->handle, ExposureMask | KeyPressMask);

    /* not resizable */
    // size_hints = XAllocSizeHints();
    // size_hints->flags = PMinSize | PMaxSize;
    // size_hints->min_width = width;
    // size_hints->max_width = width;
    // size_hints->min_height = height;
    // size_hints->max_height = height;
    // XSetWMNormalHints(priv->g_display, handle, size_hints);
    // XFree(size_hints);

    /* application name */
    // class_hint = XAllocClassHint();
    // class_hint->res_name = (char*)title;
    // class_hint->res_class = (char*)title;
    // XSetClassHint(priv->g_display, handle, class_hint);
    // XFree(class_hint);

    /* event subscription */
    // mask = KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask;
    // XSelectInput(priv->g_display, handle, mask);
    // delete_window = XInternAtom(priv->g_display, "WM_DELETE_WINDOW", True);
    // XSetWMProtocols(priv->g_display, handle, &delete_window, 1);

    return priv->handle;
}


static void rgb2bgr(char *src, char *dst, int width, int height) {
    if(!src || !dst) return;
    // aabbggrr --> ffrrggbb
    for(int i = 0; i < width * height; i++) {
        dst[i * 4 + 0] = src[i * 4 + 2];
        dst[i * 4 + 1] = src[i * 4 + 1];
        dst[i * 4 + 2] = src[i * 4 + 0];
        dst[i * 4 + 3] = src[i * 4 + 3];
    }
}

static void update_buffer_linux(renderer_linux_t *priv) {
    if(!priv || !priv->surface || !priv->pixels_buffer) return;
    rgb2bgr((char*)priv->surface->background->raw->datas, priv->pixels_buffer, priv->surface->width, priv->surface->height);
}

static void create_surface_linux(renderer_linux_t *priv, int width, int height) {
    if(!priv) return;
    int screen = XDefaultScreen(priv->g_display);
    int depth = XDefaultDepth(priv->g_display, screen);

    if(depth != 24 && depth != 32) {
        LOG_ERR("Error: depth %d is not supported\n", depth);
        return;
    }
    LOG_INFO("Depth: %d\n", depth);
    priv->surface = canvas_new(width, height);
    LOG_INFO("Surface created: %d x %d\n", width, height);
    priv->pixels_buffer = malloc(width * height * 4);
    update_buffer_linux(priv);
    priv->ximage = XCreateImage(priv->g_display, DefaultVisual(priv->g_display, screen), depth, ZPixmap, 0, priv->pixels_buffer, width, height, 32, 0);
}

static void draw_surface_linux(renderer_linux_t *priv) {
    if(!priv) return;
    int screen = XDefaultScreen(priv->g_display);
    GC gc = XDefaultGC(priv->g_display, screen);
    if(!priv->surface) return;
    update_buffer_linux(priv);
    XPutImage(priv->g_display, priv->handle, gc, priv->ximage, 0, 0, 0, 0, priv->surface->width, priv->surface->height);
    XFlush(priv->g_display);
}

void renderer_render_linux(renderer_t* rd, render_fn_t rd_fn) {
    if(!rd || !rd->priv || !rd_fn) return;
    int i = 0;
    renderer_linux_t* priv = rd->priv;
    XEvent event;
    canvas_t* cav = priv->surface;
    while(!renderer_should_close(rd)) {
        rd_fn(cav, (1.f * i) / 60.f);
        draw_surface_linux(priv);
        // Events
        while (XPending(priv->g_display)) {
            XNextEvent(priv->g_display, &event);
            if (event.type == KeyPress) {
                priv->should_close = 1;
            }
        }
        // canvas_free(cav_tmp);
        usleep(15000); // 15ms
        i++;
    }
}

#endif  // gui platform

#endif  // EVO_GUI_ENB

// ==================================================================================== //
//                                  renderer: API
// ==================================================================================== //

renderer_t* renderer_new(int width, int height, renderer_type_t type) {
    renderer_t* rd = malloc(sizeof(renderer_t));
    rd->type = type;
    if(width <= 0) {
        width = EVO_GUI_WIDTH;
    }
    if(height <= 0) {
        height = EVO_GUI_HEIGHT;
    }
    rd->width = width;
    rd->height = height;
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
#if defined(EVO_GUI_ENB)
#if defined(__linux__) && !defined(__ANDROID__)
        case RENDERER_TYPE_LINUX: {
            renderer_linux_t* priv = malloc(sizeof(renderer_linux_t));
            if (priv) {
                memset(priv, 0, sizeof(renderer_linux_t));
                // 1. Platform initialize
                LOG_INFO("Initializing X11 display\n");
                initialize_path_linux();
                open_display_linux(priv);
                // 2. Create window & surface
                LOG_INFO("Creating window & surface\n");
                create_window_linux(priv, EVO_GUI_TITLE, rd->width, rd->height);
                create_surface_linux(priv, rd->width, rd->height);
                // 3. Save context & map window
                LOG_INFO("Saving context & mapping window\n");
                // XSaveContext(priv->g_display, priv->handle, priv->g_context, (XPointer)priv);
                // XMapWindow(priv->g_display, priv->handle);
                // XFlush(priv->g_display);
                XMapWindow(priv->g_display, priv->handle);
                LOG_INFO("X11 display initialized\n");
            }
            rd->priv = priv;
            rd->render = renderer_render_linux;
            break;
        }
#endif // gui platform
#endif // EVO_GUI_ENB
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
                    break;
                }
#if defined(EVO_GUI_ENB)
#if defined(__linux__) && !defined(__ANDROID__)
                case RENDERER_TYPE_LINUX: {
                    renderer_linux_t* priv = rd->priv;
                    XUnmapWindow(priv->g_display, priv->handle);
                    XDeleteContext(priv->g_display, priv->handle, priv->g_context);
                    if (priv->ximage) { priv->ximage->data = NULL; XDestroyImage(priv->ximage); }
                    if (priv->pixels_buffer) free(priv->pixels_buffer);
                    close_display_linux(priv);
                    // XFlush(priv->g_display);
                    if (priv->surface) canvas_free(priv->surface);
                    break;
                }
#endif // gui platform
#endif // EVO_GUI_ENB
                default: break;
            }
            free(rd->priv);
        }
        free(rd);
        rd = NULL;
    }
}


int renderer_should_close(renderer_t* rd) {
    if(!rd) return 0;
    switch(rd->type) {
#if defined(EVO_GUI_ENB)
#if defined(__linux__) && !defined(__ANDROID__)
        case RENDERER_TYPE_LINUX: {
            renderer_linux_t* priv = rd->priv;
            return priv->should_close;
        }
#endif // gui platform
#endif // EVO_GUI_ENB
        default: return 0;
    }
    return 0;
}
