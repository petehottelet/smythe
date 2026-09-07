/* Smythe Glyph Rain: native X11/XScreenSaver renderer. */
#define _POSIX_C_SOURCE 200809L
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <cairo/cairo.h>
#include <cairo/cairo-xlib.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "glyph_data.h"

enum { LAYER_COUNT = 3, FPS = 40, MAX_SIZE = 8192 };
typedef struct {
    double x, y, rate, accumulator, burst;
    int glyph, phase;
    uint32_t seed;
} Column;
typedef struct {
    cairo_surface_t *trail[GLYPH_COUNT], *head[GLYPH_COUNT];
    Column *columns;
    int count;
    double cell, pad, step, speed, level;
} Layer;
typedef struct {
    Layer layers[LAYER_COUNT];
    cairo_surface_t *image;
    int width, height;
    uint64_t reference_cells, original_cells, blank_cells;
} Scene;

static volatile sig_atomic_t stopping = 0;
static int x_error = 0;
static Window live_window = 0;
static uint32_t random_state = 0x534d5954u;
static double original_mix = GLYPH_ORIGINAL_SHARE;

static void stop_signal(int number) { (void)number; stopping = 1; }
static int handle_x_error(Display *display, XErrorEvent *event) {
    /* A manager may destroy its preview between our event pump and paint.
       A vanished established target is normal shutdown, not a bad CLI ID. */
    if (live_window && event->resourceid == live_window &&
        (event->error_code == BadWindow || event->error_code == BadDrawable)) {
        stopping = 1;
        return 0;
    }
    char message[256];
    XGetErrorText(display, event->error_code, message, sizeof(message));
    fprintf(stderr, "X11: %s (request=%u resource=0x%lx)\n",
            message, event->request_code, event->resourceid);
    x_error = 1;
    stopping = 1;
    return 0;
}
static double random_unit(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return (double)random_state / 4294967296.0;
}
static int random_glyph(void) {
    return random_unit() < original_mix
        ? GLYPH_ORIGINAL_OFFSET + (int)(random_unit() * GLYPH_ORIGINAL_COUNT)
        : (int)(random_unit() * GLYPH_REFERENCE_COUNT);
}
static uint32_t mix_bits(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}
static int cell_glyph(const Column *column, int tail) {
    /* Decide the family first: a larger original catalog must not increase
       its selection probability. A stable cell seed keeps redraws identical. */
    uint32_t value = mix_bits(column->seed + (uint32_t)(column->phase - tail * 7));
    uint32_t index = mix_bits(value ^ 0xa511e9b3u);
    return (double)value / 4294967296.0 < original_mix
        ? GLYPH_ORIGINAL_OFFSET + (int)(index % GLYPH_ORIGINAL_COUNT)
        : (int)(index % GLYPH_REFERENCE_COUNT);
}
static double monotonic_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static void draw_program(cairo_t *cr, int glyph, double ox, double oy,
                         double unit,
                         double red, double green, double blue, double alpha) {
    const GlyphSpec *spec = &GLYPHS[glyph];
    cairo_set_source_rgba(cr, red / 255, green / 255, blue / 255, alpha);
    cairo_set_fill_rule(cr, CAIRO_FILL_RULE_WINDING);
    cairo_new_path(cr);
    for (int i = 0; i < spec->count; i++) {
        const GlyphCommand *command = &GLYPH_COMMANDS[spec->offset + i];
        const double *v = command->values;
        switch (command->kind) {
            case 0: cairo_move_to(cr, ox + v[0] * unit, oy + v[1] * unit); break;
            case 1: cairo_line_to(cr, ox + v[0] * unit, oy + v[1] * unit); break;
            case 2: cairo_curve_to(cr, ox + v[0] * unit, oy + v[1] * unit,
                                   ox + v[2] * unit, oy + v[3] * unit,
                                   ox + v[4] * unit, oy + v[5] * unit); break;
            case 3: cairo_close_path(cr); break;
        }
    }
    /* One compound fill preserves holes, overlapping contours, and curves. */
    cairo_fill(cr);
}

static cairo_surface_t *make_sprite(int glyph, double cell, double pad,
                                    double level, int head) {
    int size = (int)ceil(cell + 2 * pad);
    cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, size, size);
    cairo_t *cr = cairo_create(surface);
    double unit = cell / GLYPH_CANVAS_H;
    double ox = pad + (cell - GLYPH_CANVAS_W * unit) / 2;
    /* A small cached halo surrounds the original contour; the opaque core is
       never stroked or enlarged. This is Cairo glow, not the REGL bloom pass. */
    for (int pass = 0; pass < 2; pass++) {
        double radius = (pass ? .75 : 1.5) * fmax(.5, cell / 24);
        for (int direction = 0; direction < 8; direction++) {
            double angle = direction * 6.283185307179586 / 8;
            draw_program(cr, glyph, ox + cos(angle) * radius, pad + sin(angle) * radius,
                         unit, 25.5, 229.5, 83.3, (head ? .025 : .012) * level);
        }
    }
    /* Body: HSL(137deg, 80%, 50%). Head: the web explorer's #A2FFD8 mint. */
    draw_program(cr, glyph, ox, pad, unit,
                 (head ? 162 : 25.5) * level,
                 (head ? 255 : 229.5) * level,
                 (head ? 216 : 83.3) * level, 1);
    cairo_destroy(cr);
    return surface;
}

static void free_scene(Scene *scene) {
    for (int l = 0; l < LAYER_COUNT; l++) {
        Layer *layer = &scene->layers[l];
        for (int g = 0; g < GLYPH_COUNT; g++) {
            if (layer->trail[g]) cairo_surface_destroy(layer->trail[g]);
            if (layer->head[g]) cairo_surface_destroy(layer->head[g]);
        }
        free(layer->columns);
    }
    if (scene->image) cairo_surface_destroy(scene->image);
    memset(scene, 0, sizeof(*scene));
}

static int build_scene(Scene *scene, int width, int height) {
    static const double cells[] = {11, 20, 36};
    static const double spacing[] = {1.2, 1.35, 2.5};
    static const double speeds[] = {.62, .8, 1};
    static const double levels[] = {.48, .75, 1};
    if (width < 1 || height < 1 || width > MAX_SIZE || height > MAX_SIZE) {
        fprintf(stderr, "Window dimensions must be between 1 and %d\n", MAX_SIZE);
        return 0;
    }
    free_scene(scene);
    scene->width = width;
    scene->height = height;
    scene->image = cairo_image_surface_create(CAIRO_FORMAT_RGB24, width, height);
    if (cairo_surface_status(scene->image) != CAIRO_STATUS_SUCCESS) return 0;
    double scale = fmax(.5, fmin(1.5, height / 720.0));
    for (int l = 0; l < LAYER_COUNT; l++) {
        Layer *layer = &scene->layers[l];
        layer->cell = cells[l] * scale;
        layer->pad = ceil(layer->cell * .5);
        layer->step = fmax(3, layer->cell * 1.04);
        layer->speed = speeds[l];
        layer->level = levels[l];
        double lane = layer->cell * spacing[l];
        layer->count = (int)ceil(width / lane) + 1;
        layer->columns = calloc((size_t)layer->count, sizeof(Column));
        if (!layer->columns) return 0;
        for (int g = 0; g < GLYPH_COUNT; g++) {
            layer->trail[g] = make_sprite(g, layer->cell, layer->pad, layer->level, 0);
            layer->head[g] = make_sprite(g, layer->cell, layer->pad, layer->level, 1);
            if (cairo_surface_status(layer->trail[g]) != CAIRO_STATUS_SUCCESS ||
                cairo_surface_status(layer->head[g]) != CAIRO_STATUS_SUCCESS) return 0;
        }
        for (int i = 0; i < layer->count; i++) {
            Column *col = &layer->columns[i];
            col->x = i * lane + random_unit() * 7 - 3;
            col->glyph = random_glyph();
            col->phase = random_glyph();
            col->seed = (uint32_t)(random_unit() * 4294967296.0);
            col->y = random_unit() * (height + GLYPHS[col->glyph].trail * layer->step);
            col->rate = GLYPHS[col->glyph].speed * 5.6 * layer->speed;
            col->accumulator = random_unit();
        }
    }
    return 1;
}

static void render_scene(Scene *scene, double dt) {
    cairo_t *cr = cairo_create(scene->image);
    cairo_set_source_rgb(cr, 0, 0, 0);
    cairo_paint(cr);
    for (int l = 0; l < LAYER_COUNT; l++) {
        Layer *layer = &scene->layers[l];
        for (int i = 0; i < layer->count; i++) {
            Column *col = &layer->columns[i];
            col->accumulator += dt * col->rate * (col->burst > 0 ? 1.9 : 1);
            if (col->burst > 0) col->burst -= dt;
            while (col->accumulator >= 1) {
                col->accumulator -= 1;
                col->y += layer->step;
                col->phase = (col->phase + 7) % 1000000;
                double past = col->y - GLYPHS[col->glyph].trail * layer->step * 1.15;
                if (past > scene->height && random_unit() < .6) {
                    col->y = -layer->step * (int)(random_unit() * 7);
                    col->glyph = random_glyph();
                    col->rate = GLYPHS[col->glyph].speed * 5.6 * layer->speed;
                    if (random_unit() < .06) col->burst = 1.6;
                }
            }
            int length = (int)lround(GLYPHS[col->glyph].trail * 1.15);
            for (int tail = length; tail >= 0; tail--) {
                double y = col->y - tail * layer->step;
                if (y < -layer->step || y > scene->height + layer->step) continue;
                int glyph = cell_glyph(col, tail);
                if (glyph >= GLYPH_ORIGINAL_OFFSET) scene->original_cells++;
                else scene->reference_cells++;
                if (glyph == GLYPH_BLANK_INDEX) scene->blank_cells++;
                double near = 1 - (double)tail / length;
                double alpha = tail == 0 ? 1 : (.25 + .75 * sqrt(near)) * (.75 + (glyph % 5) * .0625);
                cairo_set_source_surface(cr, tail == 0 ? layer->head[glyph] : layer->trail[glyph],
                                         col->x - layer->pad, y - layer->pad);
                cairo_paint_with_alpha(cr, alpha);
            }
        }
    }
    cairo_destroy(cr);
}

static int number(const char *text, unsigned long limit, unsigned long *value) {
    char *end;
    errno = 0;
    if (!text || !*text || *text == '-') return 0;
    unsigned long result = strtoul(text, &end, 0);
    if (errno || *end || result > limit) return 0;
    *value = result;
    return 1;
}

static void print_catalog(void) {
    printf("{\"version\":\"native-mixed-svg-v1\",\"catalog_sha256\":\"%s\","
           "\"reference_sha256\":\"%s\",\"original_sha256\":\"%s\","
           "\"glyph_count\":%d,\"reference_count\":%d,\"reference_visible_count\":%d,"
           "\"original_count\":%d,\"original_offset\":%d,\"blank_index\":%d,"
           "\"canvas\":[%d,%d],\"fill_rule\":\"%s\",\"default_original_mix\":%.1f}\n",
           GLYPH_CATALOG_SHA256, GLYPH_REFERENCE_SHA256, GLYPH_ORIGINAL_SHA256,
           GLYPH_COUNT, GLYPH_REFERENCE_COUNT, GLYPH_REFERENCE_VISIBLE_COUNT,
           GLYPH_ORIGINAL_COUNT, GLYPH_ORIGINAL_OFFSET, GLYPH_BLANK_INDEX,
           GLYPH_CANVAS_W, GLYPH_CANVAS_H, GLYPH_FILL_RULE, GLYPH_ORIGINAL_SHARE);
}

static int write_glyph_sheet(const char *path) {
    enum { TILE = 128, COLUMNS = 16, MARGIN = 4 };
    int rows = (GLYPH_COUNT + COLUMNS - 1) / COLUMNS;
    cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_RGB24,
                                                         COLUMNS * TILE, rows * TILE);
    cairo_t *cr = cairo_create(surface);
    cairo_set_source_rgb(cr, 1, 1, 1);
    cairo_paint(cr);
    for (int glyph = 0; glyph < GLYPH_COUNT; glyph++) {
        draw_program(cr, glyph, (glyph % COLUMNS) * TILE + MARGIN,
                     (glyph / COLUMNS) * TILE + MARGIN,
                     (double)(TILE - 2 * MARGIN) / GLYPH_CANVAS_H, 0, 0, 0, 1);
    }
    cairo_status_t status = cairo_status(cr);
    if (status == CAIRO_STATUS_SUCCESS) status = cairo_surface_write_to_png(surface, path);
    cairo_destroy(cr);
    cairo_surface_destroy(surface);
    if (status != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "Glyph sheet: %s\n", cairo_status_to_string(status));
        return 1;
    }
    print_catalog();
    return 0;
}

static void usage(FILE *stream) {
    fprintf(stream,
        "Smythe Glyph Rain — %d reference slots + %d original SVGs, native X11\n"
        "Usage: smythe-glyph-rain [options]\n"
        "  --window                 Resizable preview window (default)\n"
        "  --fullscreen             Fullscreen standalone window\n"
        "  -root                    Paint root/XScreenSaver window\n"
        "  -window-id ID            Paint an existing X11 window\n"
        "  -display DISPLAY         Select an X server\n"
        "  --width N --height N     Preview size (default 960x640)\n"
        "  --frames N --seed N      Bounded deterministic validation run\n"
        "  --snapshot FILE.png      Save final frame (requires --frames)\n"
        "  --mix FRACTION           Original glyph share, 0 to 1 (default 0.1)\n"
        "  --glyph-sheet FILE.png   Write all filled contours without X11\n"
        "  --catalog | --license    Catalog hashes or included MIT notices\n"
        "  --help | --version       Print information and exit\n"
        "XSCREENSAVER_WINDOW is honored unless --window is explicit.\n"
        "Escape closes standalone preview. SIGTERM exits cleanly.\n",
        GLYPH_REFERENCE_COUNT, GLYPH_ORIGINAL_COUNT);
}

int main(int argc, char **argv) {
    int width = 960, height = 640, root_mode = 0, fullscreen = 0, explicit_window = 0;
    unsigned long window_id = 0, frames_limit = 0;
    const char *display_name = NULL, *snapshot = NULL, *glyph_sheet = NULL;
    random_state = (uint32_t)time(NULL) ^ (uint32_t)clock();
    if (!random_state) random_state = 1;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--help") || !strcmp(arg, "-help")) { usage(stdout); return 0; }
        if (!strcmp(arg, "--version")) {
            printf("Smythe Glyph Rain X11; glyphs=%d reference=%d original=%d catalog=%s\n",
                   GLYPH_COUNT, GLYPH_REFERENCE_COUNT, GLYPH_ORIGINAL_COUNT, GLYPH_CATALOG_SHA256);
            return 0;
        }
        if (!strcmp(arg, "--catalog")) { print_catalog(); return 0; }
        if (!strcmp(arg, "--license")) { puts(GLYPH_LICENSE_NOTICE); return 0; }
        if (!strcmp(arg, "-root") || !strcmp(arg, "--root")) { root_mode = 1; continue; }
        if (!strcmp(arg, "--window") || !strcmp(arg, "-window")) { explicit_window = 1; continue; }
        if (!strcmp(arg, "--fullscreen")) { fullscreen = 1; explicit_window = 1; continue; }
        if (i + 1 >= argc) { fprintf(stderr, "Missing value for %s\n", arg); return 2; }
        const char *value = argv[++i];
        unsigned long parsed;
        if (!strcmp(arg, "-display") || !strcmp(arg, "--display")) display_name = value;
        else if (!strcmp(arg, "--snapshot")) snapshot = value;
        else if (!strcmp(arg, "--glyph-sheet")) glyph_sheet = value;
        else if (!strcmp(arg, "--mix")) {
            char *end;
            errno = 0;
            double result = strtod(value, &end);
            if (errno || end == value || *end || !isfinite(result) || result < 0 || result > 1) goto invalid;
            original_mix = result;
        }
        else if (!strcmp(arg, "-window-id") || !strcmp(arg, "--window-id")) {
            if (!number(value, ULONG_MAX, &window_id) || !window_id) goto invalid;
        } else if (!strcmp(arg, "--width") || !strcmp(arg, "--height")) {
            if (!number(value, MAX_SIZE, &parsed) || !parsed) goto invalid;
            if (!strcmp(arg, "--width")) width = (int)parsed; else height = (int)parsed;
        } else if (!strcmp(arg, "--frames")) {
            if (!number(value, 1000000, &frames_limit) || !frames_limit) goto invalid;
        } else if (!strcmp(arg, "--seed")) {
            if (!number(value, UINT32_MAX, &parsed)) goto invalid;
            random_state = parsed ? (uint32_t)parsed : 1;
        } else { fprintf(stderr, "Unknown option: %s\n", arg); return 2; }
        continue;
invalid:
        fprintf(stderr, "Invalid value for %s: %s\n", arg, value);
        return 2;
    }
    if (glyph_sheet) return write_glyph_sheet(glyph_sheet);
    if (snapshot && !frames_limit) { fprintf(stderr, "--snapshot requires --frames\n"); return 2; }
    if (!window_id && !explicit_window) {
        const char *parent = getenv("XSCREENSAVER_WINDOW");
        if (parent && (!number(parent, ULONG_MAX, &window_id) || !window_id)) {
            fprintf(stderr, "Invalid XSCREENSAVER_WINDOW\n"); return 2;
        }
    }
    Display *display = XOpenDisplay(display_name);
    if (!display) { fprintf(stderr, "Cannot open X11 display; set DISPLAY or use -display.\n"); return 1; }
    XSetErrorHandler(handle_x_error);
    int screen = DefaultScreen(display), owned = !window_id && !root_mode;
    Window window = window_id ? (Window)window_id : RootWindow(display, screen);
    Atom wm_delete = XInternAtom(display, "WM_DELETE_WINDOW", False);
    if (owned) {
        if (fullscreen) { width = DisplayWidth(display, screen); height = DisplayHeight(display, screen); }
        window = XCreateSimpleWindow(display, RootWindow(display, screen), 0, 0,
                                     (unsigned)width, (unsigned)height, 0, 0, 0);
        if (fullscreen) {
            XSetWindowAttributes attributes;
            attributes.override_redirect = True;
            XChangeWindowAttributes(display, window, CWOverrideRedirect, &attributes);
        }
        XStoreName(display, window, "Smythe Glyph Rain");
        XSetWMProtocols(display, window, &wm_delete, 1);
        XMapWindow(display, window);
        if (fullscreen) {
            XSync(display, False);
            XSetInputFocus(display, window, RevertToParent, CurrentTime);
        }
    }
    XWindowAttributes attributes;
    if (!XGetWindowAttributes(display, window, &attributes) || x_error) {
        fprintf(stderr, "Cannot inspect target window\n"); XCloseDisplay(display); return 1;
    }
    width = attributes.width; height = attributes.height;
    XSelectInput(display, window, ExposureMask | StructureNotifyMask | (owned ? KeyPressMask : 0));
    /* Cairo owns only our offscreen pixmap. A screensaver manager can destroy
       its window at any point; that must not invalidate Cairo's resource
       construction or teardown. Only the final CopyArea touches the window. */
    Pixmap backbuffer = XCreatePixmap(display, attributes.root, (unsigned)width,
                                      (unsigned)height, (unsigned)attributes.depth);
    GC blit_gc = XCreateGC(display, backbuffer, 0, NULL);
    cairo_surface_t *surface = cairo_xlib_surface_create(display, backbuffer,
                                                        attributes.visual, width, height);
    Scene scene = {0};
    int result = 0;
    if (cairo_surface_status(surface) != CAIRO_STATUS_SUCCESS || !build_scene(&scene, width, height)) {
        fprintf(stderr, "Cannot allocate rendering surfaces\n"); result = 1; goto cleanup;
    }
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = stop_signal;
    sigemptyset(&action.sa_mask);
    sigaction(SIGTERM, &action, NULL);
    sigaction(SIGINT, &action, NULL);
    unsigned long frames = 0;
    double previous = monotonic_seconds();
    live_window = window;
    printf("window=0x%lx mode=%s glyphs=%d\n", window, owned ? "preview" : "embedded", GLYPH_COUNT);
    fflush(stdout);
    while (!stopping) {
        double started = monotonic_seconds();
        while (XPending(display)) {
            XEvent event;
            XNextEvent(display, &event);
            if (event.type == DestroyNotify) { owned = 0; stopping = 1; }
            if (
                (event.type == ClientMessage && (Atom)event.xclient.data.l[0] == wm_delete) ||
                (event.type == KeyPress && XLookupKeysym(&event.xkey, 0) == XK_Escape)) stopping = 1;
            if (event.type == ConfigureNotify &&
                (event.xconfigure.width != scene.width || event.xconfigure.height != scene.height)) {
                if (!build_scene(&scene, event.xconfigure.width, event.xconfigure.height)) { result = 1; stopping = 1; }
                if (!stopping) {
                    cairo_surface_destroy(surface);
                    XFreePixmap(display, backbuffer);
                    backbuffer = XCreatePixmap(display, attributes.root,
                                                (unsigned)scene.width, (unsigned)scene.height,
                                                (unsigned)attributes.depth);
                    surface = cairo_xlib_surface_create(display, backbuffer, attributes.visual,
                                                         scene.width, scene.height);
                    if (cairo_surface_status(surface) != CAIRO_STATUS_SUCCESS) {
                        result = 1; stopping = 1;
                    }
                }
            }
        }
        if (stopping) break;
        double dt = frames_limit ? 1.0 / FPS : fmax(0, fmin(.05, started - previous));
        previous = started;
        render_scene(&scene, frames == 0 ? 0 : dt);
        cairo_t *cr = cairo_create(surface);
        cairo_set_source_surface(cr, scene.image, 0, 0);
        cairo_paint(cr);
        if (cairo_status(cr) != CAIRO_STATUS_SUCCESS) { result = 1; stopping = 1; }
        cairo_destroy(cr);
        cairo_surface_flush(surface);
        XCopyArea(display, backbuffer, window, blit_gc, 0, 0,
                  (unsigned)scene.width, (unsigned)scene.height, 0, 0);
        XSync(display, False);
        frames++;
        if (frames_limit && frames >= frames_limit) break;
        double remaining = 1.0 / FPS - (monotonic_seconds() - started);
        if (remaining > 0) {
            struct timespec delay = {0, (long)(remaining * 1e9)};
            nanosleep(&delay, NULL);
        }
    }
    if (snapshot && !x_error && frames) {
        cairo_status_t status = cairo_surface_write_to_png(scene.image, snapshot);
        if (status != CAIRO_STATUS_SUCCESS) { fprintf(stderr, "Snapshot: %s\n", cairo_status_to_string(status)); result = 1; }
    }
    printf("frames=%lu width=%d height=%d stopped=%d\n", frames, scene.width, scene.height, stopping ? 1 : 0);
    printf("selected_reference=%llu selected_original=%llu selected_blank=%llu mix=%.6f\n",
           (unsigned long long)scene.reference_cells, (unsigned long long)scene.original_cells,
           (unsigned long long)scene.blank_cells, original_mix);
cleanup:
    free_scene(&scene);
    cairo_surface_destroy(surface);
    XFreeGC(display, blit_gc);
    XFreePixmap(display, backbuffer);
    if (owned && !x_error) XDestroyWindow(display, window);
    XCloseDisplay(display);
    return x_error ? 1 : result;
}
