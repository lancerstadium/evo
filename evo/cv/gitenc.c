#include "gifenc.h"
#include "../util/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

/* helper to write a little-endian 16-bit number portably */
#define write_num(fd, n) write((fd), (uint8_t []) {(n) & 0xFF, (n) >> 8}, 2)

static uint8_t vga[0x30] = {
    0x00, 0x00, 0x00,
    0xAA, 0x00, 0x00,
    0x00, 0xAA, 0x00,
    0xAA, 0x55, 0x00,
    0x00, 0x00, 0xAA,
    0xAA, 0x00, 0xAA,
    0x00, 0xAA, 0xAA,
    0xAA, 0xAA, 0xAA,
    0x55, 0x55, 0x55,
    0xFF, 0x55, 0x55,
    0x55, 0xFF, 0x55,
    0xFF, 0xFF, 0x55,
    0x55, 0x55, 0xFF,
    0xFF, 0x55, 0xFF,
    0x55, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF,
};

static uint8_t ge_color216_palette[216 * 3] = {
    0x00, 0x00, 0x00,  // entry 0: black
    0x00, 0x00, 0x5F,  // entry 1: navy blue
    0x00, 0x00, 0x87,  // entry 2: dark blue
    0x00, 0x00, 0xAF,  // entry 3: blue 3
    0x00, 0x00, 0xD7,  // entry 4: blue 3
    0x00, 0x00, 0xFF,  // entry 5: blue 1
    0x00, 0x5F, 0x00,  // entry 6: dark green
    0x00, 0x5F, 0x5F,  // entry 7: deep sky blue 4
    0x00, 0x5F, 0x87,  // entry 8: deep sky blue 4
    0x00, 0x5F, 0xAF,  // entry 9: deep sky blue 4
    0x00, 0x5F, 0xD7,  // entry 10: dodger blue 3
    0x00, 0x5F, 0xFF,  // entry 11: dodger blue 2
    0x00, 0x87, 0x00,  // entry 12: green 4
    0x00, 0x87, 0x5F,  // entry 13: spring green 4
    0x00, 0x87, 0x87,  // entry 14: turquoise 4
    0x00, 0x87, 0xAF,  // entry 15: deep sky blue 3
    0x00, 0x87, 0xD7,  // entry 16: deep sky blue 3
    0x00, 0x87, 0xFF,  // entry 17: dodger blue 1
    0x00, 0xAF, 0x00,  // entry 18: green 3
    0x00, 0xAF, 0x5F,  // entry 19: spring green 3
    0x00, 0xAF, 0x87,  // entry 20: dark cyan
    0x00, 0xAF, 0xAF,  // entry 21: light sea green
    0x00, 0xAF, 0xD7,  // entry 22: deep sky blue 2
    0x00, 0xAF, 0xFF,  // entry 23: deep sky blue 1
    0x00, 0xD7, 0x00,  // entry 24: green 3
    0x00, 0xD7, 0x5F,  // entry 25: spring green 3
    0x00, 0xD7, 0x87,  // entry 26: spring green 2
    0x00, 0xD7, 0xAF,  // entry 27: cyan 3
    0x00, 0xD7, 0xD7,  // entry 28: dark turquoise
    0x00, 0xD7, 0xFF,  // entry 29: turquoise 2
    0x00, 0xFF, 0x00,  // entry 30: green 1
    0x00, 0xFF, 0x5F,  // entry 31: spring green 2
    0x00, 0xFF, 0x87,  // entry 32: spring green 1
    0x00, 0xFF, 0xAF,  // entry 33: medium spring green
    0x00, 0xFF, 0xD7,  // entry 34: cyan 2
    0x00, 0xFF, 0xFF,  // entry 35: cyan 1
    0x5F, 0x00, 0x00,  // entry 36: dark red
    0x5F, 0x00, 0x5F,  // entry 37: deep pink 4
    0x5F, 0x00, 0x87,  // entry 38: purple 4
    0x5F, 0x00, 0xAF,  // entry 39: purple 4
    0x5F, 0x00, 0xD7,  // entry 40: purple 3
    0x5F, 0x00, 0xFF,  // entry 41: blue violet
    0x5F, 0x5F, 0x00,  // entry 42: orange 4
    0x5F, 0x5F, 0x5F,  // entry 43: gray 37
    0x5F, 0x5F, 0x87,  // entry 44: medium purple 4
    0x5F, 0x5F, 0xAF,  // entry 45: slate blue 3
    0x5F, 0x5F, 0xD7,  // entry 46: slate blue 3
    0x5F, 0x5F, 0xFF,  // entry 47: royal blue 1
    0x5F, 0x87, 0x00,  // entry 48: chartreuse 4
    0x5F, 0x87, 0x5F,  // entry 49: dark sea green 4
    0x5F, 0x87, 0x87,  // entry 50: pale turquoise 4
    0x5F, 0x87, 0xAF,  // entry 51: steel blue
    0x5F, 0x87, 0xD7,  // entry 52: steel blue 3
    0x5F, 0x87, 0xFF,  // entry 53: cornflower blue
    0x5F, 0xAF, 0x00,  // entry 54: chartreuse 3
    0x5F, 0xAF, 0x5F,  // entry 55: dark sea green 4
    0x5F, 0xAF, 0x87,  // entry 56: cadet blue
    0x5F, 0xAF, 0xAF,  // entry 57: cadet blue
    0x5F, 0xAF, 0xD7,  // entry 58: sky blue 3
    0x5F, 0xAF, 0xFF,  // entry 59: steel blue 1
    0x5F, 0xD7, 0x00,  // entry 60: chartreuse 3
    0x5F, 0xD7, 0x5F,  // entry 61: pale green 3
    0x5F, 0xD7, 0x87,  // entry 62: sea green 3
    0x5F, 0xD7, 0xAF,  // entry 63: aquamarine 3
    0x5F, 0xD7, 0xD7,  // entry 64: medium turquoise
    0x5F, 0xD7, 0xFF,  // entry 65: steel blue 1
    0x5F, 0xFF, 0x00,  // entry 66: chartreuse 2
    0x5F, 0xFF, 0x5F,  // entry 67: sea green 2
    0x5F, 0xFF, 0x87,  // entry 68: sea green 1
    0x5F, 0xFF, 0xAF,  // entry 69: aquamarine 1
    0x5F, 0xFF, 0xD7,  // entry 70: dark slate gray 2
    0x5F, 0xFF, 0xFF,  // entry 71: dark slate gray 1
    0x87, 0x00, 0x00,  // entry 72: dark red 3
    0x87, 0x00, 0x5F,  // entry 73: deep pink 4
    0x87, 0x00, 0x87,  // entry 74: dark magenta 4
    0x87, 0x00, 0xAF,  // entry 75: dark magenta 3
    0x87, 0x00, 0xD7,  // entry 76: dark violet
    0x87, 0x00, 0xFF,  // entry 77: purple 2
    0x87, 0x5F, 0x00,  // entry 78: orange 3
    0x87, 0x5F, 0x5F,  // entry 79: light pink 4
    0x87, 0x5F, 0x87,  // entry 80: plum 4
    0x87, 0x5F, 0xAF,  // entry 81: medium orchid 3
    0x87, 0x5F, 0xD7,  // entry 82: medium orchid 3
    0x87, 0x5F, 0xFF,  // entry 83: dark orchid
    0x87, 0x87, 0x00,  // entry 84: orange 3
    0x87, 0x87, 0x5F,  // entry 85: light pink 3
    0x87, 0x87, 0x87,  // entry 86: plum 3
    0x87, 0x87, 0xAF,  // entry 87: orchid
    0x87, 0x87, 0xD7,  // entry 88: medium purple 2
    0x87, 0x87, 0xFF,  // entry 89: medium purple 2
    0x87, 0xAF, 0x00,  // entry 90: gold 3
    0x87, 0xAF, 0x5F,  // entry 91: light coral 3
    0x87, 0xAF, 0x87,  // entry 92: pale violet red 3
    0x87, 0xAF, 0xAF,  // entry 93: orchid 2
    0x87, 0xAF, 0xD7,  // entry 94: dark goldenrod
    0x87, 0xAF, 0xFF,  // entry 95: orchid 1
    0x87, 0xD7, 0x00,  // entry 96: gold 2
    0x87, 0xD7, 0x5F,  // entry 97: light coral 2
    0x87, 0xD7, 0x87,  // entry 98: pale violet red 2
    0x87, 0xD7, 0xAF,  // entry 99: dark orange 2
    0x87, 0xD7, 0xD7,  // entry 100: dark orange 1
    0x87, 0xD7, 0xFF,  // entry 101: orange 1
    0x87, 0xFF, 0x00,  // entry 102: gold 1
    0x87, 0xFF, 0x5F,  // entry 103: light coral 1
    0x87, 0xFF, 0x87,  // entry 104: pale violet red 1
    0x87, 0xFF, 0xAF,  // entry 105: light goldenrod 2
    0x87, 0xFF, 0xD7,  // entry 106: light goldenrod 1
    0x87, 0xFF, 0xFF,  // entry 107: light yellow 2
    0xAF, 0x00, 0x00,  // entry 108: red 3
    0xAF, 0x00, 0x5F,  // entry 109: deep pink 3
    0xAF, 0x00, 0x87,  // entry 110: magenta 3
    0xAF, 0x00, 0xAF,  // entry 111: magenta 2
    0xAF, 0x00, 0xD7,  // entry 112: magenta 1
    0xAF, 0x00, 0xFF,  // entry 113: magenta 1
    0xAF, 0x5F, 0x00,  // entry 114: dark orange 3
    0xAF, 0x5F, 0x5F,  // entry 115: light pink 3
    0xAF, 0x5F, 0x87,  // entry 116: plum 3
    0xAF, 0x5F, 0xAF,  // entry 117: orchid 2
    0xAF, 0x5F, 0xD7,  // entry 118: orchid 1
    0xAF, 0x5F, 0xFF,  // entry 119: orchid 1
    0xAF, 0x87, 0x00,  // entry 120: dark orange 2
    0xAF, 0x87, 0x5F,  // entry 121: light pink 2
    0xAF, 0x87, 0x87,  // entry 122: plum 2
    0xAF, 0x87, 0xAF,  // entry 123: orchid 1
    0xAF, 0x87, 0xD7,  // entry 124: dark orange 1
    0xAF, 0x87, 0xFF,  // entry 125: orange 1
    0xAF, 0xAF, 0x00,  // entry 126: dark orange 1
    0xAF, 0xAF, 0x5F,  // entry 127: light pink 1
    0xAF, 0xAF, 0x87,  // entry 128: plum 1
    0xAF, 0xAF, 0xAF,  // entry 129: orchid
    0xAF, 0xAF, 0xD7,  // entry 130: dark goldenrod
    0xAF, 0xAF, 0xFF,  // entry 131: light goldenrod 1
    0xAF, 0xD7, 0x00,  // entry 132: orange 2
    0xAF, 0xD7, 0x5F,  // entry 133: light pink 1
    0xAF, 0xD7, 0x87,  // entry 134: plum 1
    0xAF, 0xD7, 0xAF,  // entry 135: gold 1
    0xAF, 0xD7, 0xD7,  // entry 136: light yellow 1
    0xAF, 0xD7, 0xFF,  // entry 137: light yellow 2
    0xAF, 0xFF, 0x00,  // entry 138: gold 1
    0xAF, 0xFF, 0x5F,  // entry 139: light yellow 1
    0xAF, 0xFF, 0x87,  // entry 140: pale goldenrod
    0xAF, 0xFF, 0xAF,  // entry 141: light yellow 2
    0xAF, 0xFF, 0xD7,  // entry 142: light yellow 2
    0xAF, 0xFF, 0xFF,  // entry 143: light yellow 3
    0xD7, 0x00, 0x00,  // entry 144: red 2
    0xD7, 0x00, 0x5F,  // entry 145: deep pink 2
    0xD7, 0x00, 0x87,  // entry 146: deep pink 1
    0xD7, 0x00, 0xAF,  // entry 147: magenta 2
    0xD7, 0x00, 0xD7,  // entry 148: magenta 1
    0xD7, 0x00, 0xFF,  // entry 149: magenta 1
    0xD7, 0x5F, 0x00,  // entry 150: dark orange 2
    0xD7, 0x5F, 0x5F,  // entry 151: hot pink 2
    0xD7, 0x5F, 0x87,  // entry 152: hot pink 1
    0xD7, 0x5F, 0xAF,  // entry 153: orchid 1
    0xD7, 0x5F, 0xD7,  // entry 154: orchid 1
    0xD7, 0x5F, 0xFF,  // entry 155: orchid 1
    0xD7, 0x87, 0x00,  // entry 156: dark orange 1
    0xD7, 0x87, 0x5F,  // entry 157: light pink 1
    0xD7, 0x87, 0x87,  // entry 158: plum 1
    0xD7, 0x87, 0xAF,  // entry 159: orchid
    0xD7, 0x87, 0xD7,  // entry 160: dark goldenrod
    0xD7, 0x87, 0xFF,  // entry 161: light goldenrod 1
    0xD7, 0xAF, 0x00,  // entry 162: orange 1
    0xD7, 0xAF, 0x5F,  // entry 163: light pink 1
    0xD7, 0xAF, 0x87,  // entry 164: plum 1
    0xD7, 0xAF, 0xAF,  // entry 165: gold 1
    0xD7, 0xAF, 0xD7,  // entry 166: light yellow 1
    0xD7, 0xAF, 0xFF,  // entry 167: light yellow 2
    0xD7, 0xD7, 0x00,  // entry 168: gold 1
    0xD7, 0xD7, 0x5F,  // entry 169: light yellow 1
    0xD7, 0xD7, 0x87,  // entry 170: pale goldenrod
    0xD7, 0xD7, 0xAF,  // entry 171: light yellow 2
    0xD7, 0xD7, 0xD7,  // entry 172: light yellow 2
    0xD7, 0xD7, 0xFF,  // entry 173: light yellow 3
    0xD7, 0xFF, 0x00,  // entry 174: gold 1
    0xD7, 0xFF, 0x5F,  // entry 175: light yellow 1
    0xD7, 0xFF, 0x87,  // entry 176: pale goldenrod
    0xD7, 0xFF, 0xAF,  // entry 177: light yellow 2
    0xD7, 0xFF, 0xD7,  // entry 178: light yellow 2
    0xD7, 0xFF, 0xFF,  // entry 179: light yellow 3
    0xFF, 0x00, 0x00,  // entry 180: red 2
    0xFF, 0x00, 0x5F,  // entry 181: deep pink 2
    0xFF, 0x00, 0x87,  // entry 182: deep pink 1
    0xFF, 0x00, 0xAF,  // entry 183: magenta 2
    0xFF, 0x00, 0xD7,  // entry 184: magenta 1
    0xFF, 0x00, 0xFF,  // entry 185: magenta 1
    0xFF, 0x5F, 0x00,  // entry 186: dark orange 2
    0xFF, 0x5F, 0x5F,  // entry 187: hot pink 2
    0xFF, 0x5F, 0x87,  // entry 188: hot pink 1
    0xFF, 0x5F, 0xAF,  // entry 189: orchid 1
    0xFF, 0x5F, 0xD7,  // entry 190: orchid 1
    0xFF, 0x5F, 0xFF,  // entry 191: orchid 1
    0xFF, 0x87, 0x00,  // entry 192: dark orange 1
    0xFF, 0x87, 0x5F,  // entry 193: light pink 1
    0xFF, 0x87, 0x87,  // entry 194: plum 1
    0xFF, 0x87, 0xAF,  // entry 195: orchid
    0xFF, 0x87, 0xD7,  // entry 196: dark goldenrod
    0xFF, 0x87, 0xFF,  // entry 197: light goldenrod 1
    0xFF, 0xAF, 0x00,  // entry 198: orange 1
    0xFF, 0xAF, 0x5F,  // entry 199: light pink 1
    0xFF, 0xAF, 0x87,  // entry 200: plum 1
    0xFF, 0xAF, 0xAF,  // entry 201: gold 1
    0xFF, 0xAF, 0xD7,  // entry 202: light yellow 1
    0xFF, 0xAF, 0xFF,  // entry 203: light yellow 2
    0xFF, 0xD7, 0x00,  // entry 204: gold 1
    0xFF, 0xD7, 0x5F,  // entry 205: light yellow 1
    0xFF, 0xD7, 0x87,  // entry 206: pale goldenrod
    0xFF, 0xD7, 0xAF,  // entry 207: light yellow 2
    0xFF, 0xD7, 0xD7,  // entry 208: light yellow 2
    0xFF, 0xD7, 0xFF,  // entry 209: light yellow 3
    0xFF, 0xFF, 0x00,  // entry 210: gold 1
    0xFF, 0xFF, 0x5F,  // entry 211: light yellow 1
    0xFF, 0xFF, 0x87,  // entry 212: pale goldenrod
    0xFF, 0xFF, 0xAF,  // entry 213: light yellow 2
    0xFF, 0xFF, 0xD7,  // entry 214: light yellow 2
    0xFF, 0xFF, 0xFF   // entry 215: white
};

struct Node {
    uint16_t key;
    struct Node *children[];
};
typedef struct Node Node;

static Node *
new_node(uint16_t key, int degree)
{
    Node *node = calloc(1, sizeof(*node) + degree * sizeof(Node *));
    if (node)
        node->key = key;
    return node;
}

static Node *
new_trie(int degree, int *nkeys)
{
    Node *root = new_node(0, degree);
    /* Create nodes for single pixels. */
    for (*nkeys = 0; *nkeys < degree; (*nkeys)++)
        root->children[*nkeys] = new_node(*nkeys, degree);
    *nkeys += 2; /* skip clear code and stop code */
    return root;
}

static void
del_trie(Node *root, int degree)
{
    if (!root)
        return;
    for (int i = 0; i < degree; i++)
        del_trie(root->children[i], degree);
    free(root);
}

#define write_and_store(s, dst, fd, src, n) \
do { \
    write(fd, src, n); \
    if (s) { \
        memcpy(dst, src, n); \
        dst += n; \
    } \
} while (0);

static void put_loop(ge_GIF *gif, uint16_t loop);

ge_GIF *
ge_new_gif(
    const char *fname, uint16_t width, uint16_t height,
    uint8_t *palette, int depth, int bgindex, int loop
)
{
    int i, r, g, b, v;
    int store_gct, custom_gct;
    int nbuffers = bgindex < 0 ? 2 : 1;
    ge_GIF *gif = calloc(1, sizeof(*gif) + nbuffers*width*height);
    if (!gif)
        goto no_gif;
    gif->w = width; gif->h = height;
    gif->bgindex = bgindex;
    gif->frame = (uint8_t *) &gif[1];
    gif->back = &gif->frame[width*height];
    if(palette) {
        gif->palette = palette;
    } else {
        gif->palette = ge_color216_palette;
    }
#ifdef _WIN32
    gif->fd = creat(fname, S_IWRITE);
#else
    gif->fd = creat(fname, 0666);
#endif
    if (gif->fd == -1)
        goto no_fd;
#ifdef _WIN32
    setmode(gif->fd, O_BINARY);
#endif
    write(gif->fd, "GIF89a", 6);
    write_num(gif->fd, width);
    write_num(gif->fd, height);
    store_gct = custom_gct = 0;
    if (palette) {
        if (depth < 0)
            store_gct = 1;
        else
            custom_gct = 1;
    }
    if (depth < 0)
        depth = -depth;
    gif->depth = depth > 1 ? depth : 2;
    if(gif->palette == ge_color216_palette) {
        gif->depth = 6;
    }
    write(gif->fd, (uint8_t []) {0xF0 | (depth-1), (uint8_t) bgindex, 0x00}, 3);
    if (custom_gct) {
        write(gif->fd, palette, 3 << depth);
    } else if (depth <= 4) {
        write_and_store(store_gct, palette, gif->fd, vga, 3 << depth);
    } else {
        write_and_store(store_gct, palette, gif->fd, vga, sizeof(vga));
        i = 0x10;
        for (r = 0; r < 6; r++) {
            for (g = 0; g < 6; g++) {
                for (b = 0; b < 6; b++) {
                    write_and_store(store_gct, palette, gif->fd,
                      ((uint8_t []) {r*51, g*51, b*51}), 3
                    );
                    if (++i == 1 << depth)
                        goto done_gct;
                }
            }
        }
        for (i = 1; i <= 24; i++) {
            v = i * 0xFF / 25;
            write_and_store(store_gct, palette, gif->fd,
              ((uint8_t []) {v, v, v}), 3
            );
        }
    }
done_gct:
    if (loop >= 0 && loop <= 0xFFFF)
        put_loop(gif, (uint16_t) loop);
    return gif;
no_fd:
    free(gif);
no_gif:
    return NULL;
}

static void
put_loop(ge_GIF *gif, uint16_t loop)
{
    write(gif->fd, (uint8_t []) {'!', 0xFF, 0x0B}, 3);
    write(gif->fd, "NETSCAPE2.0", 11);
    write(gif->fd, (uint8_t []) {0x03, 0x01}, 2);
    write_num(gif->fd, loop);
    write(gif->fd, "\0", 1);
}

/* Add packed key to buffer, updating offset and partial.
 *   gif->offset holds position to put next *bit*
 *   gif->partial holds bits to include in next byte */
static void
put_key(ge_GIF *gif, uint16_t key, int key_size)
{
    int byte_offset, bit_offset, bits_to_write;
    byte_offset = gif->offset / 8;
    bit_offset = gif->offset % 8;
    gif->partial |= ((uint32_t) key) << bit_offset;
    bits_to_write = bit_offset + key_size;
    while (bits_to_write >= 8) {
        gif->buffer[byte_offset++] = gif->partial & 0xFF;
        if (byte_offset == 0xFF) {
            write(gif->fd, "\xFF", 1);
            write(gif->fd, gif->buffer, 0xFF);
            byte_offset = 0;
        }
        gif->partial >>= 8;
        bits_to_write -= 8;
    }
    gif->offset = (gif->offset + key_size) % (0xFF * 8);
}

static void
end_key(ge_GIF *gif)
{
    int byte_offset;
    byte_offset = gif->offset / 8;
    if (gif->offset % 8)
        gif->buffer[byte_offset++] = gif->partial & 0xFF;
    if (byte_offset) {
        write(gif->fd, (uint8_t []) {byte_offset}, 1);
        write(gif->fd, gif->buffer, byte_offset);
    }
    write(gif->fd, "\0", 1);
    gif->offset = gif->partial = 0;
}

static void
put_image(ge_GIF *gif, uint16_t w, uint16_t h, uint16_t x, uint16_t y)
{
    int nkeys, key_size, i, j;
    Node *node, *child, *root;
    int degree = 1 << gif->depth;

    write(gif->fd, ",", 1);
    write_num(gif->fd, x);
    write_num(gif->fd, y);
    write_num(gif->fd, w);
    write_num(gif->fd, h);
    write(gif->fd, (uint8_t []) {0x00, gif->depth}, 2);
    root = node = new_trie(degree, &nkeys);
    key_size = gif->depth + 1;
    put_key(gif, degree, key_size); /* clear code */
    for (i = y; i < y+h; i++) {
        for (j = x; j < x+w; j++) {
            uint8_t pixel = gif->frame[i*gif->w+j] & (degree - 1);
            child = node->children[pixel];
            if (child) {
                node = child;
            } else {
                put_key(gif, node->key, key_size);
                if (nkeys < 0x1000) {
                    if (nkeys == (1 << key_size))
                        key_size++;
                    node->children[pixel] = new_node(nkeys++, degree);
                } else {
                    put_key(gif, degree, key_size); /* clear code */
                    del_trie(root, degree);
                    root = node = new_trie(degree, &nkeys);
                    key_size = gif->depth + 1;
                }
                node = root->children[pixel];
            }
        }
    }
    put_key(gif, node->key, key_size);
    put_key(gif, degree + 1, key_size); /* stop code */
    end_key(gif);
    del_trie(root, degree);
}

static int
get_bbox(ge_GIF *gif, uint16_t *w, uint16_t *h, uint16_t *x, uint16_t *y)
{
    int i, j, k;
    int left, right, top, bottom;
    uint8_t back;
    left = gif->w; right = 0;
    top = gif->h; bottom = 0;
    k = 0;
    for (i = 0; i < gif->h; i++) {
        for (j = 0; j < gif->w; j++, k++) {
            back = gif->bgindex >= 0 ? gif->bgindex : gif->back[k];
            if (gif->frame[k] != back) {
                if (j < left)   left    = j;
                if (j > right)  right   = j;
                if (i < top)    top     = i;
                if (i > bottom) bottom  = i;
            }
        }
    }
    if (left != gif->w && top != gif->h) {
        *x = left; *y = top;
        *w = right - left + 1;
        *h = bottom - top + 1;
        return 1;
    } else {
        return 0;
    }
}

static void
add_graphics_control_extension(ge_GIF *gif, uint16_t d)
{
    uint8_t flags = ((gif->bgindex >= 0 ? 2 : 1) << 2) + 1;
    write(gif->fd, (uint8_t []) {'!', 0xF9, 0x04, flags}, 4);
    write_num(gif->fd, d);
    write(gif->fd, (uint8_t []) {(uint8_t) gif->bgindex, 0x00}, 2);
}

static uint8_t ge_anchor_color216(uint8_t* c) {
    uint8_t c_trg[6] = { 0x00, 0x5F, 0x87, 0xAF, 0xD7, 0xFF };
    // anchor closest color from c -> c_trg
    for(int i = 0; i < 6; i++) {
        if(c_trg[i] >= *c) {
            *c = c_trg[i];
            return i;
        }
    }
    *c = 0x00;
    return 0;
}

static uint8_t ge_quantify_color216(uint32_t c) {
    uint8_t c_r = (c >> 16) & 0xFF;
    uint8_t c_g = (c >>  8) & 0xFF;
    uint8_t c_b = (c >>  0) & 0xFF;
    // anchor closest color from c -> c_trg
    uint8_t i_r = ge_anchor_color216(&c_r);
    uint8_t i_g = ge_anchor_color216(&c_g);
    uint8_t i_b = ge_anchor_color216(&c_b);
    return i_r * 6 * 6 + i_g * 6 + i_b;
}

static uint8_t ge_match_color(ge_GIF *gif, uint8_t *color, int channel) {
    if(!gif || !color || channel <= 0) return 0;
    int all = 1 << gif->depth;
    uint32_t loc_color = 0;
    for(int j = 0; j < channel; j++) {
        loc_color |= (color[j] << (j * 8));
    }
    loc_color = 0x00FFFFFF & loc_color;
    if(gif->palette == ge_color216_palette) {
        return ge_quantify_color216(loc_color);
    } else {
        for(int i = 0; i < all; i++) {
            uint32_t loc_palette = 0;
            for(int j = 0; j < channel; j++) {
                if(channel < 3) {
                    loc_palette |= (gif->palette[i * 3 + j] << (j * 8));
                }
            }
            if(loc_color == loc_palette) {
                return i;
            }
        }
    }
    return 0;
}

void
ge_render_frame(ge_GIF *gif, uint8_t *frame, int channel) {
    if(!gif || !frame || channel <= 0) return;
    int width = gif->w;
    int height = gif->h;
    for(size_t i = 0; i < width * height; i++) {
        gif->frame[i] = ge_match_color(gif, frame + i * channel, channel);
    }
}

void
ge_add_frame(ge_GIF *gif, uint16_t delay)
{
    uint16_t w, h, x, y;
    uint8_t *tmp;

    if (delay || (gif->bgindex >= 0))
        add_graphics_control_extension(gif, delay);
    if (gif->nframes == 0) {
        w = gif->w;
        h = gif->h;
        x = y = 0;
    } else if (!get_bbox(gif, &w, &h, &x, &y)) {
        /* image's not changed; save one pixel just to add delay */
        w = h = 1;
        x = y = 0;
    }
    put_image(gif, w, h, x, y);
    gif->nframes++;
    if (gif->bgindex < 0) {
        tmp = gif->back;
        gif->back = gif->frame;
        gif->frame = tmp;
    }
}

void
ge_close_gif(ge_GIF* gif)
{
    write(gif->fd, ";", 1);
    close(gif->fd);
    free(gif);
}