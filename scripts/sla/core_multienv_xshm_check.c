#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/shm.h>
#include <unistd.h>

enum { SIDE = 16 };
static int x_error;

static int record_x_error(Display *display, XErrorEvent *event) {
    (void)display;
    x_error = event->error_code;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2 || !getenv("DISPLAY")) {
        return 2;
    }
    Display *display = XOpenDisplay(NULL);
    if (!display || !XShmQueryExtension(display)) {
        return 3;
    }
    XSetErrorHandler(record_x_error);
    int screen = DefaultScreen(display);
    int depth = DefaultDepth(display, screen);
    XShmSegmentInfo segment = {0};
    XImage *image = XShmCreateImage(display, DefaultVisual(display, screen),
                                   depth, ZPixmap, NULL, &segment, SIDE, SIDE);
    if (!image || depth != 24) {
        return 4;
    }
    size_t bytes = (size_t)image->bytes_per_line * SIDE;
    segment.shmid = shmget(IPC_PRIVATE, bytes, IPC_CREAT | 0600);
    if (segment.shmid < 0) {
        return 5;
    }
    segment.shmaddr = shmat(segment.shmid, NULL, 0);
    if (segment.shmaddr == (void *)-1) {
        shmctl(segment.shmid, IPC_RMID, NULL);
        return 6;
    }
    image->data = segment.shmaddr;
    int attached = XShmAttach(display, &segment);
    XSync(display, False);
    if (shmctl(segment.shmid, IPC_RMID, NULL) < 0 || !attached || x_error) {
        return 7;
    }
    Pixmap pixmap = XCreatePixmap(display, RootWindow(display, screen), SIDE, SIDE, depth);
    GC gc = XCreateGC(display, pixmap, 0, NULL);
    printf("READY %d %d %s\n", getpid(), segment.shmid, DisplayString(display));
    fflush(stdout);

    unsigned int marker = (unsigned int)strtoul(argv[1], NULL, 10);
    unsigned int sequence;
    int status = 0;
    while (scanf("%u", &sequence) == 1) {
        unsigned long pixel = ((marker & 255U) << 16) | (sequence & 65535U);
        for (int y = 0; y < SIDE; y++) {
            for (int x = 0; x < SIDE; x++) {
                XPutPixel(image, x, y, pixel);
            }
        }
        int sent = XShmPutImage(display, pixmap, gc, image, 0, 0, 0, 0,
                                SIDE, SIDE, False);
        XSync(display, False);
        XImage *readback = XGetImage(display, pixmap, 0, 0, SIDE, SIDE,
                                     AllPlanes, ZPixmap);
        if (!sent || !readback || x_error) {
            status = 8;
            break;
        }
        for (int y = 0; y < SIDE; y++) {
            for (int x = 0; x < SIDE; x++) {
                if (XGetPixel(readback, x, y) != pixel) {
                    status = 9;
                }
            }
        }
        XDestroyImage(readback);
        if (status) {
            break;
        }
        printf("ACK %u %lx\n", sequence, pixel);
        fflush(stdout);
    }
    XFreeGC(display, gc);
    XFreePixmap(display, pixmap);
    XShmDetach(display, &segment);
    XSync(display, False);
    image->data = NULL;
    XDestroyImage(image);
    shmdt(segment.shmaddr);
    XCloseDisplay(display);
    return status;
}
