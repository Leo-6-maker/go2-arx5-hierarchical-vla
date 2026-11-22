/* Scope an LTTng 2.13 probe to owner-created files. This is an opt-in
 * nonphysical loader prototype. Keep the frozen production launch unchanged. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

static const char *wait_page(const char *name) {
    char user_name[64];
    snprintf(user_name, sizeof user_name, "/lttng-ust-wait-8-%lu", (unsigned long)getuid());
    if (strcmp(name, "/lttng-ust-wait-8") == 0)
        return "global.wait";
    if (strcmp(name, user_name) == 0)
        return "user.wait";
    return NULL;
}

int shm_open(const char *name, int flags, mode_t mode) {
    const char *realm = secure_getenv("SLA_MULTIWORLD_UST_REALM");
    const char *page = wait_page(name);
    if (realm == NULL || page == NULL) {
        int (*original)(const char *, int, mode_t) = dlsym(RTLD_NEXT, "shm_open");
        if (original == NULL) {
            errno = ENOSYS;
            return -1;
        }
        return original(name, flags, mode);
    }

    /* No traced process may create another /dev/shm route on a failed bind. */
    if (realm[0] != '/' || (flags & O_ACCMODE) != O_RDONLY) {
        errno = EACCES;
        return -1;
    }
    int dir = open(realm, O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
    if (dir < 0)
        return -1;
    struct stat check;
    int fd = -1;
    if (fstat(dir, &check) == 0 && check.st_uid == geteuid() && (check.st_mode & 077) == 0)
        fd = openat(dir, page, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    close(dir);
    long page_size = sysconf(_SC_PAGESIZE);
    if (fd < 0 || fstat(fd, &check) < 0 || !S_ISREG(check.st_mode)
        || check.st_uid != geteuid() || (check.st_mode & 077) || check.st_size != page_size) {
        if (fd >= 0)
            close(fd);
        errno = EACCES;
        return -1;
    }
    return fd;
}

static int unix_path(const struct sockaddr *address, socklen_t size, const char *path) {
    size_t length = strlen(path) + 1;
    if (address == NULL || size < offsetof(struct sockaddr_un, sun_path) + length
        || address->sa_family != AF_UNIX)
        return 0;
    return memcmp(((const struct sockaddr_un *)address)->sun_path, path, length) == 0;
}

int connect(int fd, const struct sockaddr *address, socklen_t size) {
    if (secure_getenv("SLA_MULTIWORLD_UST_REALM") != NULL
        && (unix_path(address, size, "/var/run/lttng/lttng-ust-sock-8")
            || unix_path(address, size, "/run/lttng/lttng-ust-sock-8"))) {
        errno = ENOENT;
        return -1;
    }
    int (*original)(int, const struct sockaddr *, socklen_t) = dlsym(RTLD_NEXT, "connect");
    if (original == NULL) {
        errno = ENOSYS;
        return -1;
    }
    return original(fd, address, size);
}
