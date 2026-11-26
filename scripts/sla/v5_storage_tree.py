"""Create and compare byte-complete manifests for closed Corelab run trees."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import sys


def require(condition, why):
    if not condition:
        raise ValueError(why)


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stamp(node):
    return (node.st_dev, node.st_ino, node.st_size, node.st_mode, node.st_uid,
            node.st_gid, node.st_mtime_ns, node.st_ctime_ns, node.st_nlink)


def metadata(root):
    layout = {}
    todo = [root]
    device = root.lstat().st_dev
    while todo:
        path = todo.pop()
        node = path.lstat()
        require(node.st_dev == device, f"Unexpected mounted member {path}")
        mode = node.st_mode
        kind = ("directory" if stat.S_ISDIR(mode) else "file" if stat.S_ISREG(mode)
                else "symlink" if stat.S_ISLNK(mode) else None)
        require(kind is not None, f"Unsupported special archive member {path}")
        layout[str(path.relative_to(root))] = kind, node
        if kind == "directory":
            todo.extend(path.iterdir())
    return layout


def mode_only_nfs_acl(value, mode):
    """Match the Linux nfsd projection of three POSIX permission classes."""
    entries = [(b"OWNER@", (mode >> 6) & 7), (b"GROUP@", (mode >> 3) & 7),
               (b"EVERYONE@", mode & 7)]
    encoded = [struct.pack("!I", len(entries))]
    for who, permissions in entries:
        mask = 0x120080
        if who == b"OWNER@":
            mask |= 0x040100
        if permissions & 4:
            mask |= 0x1
        if permissions & 2:
            mask |= 0x46 if stat.S_ISDIR(mode) else 0x6
        if permissions & 1:
            mask |= 0x20
        encoded.extend((struct.pack("!IIII", 0, 0, mask, len(who)), who, b"\0" * (-len(who) % 4)))
    return value == b"".join(encoded)


def scan(root, *, owner=None, progress=False, accept_trivial_nfs_acl=False):
    root = Path(root)
    require(root.resolve(strict=True) == root and root.is_dir(), "Archive root must be a real absolute directory")
    initial = metadata(root)
    rows = {}
    content_hash = {}
    logical_bytes = allocated_bytes = regular = 0
    normalized_mode_acls = 0
    for name in sorted(initial):
        kind, node = initial[name]
        path = root / name
        require(owner is None or node.st_uid == owner, f"Unexpected owner for {path}")
        require(stamp(path.lstat()) == stamp(node), f"Member changed while scanning {path}")
        attrs = {}
        for key in sorted(os.listxattr(path, follow_symlinks=False)):
            value = os.getxattr(path, key, follow_symlinks=False)
            if key == "system.nfs4_acl" and accept_trivial_nfs_acl and mode_only_nfs_acl(value, node.st_mode):
                normalized_mode_acls += 1
                continue
            attrs[key] = hashlib.sha256(value).hexdigest()
        row = dict(kind=kind, mode=stat.S_IMODE(node.st_mode), uid=node.st_uid, gid=node.st_gid, xattrs=attrs)
        if kind == "symlink":
            row["target"] = os.readlink(path)
        if kind == "file":
            key = node.st_dev, node.st_ino
            if key not in content_hash:
                content_hash[key] = name, file_sha(path)
                logical_bytes += node.st_size
                allocated_bytes += node.st_blocks * 512
            first_name, digest = content_hash[key]
            row.update(bytes=node.st_size, sha256=digest, link_member=first_name, mtime_ns=node.st_mtime_ns)
            regular += 1
            if progress and regular % 1000 == 0:
                print(f"Verified {regular} files, {logical_bytes} bytes", file=sys.stderr, flush=True)
        rows[name] = row
    final = metadata(root)
    require(initial.keys() == final.keys() and all(
        final[name][0] == kind and stamp(final[name][1]) == stamp(node)
        for name, (kind, node) in initial.items()
    ), "Archive source mutated during the complete snapshot")
    members = [dict(path=name, **row) for name, row in rows.items()]
    payload = json.dumps(members, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return dict(schema="corelab-closed-storage-tree-v1", root=str(root), device=root.stat().st_dev,
                regular_members=regular, unique_file_bytes=logical_bytes, unique_file_allocation_bytes=allocated_bytes,
                normalized_mode_only_nfs_acls=normalized_mode_acls,
                member_sha256=hashlib.sha256(payload).hexdigest(), members=members)


def compare(expected, observed):
    require(expected["schema"] == observed["schema"] == "corelab-closed-storage-tree-v1", "Unexpected manifest schema")
    require(expected["member_sha256"] == observed["member_sha256"] and expected["members"] == observed["members"],
            "Archive member set, bytes, metadata, symlink, or hardlink membership differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--same-owner", action="store_true")
    parser.add_argument("--verify-against", type=Path)
    args = parser.parse_args()
    manifest = scan(args.root, owner=os.getuid() if args.same_owner else None, progress=True)
    if args.verify_against is not None:
        compare(json.loads(args.verify_against.read_bytes()), manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
