"""Inspect metadata carried with the exact archived real RGB recordings."""

import argparse
from collections import Counter
import os
from pathlib import Path

import h5py

from source_custody import SOURCE_H5, VIEWS, pin, require, write_new


def metadata(payload):
    """Read JPEG metadata through EOI, including markers after entropy-coded scans."""
    require(payload.startswith(b"\xff\xd8"), "archived RGB is not JPEG")
    pos = 2
    in_scan = False
    result = []
    while pos < len(payload):
        if in_scan:
            pos = payload.find(b"\xff", pos)
            require(pos >= 0, "JPEG scan did not terminate")
        else:
            require(payload[pos] == 0xFF, "expected JPEG marker")
        while pos < len(payload) and payload[pos] == 0xFF:
            pos += 1
        require(pos < len(payload), "incomplete JPEG marker")
        marker = payload[pos]
        pos += 1
        if in_scan and (marker == 0 or 0xD0 <= marker <= 0xD7):
            continue
        if marker == 0xD9:
            return result, bool(any(payload[pos:]))
        require(pos + 2 <= len(payload), "incomplete JPEG segment length")
        length = int.from_bytes(payload[pos:pos+2], "big")
        require(length >= 2 and pos + length <= len(payload), "incomplete JPEG segment")
        if 0xE0 <= marker <= 0xEF or marker == 0xFE:
            data = payload[pos+2:pos+length]
            result.append((marker, data))
        in_scan = marker == 0xDA
        pos += length
    raise ValueError("JPEG end marker missing")


def label(marker, data):
    if marker == 0xE0 and data.startswith(b"JFIF\0"):
        return "APP0_JFIF"
    if marker == 0xFE and data == b"XPL-RGB1":
        return "COM_XPL_RGB1"
    if marker == 0xE1 and data.startswith(b"Exif\0\0"):
        return "APP1_EXIF"
    return f"UNINSPECTED_0x{marker:02X}"


def inventory(path):
    with h5py.File(path) as h:
        attributes = {"/": sorted(h.attrs)} if h.attrs else {}
        dataset_paths = []

        def visit(name, object):
            if object.attrs:
                attributes[name] = sorted(object.attrs)
            if isinstance(object, h5py.Dataset):
                dataset_paths.append(name)

        h.visititems(visit)
        views = {}
        for view in VIEWS:
            camera = h[f"vision/{view}"]
            marker_counts = Counter()
            nonzero_trailer = 0
            for buffer in camera["colors"]:
                fields, trailer = metadata(bytes(buffer))
                marker_counts.update(label(marker, data) for marker, data in fields)
                nonzero_trailer += trailer
            views[view] = dict(frames=len(camera["colors"]), shape=camera["shape"][:].tolist(),
                               fields=sorted(camera), metadata_segments=dict(marker_counts),
                               nonzero_post_eoi_frames=nonzero_trailer)
        return dict(archive_fps=int(h["additional_info/frequency"][()]), dataset_paths=dataset_paths,
                    object_attribute_keys=attributes, camera_views=views)


def run(root, output):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU archive inspection only")
    episodes = []
    for name, sha in SOURCE_H5.items():
        source = pin(root / "data" / name, sha)
        episodes.append(dict(episode=name, archive_sha256=sha, **inventory(source)))
    write_new(output, dict(source_recordings=episodes, new_model_forwards=0, native_ticks=0, updates=0))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    run(args.root.resolve(), args.output.resolve())
