"""Read only one licensed RoboCasa fixture from the official public archive."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import urllib.request
from urllib.error import URLError
import zipfile


URL = "https://utexas.box.com/shared/static/vckqvvkh1z8t69k8qcpcmee6k66stii4.zip"
REVISION = "7bfc60c9d26c02506d4265cbe0e86b35e6e25d42"
ASSET = ("fruit_bowl", "FruitBowl001")
ARCHIVE_BYTES = 791_939_656
CAP = 32 * 1024**2


class RangeZip:
    def __init__(self):
        self.position = 0
        self.received = 0
        self.requests = 0
        self.url = URL

    def seek(self, amount, origin=io.SEEK_SET):
        if origin == io.SEEK_SET:
            cursor = amount
        elif origin == io.SEEK_END:
            cursor = ARCHIVE_BYTES + amount
        elif origin == io.SEEK_CUR:
            cursor = self.position + amount
        else:
            raise ValueError("Unknown archive seek")
        if not 0 <= cursor <= ARCHIVE_BYTES:
            raise ValueError("Out-of-range archive seek")
        self.position = cursor
        return cursor

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def read(self, amount=-1):
        if amount < 0:
            amount = ARCHIVE_BYTES - self.position
        amount = min(amount, ARCHIVE_BYTES - self.position)
        if amount == 0:
            return b""
        if self.received + amount > CAP:
            raise ValueError("Bounded download would exceed the isolated fixture budget")
        start = self.position
        end = start + amount - 1
        for attempt in (0, 1):
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}",
                                                                "Accept-Encoding": "identity"})
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    if (response.status != 206
                            or response.headers.get("Content-Range") != f"bytes {start}-{end}/{ARCHIVE_BYTES}"):
                        raise ValueError("Official archive does not expose the exact member range")
                    self.url = response.geturl()
                    data = response.read(amount + 1)
                break
            except (URLError, ConnectionError, TimeoutError):
                if attempt:
                    raise
                self.url = URL
        if len(data) != amount:
            raise ValueError("Incomplete exact archive range")
        self.position += amount
        self.received += amount
        self.requests += 1
        return data


def acquire(output):
    remote = RangeZip()
    output.mkdir(mode=0o700)
    found = []
    with zipfile.ZipFile(remote) as archive:
        matches = []
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if len(path.parts) >= 3 and not entry.is_dir() and "__MACOSX" not in path.parts:
                pairs = zip(path.parts[:-1], path.parts[1:])
                if ASSET in pairs:
                    matches.append((entry, path))
        if not 1 <= len(matches) <= 24 or sum(entry.file_size for entry, _ in matches) > CAP:
            raise ValueError("Expected a small exact RoboCasa object directory")
        for entry, path in matches:
            anchor = path.parts.index(ASSET[1])
            relative = PurePosixPath(*path.parts[anchor + 1:])
            if not relative.parts or ".." in relative.parts:
                raise ValueError("Unsafe or missing asset member path")
            if relative.name.startswith("."):
                continue
            data = archive.read(entry)
            target = output / relative.as_posix()
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
            found.append(dict(archive_member=entry.filename, file=relative.as_posix(), bytes=len(data),
                              sha256=hashlib.sha256(data).hexdigest()))
    if not any(row["file"].endswith(".xml") for row in found):
        raise ValueError("No actual MJCF in the fixed object directory")
    with (output / "SOURCE.json").open("x") as out:
        json.dump(dict(project="RoboCasa365 / Lightwheel", license="CC-BY-4.0", commit=REVISION,
                       asset="/".join(ASSET), upstream_archive=URL, source_members=found,
                       http_range_bytes=remote.received, range_requests=remote.requests,
                       new_native_requests=0, new_physics_ticks=0), out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    acquire(parser.parse_args().output)
