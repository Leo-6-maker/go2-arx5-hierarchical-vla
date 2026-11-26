"""Read selected public Table30v2 tar members using bounded, exact HTTP ranges."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

from v6_ville_pilot import need, write


REVISION = "c58ad2cc76ce722ea54de51f9fae03012a698f47"
PART = "place_objects_into_desk_drawer.tar.part-0000"
CAP = 8 * 1024**2


def run(out):
    out.mkdir(mode=0o700)
    url = "https://huggingface.co/datasets/RoboChallenge/Table30v2/resolve/" + REVISION + "/" + PART
    reads = []
    episodes = {}
    files = []
    used = 0

    def area(offset, size):
        nonlocal used
        need(used + size <= CAP and len(reads) < 100, "Fixed sample range budget exceeded")
        request = urllib.request.Request(url, headers={
            "Range": f"bytes={offset}-{offset+size-1}", "Accept-Encoding": "identity",
            "User-Agent": "ARX5-Table30v2-bounded-pilot"})
        with urllib.request.urlopen(request, timeout=25) as response:
            data = response.read(size+1)
            need(response.status == 206 and response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
                 and len(data) == size, "Server did not provide the requested exact limited range")
        reads.append((offset, size))
        used += size
        return data

    offset = 0
    for _ in range(80):
        head = area(offset, 512)
        if not any(head):
            break
        member = tarfile.TarInfo.frombuf(head, "utf-8", "surrogateescape")
        path = Path(member.name)
        if member.isfile() and path.name in ("episode_meta.json","states.jsonl","task_info.json") and 0 < member.size <= 2**20:
            raw = area(offset+512, member.size)
            bundle = episodes.setdefault(str(path.parent), {})
            bundle[path.name] = raw
            name = f"{len(files):02d}-{path.name}"
            (out/name).write_bytes(raw)
            files.append(dict(member=member.name, file=name, bytes=member.size, sha256=hashlib.sha256(raw).hexdigest()))
        if sum("episode_meta.json" in b and "states.jsonl" in b for b in episodes.values()) >= 2:
            break
        offset += 512 + (member.size+511)//512*512
    rows = []
    for key, data in episodes.items():
        if "episode_meta.json" in data and "states.jsonl" in data:
            meta = json.loads(data["episode_meta.json"])
            samples = [json.loads(s) for s in data["states.jsonl"].splitlines()]
            need(samples, "An advertised state episode is empty")
            rows.append(dict(directory=key, robot_id=meta.get("robot_id"), declared=meta,
                             rows=len(samples), first_state_fields=list(samples[0]),
                             first_state_example=samples[0], last_state_example=samples[-1]))
    write(out/"ACTUAL.json",dict(revision=REVISION,part=PART,files=files,range_reads=reads,
                                downloaded_bytes=used,episode_state_evidence=rows,
                                public_dataset_license_verified=False,new_admitted=0,
                                rgb_frames_decoded=0,phase="BOUNDED_METADATA_PROBE"))
    print("Table30v2 files, paired states, bytes", len(files),len(rows),used)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    run(parser.parse_args().out)
