"""Compare archived PD callback timing with the native 2 ms warmup clock."""

import argparse
from collections import Counter
from pathlib import Path
import struct

import numpy as np

from go2arx5_vla.world_model.init_trace_reader import RECORD
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import write


def inspect(path, steps=2500):
    last_pd = None
    pd_count = 0
    previous_after = None
    entry = None
    observations = []
    with path.open("rb") as stream:
        header = stream.read(16)
        if header[:8] != b"WM02IT01" or struct.unpack("<I", header[8:12])[0] != RECORD.size:
            raise ValueError("Unexpected original init trace ABI")
        for ticket in range(1 << 22):
            raw = stream.read(RECORD.size)
            if len(raw) != RECORD.size:
                break
            current = RECORD.unpack(raw)
            seq, ns, seconds, kind = current[:4]
            token = current[9]
            if seq != ticket:
                raise ValueError("Noncontiguous original init records")
            if kind == 10:
                last_pd = (round(seconds / .002), ns)
                pd_count += 1
            elif kind == 11:
                if token != len(observations) + 1 or entry is not None or last_pd is None:
                    raise ValueError("Unpaired entry or missing original PD callback")
                completed = round(seconds / .002)
                if completed != token - 1 or last_pd[0] > completed or ns < last_pd[1]:
                    raise ValueError("Callback cannot follow the native step that consumes it")
                entry = dict(tick=token, measurement_tick=last_pd[0], measurement_age_ticks=completed-last_pd[0],
                             callback_count=pd_count, entry_ns=ns, callback_to_entry_ns=ns-last_pd[1],
                             previous_after_to_entry_ns=ns-previous_after if previous_after is not None else None)
                pd_count = 0
            elif kind == 12:
                if entry is None or token != entry["tick"] or round(seconds / .002) != token:
                    raise ValueError("Unpaired exit or unexpected integration clock")
                observations.append(entry)
                entry = None
                previous_after = ns
                if token == steps:
                    break
    if len(observations) != steps:
        raise ValueError("An exact original 5-second warmup prefix is required")
    return observations


def summarize(path, key_tick):
    rows = inspect(path)
    no_callback = [row for row in rows[1:] if not row["callback_count"]]
    has_callback = [row for row in rows[1:] if row["callback_count"]]
    missed_wall = [row["previous_after_to_entry_ns"] / 1e6 for row in no_callback]
    met_wall = [row["previous_after_to_entry_ns"] / 1e6 for row in has_callback]
    return dict(trace_sha256=digest(path), examined_ticks=len(rows),
                callback_count_histogram=dict(Counter(str(r["callback_count"]) for r in rows[1:])),
                age_ticks_histogram=dict(Counter(str(r["measurement_age_ticks"]) for r in rows[1:])),
                zero_callback_tick_mod25=dict(Counter(str(r["tick"] % 25) for r in no_callback)),
                wall_gap_median_ms_with_and_without_callback=[
                    float(np.median(met_wall)) if met_wall else None,
                    float(np.median(missed_wall)) if missed_wall else None],
                window=[{key: value for key, value in row.items() if key != "entry_ns"}
                        for row in rows[key_tick-3:key_tick+2]])


def parse_member(value):
    family, role, tick, path = value.split(":", 3)
    return family, role, int(tick), Path(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("members", type=parse_member, nargs="+")
    args = parser.parse_args()
    records = [dict(family=family, topology=role, **summarize(path, tick))
               for family, role, tick, path in args.members]
    write(args.output, dict(schema="V64_ARCHIVED_WARMUP_PD_CLOCK", records=records, new_native_integrations=0))
