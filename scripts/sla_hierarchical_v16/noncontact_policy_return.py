"""Replay capture and delivery of the original frozen worker messages on the owner's clock."""

import hashlib
from pathlib import Path

import numpy as np

from demonstrated_grip import pin, read, require, rows
from observation_stream import ObservationStream
from policy_query import policy_response


class ArchivedPolicy:
    def __init__(self, root, capture, ep, receipt, calls, source_pins):
        self.root = root
        self.ep = ep
        self.windows = {
            w["tick"]: w for w in capture["windows"].values() if w["episode"] == ep
        }
        self.calls = calls
        self.raw_paths = {
            v: Path(k) for k, v in source_pins.items() if k.endswith(".npy")
        }
        registration = root / "v12/TARGET_REGISTRATION.json"
        config = read(pin(registration, source_pins[str(registration)]))
        needed = {t for w in self.windows.values() for t in w["head_ticks"]}
        d = root / "v12/DENSE" / ep
        records = rows(pin(d / "frames.jsonl", receipt["frames_sha256"]))
        self.frames = {}
        offset = 0
        for shard in receipt["shards"]:
            segment = records[offset : offset + shard["count"]]
            offset += shard["count"]
            indices = [i for i, f in enumerate(segment) if f["tick"] in needed]
            if not indices:
                continue
            with np.load(
                pin(d / shard["name"], shard["sha256"]), allow_pickle=False
            ) as z:
                rgb = z["rgb"]
                require(
                    [f["tick"] for f in segment] == z["tick"].tolist(),
                    "dense shard clock",
                )
                for i in indices:
                    f = segment[i]
                    pair = rgb[i].copy()
                    require(
                        [hashlib.sha256(v.tobytes()).hexdigest() for v in pair]
                        == f["rgb_sha256"],
                        "dense pixels",
                    )
                    self.frames[f["tick"]] = (f, pair)
        require(
            needed == self.frames.keys() and offset == len(records),
            "required history missing",
        )
        self.timeline = sorted(needed)
        self.cursor = 0
        first = self.windows[min(self.windows)]
        with self.model_input(first) as z:
            self.stream = ObservationStream(
                ep,
                str(z["prompt"]),
                z["state"][:8],
                config["source_origin"],
                config["T_link6_TCP"],
            )
        self.pending = {}
        self.captured = []
        self.finished = []

    def model_input(self, w):
        path = self.root / "v12/STUDENT_INPUT" / (w["id"] + ".npz")
        return np.load(pin(path, w["input_sha256"]), allow_pickle=False)

    def capture(self, tick):
        while self.cursor < len(self.timeline) and self.timeline[self.cursor] <= tick:
            t = self.timeline[self.cursor]
            f, (body, wrist) = self.frames[t]
            self.stream.observe(
                self.ep,
                t,
                f["sim_time_s"],
                body,
                wrist,
                f["T_W_armbase"],
                f["T_W_link6"],
                f["T_W_TCP"],
                f["finger_q_m"],
            )
            self.cursor += 1
        if tick not in self.windows:
            return
        w = self.windows[tick]
        name = w["id"]
        key, obs = self.stream.begin_query()
        with self.model_input(w) as z:
            require(
                np.array_equal(obs["images"]["cam_high"], z["head"])
                and np.array_equal(obs["images"]["cam_right_wrist"], z["right_wrist"])
                and not obs["images"]["cam_left_wrist"].any()
                and np.array_equal(obs["state"], z["state"])
                and obs["prompt"] == str(z["prompt"]),
                "original worker input differs",
            )
        call = self.calls[name]
        require(
            call["input_sha256"] == w["input_sha256"] and call["tick"] == tick,
            "worker identity",
        )
        h = call["file_sha256"]
        self.pending[name] = policy_response(
            key,
            obs,
            lambda unused: np.load(pin(self.raw_paths[h], h), allow_pickle=False),
        )
        self.captured.append(name)

    def deliver(self, tick, first, schedule):
        for name in list(self.pending):
            if first + schedule[name]["return_s"] / 0.002 <= tick:
                self.stream.finish_query(*self.pending.pop(name))
                self.finished.append(name)

    def check_terminal(self):
        refused = []
        for k, msg in self.pending.items():
            try:
                self.stream.finish_query(*msg)
            except ValueError as error:
                require(str(error) == "episode stopped", "different late-return cause")
                refused.append(k)
            else:
                raise ValueError("stopped stream accepted late model output")
        for action in (self.stream.begin_query, self.stream.take_proposal):
            try:
                action()
            except ValueError as error:
                require(
                    str(error) == "episode stopped",
                    "unexpected terminal stream failure",
                )
            else:
                raise ValueError("terminal stream supplied a proposal")
        return dict(
            queries_started=self.captured,
            worker_returns_delivered=self.finished,
            genuine_archived_late_returns_refused=refused,
            terminal_query_and_mailbox_closed=True,
        )
