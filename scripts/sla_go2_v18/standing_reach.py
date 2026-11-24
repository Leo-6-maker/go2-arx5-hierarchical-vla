"""Fixed-base no-contact task and per-tick observations for Q/E1 on the two sealed geometries."""

import json

import numpy as np
from demonstrated_grip import require
from scipy.spatial.transform import Rotation


class ReachProposalRejected(ValueError):
    """A well-formed proposal failed the Reach-specific geometry gate."""


def angle(a, b):
    return float(np.degrees(Rotation.from_matrix(a @ b.T).magnitude()))


class StateMonitor:
    def __init__(self, model):
        import mujoco

        self.model = model
        self.data = mujoco.MjData(model)
        self.names = [model.geom(i).name for i in range(model.ngeom)]
        self.root = model.body("base_link").id
        self.feet = [model.geom(x).id for x in ("FL", "FR", "RL", "RR")]
        self.objects = [
            model.geom(n).id for n in ("task_cube_geom", "task_blue_cube_geom")
        ]
        self.tool = model.site("x5_ee").id
        self.geoms = []
        for g in range(model.ngeom):
            b = int(model.geom_bodyid[g])
            while b and b != self.root:
                b = int(model.body_parentid[b])
            if b == self.root and (model.geom_contype[g] or model.geom_conaffinity[g]):
                self.geoms.append(g)
        require(
            len(self.geoms) == 30 and (model.nq, model.nv, model.njnt) == (41, 38, 23),
            "registered collision geometry",
        )
        self.entry = None
        self.last = None
        self.stable = []
        self.task_rows = []

    def clearance(self, data):
        import mujoco

        return min(
            mujoco.mj_geomDistance(self.model, data, g, o, 0.25, None)
            for g in self.geoms
            for o in self.objects
        )

    @staticmethod
    def permitted(pair, distance):
        from go2arx5_vla.world_model.multitask_contact_v6 import observer_permitted

        return observer_permitted(*pair, distance, "red") and not (
            {"task_cube_geom", "task_blue_cube_geom"} & set(pair)
            and "task_table" not in pair
        )

    def observe(self, p):
        import mujoco
        from fault_reference import floor_feet

        require(self.last is None or p["step"] == self.last + 1, "missing native step")
        self.last = p["step"]
        require(
            abs(p["time_s"] - self.last * 0.002) < 1e-8
            and set(floor_feet(self.names, p)) == {"FL", "FR", "RL", "RR"},
            "standing clock or support",
        )
        for c in p["contacts"]:
            require(
                self.permitted(
                    (self.names[c["geom0"]], self.names[c["geom1"]]), c["distance_m"]
                ),
                "prohibited task contact",
            )
        d = self.data
        d.qpos[:] = p["qpos"]
        mujoco.mj_fwdPosition(self.model, d)
        row = (
            d.site_xpos[self.tool].copy(),
            d.site_xmat[self.tool].reshape(3, 3).copy(),
            d.xpos[self.root].copy(),
            d.xmat[self.root].reshape(3, 3).copy(),
            d.geom_xpos[self.feet].copy(),
            float(self.clearance(d)),
            d.geom_xpos[self.objects[0]].copy(),
        )
        require(row[5] >= 0.015, "measured robot/cube clearance below 15 mm")
        if self.entry is not None:
            r = self.entry
            require(
                np.linalg.norm(row[2] - r[2]) <= 0.020
                and angle(row[3], r[3]) <= 5.0
                and max(np.linalg.norm(row[4] - r[4], axis=1)) <= 0.005,
                "standing task limits",
            )
            self.task_rows.append(row)
        self.stable.append(row)
        if len(self.stable) > 501:
            self.stable.pop(0)
        return row

    def stable_now(self):
        if len(self.stable) != 501:
            return False
        a = self.stable[0]
        return all(
            np.linalg.norm(r[0] - a[0]) <= 0.001
            and np.linalg.norm(r[2] - a[2]) <= 0.002
            and angle(r[1], a[1]) <= 0.5
            for r in self.stable
        )

    def score(self, direction, pregrasp):
        r = self.entry
        tail = self.task_rows[-251:]
        require(len(tail) == 251, "missing terminal half-second")
        endpoints = []
        cross = []
        for v in self.task_rows:
            delta = v[0] - r[0]
            cross.append(
                float(1000 * np.linalg.norm(delta - (delta @ direction) * direction))
            )
        for v in tail:
            delta = v[0] - r[0]
            endpoints.append(
                (
                    float(1000 * delta @ direction),
                    float(
                        1000
                        * (
                            np.linalg.norm(pregrasp - r[0])
                            - np.linalg.norm(pregrasp - v[0])
                        )
                    ),
                    float(1000 * np.linalg.norm(delta - 0.02 * direction)),
                    float(
                        1000
                        * (np.linalg.norm(r[6] - r[0]) - np.linalg.norm(v[6] - v[0]))
                    ),
                )
            )
        vals = np.asarray(endpoints)
        amax = max(angle(v[1], r[1]) for v in self.task_rows)
        return {
            "minimum_terminal_projection_mm": float(vals[:, 0].min()),
            "minimum_terminal_pregrasp_progress_mm": float(vals[:, 1].min()),
            "minimum_terminal_object_progress_mm": float(vals[:, 3].min()),
            "maximum_terminal_endpoint_error_mm": float(vals[:, 2].max()),
            "maximum_cross_track_mm": max(cross),
            "maximum_orientation_error_deg": amax,
            "minimum_clearance_mm": 1000 * min(v[5] for v in self.task_rows),
            "passed": bool(
                vals[:, 0].min() >= 12
                and vals[:, 1].min() >= 10
                and vals[:, 3].min() >= 10
                and vals[:, 2].max() <= 8
                and max(cross) <= 8
                and amax <= 2
            ),
        }


def knots(anchor, direction, elapsed):
    t = elapsed + np.arange(41) * 0.05
    u = np.clip(t / 4, 0, 1)
    v = np.repeat(anchor[None], 41, axis=0)
    v[:, :3, 3] += (0.02 * u**3 * (10 - 15 * u + 6 * u * u))[:, None] * direction
    v[-1] = v[-2]
    return v


def drive(a, case, card, profile, fault, result, student=None):
    from b2_native import fixed, paused
    from d1_factorial_reference_matrix import boundary
    from d1_geometry import sweep_profile
    from go2arx5_vla.world_model.mobile_safety_profile_v1 import binding
    from go2arx5_vla.world_model.multitask_state_v6 import read_journal_packet
    from identity import read, record
    from known_dynamic_panel import KnownCurve
    from native_known import atom_receipt, install, read_line
    from scripts.wm02 import mobile_boundary_runner_v1 as mobile

    out = a.output
    a.prepare_dynamic()
    require(
        read(out / "GUARD_READY.json")
        == {
            "episode": out.name,
            "pid": next(p.pid for n, p in a.processes if n == "planner"),
        },
        "receiver owner",
    )
    mount = mobile.arm(a)
    sample = a.measured_task_state()
    action, base = fixed(sample, mount)
    for i in range(128):
        if i % 7 == 0:
            mobile.install(a, action, base)
        sample, _ = mobile.consume(a, action, sequence=i)
    paused(a, 5703)
    last = read_line(out / "MOBILE_SAMPLES.private.jsonl")
    parts, wire, seq = boundary(out, last)
    source = round(a.active_reference["source_mpc_s"] / 0.002)
    receipt = atom_receipt(out, last, sample, source, wire)
    model = a.actual_state.model
    monitor = StateMonitor(model)
    guard = binding(a.spec["native_root"])
    pregrasp = np.array(profile["original_teacher_pregrasp"])
    tick = 5703
    entry = None
    direction = None
    steady = parts["old_ee"]
    with (
        (out / "state.bin.journal").open("rb") as journal,
        (case / "REACH_WIRES.jsonl").open("x", buffering=1) as ledger,
    ):
        for i in range(1, tick + 1):
            p = read_journal_packet(journal)
            require(p["step"] == i, "initial clock")
        monitor.observe(p)
        n = 0
        while entry is None or tick < entry + (4500 if student else 2250):
            paused(a, tick)
            if entry is None and monitor.stable_now():
                entry = tick
                monitor.entry = monitor.stable[-1]
                direction = pregrasp - monitor.entry[0]
                direction /= np.linalg.norm(direction)
                steady = receipt["end_pose"].copy()
                record(
                    case / "REACH_ENTRY.json",
                    {
                        "tick": entry,
                        "actual_xyz": monitor.entry[0].tolist(),
                        "consumed": steady.tolist(),
                        "direction": direction.tolist(),
                        "reference_wire": receipt["consumed"],
                        "stable_samples": 501,
                    },
                )
                if card == "G00-STOP":
                    fault[entry + 1025] = (entry + 1000) * 0.002 + 0.021
            require(entry is not None or tick < 15500, "stable entry deadline")
            if entry is None:
                v = np.repeat(steady[None], 41, axis=0)
            preview_finished = False
            try:
                if entry is not None:
                    if student is None:
                        v = knots(steady, direction, (tick - entry) * 0.002)
                    else:
                        v = student.preview(
                            a, tick, receipt, sample, monitor, steady, direction
                        )
                preview_finished = True
                KnownCurve.admit(v, receipt, sample)
                for pose in np.unique(v.reshape(41, 16), axis=0).reshape(-1, 4, 4):

                    def witness(data):
                        clearance = monitor.clearance(data)
                        require(
                            clearance >= 0.015,
                            f"prospective robot/cube clearance {1000 * clearance:.5f} mm below 15 mm",
                        )

                    sweep_profile(
                        model,
                        sample,
                        dict(parts, old_ee=receipt["end_pose"]),
                        pose,
                        parts["old_base"],
                        guard,
                        (41, 38, 23),
                        monitor.permitted,
                        witness,
                    )
            except (ValueError, RuntimeError, AssertionError) as exc:
                if (
                    student is None
                    or entry is None
                    or (
                        not preview_finished
                        and not isinstance(exc, ReachProposalRejected)
                    )
                ):
                    raise
                student.refusal = {
                    "kind": "REFERENCE_GATE",
                    "tick": tick,
                    "reason": str(exc),
                }
                student.log("REFUSED", reason=student.refusal)
                result["refusal"] = student.refusal
                result["student"] = student.summary()
                result["outcome"] = "A_REFERENCE_REFUSED"
                result["end"] = tick
                result["cancel"] = a.cancel_reference()
                require(
                    result["cancel"]["native_step"] == tick
                    and result["cancel"]["mode"] == "HOLD",
                    "refusal terminal hold",
                )
                a.seal_trace(tick)
                return
            action, wire = install(a, v, parts["old_base"], seq + 1 + n, tick)
            source = tick
            ledger.write(
                json.dumps(
                    {
                        "tick": tick,
                        "wire": wire,
                        "start": v[0, :3, 3].tolist(),
                        "next": v[2, :3, 3].tolist(),
                        "entry": entry,
                        "actual": sample["tcp"][:3, 3].tolist(),
                    }
                )
                + "\n"
            )
            for j in range(2):
                sample, _ = mobile.consume(a, action, sequence=128 + 2 * n + j)
                receipt = atom_receipt(
                    out,
                    read_line(out / "MOBILE_SAMPLES.private.jsonl"),
                    sample,
                    source,
                    wire,
                )
                require(
                    receipt["end"] == tick + 25
                    and np.allclose(receipt["end_pose"], v[j + 1], rtol=0, atol=2e-7),
                    "consumption identity",
                )
                for k in range(25):
                    p = read_journal_packet(journal)
                    require(p["step"] == tick + k + 1, "live state/receipt join")
                    monitor.observe(p)
                tick += 25
                if student is not None and entry is not None:
                    student.consumed(tick, receipt, sample)
            n += 1
        require(card != "G00-STOP", "registered fault not observed")
        result["score"] = monitor.score(direction, pregrasp)
        result["end"] = tick
        result["outcome"] = (
            "H0_PASS" if result["score"]["passed"] else "H0_REACH_FAILED"
        )
        if student is not None:
            result["student"] = student.summary()
            result["outcome"] = (
                "A_PASS"
                if result["score"]["passed"]
                and (
                    result["student"]["post_motion_forward"]
                    or result["student"].get("post_motion_geometric_update", False)
                )
                else "A_REACH_FAILED"
            )
        result["cancel"] = a.cancel_reference()
        require(
            result["cancel"]["native_step"] == tick
            and result["cancel"]["mode"] == "HOLD",
            "terminal hold",
        )
        a.seal_trace(tick)
