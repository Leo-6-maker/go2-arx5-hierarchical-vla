"""Consume the preregistered dynamic reference from the last measured and acknowledged F2 boundary."""

import json
from pathlib import Path

import numpy as np
from demonstrated_grip import require
from known_dynamic_curve import trajectory

RATES = {"SLOW": (0.004, np.deg2rad(0.6)), "SOURCE_SPEED": (0.01125, 0.045)}
CARDS = ("SLOW_A0", "SLOW_A1", "SOURCE_SPEED_A1", "SOURCE_SPEED_A0")


class KnownCurve:
    def __init__(self, card):
        require(card in CARDS, "unknown scientific slot")
        rate, self.mode = card.rsplit("_", 1)
        self.rate = RATES[rate]

    @staticmethod
    def support(names, packet):
        from fault_reference import floor_feet

        from scripts.pi0294.task_contacts import allowed

        require(
            set(floor_feet(names, packet)) == {"FL", "FR", "RL", "RR"},
            "DYNAMIC_FOUR_FOOT_CONTACT",
        )
        for pair in packet["contacts"]:
            a, b = (names[pair[k]] for k in ("geom0", "geom1"))
            require(
                allowed(a, b)
                and ("task_cube_geom" not in (a, b) or "task_table" in (a, b)),
                "DYNAMIC_PROHIBITED_CONTACT",
            )

    @staticmethod
    def admit(knots, receipt, sample):
        from feedback import OPEN
        from go2arx5_vla.world_model.mobile_wbc_reference_v1 import full_eef_gate

        from scripts.pi0291_arx5_data.native_dataset import pose10

        require(
            knots.shape == (41, 4, 4)
            and np.isfinite(knots).all()
            and receipt["finger_fraction"] == OPEN
            and np.allclose(knots[0], receipt["end_pose"], atol=2e-7, rtol=0),
            "consumed pose or gripper seam",
        )
        action = pose10(knots[:40], np.full(40, OPEN))
        full_eef_gate(action, sample["tcp"])
        before = pose10(
            np.asarray([receipt["start_pose"], receipt["end_pose"]]), np.full(2, OPEN)
        )
        full_eef_gate(np.vstack((before, action[1:39])), sample["tcp"])

    def preview(self, manager, receipt, measured, goal, entry_anchor, entry, tick):
        dt = (tick - entry) * 0.002
        require(
            tick >= entry
            and (tick - entry) % 50 == 0
            and tick == manager.next_tick
            and manager.last_wire == receipt["consumed"]
            and np.allclose(manager.anchor, receipt["end_pose"], atol=2e-7, rtol=0),
            "misjoined dynamic clock or wire",
        )
        times = dt + np.arange(41) * 0.05
        nominal = trajectory(goal, times, *self.rate)
        if self.mode == "A1":
            knots = manager.propose(
                "A1", tick, receipt["consumed"], receipt, measured, nominal[0]
            )
        else:
            # The final receiver knot is its held padding after the real H40.
            knots = trajectory(entry_anchor, times, *self.rate)
            knots[-1] = knots[-2]
        return knots, nominal

    def follow(
        self,
        adapter,
        folder,
        parts,
        sample,
        seq,
        receipt,
        manager,
        goal,
        entry,
        atoms,
        installs,
        stream,
        dwell,
    ):
        import mujoco
        from b2_native import paused
        from d1_geometry import sweep
        from feedback import START
        from go2arx5_vla.world_model.mobile_safety_profile_v1 import binding
        from native_known import atom_receipt, install, packet, read_line, tcp
        from native_servo import measure, verify_finger_file
        from scipy.spatial.transform import Rotation, Slerp
        from scripts.wm02 import mobile_boundary_runner_v1 as mobile

        require(
            entry == dwell + 500 and entry <= START + 12000, "unverified entry dwell"
        )
        out = adapter.output
        model = adapter.actual_state.model
        data = mujoco.MjData(model)
        names = [model.geom(i).name for i in range(model.ngeom)]
        tool = model.site("x5_ee").id
        guard = binding(adapter.spec["native_root"])
        anchor = manager.anchor.copy()
        # Preserve independent channels throughout, including the common entry boundary.
        actuals = [sample["tcp"].copy()]
        issued = [anchor.copy()]
        wire = receipt["consumed"]
        from identity import record

        record(
            folder / "DYNAMIC_ENTRY.json",
            {
                "tick": entry,
                "entry_dwell_start": dwell,
                "entry_wire": wire,
                "nominal": goal.tolist(),
                "actual": sample["tcp"].tolist(),
                "consumed": anchor.tolist(),
                "mode": self.mode,
            },
        )
        with (folder / "DYNAMIC_WIRE.jsonl").open("x", buffering=1) as log:
            for update in range(70):
                tick = entry + 50 * update
                paused(adapter, tick)
                require(
                    not any(
                        (p / "STOP.json").exists()
                        for p in (folder.parent, Path(adapter.spec["campaign_root"]))
                    ),
                    "operator STOP",
                )
                knots, nominal = self.preview(
                    manager,
                    receipt,
                    measure(out, tick, sample),
                    goal,
                    anchor,
                    entry,
                    tick,
                )
                self.admit(knots, receipt, sample)
                # Check each distinct exposed preview pose with the same native-scene geometry guard.
                for pose in np.unique(knots.reshape(41, 16), axis=0).reshape(-1, 4, 4):
                    sweep(
                        model,
                        sample,
                        dict(parts, old_ee=manager.anchor),
                        pose,
                        parts["old_base"],
                        guard,
                    )
                action, wire = install(
                    adapter, knots, parts["old_base"], seq + 1 + installs, tick
                )
                installs += 1
                log.write(
                    json.dumps(
                        {
                            "source_tick": tick,
                            "previous_wire": receipt["consumed"],
                            "wire": wire,
                            "nominal": nominal[:3].tolist(),
                            "installed_0_50_100ms": knots[:3].tolist(),
                            "selected_MRT_policy_id": None,
                        }
                    )
                    + "\n"
                )
                for j in (0, 1):
                    sample, _ = mobile.consume(adapter, action, sequence=128 + atoms)
                    atoms += 1
                    last = read_line(out / "MOBILE_SAMPLES.private.jsonl")
                    receipt = atom_receipt(out, last, sample, tick, wire)
                    end = tick + (j + 1) * 25
                    require(
                        receipt["end"] == end
                        and receipt["start"] == end - 25
                        and np.allclose(
                            receipt["start_pose"], knots[j], rtol=0, atol=2e-7
                        )
                        and np.allclose(
                            receipt["end_pose"], knots[j + 1], rtol=0, atol=2e-7
                        ),
                        "wrong dynamic atom",
                    )
                    verify_finger_file(out, end, 127 + atoms)
                    start_pose, end_pose = receipt["start_pose"], receipt["end_pose"]
                    quats = Slerp(
                        [0, 1],
                        Rotation.from_matrix(
                            np.stack((start_pose[:3, :3], end_pose[:3, :3]))
                        ),
                    )(np.arange(1, 26) / 25).as_matrix()
                    for k in range(1, 26):
                        p = packet(stream)
                        require(
                            p["step"] == end - 25 + k
                            and abs(p["time_s"] - p["step"] * 0.002) < 1e-8,
                            "dynamic journal gap",
                        )
                        self.support(names, p)
                        actuals.append(tcp(model, data, tool, p))
                        pos = np.eye(4)
                        pos[:3, 3] = start_pose[:3, 3] + k / 25 * (
                            end_pose[:3, 3] - start_pose[:3, 3]
                        )
                        pos[:3, :3] = quats[k - 1]
                        issued.append(pos)
                    require(
                        np.allclose(actuals[-1], sample["tcp"], atol=1e-10, rtol=0),
                        "measured TCP disagrees",
                    )
                manager.consumed(
                    tick + 50, wire, knots, receipt, measure(out, tick + 50, sample)
                )
        goal_curve = trajectory(goal, np.arange(3501) * 0.002, *self.rate)
        np.savez_compressed(
            folder / "DYNAMIC_TRACK.npz",
            actual=actuals,
            issued=issued,
            nominal=goal_curve,
            tick=entry + np.arange(3501),
        )
        return {
            "outcome": "DYNAMIC_COMPLETE",
            "end": entry + 3500,
            "atoms": atoms,
            "installs": installs,
            "entry_tick": entry,
            "dwell_start": dwell,
            "motion_panel_ticks": 3500,
            "dynamic_installs": 70,
        }
