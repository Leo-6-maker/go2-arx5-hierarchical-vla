"""Attribute archived blue D1 TCP motion without integrating or changing its outcome."""

import json
from pathlib import Path
import sys
from tempfile import mkdtemp

import numpy as np

from m38_gpu_preflight_v1 import pin, read, sha
from v64_adapt_pregrasp_attribution import journal
from v64_recovery_release_prefix import forbid_integration
from go2arx5_vla.world_model.multitask_state_v6 import packets


def run(work):
    import mujoco

    case = work / "V7-D1-EARLY-NATIVE-02/V7_D1_EARLY_TRAIN_17"
    result = read(pin(case / "RESULT.private.json", "d88e246379089963f253a9d5faf31ebfb4c62c69f03eda82ffc299c2f242926c"))
    assert result["unsafe"] and result["shutdown_clean"] and not result["technical_failure"]
    assert result["actual_native_integrations"] == 28584
    assert result["trace_audit"]["integrations_after_first_STOP"] == 0
    for owner in (case, case.parent, case.parent / "worker"):
        assert read(owner / "control/LEASE.json")["status"] == "CLOSED"
    spec = read(pin(case / "EXECUTION_SPEC.json", read(case / "control/LEASE.json")["spec_sha256"]))
    raw = case / "native-01"
    cap = read(pin(raw / "LIVE_DATA/MANIFEST.private.json", "09c8f3f2d9a53e63d2bac4226ca09b087e1c19838b7bdfb88da10dbc5f5254a4"))
    queries = read(pin(cap["labels"]["path"], cap["labels"]["sha256"]))
    assert len(queries) == 42 and [q["phase"] for q in queries[-3:]] == ["DESCEND", "CLOSE", "CLOSE"]
    starts = [q["native_step"] for q in queries]
    samples = {r["native_end_step"]: r for r in journal(raw / "MOBILE_SAMPLES.private.jsonl")}
    full = [r for r in samples.values() if r["native_start_step"] >= starts[0]]
    assert len(full) == 415 and all(r["ticks"] == 25 and r["safety_valid"] for r in full)
    sequences = [r["gripper_command"]["sequence"] for r in full]
    assert sequences == list(range(628, 1043))
    boundaries = {r["native_step"]: r for r in journal(raw / "COMMANDS.private.jsonl")
                  if r["phase"] == "mobile-accepted-boundary-050"}
    chosen = [queries[-3]["native_step"], queries[-2]["native_step"], queries[-1]["native_step"], 28578, 28584]
    states = {r["step"]: r for r in packets(raw / "state.bin.journal") if r["step"] in chosen}
    assert set(states) == set(chosen)
    geometry = pin(spec["model_path"], spec["frozen_files"][spec["model_path"]])
    model = mujoco.MjModel.from_binary_path(str(geometry))
    assert model.jnt_qposadr[0] == 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    tcp = model.site("x5_ee").id
    mount = model.body("x5_base_link").id
    arm = [int(model.jnt_qposadr[model.joint(f"x5_joint{i}").id]) for i in range(1, 7)]
    scratch = mujoco.MjData(model)

    def fk(q):
        scratch.qpos[:] = q
        mujoco.mj_fwdPosition(model, scratch)
        return scratch.site_xpos[tcp].copy(), scratch.xpos[mount].copy()

    snapshots = []
    with forbid_integration(mujoco) as attempted:
        for tick in chosen:
            p = states[tick]
            xyz, base = fk(p["qpos"])
            if tick in starts:
                q = queries[starts.index(tick)]
                assert np.max(np.abs(xyz - np.asarray(q["actual_state"]["tcp"])[:3, 3])) < 1e-10
                accepted = q["recent_reference"]
            elif tick in samples:
                assert np.max(np.abs(xyz - np.asarray(samples[tick]["actual_TCP"])[:3, 3])) < 1e-10
                accepted = boundaries[tick]
            else:
                accepted = None
            if accepted is not None:
                assert accepted["native_step"] == tick and accepted["exact_receiver_knot"]
            snapshots.append(dict(tick=tick, tcp_world_mm=(1000*xyz).tolist(),
                                  mount_world_mm=(1000*base).tolist(),
                                  accepted_reference_world_mm=(1000*np.asarray(accepted["accepted_pose7"][:3])).tolist()
                                  if accepted else None))
        decompositions = []
        for first in chosen[:3]:
            for last in (28578, 28584):
                q0, q1 = (np.asarray(states[t]["qpos"]).copy() for t in (first, last))
                qb, qa = q0.copy(), q0.copy()
                qb[:7] = q1[:7]
                qa[arm] = q1[arm]
                qba = qb.copy()
                qba[arm] = q1[arm]
                z0, z1, zb, za, zba = [fk(q)[0][2] for q in (q0, q1, qb, qa, qba)]
                assert abs(z1-zba) < 1e-10, "Other joints must not affect this TCP pose"
                base = .5*((zb-z0)+(zba-za))
                arm_delta = .5*((za-z0)+(zba-zb))
                assert abs(base+arm_delta-(z1-z0)) < 1e-12
                decompositions.append(dict(first_tick=first, last_tick=last, actual_tcp_z_change_mm=1000*(z1-z0),
                    floating_base_shapley_z_mm=1000*base, six_arm_joints_shapley_z_mm=1000*arm_delta,
                    other_joint_fk_residual_mm=1000*(z1-zba)))
    assert not attempted
    contact = []
    for row in states[28584]["contacts"]:
        names = [model.geom(row[k]).name for k in ("geom0", "geom1")]
        if set(names) == {"task_table", "task_right_pad"}:
            contact.append(dict(geoms=names, distance_m=row["distance_m"], normal_force_N=row["wrench"][0]))
    assert contact
    out = Path(mkdtemp(prefix="v7-d1-blue-contact-audit.", dir=work)) / "ACTUAL.json"
    value = dict(status="ARCHIVED_BLUE_D1_CONTACT_FAILURE_ATTRIBUTED_WITH_STATIC_FK", case=case.name,
        result_sha256=sha(case / "RESULT.private.json"), capture_sha256=sha(raw / "LIVE_DATA/MANIFEST.private.json"),
        original_queries_sha256=sha(cap["labels"]["path"]), original_model_sha256=sha(geometry),
        state_journal_sha256=sha(raw / "state.bin.journal"), native_samples_sha256=sha(raw / "MOBILE_SAMPLES.private.jsonl"),
        native_commands_sha256=sha(raw / "COMMANDS.private.jsonl"),
        teacher_queries=len(queries), full_teacher_atoms=len(full), first_teacher_sequence=sequences[0],
        last_full_teacher_sequence=sequences[-1], partial_inflight_ticks=28584-28578,
        stop_tick=28584, stop_phase=queries[-1]["phase"], stop_contacts=contact,
        phase_queries={p:sum(q["phase"] == p for q in queries) for p in sorted({q["phase"] for q in queries})},
        snapshots=snapshots, static_TCP_z_decompositions=decompositions,
        attribution_scope="Static kinematic decomposition of archived states; not a causal controller intervention or pad-clearance decomposition",
        controller_selected_reference_directly_observed=False, admitted_recovery=False,
        post_STOP_integrations=0, new_native_integrations=0, new_model_forwards=0, new_optimizer_updates=0)
    out.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    print(out, sha(out), flush=True)
    print(json.dumps(value, allow_nan=False), flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
