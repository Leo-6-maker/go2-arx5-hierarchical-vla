"""Independent readback of the complete Expert-reset panel and its exact reused outputs."""

import fcntl
import hashlib
import json
from pathlib import Path
import sys
from tempfile import mkdtemp

import numpy as np

from m38_gpu_preflight_v1 import pin, read, sha


def run(work):
    root = work/"V7-POSTRESET-EXPERT-37-01"
    registry = read(pin(root/"REGISTRY.json","5ad324eb7d8c9377ea3139d5bee8eaa8ed04a60fa98878e0630150aec6500d0a"))
    actual = read(pin(root/"OUTPUT/ACTUAL.json","9708cb9e2b7bca384a79e4364d3fc98ccd090cc94901d371e39c562303bd95b0"))
    summary = read(pin(root/"OUTPUT/SUMMARY.json","2a4aba8a7fe1dae70dc7ed303e7cad00692fba36fef4ae7959caad018d7f92af"))
    assert actual["complete"] and len(actual["rows"]) == 74 and actual["actual_new_frozen_forwards"] == 66
    assert summary["actual_sha256"] == sha(root/"OUTPUT/ACTUAL.json") and len(summary["rows"]) == 148
    frames = {r["id"]:r for r in registry["frames"]}
    assert len(frames) == 37
    for frame in frames.values():
        pin(frame["input"]["path"],frame["input"]["sha256"])
    old = {(r["frame"],r["repeat"]):r for r in registry["reused_hybrid"]["rows"]}
    seen,actions,copied = set(),{},0
    for row in actual["rows"]:
        key = row["frame"],row["repeat"]
        assert key not in seen
        seen.add(key)
        p = pin(root/"OUTPUT"/row["artifact"],row["artifact_sha256"])
        with np.load(p,allow_pickle=False) as f:
            physical,normalized=f["physical_H40"],f["normalized"]
        assert physical.shape == (40,10) and np.isfinite(physical).all() and np.isfinite(normalized).all()
        actions[key] = physical.copy()
        if key in old:
            assert row["artifact_sha256"] == old[key]["artifact_sha256"]
            assert row["exactly_reused_from_receipt_sha256"] == registry["reused_hybrid"]["receipt_sha256"]
            pin(Path(registry["reused_hybrid"]["root"])/old[key]["artifact"],row["artifact_sha256"])
            copied += 1
    assert seen == {(frame,repeat)for frame in frames for repeat in (0,1)} and copied == 8
    assert all(np.array_equal(actions[frame,0],actions[frame,1]) for frame in frames)
    source = registry["original_outputs"]
    baseline = {(r["model"],r["frame"],r["repeat"]):r for r in source["rows"]}
    assert len(baseline) == 222
    assert all(row["noise_sha256"] == baseline["R4",row["frame"],row["repeat"]]["noise_sha256"]
               for row in actual["rows"])
    for row in baseline.values():pin(Path(source["root"])/row["artifact"],row["action_sha256"])
    measured = {(r["model"],r["frame"]):r for r in summary["rows"]}
    for frame in frames:
        for name in ("R4","LAMBDA_ZERO","SHARED_L2SP",registry["model"]["id"]):
            if name == registry["model"]["id"]:
                physical=actions[frame,0]
            else:
                row=baseline[name,frame,0]
                with np.load(Path(source["root"])/row["artifact"],allow_pickle=False)as f:physical=f["physical_H40"]
            rates=np.abs(np.diff(np.r_[frames[frame]["previous_command"]["width_m"],.002+.040*physical[:,9]]))/.05
            m=measured[name,frame]
            assert abs(1000*rates.max()-m["full_H40_max_mm_s"]) < 1e-9
            assert abs(1000*rates[0]-m["first_seam_mm_s"]) < 1e-9
            assert bool(rates.max()>.040) == m["raw_illegal"]
    for frame in registry["frames"]:
        label=baseline["R4",frame["id"],0]
        with np.load(Path(source["root"])/label["artifact"],allow_pickle=False)as f:
            assert ("teacher_H40" in f) == frame["has_real_teacher_label"]
    assert sum(r["has_real_teacher_label"] for r in frames.values()) == 30
    assert not (Path("/proc")/(root/"measure.pid").read_text().strip()).exists()
    spec=read(registry["worker"]["path"])
    with Path(spec["neural_owner_lock"]).open("a")as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
    p=Path(mkdtemp(prefix="v7-postreset-expert-full-readback.",dir=work))/"ACTUAL.json"
    value=dict(status="FULL_37_POSTRESET_PANEL_AND_296_COMPARISON_OUTPUTS_INDEPENDENTLY_VERIFIED",
        verifier_sha256=sha(Path(__file__)), verified_input_files=37, matched_actual_noise_outputs=74,
        registry_sha256=sha(root/"REGISTRY.json"),forward_sha256=sha(root/"OUTPUT/ACTUAL.json"),summary_sha256=sha(root/"OUTPUT/SUMMARY.json"),
        unique_frames=37,hybrid_outputs=74,new_frozen_forwards=66,exact_reused_hybrid_outputs=8,
        exact_reused_baseline_outputs=222,total_comparison_outputs=296,exact_repeat=True,
        all_raw_physical_speeds_independently_recomputed=True,real_teacher_label_conditions=30,
        input_files_read_back=True,process_gone=True,owner_released=True,
        raw_illegal_frames=summary["raw_illegal_frames"],new_raw_illegal_vs_R4=summary["new_illegal_vs_R4"],
        no_dynamic_L1_or_WBC_execution=True,native_admission=False,new_model_forwards=0,optimizer_updates=0,physics_integrations=0)
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    print(p,sha(p),flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
