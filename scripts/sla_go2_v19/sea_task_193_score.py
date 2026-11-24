"""Score the source task and holding-language panels without running a policy."""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

from sea_task_193 import NAMES, PLAN_SHA, case_id, load_roster
from sea1b_panel import recorded
from sea1b_score import score_cell
from sea1_archive_readout import TIMES, alignment, pose_error, pose_rows
from source_custody import digest, pin, read, require, write_new
from demonstrated_grip import events, rows
from policy_query import tensor_sha


def longest_run(values):
    changes = np.flatnonzero(np.diff(np.r_[False, values, False]))
    return max(changes[1::2]-changes[::2], default=0)


def grip_intent(pred, measured, release_row):
    width = np.r_[measured[15],pred[:,15]]
    transition = events(np.arange(51)/30.,width,step=True)
    first = next((int(round(30*r["s"]))-1 for r in transition if r["kind"] == "close"),None)
    is_closed = width[0] <= .55
    closed = []
    for frac in width[1:]:
        if frac <= .55:
            is_closed = True
        elif frac >= .65:
            is_closed = False
        closed.append(is_closed)
    elevated = np.zeros(50,dtype=bool)
    z_at_close = None
    if first is not None:
        z_at_close = float(pred[first,10])
        elevated[first:] = pred[first:,10]-z_at_close >= .02
    above = np.array(closed) & elevated
    hold_s = max(0,int(longest_run(above))-1)/30.
    beyond = closed[release_row:] if 0 <= release_row < 50 else []
    return dict(
        first_close_zero_based_row=first, close_z_source_m=z_at_close,
        max_link6_z_above_close_mm=None if first is None else float(1000*np.max(pred[first:,10]-z_at_close)),
        contiguous_elevated_closed_forecast_s=hold_s, one_second_forecast=hold_s >= 1.,
        final_closed=closed[-1], final_elevated_closed=bool(above[-1]),
        recorded_release_inside_horizon=bool(beyond), remains_closed_after_recorded_release=all(beyond) if beyond else None,
        action_threshold_events=transition,
    )


def scored(root, batch, cell, entry, path, fingerprints, obs, prior):
    slot = case_id(cell)
    a,c = cell["anchor"],cell["condition"]
    label = fingerprints[a]
    frame = label["frame"]
    with np.load(path,allow_pickle=False) as z:
        pred = z["absolute"].copy()
        require(pred.shape == (50,16) and z["actions"].shape == (50,32) and np.isfinite(pred).all(),
                "invalid sealed output")
        hashes = dict(decoded_sha256=tensor_sha(pred),raw_sha256=tensor_sha(z["actions"]),
                      raw_dtype=str(z["actions_dtype"]), sampler_state_sha256=tensor_sha(z["sampler_state"]))
    with h5py.File(pin(root / "data" / label["episode"],label["source_sha256"])) as history:
        targets = recorded(history,"state",slice(frame+1,frame+51))
        require(np.array_equal(targets,recorded(history,"action",slice(frame,frame+50))),
                "source actions no longer exact successor states")
    measured = obs[a,c]["state"]
    close,release = (prior[a][f"fruit_{k}_frame"]-frame-1 for k in ("close","release"))
    active = min(release,50)
    phases = {}
    for name, i, j in (("approach",0,close),("lift_before_release",close,active)):
        x,y = (pose_rows(k[i:j,8:]) for k in (pred,targets))
        start = pose_rows(measured[None,8:])[0,:3,3] if i == 0 else pose_rows(targets[i-1:i,8:])[0,:3,3]
        phases[name] = dict(rows=[i,j],score=pose_error(x,y),
                            toward_recorded_end_cos=alignment(x[-1,:3,3]-start,y[-1,:3,3]-start))
    if batch == "TL":
        physical = dict(
            shared_before_release=dict(rows=active,score=pose_error(pose_rows(pred[:active,8:]),pose_rows(targets[:active,8:])),
                                       hold=pose_error(np.repeat(pose_rows(measured[None,8:]),active,axis=0),pose_rows(targets[:active,8:]))),
        )
    else:
        physical = score_cell(pred,measured,targets)
    return dict(
        **cell, slot=slot, status="RETURNED", actual_noise_sha256=entry["noise_sha256"],
        frozen_file_sha256=digest(path), **hashes, phases=phases, source_close_row=close,source_release_row=release,
        **physical, holding_intent=grip_intent(pred,measured,release),
    )


def run(root,batch,output):
    reg, obs, cells = load_roster(root)
    dest = root / "v19" / NAMES[batch]
    old = root / "v19/SEA1B-SOURCE-FACTORIAL-01"
    start,end = (read(dest / x) for x in ("START.json","OWNER_EXIT.json"))
    require(end["project_lock_released"] and end["physics_ticks"] == end["optimizer_updates"] == 0
            and start["plan_sha"] == PLAN_SHA
            and not (Path("/proc") / str(start["owner_pid"])).exists(),"owner did not exit before scoring")
    reuse_file = dest / "REUSE.json"
    reuse = read(reuse_file) if reuse_file.exists() else {"exact":{}}
    guesses = {a["anchor"]:a for a in reg["anchors"]}
    events_by_ep = {a["episode"]:a for a in reg["observations"]}
    events_by_anchor = {a["anchor"]:events_by_ep[a["episode"]] for a in reg["anchors"]}
    out = []
    for cell in cells[batch]:
        slot = case_id(cell)
        n = cell["seed"][1]
        entry = dest / f"{slot}.entered.json"
        returned = dest / f"{slot}.returned.json"
        if slot in reuse["exact"]:
            archived = reuse["exact"][slot]
            path = pin(old / (archived["archived_slot"] + ".npz"),archived["output_sha256"])
            e = {"noise_sha256":archived["noise_sha256"]}
            kind = "EXACT_REUSE"
            require(not entry.exists(),"reused output also entered a new sampler")
        elif returned.exists():
            e = read(entry)
            require(all(e[k] == cell[k] for k in cell),"returned query identity changed")
            path = pin(dest / f"{slot}.npz",read(returned)["file_sha256"])
            kind = "NEW"
        else:
            out.append(dict(**cell,slot=slot,status="TECHNICAL_ENTERED" if entry.exists() else "UNSTARTED"))
            continue
        with np.load(dest / f"noise-{n}.npz",allow_pickle=False) as z:
            require(tensor_sha(z["noise"]) == e["noise_sha256"],"paired noise storage mismatch")
        result = scored(root,batch,cell,e,path,guesses,obs,events_by_anchor)
        out.append(dict(result,origin=kind))
    require(sum(x.get("origin") == "NEW" for x in out) == end["returned"]
            and sum(x["status"] == "TECHNICAL_ENTERED" for x in out) == end["entered"]-end["returned"],
            "scientific counts do not match the owner")
    pairs = []
    if batch == "T0T1":
        r = {(a["anchor"],a["seed"][1],a["condition"]):a for a in out if a["status"] == "RETURNED"}
        for a in guesses:
            for n in (0,1):
                x,y = (r.get((a,n,c)) for c in ("T0","T1"))
                if x and y:
                    pairs.append(dict(anchor=a,seed=[0,n],xyz_rms_delta_mm={
                        h:y[h]["model"]["xyz_rms_mm"]-x[h]["model"]["xyz_rms_mm"] for h in ("P10","FULL50")
                    }))
    calls = dest / "calls.jsonl"
    stats = list(rows(calls)) if calls.exists() else []
    write_new(output,dict(
        batch=batch,run=NAMES[batch],plan_sha256=PLAN_SHA,exact_reused=len(reuse["exact"]),reuse_proof=reuse,
        owner_receipt=end, allocation_sha256=start["allocation_sha"],cold_sha256=start["cold_sha256"],
        model_load_and_return_events=stats, device_inputs_sha256=digest(dest / "DEVICE_INPUTS.json") if (dest / "DEVICE_INPUTS.json").exists() else None,
        unstarted=sum(r["status"] == "UNSTARTED" for r in out), results=out,pairs=pairs,physics_ticks=0,optimizer_updates=0,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root",type=Path)
    cli.add_argument("batch",choices=NAMES)
    cli.add_argument("output",type=Path)
    a = cli.parse_args()
    os.umask(0o077)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU score only")
    run(a.root.resolve(),a.batch,a.output.resolve())
