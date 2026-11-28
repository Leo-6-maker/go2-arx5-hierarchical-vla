"""Reconstruct published labels and compare them with original native command receipts."""
import io
import json
from pathlib import Path
import sys
import tarfile
from tempfile import mkdtemp

import numpy as np
from scipy.spatial.transform import Rotation

from m38_gpu_preflight_v1 import pin, read, sha
from scripts.sla.core_hopper_native_scope import accepted
from scripts.sla.hopper_site import pinned
from scripts.pi0291_arx5_data.native_dataset import matrix_from10
from go2arx5_vla.sla_adapter.codec_v1 import Quantiles
from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec
from v64_adapt_pregrasp_attribution import journal

def replay_case(case, ep, package, quant):
    """Compare one immutable exported episode to the actual original interval-start commands."""
    lease=read(case/"control/LEASE.json")
    assert lease["status"]=="CLOSED"
    spec=read(pin(case/"EXECUTION_SPEC.json",lease["spec_sha256"]))
    terminal=read(case/"RESULT.private.json")
    assert terminal["shutdown_clean"] and not terminal["technical_failure"] and not terminal["unsafe"]
    cap=terminal["detail"]["capture"]
    manifest=read(pin(cap["path"],cap["sha256"]))
    query=read(pin(manifest["labels"]["path"],manifest["labels"]["sha256"]))
    with tarfile.open(pin(package/ep["path"],ep["archive_sha256"]),"r") as archive:
        with archive.extractfile("labels.npz") as member,np.load(io.BytesIO(member.read()),allow_pickle=False) as f:
            data={k:f[k] for k in f.files}
    assert len(query)==len(data["native_step"])==ep["windows"]
    samples={r["native_end_step"]:r for r in journal(case/"native-01/MOBILE_SAMPLES.private.jsonl")}
    commands={r["native_step"]:r for r in journal(case/"native-01/COMMANDS.private.jsonl") if r["phase"]=="mobile-accepted-boundary-050"}
    policy=np.asarray(spec["policy_from_world"])
    assert np.array_equal(ep["T_policy_from_mount"],query[0]["episode_fixed_T_policy_from_mount"])
    codec=GripLogitCodec(ep["T_policy_from_mount"])
    residual=[]; p10=[]; endpoint_difference=[]; tracking=[]; clock_errors=[]; phases={}
    for i,q in enumerate(query):
        tick=q["native_step"]
        assert tick==data["native_step"][i] and np.array_equal(q["state_eef10"],data["state_eef10"][i])
        label=data["action"][i]
        # Reproduce the deployed output normalization precision.
        decoded=codec.decode(data["state_eef10"][i],np.pad(quant.unnormalize(quant.normalize(label).astype(np.float32)),((0,0),(0,25))))
        world=np.linalg.inv(policy)@matrix_from10(decoded)
        actual=np.asarray(q["actual_state"]["tcp"])
        latest=q["recent_reference"]
        assert latest["native_step"]==tick and latest["exact_receiver_knot"]
        reference=np.eye(4); reference[:3,3]=latest["accepted_pose7"][:3]
        reference[:3,:3]=Rotation.from_quat(latest["accepted_pose7"][3:]).as_matrix()
        world[:,:3,3]+=reference[:3,3]-actual[:3,3]
        world[:,:3,:3]=reference[:3,:3]@actual[:3,:3].T@world[:,:3,:3]
        tracking.append(1000*float(np.linalg.norm(reference[:3,3]-actual[:3,3])))
        count=int(data["valid"][i].sum())
        assert np.array_equal(data["valid"][i],np.arange(40)<count)
        for j in range(count):
            t=tick+25*(j+1)
            sample,boundary=samples[t],commands[t]
            assert sample["native_start_step"]==t-25 and sample["ticks"]==25 and sample["safety_valid"]
            assert boundary["exact_receiver_knot"] and boundary["accepted_wire_sha256"]==sample["installed_reference_sha256"]
            # Labels name the command at the interval start. The accepted
            # receiver boundary names the end, 50 ms later.
            installed=matrix_from10(np.asarray(sample["installed_EEF10"]))
            distance=1000*float(np.linalg.norm(world[j,:3,3]-installed[:3,3]))
            angle=float(np.degrees(Rotation.from_matrix(world[j,:3,:3]@installed[:3,:3].T).magnitude()))
            endpoint_difference.append(1000*float(np.linalg.norm(world[j,:3,3]-np.asarray(boundary["accepted_pose7"][:3]))))
            width=1000*abs(.002+.040*decoded[j,9]-.044*sample["installed_EEF10"][9])
            residual.append([distance,angle,width])
            if j<10:
                p10.append([distance,angle,width])
            clock_errors.append(abs(boundary["native_time_s"]-.002*t))
        phases[str(q["phase"])]=phases.get(str(q["phase"]),0)+1
    return dict(episode=case.name,windows=len(query),phases=phases,valid_action_points=len(residual),p10_points=len(p10),
                        max_position_mm_SO3_deg_width_mm=np.max(residual,axis=0).tolist(),
                        p10_max_position_mm_SO3_deg_width_mm=np.max(p10,axis=0).tolist(),
                        query_reference_residual_mm={"median":float(np.median(tracking)),"max":float(np.max(tracking))},
                        wrong_end_boundary_max_difference_mm=max(endpoint_difference),
                        label_time="INTERVAL_START",receiver_boundary_time="INTERVAL_END_50MS_LATER",
                        action_boundary_clock_error_s=max(clock_errors),partial_tail_windows=int((~data["valid"].all(axis=1)).sum()),
                        result_sha256=sha(case/"RESULT.private.json"),native_commands_sha256=sha(case/"native-01/COMMANDS.private.jsonl"),
                        native_samples_sha256=sha(case/"native-01/MOBILE_SAMPLES.private.jsonl"),
                        original_queries_sha256=sha(manifest["labels"]["path"]))

def run(w):
    root=Path("/mnt/local/hdd1/UserData/liuyu-nfs/Projects/GR00T-WBC/WM02")
    package=root/"slago2-embodiment-a93-r3-hub-20261127-01"
    index=read(pin(package/"index.json","9d9396b503064b9fc94646db1e38c0c3a3ca1a1a24b254310644bd21fca43f56"))
    spec=read(w/"ADAPT0250-NATIVE-PLACE-01/worker/EXECUTION_SPEC.json")
    _,_,receiver=accepted(spec)
    stats=pinned(receiver["action"])["actions"]
    quant=Quantiles(stats["q01"],stats["q99"])
    cases=[Path("/var/tmp/sla-v6-core-uid2009-owned-20261118/sla-core-hopper-v1-expert-chain-20261121-02/cases/CHV1-TRAIN-F2-HOLD-blue-00"),
           w/"DIVERSE-PLACE-TRAIN-01/cases/EF5_DIVERSE_PLACE_001",
           w/"RECOVERY-TRAIN-LIVE-01/V64_RECOVERY_TRAIN_02"]
    results=[replay_case(case,next(e for e in index["episodes"] if e["episode"]==case.name),package,quant)
             for case in cases]
    out=Path(mkdtemp(prefix="v7-physical-label-replay.",dir=w))/"ACTUAL.json"
    out.write_text(json.dumps(dict(scope="OLD_NEW_RECOVERY_LABELS_TO_ORIGINAL_NATIVE_COMMANDS",episodes=results,
                                   new_native_integrations=0,new_model_forwards=0,new_optimizer_updates=0),indent=2,allow_nan=False)+"\n")
    print(out,sha(out))
    print(json.dumps(results))

if __name__=="__main__":
    run(Path(sys.argv[1]))
