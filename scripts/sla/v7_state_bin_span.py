"""Measure the physical span of state bins under the unchanged R4 statistics."""
from collections import defaultdict
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
from go2arx5_vla.sla_adapter.codec_v1 import Codec, Quantiles

INDEX = "9d9396b503064b9fc94646db1e38c0c3a3ca1a1a24b254310644bd21fca43f56"
CHANNELS = ("x", "y", "z", "rotvec_x", "rotvec_y", "rotvec_z", "measured_gripper")
UNITS = ("mm", "mm", "mm", "degree", "degree", "degree", "mm")
SCALES = np.array([1000., 1000., 1000., 180/np.pi, 180/np.pi, 180/np.pi, 44.])

def run(site, package):
    index = read(pin(package / "index.json", INDEX))
    _, _, receiver = accepted(read(site / "ADAPT0250-NATIVE-PLACE-01/worker/EXECUTION_SPEC.json"))
    ref = pinned(receiver["input"])["target_D1"]["statistics"]
    stats = pinned(ref)["state"]
    quantiles = Quantiles(stats["q01"], stats["q99"])
    values, labels, episodes, phases, original_bins = [], [], [], [], []
    for number, entry in enumerate(index["episodes"]):
        with tarfile.open(pin(package / entry["path"], entry["archive_sha256"]), "r") as archive:
            with archive.extractfile("labels.npz") as member:
                with np.load(io.BytesIO(member.read()), allow_pickle=False) as payload:
                    state = payload["state_eef10"]
                    phase = payload["phase"]
        codec = Codec(entry["T_policy_from_mount"])
        physical = np.array([codec.encode(row, np.tile(row, (40,1)))[0] for row in state])
        group = "old" if number < 52 else "new" if number < 93 else "recovery"
        normalized = quantiles.normalize(physical).astype(np.float32)
        values.extend(physical)
        labels.extend([group] * len(state))
        episodes.extend([(entry["episode"], i) for i in range(len(state))])
        phases.extend(phase.tolist())
        original_bins.extend(np.digitize(normalized, np.linspace(-1,1,257)[:-1]) - 1)
    physical = np.asarray(values)
    bins = np.asarray(original_bins)
    groups, stages = np.asarray(labels), np.asarray(phases)
    assert physical.shape == bins.shape == (8031,7)
    table = {}
    for group in ("old", "new", "recovery"):
        cohorts = {"ALL": groups == group}
        cohorts.update({name:(groups == group) & (stages == name) for name in sorted(set(stages[groups == group]))})
        table[group] = {}
        for phase, selected in cohorts.items():
            data = physical[selected]
            code = bins[selected]
            channels = {}
            for i, (name, unit, scale) in enumerate(zip(CHANNELS,UNITS,SCALES,strict=True)):
                boundaries = {}
                for boundary in (-1,255):
                    members = data[code[:,i] == boundary,i] * scale
                    boundaries[str(boundary)] = dict(count=len(members), minimum=None if not len(members) else float(members.min()),
                                                     median=None if not len(members) else float(np.median(members)),
                                                     maximum=None if not len(members) else float(members.max()))
                channels[name] = dict(unit=unit, occupied_bins=len(set(code[:,i].tolist())), outer_bins=boundaries)
            table[group][phase] = dict(windows=len(data),channels=channels)
    center = (quantiles.q01[3:6] + quantiles.q99[3:6])/2
    deviation = np.degrees((Rotation.from_rotvec(physical[:,3:6]) * Rotation.from_rotvec(center).inv()).magnitude())
    normalized = quantiles.normalize(physical)
    extremes = np.argsort(np.max(np.abs(normalized[:,3:6]),axis=1))[-8:][::-1]
    examples = [dict(episode=episodes[i][0],window=int(episodes[i][1]),phase=str(stages[i]),group=str(groups[i]),
                     rotvec_degree=(physical[i,3:6]*180/np.pi).tolist(),normalized_rotation=normalized[i,3:6].tolist(),
                     SO3_distance_from_stats_center_degree=float(deviation[i]),state_codes=bins[i].tolist()) for i in extremes]
    new_low, new_high = np.quantile(physical, [.01,.99], axis=0)
    hypothetical = Quantiles(new_low,new_high)
    proposal_codes = np.digitize(hypothetical.normalize(physical).astype(np.float32),np.linspace(-1,1,257)[:-1])-1
    counterfactual = dict(
        lower_bounds_physical=(new_low*SCALES).tolist(),upper_bounds_physical=(new_high*SCALES).tolist(),
        by_group={group:dict(windows=int((groups==group).sum()),
            lower_code_counts=(proposal_codes[groups==group]==-1).sum(0).tolist(),
            upper_code_counts=(proposal_codes[groups==group]==255).sum(0).tolist()) for group in ("old","new","recovery")},
        changed_training_or_deployment_statistics=False)
    result = dict(index_sha256=INDEX,state_statistics_sha256=ref["sha256"],
                  lower_bounds_physical=(quantiles.q01*SCALES).tolist(),upper_bounds_physical=(quantiles.q99*SCALES).tolist(),
                  channels=list(CHANNELS),units=list(UNITS),by_group_and_phase=table,rotation_tail_examples=examples,
                  stats_center_rotvec_degree=(center*180/np.pi).tolist(),fresh_global_quantiles_counterfactual=counterfactual,
                  new_optimizer_updates=0,new_model_forwards=0,new_native_integrations=0)
    output = Path(mkdtemp(prefix="v7-state-bin-span.",dir=site))/"ACTUAL.json"
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(output,sha(output),flush=True)
    print(json.dumps(dict(recovery=table["recovery"]["ALL"],rotation_examples=examples[:3])),flush=True)

if __name__=="__main__":
    run(Path(sys.argv[1]),Path(sys.argv[2]))
