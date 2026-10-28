"""Read existing arrays only: label/gate alignment and FULL retention attribution."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import dataclasses
import hashlib
import json
import sys
import numpy as np
import pandas as pd

BASE = Path('/home/liuyu-nfs/Projects/go2-arx5')
ROOT = BASE / 'PI0253'
OUT = ROOT / 'p0_review'
CODE = BASE / 'PI0250/runtime/reference-317a'
sys.path[:0] = [str(ROOT / 'runtime'), str(CODE / 'scripts'), str(CODE / 'src'), '/home/liuyu-nfs/Projects/isaac-gr00t']
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix as rot, geodesic_angle as angle
from go2arx5_vla.safety import TrajectorySafetyGateV1
import pi0228_train_calib_checkpoint_selection as selector


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def arr_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def table(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def errors(decoded, expert):
    e = decoded[:, :3].astype(float) - expert[:, :3].astype(float)
    offset = e.mean(0)
    rmse = float(np.sqrt(np.mean(np.sum(e * e, axis=1))))
    shape = float(np.sqrt(np.mean(np.sum((e - offset) ** 2, axis=1))))
    assert np.isclose(rmse * rmse, float(offset @ offset) + shape * shape, atol=1e-12)
    th = angle(rot(decoded[:, 3:9]), rot(expert[:, 3:9]))
    return dict(xyz_rmse_m=rmse, mean_offset_m=float(np.linalg.norm(offset)),
                offset_x_m=float(offset[0]), offset_y_m=float(offset[1]), offset_z_m=float(offset[2]),
                centered_shape_rmse_m=shape, first_xyz_error_m=float(np.linalg.norm(e[0])),
                last_xyz_error_m=float(np.linalg.norm(e[-1])), so3_rmse_rad=float(np.sqrt(np.mean(th * th))))


def main():
    OUT.mkdir(exist_ok=True)
    historic_path = OUT / 'inputs/pi0237_p2_expert_gate.json'
    historic = read(historic_path)
    old = {r['source_episode_id']: r for r in historic['rows'] if r['split'] == 'CALIB'}
    assert len(old) == 1088 and sum(r['decision'] == 'REJECT' for r in old.values()) == 12
    gate_hash = hashlib.sha256(json.dumps(dataclasses.asdict(TrajectorySafetyGateV1().config), sort_keys=True).encode()).hexdigest()
    assert gate_hash == historic['gate_config_sha256']
    assert all(r['provider_selector_sha256'] == sha(CODE / 'scripts/pi0228_train_calib_checkpoint_selection.py') for r in old.values())
    profiles = {s: read(ROOT / f'calib/profile-{s}/profile.json') for s in (2500, 5000, 7500, 10000)}
    records = {s: {r['source_episode_id']: r for r in p['records']} for s, p in profiles.items()}
    assert all(set(p) == set(old) for p in records.values())
    reject_sets = {s: {k for k, r in p.items() if r['safety_gate']['decision'] == 'REJECT'} for s, p in records.items()}
    historic_reject = {k for k, r in old.items() if r['decision'] == 'REJECT'}
    union = set.union(historic_reject, *reject_sets.values())
    bindings = read(ROOT / 'evidence/FULL/FULL_CALIB_BINDINGS.json')
    dataset = Path(bindings['calib_dataset']['path'])
    metadata = {r['source_episode_id']: r for r in (json.loads(x) for x in (dataset / 'meta/episodes.jsonl').read_text().splitlines())}
    # Reuse the already qualified processor construction; never call inference/materialize.
    from pi0253_full_screen_v2 import context
    ctx = context()[-1]; proc, emb = ctx['processor'], ctx['embodiment']
    label_rows, evidence = [], []
    for source in sorted(union):
        h = old[source]; i = h['episode_index']; assert metadata[source]['episode_index'] == i
        parquet = dataset / f'data/chunk-{i//1000:03d}/episode_{i:06d}.parquet'
        d = pd.read_parquet(parquet)
        state = np.asarray(d.iloc[0]['observation.state'], np.float32)
        expert = np.stack(d['action'].map(lambda x: np.asarray(x, np.float32)))
        times = np.asarray(d['timestamp'], float)
        assert arr_sha(state) == h['state_sha256'] and arr_sha(expert) == h['label_sha256'] and arr_sha(times) == h['timestamp_sha256']
        assert np.allclose(np.diff(times), .05, atol=1e-6, rtol=0)
        _, groups = proc.state_action_processor.apply(state={'eef_9d':state[None,:9], 'gripper_position':state[None,9:]},
            action={'eef_9d':expert[:,:9], 'gripper_position':expert[:,9:]}, embodiment_tag=emb.value)
        keys = proc.modality_configs[emb.value]['action'].modality_keys
        norm = np.concatenate([np.asarray(groups[k]) for k in keys], axis=-1).astype(np.float32)
        dg = proc.decode_action(norm[None], emb, {'eef_9d':state[None,None,:9], 'gripper_position':state[None,None,9:]})
        roundtrip = np.concatenate([dg['eef_9d'], dg['gripper_position']], axis=-1)[0]
        codec_so3 = float(angle(rot(roundtrip[:,3:9]),rot(expert[:,3:9])).max())
        expert_jump = float(angle(rot(expert[:1,3:9]), rot(state[None,3:9]))[0])
        assert abs(expert_jump - h['metrics']['initial_rotation_discontinuity_rad']) < 1e-7
        same_rollout = sorted([k for k in metadata if k.split(':obs-')[0] == source.split(':obs-')[0]], key=lambda k:int(k.split(':obs-')[1]))
        phase = min(3, 4 * same_rollout.index(source) // len(same_rollout))
        for s in profiles:
            r = records[s][source]; cell = ROOT / f'calib/profile-{s}/cells/source-{i:06d}'
            receipt = read(cell / 'receipt.json'); assert sha(cell/'tensors.npz') == receipt['artifact_sha256']
            with np.load(cell/'tensors.npz',allow_pickle=False) as z:
                pred = z['decoded'][0].copy(); assert np.array_equal(z['state'],state)
                assert arr_sha(z['decoded']) == r['decoded_sha256']
            jump = float(angle(rot(pred[:1,3:9]),rot(state[None,3:9]))[0])
            assert abs(jump-r['safety_gate']['metrics']['initial_rotation_discontinuity_rad']) < 1e-7
            rejected = source in reject_sets[s]
            classification = ('EXPERT_ALSO_REJECTS' if source in historic_reject else 'MODEL_NEW_VIOLATION') if rejected else 'MODEL_NONREJECT_EXPERT_REJECT' if source in historic_reject else 'NONREJECT'
            label_rows.append(dict(source_episode_id=source, checkpoint=s, episode_index=i, phase=phase,
                source_time_s=float(times[0]), expert_reject=source in historic_reject, model_reject=rejected,
                classification=classification, theta_expert_current_rad=expert_jump, theta_model_current_rad=jump,
                theta_model_expert_rad=float(angle(rot(pred[:1,3:9]),rot(expert[:1,3:9]))[0]),
                threshold_rad=1.5, model_excess_rad=jump-1.5, expert_excess_rad=expert_jump-1.5,
                codec_max_so3_rad=codec_so3, codec_max_xyz_m=float(np.linalg.norm(roundtrip[:,:3]-expert[:,:3],axis=1).max()),
                label_identity='EXACT_HISTORICAL_HASH_MATCH', actual_clip_events='UNOBSERVED'))
            evidence.append(dict(path=str(cell/'tensors.npz'),sha256=receipt['artifact_sha256']))
        evidence.append(dict(path=str(parquet),sha256=sha(parquet)))
    table('expert_model_gate.csv',label_rows)
    exposure = read(OUT/'inputs/actual_exposure.json'); schedule = read(OUT/'inputs/full_schedule.json')
    assert exposure['schedule_sha256'] == sha(OUT/'inputs/full_schedule.json') and len(exposure['rows']) == 10000
    assert all(a['source_episode_id']==b['source_episode_id'] for a,b in zip(exposure['rows'],schedule['rows']))
    sources = read(BASE/'PI0250/inputs/frozen_v1/source64/manifest.json')['files']
    source_meta = {r['source_episode_id']:r for r in sources}
    old16 = set(read(BASE/'PI0252/screen/G2000/screen_report.json')['score_card']['support']['old16_source_ids'])
    rows, ref = [], {}
    for name, step, path in [('G2000',0,BASE/'PI0252/screen/G2000')] + [(f'FULL{s}',s,ROOT/f'screen/FULL{s}') for s in (1000,2500,5000,7500,10000)]:
        report = read(path/'screen_report.json')
        count = Counter(r['source_episode_id'] for r in exposure['rows'] if step and r['step'] <= step)
        for cell in report['cells']:
            p = Path(cell['source_cell'])/'tensors.npz'; assert sha(p) == cell['tensor_sha256']
            with np.load(p,allow_pickle=False) as z:
                pred,expert = z['decoded'][0].copy(),z['expert'].copy(); raw=z['raw_scored'][0].copy()
                key=(cell['source_episode_id'],cell['noise_name'])
                if name=='G2000':ref[key]=(expert.copy(),z['state'].copy())
                else:assert np.array_equal(expert,ref[key][0]) and np.array_equal(z['state'],ref[key][1])
            m=source_meta[key[0]]; metric=errors(pred,expert)
            assert abs(metric['xyz_rmse_m']-cell['metrics']['xyz_rmse_m']) < 1e-10
            rows.append(dict(model=name,checkpoint=step,source_episode_id=key[0],noise=key[1],old16=key[0] in old16,
                task=m['target_family'],phase=m['phase'],full_exposures=count[key[0]],
                **metric,normalized_xyz_abs_max=float(np.abs(raw[:,:3]).max()),
                normalized_xyz_abs_gt1_fraction=float(np.mean(np.abs(raw[:,:3])>1)),
                normalized_xyz_exact_boundary_fraction=float(np.mean(np.abs(raw[:,:3])==1)),
                actual_clip_events='UNOBSERVED_NOT_INFERRED_FROM_NORMALIZED_RANGE'))
            evidence.append(dict(path=str(p),sha256=cell['tensor_sha256']))
    assert len(rows)==704 and len({(r['model'],r['source_episode_id'],r['noise']) for r in rows})==704
    table('retention_decomposition.csv',rows)
    grouped=[]
    for model in dict.fromkeys(r['model'] for r in rows):
        for noise in ('B0','B1'):
            for stratum,predicate in [('all64',lambda r:True),('old16',lambda r:r['old16']),('other48',lambda r:not r['old16'])]+[(f'task_{k}',lambda r,k=k:r['task']==k) for k in 'abcd']+[(f'phase_{k}',lambda r,k=k:r['phase']==k) for k in range(4)]:
                rr=[r for r in rows if r['model']==model and r['noise']==noise and predicate(r)]
                if not rr:continue
                o=dict(model=model,noise=noise,stratum=stratum,n=len(rr))
                for k in ('xyz_rmse_m','mean_offset_m','offset_x_m','offset_y_m','offset_z_m','centered_shape_rmse_m','first_xyz_error_m','last_xyz_error_m','so3_rmse_rad'):
                    o['median_'+k]=float(np.median([r[k] for r in rr]));o['p95_'+k]=float(np.percentile([r[k] for r in rr],95))
                o['zero_full_exposure_windows']=sum(r['full_exposures']==0 for r in rr)
                grouped.append(o)
    table('retention_groups.csv',grouped)
    write('summary.json',dict(schema='PI0253_P0_POSTMORTEM_V1',status='COMPLETE_ZERO_MODEL_QUERY',
        expert_reject_count=len(historic_reject),reject_union_count=len(union),
        per_checkpoint={str(s):dict(model_reject=len(reject_sets[s]),expert_overlap=len(reject_sets[s]&historic_reject),
            model_new=len(reject_sets[s]-historic_reject),expert_rescued=len(historic_reject-reject_sets[s])) for s in profiles},
        four_model_reject_intersection=sorted(set.intersection(*reject_sets.values())),
        union_sources=sorted(union),retention_rows=len(rows),full_window_exposure_histogram=dict(Counter(Counter(r['source_episode_id'] for r in exposure['rows']).values())),
        old64_exposure_histogram=dict(Counter(Counter(r['source_episode_id'] for r in exposure['rows'])[k] for k in source_meta)),
        identity_checks='Historical expert label/state/time/gate/selector match; actual 10000 sources match sealed schedule; saved tensors match original receipts',
        actual_clip_events='UNOBSERVED; normalized boundaries descriptive only',new_model_queries=0,new_optimizer_updates=0,
        new_physical_episodes=0,selected_checkpoint='NONE',P1='NOT_AUTHORIZED_NEW_QUERIES_REQUIRE_FROZEN_PROTOCOL'))
    write('input_inventory.json',dict(files=evidence,script_sha256=sha(__file__),historic_p2_sha256=sha(historic_path),actual_exposure_sha256=sha(OUT/'inputs/actual_exposure.json')))
    print(json.dumps(read(OUT/'summary.json')),flush=True)


if __name__ == '__main__':
    main()
