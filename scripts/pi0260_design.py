"""Prospective PI0260 population and cue design; no renderer, IK or model calls."""
from pathlib import Path
import copy
import itertools
import json
import shutil
import numpy as np
from pi0255_motion3 import ordering, scene, sha, write
from pi0258_kinematic import instructions

ROOT = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260')
OLD = ROOT.parent/'PI0259'
AUTHORITY = '90cbb926a842f08c3f9a78756d7a24e0d7a63f78'
KEYS = ('x_shift', 'y_shift', 'z_shift', 'radius')


def key(family, registry):
    # Compare world geometry, not namespace, color alias or floating-point spelling.
    center = family.get('center_world')
    if center is None:
        center = np.asarray(registry['geometry']['nominal_center_world']) + [family['factors'][k] for k in KEYS[:3]]
    return tuple(round(float(x), 10) for x in [*center, family['factors']['radius']])


def freeze():
    assert not (ROOT/'design/registry.json').exists()
    reg = copy.deepcopy(json.loads((OLD/'design/registry.json').read_text()))
    sources = [ROOT.parent/'PI0258/design/registry.json', OLD/'design/registry.json',
               Path('/home/liuyu-nfs/Projects/go2-arx5/PI0255/design/task_registry_v1.json')]
    exclusions = []
    for path in sources:
        prior = json.loads(path.read_text())
        for f in prior['families'] + prior.get('calibration_families', []):
            exposed = f['split'] == 'CALIBRATION'
            # PI0258 executed exactly its first six development families.
            if path == sources[0] and f in prior['families'][:6]: exposed = True
            if path == sources[2]:
                attempted={json.loads(line)['episode']['family_id'] for line in (path.parents[1]/'attempt_results.jsonl').read_text().splitlines() if line.strip()}
                exposed=f['family_id'] in attempted
            reserved = f['split'].startswith('TEST')
            if exposed or reserved:
                exclusions.append(dict(source=str(path), source_sha256=sha(path), family_id=f['family_id'],
                    geometry_key=key(f, prior), reason='PRIOR_EXPOSURE' if exposed else 'OLD_TEST_RESERVED'))
    excluded = {tuple(x['geometry_key']) for x in exclusions}
    factors = [dict(zip(KEYS, x)) for x in itertools.product(
        [-.04,-.02,0,.02,.04], [-.04,-.02,0,.02,.04], [0,.015,.03], [.035,.045,.055])]
    remaining, overlaps = [], []
    for f in factors:
        k = key(dict(factors=dict(f,colors=['red','blue'])), reg)
        (overlaps if k in excluded else remaining).append(f)
    remaining.sort(key=lambda f: ordering(f, 'PI0260_GEOMETRY_PROPOSALS:26001'))
    write(ROOT/'design/exclusions.json', dict(sources=exclusions, excluded_grid=overlaps,
        parent_count=225, remaining_count=len(remaining), numeric_key='rounded 1e-10 world center XYZ and radius'))
    assert len(remaining)>=160, ('ELIGIBLE_SCENE_SHORTFALL_REGISTRY', len(remaining))
    families=[]; sequences=[]
    for i,f in enumerate(remaining):
        split='DEVELOPMENT' if i<20 else 'TRAIN' if i<120 else 'KIN-VAL' if i<160 else 'UNUSED_RESERVED'
        colors=[['red','blue'],['blue','green'],['green','red']][i%3]
        family=dict(family_id='pi0260-family-%03d'%i, split=split, factors=dict(f,colors=colors),
            proposal_index=i, geometry_key=f, state_origin='KINEMATIC_FK_NOT_DYNAMIC_MEASUREMENT',
            center_world=(np.asarray(reg['geometry']['nominal_center_world'])+[f[k] for k in KEYS[:3]]).tolist())
        assert key(family,reg)==key(dict(factors=family['factors']),reg)
        families.append(family)
        for task in ('M1','M3'):
            for command in (0,1):
                texts=instructions(task,command,colors); template=(i+command)%2+(2 if split=='KIN-VAL' else 0)
                sequences.append(dict(sequence_id=family['family_id']+'-'+task+'-c'+str(command),
                    family_id=family['family_id'],task=task,command=command,split=split,
                    instruction=texts[template],paraphrases=texts,execute=i<160))
    oldsequences=[s for s in reg['sequences'] if s['split']=='CALIBRATION' and s['task'] in ('M1','M3')]
    reg.update(schema='PI0260_KIN_MOTION2_V1', authority=AUTHORITY, source_registry_sha256=sha(OLD/'design/registry.json'),
        families=families,sequences=sequences,calibration_sequences=oldsequences,
        family_order='PI0260_GEOMETRY_PROPOSALS:26001',ordering_serialization='namespace + colon + json.dumps(factors,sort_keys=True)',
        language_order='fixed proposal index modulo 3 colors; (index+command)%2 template; VAL templates2/3',
        camera_identity='PI0260_CUE_V1',reference_identity='PI0259_KIN_REF_V2',
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    reg['geometry'].update(axes_length_m=.060,axes_marker_radius_m=.010,start_marker_radius_m=.018,contour_marker_radius_m=.006)
    # Freeze the envelope from all declared non-TEST proposals, never successful outcomes.
    envelope=[np.asarray(reg['initial_TCP_world'])[:3,3]];g=reg['geometry']
    from scipy.spatial.transform import Rotation
    for f in reg['calibration_families']+families:
        d=scene(reg,f)
        for i,p in enumerate(d['targets']):
            envelope.extend([p-g['target_radius_m'],p+g['target_radius_m']])
            r=np.asarray(reg['initial_TCP_world'])[:3,:3]@Rotation.from_euler('z',g['target_yaw_rad'][i]).as_matrix()
            for a in range(3):
                end=p+r[:,a]*g['axes_length_m'];envelope.extend([end-g['axes_marker_radius_m'],end+g['axes_marker_radius_m']])
        envelope.extend([d['obstacle']-g['obstacle_half_extents_m'],d['obstacle']+g['obstacle_half_extents_m'],
            d['goal']-g['target_radius_m'],d['goal']+g['target_radius_m'],
            d['center']+[-d['radius']-.018,-d['radius']-.018,-.018],d['center']+[d['radius']+.018,d['radius']+.018,.018]])
    low=np.min(envelope,axis=0);high=np.max(envelope,axis=0)
    reg['renderer'].update(rigs_base_positions=[[.65,0,.75],[.65,.45,.85],[.65,-.45,.85]],
        look_at_world=((low+high)/2).tolist(),envelope_min=low.tolist(),envelope_max=high.tolist(),selected_rig=None,
        envelope_population='all remaining proposals including UNUSED_RESERVED plus OLD calibration; no TEST geometry queries')
    reg['budgets']=dict(calibration_image_pairs=512,proposal_families=160,proposal_sequences=640,
        technical_sequence_reserve=24,active_windows=60000,production_image_pairs=120000,
        main_updates=12000,canary_updates=32,technical_update_reserve=64,planned_quality_queries=1760,
        invalid_quality_reserve=64,physics_steps=0,WBC_actor_calls=0)
    write(ROOT/'design/registry.json',reg)
    reused=[]
    for s in oldsequences:
        directory=ROOT/'calibration'/s['sequence_id'];directory.mkdir(parents=True)
        for name in ('geometry.npz','result.json','solver_ledger.json','started.json','raw_gate.json'):
            src=OLD/'calibration'/s['sequence_id']/name;dest=directory/name
            shutil.copyfile(src,dest);assert sha(src)==sha(dest)
            reused.append(dict(source=str(src),path=str(dest.relative_to(ROOT)),sha256=sha(dest),bytes=dest.stat().st_size))
    write(ROOT/'design/reused_geometry.json',reused)
    assert len({key(f,reg) for f in families})==len(families)
    assert not {key(f,reg) for f in families}&excluded
    assert [sum(f['split']==s for f in families) for s in ('DEVELOPMENT','TRAIN','KIN-VAL')]==[20,100,40]
    write(ROOT/'evidence/design_cpu_check.json',dict(status='PASS',unique_geometry=len(families),
        exclusions_disjoint=True,partitions=[20,100,40],old_start_difference=reg['old_initial_difference'],model_queries=0))
    print(json.dumps(dict(registry_sha256=sha(ROOT/'design/registry.json'),remaining=len(remaining),excluded=len(overlaps),
        look_at_world=reg['renderer']['look_at_world'],new_IK_calls=0)))


if __name__=='__main__':freeze()
