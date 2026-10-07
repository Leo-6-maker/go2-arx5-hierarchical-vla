"""CPU-only original-data provenance, complete TRAIN statistics and fixed plans."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from codec import ROOT, PRISTINE, OFFICIAL, COSMOS, TAG, sha, modality

CARD = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0276/design/DATASET_CARD.json')
CARD_SHA = '4ac409b6742401266862707421c7c132ecd05f54d2cab54d08f098c672c28e6b'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text() == payload, 'Immutable artifact differs: ' + str(path)
    else:
        path.write_text(payload)


def order_key(value):
    return hashlib.sha256(('PI0282-28203-' + str(value)).encode()).hexdigest()


def plans(rows, root):
    import numpy as np
    groups = defaultdict(list)
    for row in rows:
        if row['split'] == 'TRAIN':
            groups[(row['seed'], row['group'])].append(row)
    seeds = sorted({seed for seed, phase in groups})
    assert len(seeds) == 93 and len(groups) == 93 * 4
    streams = {}
    coverage = []
    for (seed, phase), group in sorted(groups.items()):
        group.sort(key=lambda x: (x['anchor'], x['uid']))
        strata = [list(x) for x in np.array_split(np.arange(len(group)), min(8, len(group))) if len(x)]
        orders = [sorted(x, key=lambda i: order_key(group[i]['uid'])) for x in strata]
        stream = [dict(group[0], stratum=0), dict(group[-1], stratum=len(strata)-1)]
        for visit in range(43):
            sid = visit % len(orders)
            index = orders[sid][(visit // len(orders)) % len(orders[sid])]
            stream.append(dict(group[index], stratum=sid))
        streams[(seed, phase)] = stream
    plan = []
    visits = Counter()
    for phase_visit in range(4096):
        # Alternate phases on the shared global plan, with balanced episode cycles.
        for phase in range(4):
            seed = seeds[phase_visit % 93]
            row = dict(streams[(seed, phase)][phase_visit // 93])
            row.update(index=len(plan), phase=phase)
            plan.append(row)
            visits[(seed, phase)] += 1
    assert len(plan) == 16384 and min(visits.values()) == 44 and max(visits.values()) == 45
    for key, count in sorted(visits.items()):
        observed = [r for r in plan if (r['seed'], r['phase']) == key]
        full = groups[key]
        coverage.append({'seed': key[0], 'phase': key[1], 'visits': count,
                         'first': full[0]['uid'], 'last': full[-1]['uid'],
                         'strata': sorted({r['stratum'] for r in observed}),
                         'unique_uids': len({r['uid'] for r in observed})})
    diag = []
    for seed in range(10000, 10004):
        gs = [groups[(seed, p)] for p in range(4)]
        chosen = [gs[p][0] for p in range(4)] + [gs[3][len(gs[3]) // 2], gs[3][-1]]
        for r in chosen:
            diag.append(dict(r, phase=r['group'], diagnostic_index=len(diag)))
    val_seeds = sorted({r['seed'] for r in rows if r['split'] == 'VAL'})[:6]
    for seed in val_seeds:
        for phase in range(4):
            eligible = sorted([r for r in rows if r['split'] == 'VAL' and r['seed'] == seed and r['group'] == phase], key=lambda x:(x['anchor'],x['uid']))
            assert eligible
            diag.append(dict(eligible[0], phase=phase, diagnostic_index=len(diag)))
    assert len(diag) == 48
    for name, values in [('PLAN16384.jsonl', plan), ('DIAG48.jsonl', diag)]:
        payload = ''.join(json.dumps(x, sort_keys=True) + '\n' for x in values)
        p = root / name
        if p.exists():
            assert p.read_text() == payload
        else:
            p.write_text(payload)
    write(root / 'PLAN_AUDIT.json', {'phase_counts': dict(Counter(r['phase'] for r in plan)),
                                  'episode_phase_visits': coverage, 'DIAG48': {'TRAIN':24,'VAL':24},
                                  'plan_sha256': sha(root/'PLAN16384.jsonl'), 'diag_sha256': sha(root/'DIAG48.jsonl'),
                                  'phase_is_sampler_metadata_only':True})


def summary(x):
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    return {k: v.tolist() for k, v in {'min':x.min(0), 'max':x.max(0), 'mean':x.mean(0), 'std':x.std(0),
                                      'q01':np.quantile(x,.01,axis=0), 'q99':np.quantile(x,.99,axis=0)}.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(ROOT))
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    resource.setrlimit(resource.RLIMIT_AS, (16*1024**3, 16*1024**3))
    import numpy as np
    import sys
    sys.path.insert(0,str(OFFICIAL))
    from gr00t.data.state_action.state_action_processor import StateActionProcessor
    from gr00t.data.types import ActionType, ActionFormat
    root = Path(args.root);root.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    assert sha(CARD) == CARD_SHA
    card = json.loads(CARD.read_text())
    rows = [r for r in card['rows'] if r['eligible']]
    train = [r for r in rows if r['split']=='TRAIN']
    assert len(train)==55893 and len(rows)==64908
    admitted = [e for e in card['episodes'] if e['admitted']]
    assert Counter(e['split'] for e in admitted)=={'TRAIN':93,'VAL':15}
    receipts = []
    by_episode = defaultdict(list)
    for r in rows:by_episode[r['episode']].append(r)
    for e in admitted:
        path = Path(e['path']); rr=by_episode[e['path']][0]
        for name,key in [('commands.npz','command_sha256'),('observations.npz','observation_sha256')]:
            assert sha(path/name)==rr[key]
        with np.load(path/'observations.npz') as obs:
            for k in ('base_camera_raw','hand_camera_raw'):
                assert obs[k].shape==(640,128,128,3) and obs[k].dtype==np.uint8
            assert obs['state_v1'].shape==(640,10)
        receipts.append(dict(e,command_sha256=rr['command_sha256'],observation_sha256=rr['observation_sha256']))
    plans(rows,root)
    processor_config = json.loads((PRISTINE/'processor_config.json').read_text())
    kwargs = processor_config['processor_kwargs']
    kwargs['modality_configs'][TAG]=modality()
    kwargs.update(use_percentiles=False,clip_outliers=True,shortest_image_edge=256,crop_fraction=1.0,
                  image_crop_size=[256,256],image_target_size=[256,256],random_rotation_angle=0,
                  color_jitter_params={k:0.0 for k in ('brightness','contrast','saturation','hue')},
                  extra_augmentation_config=None,state_dropout_prob=0.0,exclude_state=False,
                  max_state_dim=132,max_action_dim=132,max_action_horizon=40,use_relative_action=True,model_name=str(COSMOS),model_type='qwen')
    converter = StateActionProcessor({TAG:modality()},use_relative_action=True)
    state_parts=[];action_parts=[];relative_parts=[]
    for e in admitted:
        if e['split']!='TRAIN':continue
        ep=Path(e['path'])
        with np.load(ep/'observations.npz') as o:states=o['state_v1']
        with np.load(ep/'commands.npz') as c:commands=c['action']
        ep_rows=by_episode[e['path']]
        state_parts.append(states[[r['anchor'] for r in ep_rows]])
        absolute=[];relative=[]
        for r in ep_rows:
            a=r['anchor']; chunk=commands[a:a+40]
            assert chunk.shape==(40,10)
            absolute.append(chunk)
            relative.append(converter._convert_to_relative_action(chunk[:,:9],states[a,:9],ActionType.EEF,ActionFormat.XYZ_ROT6D))
        action_parts.append(np.concatenate(absolute));relative_parts.append(np.concatenate(relative))
        print('STATS_EPISODE',e['seed'],len(ep_rows),flush=True)
    state=np.concatenate(state_parts);action=np.concatenate(action_parts);relative=np.concatenate(relative_parts)
    new={'state':{'eef_9d':summary(state[:,:9]),'gripper_position':summary(state[:,9:])},
         'action':{'eef_9d':summary(action[:,:9]),'gripper_position':summary(action[:,9:])},
         'relative_action':{'eef_9d':summary(relative)}}
    statistics=json.loads((PRISTINE/'statistics.json').read_text());assert TAG not in statistics
    statistics[TAG]=new
    write(root/'processor_config.json',processor_config);write(root/'statistics.json',statistics)
    write(root/'embodiment_id.json',json.loads((PRISTINE/'embodiment_id.json').read_text()))
    constants={}
    for m,groups in new.items():
        constants[m]={}
        for k,stats in groups.items():
            lo=np.asarray(stats['min']);hi=np.asarray(stats['max'])
            constants[m][k]={'exact_constant':np.where(lo==hi)[0].tolist(),
                             'native_isclose_constant':np.where(np.isclose(lo,hi))[0].tolist(),
                             'range':(hi-lo).tolist()}
    write(root/'DATA_CONTRACT.json',{'schema':'PI0282_NATIVE_EEF10_DATA_V1','original_card_sha256':CARD_SHA,
          'admitted_episodes':receipts,'all_original_attempts':card['episodes'],'original_exclusions':card['coverage_failures'],
          'eligible_windows':dict(Counter(r['split'] for r in rows)),'stats_population_state_rows':len(state),
          'stats_population_action_steps':len(action),'stats_fit_split':'TRAIN_ONLY_FULL_POPULATION',
          'constant_dimensions':constants,'coordinate':'absolute source world TCP XYZ + first two rotation matrix rows + open fraction',
          'command_and_actual_state_separate':True,'native_relative_action':True,'new_science_calls':0,
          'processor_class':'Gr00tN1d7Processor','official_code_commit':'51d4c89f72fda44cbf77285c6a8114b52676b8a1',
          'files':{n:sha(root/n) for n in ('processor_config.json','statistics.json','embodiment_id.json','PLAN16384.jsonl','DIAG48.jsonl')},
          'original_processor_sha256':sha(PRISTINE/'processor_config.json'),'original_statistics_sha256':sha(PRISTINE/'statistics.json')})
    write(root/'DATA_BUILD_USAGE.json',{'elapsed_seconds':time.monotonic()-started,'rusage_self':list(resource.getrusage(resource.RUSAGE_SELF)),
                                    'model_forward':0,'physics':0,'training':0,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES']})
    print('DATA_STATISTICS_PLAN_FROZEN',flush=True)


if __name__=='__main__':main()
