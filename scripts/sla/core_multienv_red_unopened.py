"""Project the measured F2 red approach candidate onto the next registered TRAIN layout."""
import argparse
from pathlib import Path

import mujoco
import numpy as np

from go2arx5_vla.world_model.native_replay import forbid_integration
from scripts.pi0292.durable import read
from scripts.sla.core_multienv_red_descent import CASES, CONFIGS, compare_path
from scripts.sla.core_multienv_place_sweep import ArmShadow, scene_delta
from scripts.sla.core_multienv_train import PRODUCER_SPEC
from scripts.sla.hopper_parent_custody import digest, write_once
from scripts.wm02.multitask_stage_cards_v6 import require_unstarted_cases

SCENE = 'MEV1-TRAIN-F2-NEAR_NEG-01'
SCENE_SHA = 'b2bbb276d0a0b7bbb65c93841e35e5b35168d57ac77f7c89abc545031d52629f'
PROOF_SHA = '92f3aae542bea55faf48e43739998b27a3e92dbda84942d52f45365db040f804'
CONTROLS = ((0,0,0),(20,0,0))


def run(root,saved):
    spec = read(root/'EXECUTION_SPEC.json')
    if (digest(root/'EXECUTION_SPEC.json')!=PRODUCER_SPEC or digest(saved)!=PROOF_SHA
            or mujoco.__version__!='3.2.7' or mujoco.get_mjcb_control() is not None):
        raise PermissionError('Require the exact original producer and completed full-arm comparison')
    previous = read(saved)
    if [list(config) for config in CONFIGS]!=previous['configs']:
        raise ValueError('The saved path comparison changed its fixed candidates')
    candidates = [config for config,count in zip(CONFIGS,previous['passed_anchor_counts'],strict=True)
                  if count==6 and config not in CONTROLS]
    if not candidates:
        raise ValueError('No fixed TRAIN candidate passed its source anchors')
    configs = (*CONTROLS,*candidates)
    pop = spec['compiled_population']
    if digest(Path(pop['path']))!=pop['sha256']:
        raise ValueError('Registered population changed')
    pop = read(Path(pop['path']))
    ids = {f'{SCENE}-{name.split("-00-",1)[1]}' for name in CASES}
    cards = {r['id'] for r in pop['conditions']['TRAIN'] if r['scene_family']==SCENE and r['id'] in ids}
    if cards!=ids or any((root/'cases'/name).exists() for name in cards):
        raise ValueError('The exact three original next TRAIN cards must remain unstarted')
    require_unstarted_cases(spec['registry'],ids)
    scene = next(v['scene'] for v in pop['clusters'] if v['id']==SCENE)
    if scene['sha256']!=SCENE_SHA or digest(Path(scene['path']))!=SCENE_SHA:
        raise ValueError('Unopened geometry bytes differ')
    outcomes = {}
    with forbid_integration(mujoco) as calls:
        shadow = ArmShadow(mujoco.MjModel.from_binary_path(scene['path']))
        for cid in CASES:
            case = root/'cases'/cid
            native = read(case/'EXECUTION_SPEC.json')
            q = case/'native-01/LIVE_DATA/QUERIES.private.json'
            if digest(q)!=previous['cases'][cid]['witness']['query_sha256']:
                raise ValueError('Observed source states changed')
            old = mujoco.MjModel.from_binary_path(native['model_path'])
            shifts = scene_delta(old,shadow.model)
            q = read(q)
            i = next(i for i,x in enumerate(q) if x['phase']=='DESCEND')
            intent = dict(q[i]['raw_intent'],goal=np.asarray(q[i]['raw_intent']['goal'])+shifts[0][1])
            anchors = []
            for row in q[i-1:i+1]:
                qpos = np.asarray(row['actual_state']['qpos']).copy()
                for slots,delta in shifts:
                    qpos[slots] += delta
                anchors.append(dict(row,actual_state=dict(row['actual_state'],qpos=qpos)))
            outcomes[cid] = dict(translated_cube_centres_mm=[(1000*d).tolist() for _,d in shifts],
                configs=[dict(config=config,anchors=[compare_path(shadow,a,intent,config) for a in anchors])
                         for config in configs])
        if calls:
            raise RuntimeError('Unexpected integration in scene projection')
    require_unstarted_cases(spec['registry'],ids)
    counts = [sum(x['passed'] for v in outcomes.values() for x in v['configs'][i]['anchors'])
              for i in range(len(configs))]
    return dict(status='PROJECTED_UNOPENED_TRAIN_GEOMETRY',unopened_scene=scene,
                unopened_conditions=sorted(ids),prior_proof_sha256=PROOF_SHA,anchors=6,
                configs=configs,pass_counts=counts,outcomes=outcomes,native_requests=0,native_integrations=0,
                optimizer_updates=0,model_generations=0,VAL_inputs=0)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('producer',type=Path)
    cli.add_argument('saved_proof',type=Path)
    cli.add_argument('output',type=Path)
    args=cli.parse_args()
    write_once(args.output,run(args.producer,args.saved_proof))
