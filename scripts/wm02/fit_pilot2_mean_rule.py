"""Fit the strong response baseline from completed randomized TRAIN episodes."""
import argparse
import json
from pathlib import Path


def run(manifest,output):
    import numpy as np
    from go2arx5_vla.world_model.storage import digest,write_json
    from go2arx5_vla.world_model.pilot2_mean_rule import fit
    if manifest.get('schema')!='WM02_PILOT2_MEAN_RULE_INPUT_V1':raise ValueError('DECLARED_TRAIN_RULE_FIT_MANIFEST_REQUIRED')
    if digest(manifest['task'])!=manifest['task_sha256']:raise ValueError('FROZEN_ORIGINAL_TASK_CHANGED')
    if digest(manifest['panel'])!=manifest['panel_sha256']:raise ValueError('FROZEN_PHYSICAL_PARENT_PANEL_CHANGED')
    panel=json.loads(Path(manifest['panel']).read_text())
    expected=[s['parent']for s in panel['scenes']if s['split']=='TRAIN']
    if (panel['schema']!='WM02_PILOT2_B0_FROZEN_ACTION_PANEL_V1'
            or manifest['TRAIN_parents']!=expected or len(expected)!=12):
        raise ValueError('EXACT_TWELVE_PREDECLARED_TRAIN_PHYSICAL_PARENTS_REQUIRED')
    records=[]
    for item in manifest['episodes']:
        root=Path(item['run']);native=root/'native-01'
        paths={k:p for k,p in [('spec',root/'EXECUTION_SPEC.json'),('episode',root/'EPISODE_MANIFEST.private.json'),
            ('result',native/'RESULT.json'),('decision',native/'PILOT1_DECISION.private.json'),('score',native/'PILOT1_SCORE.private.json')]}
        for key,path in paths.items():
            if digest(path)!=item[key+'_sha256']:raise ValueError('SEALED_TRAIN_BASELINE_EVIDENCE_CHANGED:'+key)
        spec,episode,result,decision,score=[json.loads(paths[k].read_text())for k in ['spec','episode','result','decision','score']]
        if (episode['schema']!='WM02_PILOT2_COVERAGE_EPISODE_V1' or episode['purpose']!='ACTION_COVERAGE'
                or episode['split']!='TRAIN' or result['status']!='NATIVE_STAGE_PASSED'
                or not result['shutdown_clean'] or result['physics_stops'] or result['actual_target_integrations']!=4103
                or decision['future_response_observed']is not False or decision['native_step']!=3703
                or decision['selected_action']!=episode['forced_action'] or score['action']!=episode['forced_action']
                or spec['task_sha256']!=manifest['task_sha256'] or result['native_binary_sha256']!=manifest['native_binary_sha256']):
            raise ValueError('REAL_PREREGISTERED_TRAIN_ACTION_RESPONSE_REQUIRED')
        if not spec.get('parallel_batch_spec'):raise ValueError('ORIGINAL_WHOLE_BATCH_ELIGIBILITY_REQUIRED')
        batch=Path(spec['parallel_batch_spec']).parent/'batch-01/RESULT.json'
        if json.loads(batch.read_text())['status']!='TWO_ENVIRONMENT_NATIVE_PILOT_PASSED':
            raise ValueError('FAILED_PEER_BATCH_NOT_RULE_TRAINING_DATA')
        response=np.asarray(score['actual_endpoint_TCP_m'])[:2]-np.asarray(decision['actual_decision_TCP_m'])[:2]
        records.append(dict(run=root.name,parent=episode['parent'],split=episode['split'],action=episode['forced_action'],
            repeat=episode['repeat_index'],response_xy_m=response.tolist(),evidence_sha256={k:digest(v)for k,v in paths.items()}))
    value=fit(records,manifest['TRAIN_parents'],manifest['task_sha256']);value.update(
        native_binary_sha256=manifest['native_binary_sha256'],actual_episode_evidence=records)
    output=Path(output);output.mkdir(exist_ok=False);write_json(output/'MEAN_RESPONSE.json',value)
    result=dict(status='REAL_TRAIN_PARENT_BALANCED_MEAN_RULE_FITTED',model_sha256=digest(output/'MEAN_RESPONSE.json'),
        TRAIN_parents=len(value['physical_parents']),real_episodes=len(records),native_integrations=0,optimizer_updates=0,
        TEST_read=False,DEV_used_for_fit=False,task_or_score_changed=False)
    write_json(output/'RESULT.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    print(json.dumps(run(json.loads(a.manifest.read_text()),a.output)));return 0


if __name__=='__main__':raise SystemExit(main())
