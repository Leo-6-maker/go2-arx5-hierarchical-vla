"""Freeze the finite quality entry and its reused numerical/control dependencies."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import importlib.util
import sys
from pathlib import Path
from infer_worker import ROOT,BASE,read,sha,write
import offline


def main():
    assert not (ROOT/'evidence/model_attempts.jsonl').exists()
    check=read(ROOT/'evidence/EVALUATION_CPU_CHECK.json')
    assert len(check['checks'])==832 and check['model_queries']==0
    card=read(ROOT/'design/FIXED_OFFLINE_CARD.json')
    assert [len(card[k]) for k in ('fit_rows','factor_rows','legacy_rows')]==[224,304,440]
    assert card['rows_per_point']==968 and card['logical_rows']==3872
    assert all(r['admitted'] for r in card['factor_rows'])
    paths=[ROOT/'code'/name for name in ('infer_worker.py','offline.py','virtual_episode.py','evaluate.py','reliability.py','checkpoint.py')]
    paths.extend(ROOT/'design'/name for name in ('EVALUATION_CARD.json','FIXED_OFFLINE_CARD.json','FIT112_FIRST2048.json','PAIRING_PLAN.json'))
    paths.extend(ROOT/'evidence'/name for name in ('EVALUATION_CPU_CHECK.json','VIRTUAL_CPU_CHECK.json','OLD0_EXACT_CACHE_BINDING.json'))
    paths.extend(ROOT/'contracts'/arm/'source/dataset/statistics.json' for arm in ('OLD','ROT'))
    modules=('pi0260_r3_inputs','pi0260_r2_codec','pi0267_virtual_renderer','pi0267_virtual_data','pi0267_cpu',
        'pi0267_metrics_v3','pi0266_expert','pi0255_motion3','pi0242_train_infer_bridge','stage21_same_scene_groot_inference',
        'stage40_b0_model_control','pi0243_staged_learning_to_sim','pi0228_train_calib_checkpoint_selection','pi0264_freeze_eval',
        'go2arx5_vla.safety','go2arx5_vla.providers','go2arx5_vla.trajectory.so3',
        'gr00t.data.state_action.state_action_processor')
    for name in modules:
        spec=importlib.util.find_spec(name);assert spec and spec.origin,name
        paths.append(Path(spec.origin))
    scorer=next(p for p in paths if p.name=='pi0267_metrics_v3.py')
    assert sha(scorer)==read(ROOT/'design/EVALUATION_CARD.json')['scorer_sha256']
    for p in paths:assert p.is_file()
    result=dict(status='FROZEN_BEFORE_NEW_QUALITY_QUERIES',files=[dict(path=str(p),sha256=sha(p)) for p in sorted(set(paths))],
        model_queries=0,gradient_queries=0,new_physics=0,rows_per_point=968,logical_offline=3872,expected_new_offline=3432,
        logical_virtual=256,exact_OLD0_reuse=64,new_virtual=192,CPU_input_checks=832,
        admission='Both2048 actual integrity audit and four complete Trainer inventories; fixed matching input/noise cards',
        first_valid_failures_retained=True,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    target=ROOT/'design/QUALITY_EXECUTION_FILESET.json'
    if target.exists():assert read(target)==result
    else:write(target,result)
    print(result['status'],len(paths),'CPU_ONLY',flush=True)


if __name__=='__main__':main()
