#!/usr/bin/env python3
"""Independently replay saved M64 tensors on CPU, without constructing a model."""
import argparse
import json
import math
import os
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np
import torch
import pi0247_m64_screen as m64


def same(a,b):
    return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step',type=int,choices=(0,1000,2500,5000),required=True)
    args = parser.parse_args()
    card, fit, ctx = m64.inputs()
    import pi0247_d1_screen as d1
    import pi0242_train_infer_bridge as bridge
    import pi0243_staged_learning_to_sim as stage43
    import pi0228_train_calib_checkpoint_selection as selector
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat,rot6d_rows_to_matrix
    root = m64.M64/f'screen_step{args.step}'
    report_path = root/'screen_report.json'
    report = m64.read(report_path)
    score = report['score_card']
    assert report['selected_checkpoint']=='NONE' and report['checkpoint_step']==args.step
    assert score['run_card_sha256']==m64.CARD_SHA and score['runtime_id']=='S4'
    assert score['input_manifest_sha256']==m64.sha(m64.M64/'screen_inputs_manifest.json')
    assert score['processor_identity']==fit['processor_identity'] and score['loss_contract']==fit['loss_contract']
    for entry in report['execution_cards']:
        assert m64.sha(entry['path'])==entry['sha256']
    sources = {r['source_episode_id'] for r in ctx['rows']}
    noises = ('B0','B1') if args.step==5000 else ('B0',)
    expected = {(source,n) for source in sources for n in noises}
    seen, replay = set(), []
    for row in report['cells']:
        source, noise = row['source_episode_id'],row['noise_name']
        assert (source,noise) in expected and (source,noise) not in seen
        seen.add((source,noise))
        cell = Path(row['source_cell'])
        record = d1.collect_record(cell/'record.json',ctx['p1'])
        assert record['status']=='VALID' and record['tensor_sha256']==row['tensor_sha256']
        assert record['source_episode_id']==source and record['noise_name']==noise
        assert record['model_manifest_sha256']==score['inference_restore_sha256']
        score_path = (m64.ROOT/'lu/U_screen_step2000/score_card.json' if row['status']=='REUSED_VALID'
                      else Path(row['score_card_path']))
        assert record['run_card_sha256']==m64.sha(score_path)
        source_parent = Path(ctx['source_cells'][source][noise]['cell_path'])
        parent = m64.read(source_parent/'record.json')
        assert record['condition_identity']==ctx['source_cells'][source][noise]['condition_identity']
        assert record['condition_sha256']==m64.sha(source_parent/'condition.pt')==parent['condition_sha256']
        assert record['source_tensor_sha256']==parent['source_tensor_sha256']
        with np.load(cell/'tensors.npz',allow_pickle=False) as data:
            raw, decoded_saved, state, expert = [data[k].copy() for k in ('raw_scored','decoded','state','expert')]
            assert raw.shape==(1,40,132) and decoded_saved.shape==(1,40,10) and expert.shape==(40,10)
            assert all(np.isfinite(data[k]).all() for k in data.files)
            assert np.array_equal(raw,data['raw_native']) and not record['m_path']['applicable']
            fixed = ctx['m64_noise'][source][noise]
            native = torch.from_numpy(fixed).to(torch.bfloat16).float().numpy()
            assert np.array_equal(data['noise_source_float32'],fixed)
            assert np.array_equal(data['noise_initial'],native) and np.array_equal(data['trace_x'][0],native)
            assert data['trace_x'].shape==(4,1,40,132) and data['trace_v'].shape==(4,1,40,132)
        decoded = bridge._decode_physical(ctx['processor'],ctx['embodiment'],raw,state)
        error = float(np.max(np.abs(decoded-decoded_saved)))
        assert error<=1e-7
        metrics = stage43._screen_metrics(decoded[0].astype(np.float64),expert.astype(np.float64),None)
        residual = decoded[0,:,:3].astype(np.float64)-expert[:,:3].astype(np.float64)
        squared = np.sum(residual*residual,axis=1)
        metrics['physical_residual_delta_max_mps'] = float(np.linalg.norm(np.diff(residual,axis=0),axis=1).max()/.05)
        metrics['tail35_39_pooled_sse_fraction'] = float(squared[35:].sum()/squared.sum()) if squared.sum() else 0.0
        assert set(metrics)==set(row['metrics']) and all(same(v,row['metrics'][k]) for k,v in metrics.items())
        gate = TrajectorySafetyGateV1().evaluate(
            selector._build_chunk(decoded,chunk_id=f"pi0247-S4-{record['model_name']}-{source}-{noise}"),
            now=0.0,achieved_position=state[:3],achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(state[3:9].reshape(1,6)))[0])
        assert gate.decision.value==record['gate_decision']==row['gate_decision']
        issues = gate.as_dict()['issues']
        assert len(issues)==len(record['gate']['issues'])
        for a,b in zip(issues,record['gate']['issues']):
            assert set(a)==set(b) and all(same(v,b[k]) if isinstance(v,(int,float)) else v==b[k] for k,v in a.items())
        replay.append({'source_episode_id':source,'noise_name':noise,'status':'VALID','metrics':metrics,
                       'gate_decision':gate.decision.value,'gate_issues':issues,'tensor_sha256':row['tensor_sha256'],
                       'decoded_max_abs':error,'model_source_card_noise_binding_verified':True})
    assert seen==expected
    old16 = set(card['support']['old16_source_ids'])
    groups = {n:m64.group([r for r in replay if r['noise_name']==n],64,args.step==5000) for n in noises}
    subsets = {n:m64.group([r for r in replay if r['noise_name']==n and r['source_episode_id'] in old16],16,args.step==5000) for n in noises}
    for actual,saved in [(groups,report['group_results']),(subsets,report['old16_group_results'])]:
        assert actual==saved
    result = {'schema':'PI0247_M64_INDEPENDENT_CPU_REPLAY_V1','status':'PASS_INDEPENDENT_CPU_REPLAY',
              'checkpoint_step':args.step,'screen_report_sha256':m64.sha(report_path),'run_card_sha256':m64.CARD_SHA,
              'counts':{'cells_replayed':len(replay),'model_loads':0,'model_forwards':0},
              'group_results':groups,'old16_group_results':subsets,'cells':replay}
    m64.write(root/'cpu_replay.json',result)
    print(json.dumps({'status':result['status'],'step':args.step,'cells':len(replay),'sha256':m64.sha(root/'cpu_replay.json')}))


if __name__=='__main__':
    main()
