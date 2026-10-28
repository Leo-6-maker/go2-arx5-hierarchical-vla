#!/usr/bin/env python3
"""Registered M64 screens, using the frozen D1 query/decoder/gate helpers."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path('/var/cache/users/liuyu-nfs/pi0247_generation_to_sim_20260926')
M64 = ROOT/'m64'
PUBLIC = M64/'public_inputs/reports/PI0247/M64_U_5845272077'
START = PUBLIC/'start'
SOURCES = PUBLIC/'source64/source64'
CARD_SHA = '53bc452b5ca77f4ee9030663415227f471258f0b5b7d76c708acf60527306664'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    from pi0247_lu_final_screen import write_immutable
    write_immutable(path, value)


def inputs():
    import torch
    import pi0247_d1_screen as d1
    import pi0247_d2_full16_screen as d2
    card = read(START/'run_card.json')
    assert sha(START/'run_card.json') == CARD_SHA == read(START/'run_card.seal.json')['sha256']
    assert card['schema'] == 'PI0247_M64_RUN_CARD_V1' and card['selected_checkpoint'] == 'NONE'
    fit_path = ROOT/'lu/fit_recipe_manifest.json'
    fit = read(fit_path)
    assert sha(fit_path) == card['source']['fit_recipe_manifest_sha256']
    d2_path = ROOT/'d2/PI0247_D2_RUN_CARD_20260926_R2.json'
    d2_card, d2_sha = d2.read_card(d2_path, d2_path.with_suffix('.json.sha256'))
    facts, ctx = d1.prepare(Path(d2_card['paths']['stage_root']), Path(d2_card['paths']['code_root']))
    assert d1.canonical_sha(facts) == d2_card['static_preflight_canonical_sha256']
    ctx['facts'] = facts
    d2.prepare_inputs(d2_card, ctx)
    old16 = set(card['support']['old16_source_ids'])
    assert old16 == {r['source_episode_id'] for r in ctx['rows']}
    source_doc = read(SOURCES/'manifest.json')
    assert sha(SOURCES/'manifest.json') == '3560fd0c21432b01ccd4d47fd9e578c4dbfb211fb518be1e29f72fa7b54917f8'
    assert source_doc['run_card_sha256'] == CARD_SHA and source_doc['cohort_sha256'] == card['cohort']['sha256']
    ct = fit['processor_identity']['frozen_C_T']
    assert source_doc['processor_config_sha256'] == ct['processor_config_sha256']
    assert source_doc['train_stats_sha256'] == ct['NEW_EMBODIMENT_stats_sha256']
    assert source_doc['relative_stats_sha256'] == ct['relative_stats_sha256']
    assert source_doc['helper_sha256'] == facts['source_code_files_sha256']['screen_metrics']
    noise_path = START/'screen_noise/manifest.json'
    noise_doc = read(noise_path)
    assert sha(noise_path) == (noise_path.with_name('manifest.sha256')).read_text().strip()
    assert noise_doc['run_card_sha256'] == CARD_SHA and noise_doc['old16_B0'] == fit['noise_identity']['B0']
    assert noise_doc['all64_B1'] == fit['noise_identity']['B1']
    b0_path, b1_path = START/'screen_noise/new48_B0.npz', START/'screen_noise/B1_N1.npz'
    assert sha(b0_path) == noise_doc['new48_B0']['packet_sha256']
    assert sha(b1_path) == fit['noise_identity']['B1']['file_sha256']
    with np.load(b0_path, allow_pickle=False) as data:
        new_b0 = data['source_float32'].copy()
        native = torch.from_numpy(new_b0).to(torch.bfloat16).view(torch.uint16).numpy()
        assert np.array_equal(native, data['native_bfloat16_uint16'])
    assert hashlib.sha256(new_b0.tobytes()).hexdigest() == noise_doc['new48_B0']['source_tensor_sha256']
    assert hashlib.sha256(native.tobytes()).hexdigest() == noise_doc['new48_B0']['native_bytes_sha256']
    with np.load(b1_path, allow_pickle=False) as data:
        b1 = data['float32'].copy()
    assert np.array_equal(b1, ctx['noise']['N1'])
    import pi0242_train_infer_bridge as bridge
    import pi0243_staged_learning_to_sim as stage43
    from gr00t.data.types import MessageType, VLAStepData
    old_rows = {r['source_episode_id']:r for r in ctx['rows']}
    all_rows = source_doc['files']
    assert len(all_rows) == 64 and len({r['source_episode_id'] for r in all_rows}) == 64
    assert old16.issubset({r['source_episode_id'] for r in all_rows})
    assert set(noise_doc['new48_source_ids']) == {r['source_episode_id'] for r in all_rows}-old16
    old_cells = ctx['source_cells']
    ctx['source_cells'], ctx['m64_noise'], ctx['m64_packets'] = {}, {}, {}
    provenance = []
    for row in all_rows:
        source = row['source_episode_id']
        tensor = SOURCES/row['path']
        assert sha(tensor) == row['sha256'] and tensor.stat().st_size == row['bytes']
        with np.load(tensor, allow_pickle=False) as data:
            arrays = {k:data[k].copy() for k in data.files}
        state, expert, mask = arrays['state'], arrays['expert'], arrays['action_mask']
        assert state.shape == (10,) and expert.shape == (40,10) and mask.shape == (40,132)
        assert np.all(mask[:,:10] == 1) and not np.count_nonzero(mask[:,10:])
        assert all(np.isfinite(v).all() for v in arrays.values())
        content = VLAStepData(images={k:arrays[v][None] for k,v in [('external_rgb','body_rgb'),('wrist_rgb','wrist_rgb')]},
                              states={'eef_9d':state[None,:9],'gripper_position':state[None,9:]},
                              actions={}, text=stage43.LANGUAGE, embodiment=ctx['embodiment'])
        processed = ctx['processor']([{'type':MessageType.EPISODE_STEP.value,'content':content}])
        policy = bridge._policy_inputs(ctx['processor'].collator([processed]))
        inventory = bridge._leaf_inventory(policy)
        assert not any(any(token in k.lower() for token in bridge.FORBIDDEN_INPUT_TOKENS) for k in inventory)
        identity = bridge._canonical_hash(inventory)
        if source in old16:
            with np.load(old_rows[source]['resolved_tensor_path'], allow_pickle=False) as old:
                assert all(np.array_equal(arrays[k],old[k]) for k in arrays), f'old16 array identity drift: {source}'
            info = old_cells[source]
            assert identity == info['B0']['condition_identity'] == info['B1']['condition_identity']
            ctx['source_cells'][source] = info
            packet = Path(ctx['d2_noise_packet_paths'][source])
            assert sha(packet) == noise_doc['old16_B0'][source]['noise_packet_sha256']
            b0 = ctx['d2_noise'][source]['B0']
        else:
            parent = M64/'source_inputs'/f"source-{row['episode_index']:06d}"
            parent.mkdir(parents=True,exist_ok=True)
            if not (parent/'condition.pt').exists():
                ctx['p1'].durable_torch(parent/'condition.pt',policy)
            saved = torch.load(parent/'condition.pt',map_location='cpu',weights_only=False)
            assert bridge._canonical_hash(bridge._leaf_inventory(saved)) == identity
            if not (parent/'tensors.npz').exists():
                (parent/'tensors.npz').symlink_to(tensor)
            assert (parent/'tensors.npz').resolve() == tensor.resolve()
            record = {'source_episode_id':source,'source_tensor_sha256':row['sha256'],
                      'condition_identity':identity,'condition_sha256':sha(parent/'condition.pt'),
                      'status':'SOURCE_INPUT_ONLY','target_xyz_in_model_input':False}
            write(parent/'record.json',record)
            ctx['source_cells'][source] = {n:{'cell_path':str(parent),'condition_identity':identity} for n in ('B0','B1')}
            b0, packet = new_b0, b0_path
        ctx['m64_noise'][source] = {'B0':b0,'B1':b1}
        ctx['m64_packets'][source] = {'B0':str(packet),'B1':str(b1_path)}
        provenance.append({'source_episode_id':source,'source_tensor_sha256':row['sha256'],
                           'condition_identity':identity,'old16_exact_arrays_and_condition':source in old16})
    ctx['rows'] = all_rows
    manifest = {'schema':'PI0247_M64_SCREEN_INPUTS_V1','run_card_sha256':CARD_SHA,'d2_run_card_sha256':d2_sha,
                'source64_manifest_sha256':sha(SOURCES/'manifest.json'),'noise_manifest_sha256':sha(noise_path),
                'processor_identity':fit['processor_identity'],'model_loads':0,'model_forwards':0,'sources':provenance}
    write(M64/'screen_inputs_manifest.json',manifest)
    return card, fit, ctx


def group(rows, count, final):
    valid = [r for r in rows if r.get('metrics') and r['status'] in ('VALID','REUSED_VALID')]
    xyz = [r['metrics']['xyz_rmse_m'] for r in valid]
    so3 = [r['metrics']['so3_rmse_rad'] for r in valid]
    result = {'cell_count':len(rows),'valid_count':len(valid),'invalid_count':len(rows)-len(valid),
              'gate_counts':dict(Counter(r['gate_decision'] for r in rows)),
              'reject_count':sum(r['gate_decision']=='REJECT' for r in rows),
              'median_xyz_rmse_m':float(np.median(xyz)) if xyz else None,
              'p95_xyz_rmse_m':float(np.quantile(xyz,.95,method='linear')) if xyz else None,
              'median_so3_rmse_rad':float(np.median(so3)) if so3 else None,'quality_gate_applied':final}
    if final:
        result['quality_pass'] = (len(rows)==len(valid)==count and result['reject_count']==0
                                  and result['median_xyz_rmse_m']<=.020 and result['p95_xyz_rmse_m']<=.035
                                  and result['median_so3_rmse_rad']<=.050)
    return result


def archive_prefix(card, receipt, resume=None):
    step = receipt['checkpoint_step']
    output = Path(card['paths']['output_root'])
    prefix = card['archive']['path_prefix']
    if receipt['source'] != str(output/'training'/f'checkpoint-{step}'):
        job = receipt['job_id']
        assert job and all(c.isalnum() or c in '._-' for c in job)
        assert receipt['source'] == str(output/'recovery_attempts'/job/'training'/f'checkpoint-{step}')
        assert resume and resume['status']=='SEALED_RESUME_SOURCE' and resume['run_card_sha256']==CARD_SHA
        assert resume['schedule_sha256']==card['schedule']['sha256'] and resume['resume_after_step']<step
        assert resume['source_checkpoint_path']==str(output/'training'/f"checkpoint-{resume['resume_after_step']}")
        prefix += '/recovery/'+job
    assert receipt['durable_path']==card['archive']['repo_id']+'/'+prefix+f'/checkpoint-{step}'
    return prefix


def materialize(step):
    from huggingface_hub import snapshot_download
    card = read(START/'run_card.json')
    assert sha(START/'run_card.json') == CARD_SHA
    receipt_path = M64/f'checkpoint_evidence/{step}/durable_receipt.json'
    readback_path = receipt_path.with_name('hf_readback.json')
    receipt, rb = read(receipt_path), read(readback_path)
    record = receipt['source_record']
    inventory = record['inventory']
    resume_path = M64/'resume_evidence'/receipt['job_id']/'resume_record.json'
    base_prefix = archive_prefix(card,receipt,read(resume_path) if resume_path.exists() else None)
    prefix = base_prefix+f'/checkpoint-{step}'
    assert receipt['status']=='DURABLE_VERIFIED' and receipt['family']==card['family']
    assert receipt['checkpoint_step']==step and receipt['max_steps']==5000
    assert rb['status']=='PASS_PT_FULL_HF_READBACK' and rb['pi0247_status']=='PASS_PI0247_LU_FULL_HF_READBACK'
    assert rb['revision']==receipt['hf_manifest_commit_id'] and rb['source_tree_sha256']==inventory['tree_sha256']
    assert rb['repo_id']==card['archive']['repo_id'] and rb['path_prefix']==base_prefix
    expected = {prefix+'/'+r['path']:r for r in inventory['entries']}
    observed = {r['path']:r for r in rb['files']}
    assert set(expected)==set(observed)
    assert all(observed[n]['sha256']==r['sha256'] and observed[n]['bytes']==r['bytes'] for n,r in expected.items())
    files = [r for r in inventory['entries'] if r['path'] in ('config.json','model.safetensors.index.json') or r['path'].endswith('.safetensors')]
    assert len(files)==4
    download = M64/f'weights/checkpoint-{step}'
    snapshot_download(card['archive']['repo_id'], revision=receipt['hf_manifest_commit_id'],
                      allow_patterns=[prefix+'/'+r['path'] for r in files],local_dir=str(download),token=False)
    model = M64/f'model_views/checkpoint-{step}'
    model.mkdir(parents=True,exist_ok=True)
    for entry in files:
        path = download/prefix/entry['path']
        assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
        target = model/entry['path']
        if not target.exists():target.symlink_to(path)
        assert target.resolve()==path.resolve()
    write(model/'pi0245_inference_restore.json',{'schema':'PI0247_M64_INFERENCE_VIEW_V1','optimizer_restored':False,
          'checkpoint_step':step,'run_card_sha256':CARD_SHA,'prefix':prefix,'revision':receipt['hf_manifest_commit_id'],
          'checkpoint_tree_sha256':inventory['tree_sha256'],'durable_receipt_sha256':sha(receipt_path),
          'hf_readback_sha256':sha(readback_path),'files':files})
    print(json.dumps({'status':'PASS_PINNED_M64_INFERENCE_MATERIALIZATION','step':step,'model_root':str(model)}))


def screen(step):
    import torch
    import pi0247_d1_screen as d1
    card, fit, ctx = inputs()
    import pi0242_train_infer_bridge as bridge
    import pi0228_train_calib_checkpoint_selection as selector
    import pi0243_staged_learning_to_sim as stage43
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from transformers import AutoModel
    root = M64/f'screen_step{step}'
    root.mkdir(parents=True,exist_ok=True)
    model_root = (ROOT/'lu/U_screen_step2000/inference_view_step_2000' if step==0
                  else M64/f'model_views/checkpoint-{step}')
    model_files = ctx['p1'].verify_model_files(model_root)
    if step==0:
        baseline = read(ROOT/'lu/U_screen_step2000/final_screen_report.json')
        assert sha(ROOT/'lu/U_screen_step2000/final_screen_report.json')==fit['final_screen_report_sha256']
        binding = fit['checkpoint_binding']
        assert model_files['manifest_sha256']==read(ROOT/'lu/U_screen_step2000/score_card.json')['inference_restore_sha256']
    else:
        restore = read(model_root/'pi0245_inference_restore.json')
        assert restore['run_card_sha256']==CARD_SHA and restore['checkpoint_step']==step
        binding = {'revision':restore['revision'],'tree_sha256':restore['checkpoint_tree_sha256'],
                   'durable_receipt_sha256':restore['durable_receipt_sha256'],'hf_readback_sha256':restore['hf_readback_sha256']}
        if step==5000:
            training = read(M64/'training_manifest.json')
            assert training['status']=='PASS_TRAINING_INTEGRITY' and training['global_step']==5000
            assert training['run_card_sha256']==CARD_SHA and training['selected_checkpoint']=='NONE'
            assert training['checkpoints']['5000']['tree_sha256']==restore['checkpoint_tree_sha256']
    score = {'schema':'PI0247_M64_SCREEN_CARD_V1','stage':'M64','checkpoint_step':step,'runtime_id':'S4',
             'run_card_sha256':CARD_SHA,'family':card['family'],'selected_checkpoint':'NONE',
             'input_manifest_sha256':sha(M64/'screen_inputs_manifest.json'),'checkpoint_binding':binding,
             'inference_restore_sha256':model_files['manifest_sha256'],'runner_sha256':sha(__file__),
             'loss_contract':fit['loss_contract'],'processor_identity':fit['processor_identity'],
             'resource':{'device':'cuda:0'}}
    base_score_path = root/'score_card.json'
    active_score_path = base_score_path
    if base_score_path.exists():
        saved = read(base_score_path)
        assert {k:v for k,v in saved.items() if k!='runner_sha256'} == {k:v for k,v in score.items() if k!='runner_sha256'}
        if saved['runner_sha256'] != score['runner_sha256']:
            score['engineering_parent_score_card_sha256'] = sha(base_score_path)
            active_score_path = root/f"score_card_repair_{score['runner_sha256'][:12]}.json"
    write(active_score_path,score)
    score_sha = sha(active_score_path)
    model = AutoModel.from_pretrained(model_root,local_files_only=True,trust_remote_code=True)
    model.eval().to(device='cuda:0',dtype=torch.bfloat16)
    assert model.action_head.num_inference_timesteps==4
    old16 = set(card['support']['old16_source_ids'])
    noises = ('B0','B1') if step==5000 else ('B0',)
    rows = []
    for source_row in ctx['rows']:
        source = source_row['source_episode_id']
        for noise in noises:
            if step==0 and source in old16:
                row = next(r for r in baseline['cells'] if r['source_episode_id']==source and r['noise_name']==noise)
                record = d1.collect_record(Path(row['source_cell'])/'record.json',ctx['p1'])
                assert record['tensor_sha256']==row['tensor_sha256'] and record['status']=='VALID'
                assert record['model_manifest_sha256']==model_files['manifest_sha256']
                assert record['condition_identity']==ctx['source_cells'][source][noise]['condition_identity']
                rows.append({**row,'status':'REUSED_VALID','reused_record_sha256':sha(Path(row['source_cell'])/'record.json')})
                continue
            ctx['noise'][noise] = ctx['m64_noise'][source][noise]
            ctx['facts']['noise_files_sha256'][noise] = sha(ctx['m64_packets'][source][noise])
            old_cell = root/'cells'/f"M64{step}__S4__{source.split(':',1)[0]}__{noise}"
            if (old_cell/'record.json').is_file():
                cached = d1.collect_record(old_cell/'record.json',ctx['p1'])
                if cached['source_episode_id']==source:
                    assert cached['condition_identity']==ctx['source_cells'][source][noise]['condition_identity']
                    assert cached['model_manifest_sha256']==model_files['manifest_sha256']
                    assert cached['noise_source_sha256']==hashlib.sha256(ctx['noise'][noise].tobytes()).hexdigest()
                    rows.append({**cached,'noise_name':noise,'source_cell':str(old_cell),
                                 'record_sha256':sha(old_cell/'record.json'),'score_card_path':str(base_score_path)})
                    continue
            # Full window identity belongs in the directory; the frozen D1 helper keeps its original cell key.
            query_root = root/f"observations/source-{source_row['episode_index']:06d}"
            record = d1.run_one(args=argparse.Namespace(output_root=query_root),card=score,card_sha=score_sha,
                               context=ctx,model_name=f'M64{step}',runtime={'id':'S4','steps':4,'inactive':'stock'},
                               source=source,noise_name=noise,model=model,model_files=model_files,
                               bridge=bridge,selector=selector,stage43=stage43,gate=TrajectorySafetyGateV1())
            cell = query_root/'cells'/record['cell_key']
            assert record['source_episode_id']==source
            rows.append({**record,'noise_name':noise,'source_cell':str(cell),'record_sha256':sha(cell/'record.json'),
                         'score_card_path':str(active_score_path)})
    assert len(rows)==64*len(noises) and len({(r['source_episode_id'],r['noise_name']) for r in rows})==len(rows)
    groups = {n:group([r for r in rows if r['noise_name']==n],64,step==5000) for n in noises}
    subsets = {n:group([r for r in rows if r['noise_name']==n and r['source_episode_id'] in old16],16,step==5000) for n in noises}
    passed = step==5000 and all(g['quality_pass'] for g in list(groups.values())+list(subsets.values()))
    report = {'schema':'PI0247_M64_SCREEN_REPORT_V1','stage':'M64','checkpoint_step':step,'selected_checkpoint':'NONE',
              'status':('PASS_M64_SUPPORT_EXPANSION' if passed else 'SCIENCE_FAIL_M64_SUPPORT_EXPANSION') if step==5000 else 'OBSERVATION_ONLY',
              'score_card_sha256':score_sha,'score_card':score,'group_results':groups,'old16_group_results':subsets,
              'execution_cards':[{'path':str(p),'sha256':sha(p)} for p in dict.fromkeys((base_score_path,active_score_path))],
              'counts':{'logical_cells':len(rows),'reused_valid_cells':16 if step==0 else 0,
                        'new_queries':len(rows)-(16 if step==0 else 0),'new_head_evaluations':4*(len(rows)-(16 if step==0 else 0))},'cells':rows}
    write(root/'screen_report.json',report)
    print(json.dumps({k:report[k] for k in ('status','checkpoint_step','counts','group_results','old16_group_results')}))


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-inputs',action='store_true')
    mode.add_argument('--materialize',action='store_true')
    mode.add_argument('--screen',action='store_true')
    mode.add_argument('--self-check',action='store_true')
    parser.add_argument('--step',type=int,choices=(0,1000,2500,5000),default=0)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='0' if args.materialize else '1',TRANSFORMERS_OFFLINE='1')
    if args.self_check:
        rows = [{'status':'VALID','gate_decision':'WARN','metrics':{'xyz_rmse_m':.010,'so3_rmse_rad':.030}} for _ in range(64)]
        assert group(rows,64,True)['quality_pass']
        assert not group(rows[:16],64,True)['quality_pass']
        rows[0]['gate_decision']='REJECT'
        assert not group(rows,64,True)['quality_pass']
        assert 'quality_pass' not in group(rows,64,False)
        card={'paths':{'output_root':'/bound'},'archive':{'path_prefix':'arms/M64','repo_id':'public/repo'},'schedule':{'sha256':'fixed'}}
        receipt={'source':'/bound/training/checkpoint-1000','checkpoint_step':1000,'durable_path':'public/repo/arms/M64/checkpoint-1000'}
        assert archive_prefix(card,receipt)=='arms/M64'
        receipt.update(source='/bound/recovery_attempts/job.1/training/checkpoint-2500',checkpoint_step=2500,
                       job_id='job.1',durable_path='public/repo/arms/M64/recovery/job.1/checkpoint-2500')
        resume={'status':'SEALED_RESUME_SOURCE','run_card_sha256':CARD_SHA,'schedule_sha256':'fixed',
                'resume_after_step':1000,'source_checkpoint_path':'/bound/training/checkpoint-1000'}
        assert archive_prefix(card,receipt,resume)=='arms/M64/recovery/job.1'
        try: archive_prefix(card,receipt)
        except AssertionError: pass
        else: raise AssertionError('recovery requires its sealed resume record')
        print('PASS_M64_GROUP_CARDINALITY_AND_ORIGINAL_GATES')
        print('PASS_M64_ORIGINAL_AND_EXACT_RESUME_ARCHIVE_BINDING')
    elif args.materialize:
        assert args.step in (1000,2500,5000)
        materialize(args.step)
    elif args.prepare_inputs:
        os.environ['CUDA_VISIBLE_DEVICES']=''
        inputs()
        print(json.dumps({'status':'PASS_M64_CPU_INPUT_BINDING','sources':64,'old16_exact':16,'model_queries':0,
                          'manifest_sha256':sha(M64/'screen_inputs_manifest.json')}))
    else:
        screen(args.step)


if __name__=='__main__':
    main()
