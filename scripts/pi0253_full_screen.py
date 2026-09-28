#!/usr/bin/env python3
"""Fixed FULL retention using the qualified Q runner and original actual packets."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import math
import os
import sys

BASE = Path('/home/liuyu-nfs/Projects/go2-arx5')
ROOT = BASE / 'PI0253'
INPUTS = BASE / 'PI0250/inputs/frozen_v1'
CODE = BASE / 'PI0250/runtime/reference-317a'
STAGE10 = Path('/home/liuyu-nfs/Projects/isaac-gr00t')
OLD = BASE / 'PI0252/screen/G2000/screen_report.json'
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                  HF_HOME=str(BASE / 'PI0250/runtime/hf_home'))
sys.path[:0] = [str(CODE / 'scripts'), str(CODE / 'src'), str(STAGE10)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert read(path) == value, f'first-valid identity conflict: {path}'
        return
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def materialize(step):
    """Stream pinned model bytes on Corelab; optimizer state stays on Hopper/HF."""
    import requests
    from huggingface_hub import hf_hub_url
    evidence = ROOT / 'evidence/FULL'
    card = read(evidence / 'FULL_run_card.json')
    card_sha = sha(evidence / 'FULL_run_card.json')
    assert card['schema'] == 'PI0253_FULL_RUN_CARD_V1' and card['family'] == 'PI0253_FULL_G_SUPPORT_V1'
    assert card['selected_checkpoint'] == 'NONE' and card['stage']['steps'] == 10000
    receipt_path = evidence / ('model-only-1000.hf_readback.json' if step == 1000 else f'checkpoint-{step}.json')
    receipt = read(receipt_path)
    if step == 1000:
        assert receipt['status'] == 'PASS_FULL_FILE_READBACK' and receipt['candidate'] is False and receipt['resumable'] is False
        assert receipt['run_card_sha256'] == card_sha
        prefix = receipt['prefix']; entries = receipt['files']; tree = None
    else:
        durable_path = evidence / f'checkpoint-{step}.durable_receipt.json'
        durable = read(durable_path)
        assert durable['status'] == 'DURABLE_VERIFIED' and durable['family'] == card['family'] and durable['checkpoint_step'] == step
        assert receipt['status'] == 'PASS_PT_FULL_HF_READBACK' and receipt['revision'] == durable['hf_manifest_commit_id']
        inventory = durable['source_record']['inventory']; tree = inventory['tree_sha256']
        assert receipt['source_tree_sha256'] == tree
        prefix = receipt['path_prefix'] + f'/checkpoint-{step}'
        entries = inventory['entries']
        observed = {r['path'].split(f'/checkpoint-{step}/',1)[-1]:r for r in receipt['files']}
        assert set(observed) == {r['path'] for r in entries}
        assert all(r['bytes'] == observed[r['path']]['bytes'] and r['sha256'] == observed[r['path']]['sha256'] for r in entries)
    repo, revision = receipt['repo_id'], receipt['revision']
    selected = {r['path']:r for r in entries if r['path'] in {'config.json','model.safetensors.index.json',
                'model-00001-of-00002.safetensors','model-00002-of-00002.safetensors'}}
    assert len(selected) == 4
    view = ROOT / f'models/FULL{step}/inference_view'
    view.mkdir(parents=True, exist_ok=True)
    token = Path('/home/liuyu-nfs/.config/huggingface/token').read_text().strip()
    for name, row in selected.items():
        path = view / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.partial')
            h, size = hashlib.sha256(), 0
            with requests.get(hf_hub_url(repo, filename=prefix+'/'+name, revision=revision),
                              headers={'Authorization':'Bearer '+token}, stream=True, timeout=(30,180)) as response:
                response.raise_for_status()
                with temporary.open('wb') as stream:
                    for block in response.iter_content(8 << 20): stream.write(block); h.update(block); size += len(block)
                    stream.flush(); os.fsync(stream.fileno())
            assert size == row['bytes'] and h.hexdigest() == row['sha256'], name
            os.replace(temporary, path)
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], name
        print(json.dumps(dict(status='PINNED_MODEL_FILE',step=step,path=name)),flush=True)
    restore = dict(schema='PI0253_PINNED_INFERENCE_VIEW_V1', optimizer_restored=False, checkpoint_step=step,
                   run_card_sha256=card_sha, repo_id=repo, revision=revision, prefix=prefix, checkpoint_tree_sha256=tree,
                   full_hf_readback_sha256=sha(receipt_path), files=list(selected.values()), candidate=step != 1000)
    write(view / 'pi0245_inference_restore.json', restore)
    return card, card_sha, view, restore


def context():
    import numpy as np
    import pi0245_p1_stock as p1
    import stage21_same_scene_groot_inference as s21
    import stage40_b0_model_control as s40
    old = read(OLD); old_score = old['score_card']
    assert sha(INPUTS/'source64/manifest.json') == old_score['source64_manifest_sha256']
    assert sha(INPUTS/'noise/manifest.json') == old_score['noise_manifest_sha256']
    assert sha(INPUTS/'conditions_recomputed_manifest.json') == old_score['conditions_recomputed_manifest_sha256']
    assert sha(INPUTS/'conditions_baseline_manifest.json') == old_score['conditions_baseline_manifest_sha256']
    assert sha(INPUTS/'pi0250_frozen_screen_inputs.json') == old_score['frozen_screen_inputs_sha256']
    s40._bind_local_cosmos_processor(STAGE10, INPUTS/'cosmos_snapshot')
    processor, embodiment = s21._build_processor(STAGE10, INPUTS/'processor_base', INPUTS/'train_dataset/meta', INPUTS/'modality/go2_arx5_modality.py')
    processor.eval()
    sources = sorted(read(INPUTS/'source64/manifest.json')['files'], key=lambda r:int(r['episode_index']))
    assert len(sources) == 64 and len({r['source_episode_id'] for r in sources}) == 64
    old16 = set(old_score['support']['old16_source_ids'])
    noise, packets = {}, old_score['noise_bindings']
    for key, row in packets.items():
        path = Path(row['path'])
        assert sha(path) == row['file_sha256']
        with np.load(path,allow_pickle=False) as data:
            value = data[{'old16_B0':'noise_source_float32','new48_B0':'source_float32','B1':'float32'}[key]].copy()
        assert hashlib.sha256(value.tobytes()).hexdigest() == row['raw_tensor_sha256']
        noise[key] = value
    cells = {}
    old_by_id = {(r['source_episode_id'],r['noise_name']):r for r in old['cells']}
    for row in sources:
        source = row['source_episode_id']; parent = BASE / 'PI0252/screen/G2000/source_inputs' / f"source-{row['episode_index']:06d}"
        record = read(parent/'record.json')
        assert sha(INPUTS/'source64'/row['path']) == row['sha256'] == record['source_tensor_sha256']
        assert sha(parent/'condition.pt') == record['condition_sha256']
        assert record['condition_identity'] == old_by_id[(source,'B0')]['condition_identity']
        cells[source] = {n:dict(cell_path=str(parent),condition_identity=record['condition_identity']) for n in ('B0','B1')}
    ctx = dict(p1=p1,p1_root=INPUTS,source_cells=cells,noise={},facts={'noise_files_sha256':{}},processor=processor,embodiment=embodiment)
    return old, sources, old16, noise, packets, ctx


def cpu_replay(root, sources, old16, noises, packets, ctx):
    """Decode and score actual saved tensors independently; no model construction."""
    import numpy as np
    import torch
    import pi0242_train_infer_bridge as bridge
    import pi0243_staged_learning_to_sim as s43
    import pi0228_train_calib_checkpoint_selection as selector
    import pi0247_m64_screen as m64
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat,rot6d_rows_to_matrix
    report = read(root/'screen_report.json'); replay=[]
    expected={(r['source_episode_id'],n) for r in sources for n in report['noise_names']}
    seen=set()
    for row in report['cells']:
        key=row['source_episode_id'],row['noise_name'];assert key in expected and key not in seen;seen.add(key)
        cell=Path(row['source_cell']);record=read(cell/'record.json')
        assert record['status']=='VALID' and sha(cell/'tensors.npz')==row['tensor_sha256']
        assert record['run_card_sha256']==report['score_card_sha256'] and record['model_revision']==report['hf_revision']
        assert record['condition_identity']==ctx['source_cells'][key[0]][key[1]]['condition_identity']
        packet = ('old16_B0' if key[0] in old16 else 'new48_B0') if key[1]=='B0' else 'B1'
        assert record['noise_file_sha256']==packets[packet]['file_sha256']
        with np.load(cell/'tensors.npz',allow_pickle=False) as data:
            raw,decoded_saved,state,expert=[data[k].copy() for k in ('raw_scored','decoded','state','expert')]
            assert np.array_equal(raw,data['raw_native']) and not record['m_path']['applicable']
            assert np.array_equal(data['noise_source_float32'],noises[packet])
            native=torch.from_numpy(noises[packet]).to(torch.bfloat16).float().numpy()
            assert np.array_equal(data['noise_initial'],native) and np.array_equal(data['trace_x'][0],native)
            assert data['trace_x'].shape==data['trace_v'].shape==(4,1,40,132)
        decoded=bridge._decode_physical(ctx['processor'],ctx['embodiment'],raw,state)
        assert np.max(np.abs(decoded-decoded_saved))<=1e-7
        metrics=s43._screen_metrics(decoded[0].astype(np.float64),expert.astype(np.float64),None)
        residual=decoded[0,:,:3].astype(np.float64)-expert[:,:3].astype(np.float64);squared=np.sum(residual*residual,axis=1)
        metrics.update(physical_residual_delta_max_mps=float(np.linalg.norm(np.diff(residual,axis=0),axis=1).max()/.05),
                       tail35_39_pooled_sse_fraction=float(squared[35:].sum()/squared.sum()) if squared.sum() else 0.)
        assert set(metrics)==set(row['metrics']) and all(math.isclose(v,row['metrics'][k],rel_tol=1e-10,abs_tol=1e-12) for k,v in metrics.items())
        gate=TrajectorySafetyGateV1().evaluate(selector._build_chunk(decoded,chunk_id=record['gate']['metrics']['chunk_id']),now=0.,
                achieved_position=state[:3],achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(state[3:9].reshape(1,6)))[0])
        assert gate.decision.value==row['gate_decision'] and gate.as_dict()['issues']==record['gate']['issues']
        replay.append(dict(source_episode_id=key[0],noise_name=key[1],status='VALID',metrics=metrics,gate_decision=gate.decision.value,
                           old16=key[0] in old16,linear_speed_reject=any(i['code']=='LINEAR_SPEED_REJECT' for i in gate.as_dict()['issues'])))
    assert seen==expected
    groups={n:m64.group([r for r in replay if r['noise_name']==n],64,True) for n in report['noise_names']}
    old_groups={n:m64.group([r for r in replay if r['noise_name']==n and r['old16']],16,True) for n in report['noise_names']}
    result=dict(schema='PI0253_FULL_RETENTION_CPU_REPLAY_V1',status='PASS_INDEPENDENT_CPU_REPLAY',checkpoint_step=report['checkpoint_step'],
                report_sha256=sha(root/'screen_report.json'),hf_revision=report['hf_revision'],group_results=groups,old16_group_results=old_groups,
                B0_linear_speed_reject_count=sum(r['noise_name']=='B0' and r['linear_speed_reject'] for r in replay),
                cells=replay,model_queries=0,optimizer_updates=0,selected_checkpoint='NONE')
    write(root/'cpu_replay.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='cells'}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,choices=(1000,2500,5000,7500,10000),required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--cpu',action='store_true');args=parser.parse_args()
    if args.prepare_only:
        materialize(args.step);return
    import numpy as np
    import torch
    import pi0247_d1_screen as d1
    import pi0247_m64_screen as m64
    import pi0242_train_infer_bridge as bridge
    import pi0243_staged_learning_to_sim as stage43
    import pi0228_train_calib_checkpoint_selection as selector
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    old,sources,old16,noises,packets,ctx=context()
    root=ROOT/f'screen/FULL{args.step}'
    if args.cpu:
        cpu_replay(root,sources,old16,noises,packets,ctx);return
    card,card_sha,view,restore=materialize(args.step)
    files=ctx['p1'].verify_model_files(view)
    score=dict(schema='PI0253_FULL_RETENTION_SCORE_CARD_V1',checkpoint_step=args.step,selected_checkpoint='NONE',runtime_id='S4',sampler_steps=4,
               runner_sha256=sha(Path(d1.__file__)),orchestrator_sha256=sha(Path(__file__)),run_card_sha256=card_sha,
               initializer_G_screen_report_sha256=sha(OLD),inference_restore_sha256=files['manifest_sha256'],hf_revision=restore['revision'],
               source64_manifest_sha256=sha(INPUTS/'source64/manifest.json'),noise_manifest_sha256=sha(INPUTS/'noise/manifest.json'),
               policy_target_in_input=False,scientific_scope='fixed retention; intermediate scores cannot tune training')
    score_path=root/'score_card.json';write(score_path,score);score_sha=sha(score_path)
    report_path=root/'screen_report.json'
    if report_path.exists():
        report=read(report_path);assert report['score_card_sha256']==score_sha;print('REUSED_COMPLETE_FIRST_VALID');return
    assert torch.cuda.is_available()
    from transformers import AutoModel
    model=AutoModel.from_pretrained(view,local_files_only=True,trust_remote_code=True).eval().to(device='cuda:0',dtype=torch.bfloat16)
    assert model.action_head.num_inference_timesteps==4
    cells=[];reused=0;model_name=f'FULL{args.step}';noise_names=('B0',) if args.step==1000 else ('B0','B1')
    for row in sources:
        source=row['source_episode_id']
        for n in noise_names:
            packet=('old16_B0' if source in old16 else 'new48_B0') if n=='B0' else 'B1'
            ctx['noise'][n]=noises[packet];ctx['facts']['noise_files_sha256'][n]=packets[packet]['file_sha256']
            observation=root/'observations'/f"source-{row['episode_index']:06d}"
            key=f"{model_name}__S4__{source.split(':',1)[0]}__{n}";exists=(observation/'cells'/key/'record.json').is_file()
            record=d1.run_one(args=SimpleNamespace(output_root=observation),card=score,card_sha=score_sha,context=ctx,
                model_name=model_name,runtime={'id':'S4','steps':4,'inactive':'stock'},source=source,noise_name=n,model=model,
                model_files=files,bridge=bridge,selector=selector,stage43=stage43,gate=TrajectorySafetyGateV1())
            cell=observation/'cells'/record['cell_key'];reused+=int(exists)
            assert record['source_episode_id']==source and record['noise_name']==n
            cells.append(dict(record,source_cell=str(cell),record_sha256=sha(cell/'record.json'),score_card_path=str(score_path),old16=source in old16))
    assert len(cells)==64*len(noise_names)
    report=dict(schema='PI0253_FULL_RETENTION_SCREEN_REPORT_V1',status='SCREEN_COMPLETE',checkpoint_step=args.step,selected_checkpoint='NONE',
                score_card_sha256=score_sha,score_card=score,hf_revision=restore['revision'],noise_names=noise_names,cells=cells,
                group_results={n:m64.group([r for r in cells if r['noise_name']==n],64,True) for n in noise_names},
                old16_group_results={n:m64.group([r for r in cells if r['noise_name']==n and r['old16']],16,True) for n in noise_names},
                counts=dict(logical_cells=len(cells),new_queries=len(cells)-reused,reused_first_valid=reused,head_forwards=4*(len(cells)-reused)))
    write(report_path,report);print(json.dumps({k:v for k,v in report.items() if k not in ('cells','score_card')}),flush=True)


if __name__=='__main__':main()
