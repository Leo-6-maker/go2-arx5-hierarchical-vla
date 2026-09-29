"""Fixed PI0254 matrix over the original qualified S4 query/decoder/gate."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import sys

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0254'
sys.path[:0]=[str(ROOT/'execution'),str(BASE/'PI0253/runtime')]
import pi0253_full_screen_v2 as inherited


def plan():
    cohorts={name:inherited.read(ROOT/f'design/{name}.json')['rows'] for name in ('O64','E64','P64')}
    rows=[]
    for point,groups,noises in [('G2000',('E64','P64'),('B0','B1')),('FULL10000',('P64',),('B0',))]+[
            (f'{arm}_{step}',('O64','E64','P64'),('B0','B1')) for arm in ('U576','R576') for step in (1152,4608)]:
        for group in groups:
            for source in cohorts[group]:
                for noise in noises:rows.append(dict(point=point,cohort=group,noise_name=noise,**source))
    assert len(rows)==1856 and len({(r['point'],r['source_episode_id'],r['noise_name']) for r in rows})==1856
    return rows


def prepare():
    import requests
    from huggingface_hub import hf_hub_url
    receipt=inherited.read(ROOT/'bindings/G2000_FULL_STREAM_RECEIPT.json')
    assert receipt['revision']=='efecea485ae77000599d19171b02491f8d514f42'
    assert receipt['tree_sha256']=='cfbfeb54bea2b5ac266047dfd764cfa6b041a8a77b9688737ea70b56935d5226'
    entries={Path(row['path']).name:row for row in receipt['files'] if Path(row['path']).name in (
                'config.json','model.safetensors.index.json','model-00001-of-00002.safetensors','model-00002-of-00002.safetensors')}
    assert len(entries)==4
    view=ROOT/'models/G2000';view.mkdir(parents=True,exist_ok=True)
    for name,row in entries.items():
        path=view/name
        if not path.exists():
            temporary=path.with_suffix('.partial');h=hashlib.sha256();size=0
            with requests.get(hf_hub_url(receipt['repo_id'],filename=row['path'],revision=receipt['revision']),stream=True,timeout=(30,180)) as response:
                response.raise_for_status()
                with temporary.open('wb') as stream:
                    for block in response.iter_content(8<<20):stream.write(block);h.update(block);size+=len(block)
                    stream.flush();os.fsync(stream.fileno())
            assert h.hexdigest()==row['sha256'] and size==row['bytes'];os.replace(temporary,path)
        assert path.stat().st_size==row['bytes'] and inherited.sha(path)==row['sha256']
        print(json.dumps(dict(status='PINNED_G_MODEL_FILE',path=name)),flush=True)
    restore=dict(schema='PI0254_PINNED_MODEL_ONLY_V1',revision=receipt['revision'],repo_id=receipt['repo_id'],
                 checkpoint_tree_sha256=receipt['tree_sha256'],optimizer_restored=False,
                 prefix=str(Path(next(iter(entries.values()))['path']).parent),
                 files=[{**row,'path':name} for name,row in entries.items()])
    inherited.write(view/'pi0245_inference_restore.json',restore)
    old=inherited.read(inherited.OLD);score=old['score_card']
    import pi0247_d1_screen as d1
    import pi0247_m64_screen as m64
    card=dict(schema='PI0254_FIXED_EVAL_CARD_V1',authority_commit='7e5a4aae69912ba8e593d50d313f288abc16a914',
         deployment_selected_checkpoint='NONE',planned_queries=1856,query_cap=1920,inference_head_cap=7680,
         runner_sha256=inherited.sha(Path(d1.__file__)),orchestrator_sha256=inherited.sha(Path(__file__)),
         qualification_code_sha256=inherited.sha(Path(m64.__file__)),
         noise_manifest_sha256=inherited.sha(ROOT/'noise/manifest.json'),
         codec_conditions_sha256=inherited.sha(ROOT/'cpu_codec_conditions.json'),
         original_O64_score_card=score,models=dict(G2000=str(view),FULL10000=str(BASE/'PI0253/models/FULL10000/inference_view')),
         inherited_thresholds=dict(reject_count=0,median_xyz_m=.020,p95_xyz_m=.035,median_so3_rad=.050),
         new_research_thresholds=dict(E64_median_xyz_ratio=.8,P64_median_xyz_ratio=.9,no_worse_tolerance=1e-6),
         evaluation_steps=[1152,4608],rows=plan())
    inherited.write(ROOT/'eval_card.json',card)
    print('PASS_SEALED_FIXED_EVAL_MATRIX_NOT_QUERIED',flush=True)


def context():
    old,_,old16,old_noise,old_packets,ctx=inherited.context()
    noise_manifest=inherited.read(ROOT/'noise/manifest.json')
    noise_by_id={(r['source_episode_id'],r['noise_name']):r for r in noise_manifest['rows']}
    conditions={r['source_episode_id']:r for r in inherited.read(ROOT/'cpu_codec_conditions.json')['rows']}
    return old,old16,old_noise,old_packets,ctx,noise_by_id,conditions


def bind_cell(row,ctx,old16,old_noise,old_packets,noise_by_id,conditions):
    import numpy as np
    source,noise=row['source_episode_id'],row['noise_name']
    if row['cohort']=='O64':
        key=('old16_B0' if source in old16 else 'new48_B0') if noise=='B0' else 'B1'
        packet=old_packets[key];value=old_noise[key]
    else:
        record=conditions[source];parent=ROOT/'conditions'/f"source-{row['episode_index']:06d}"
        assert inherited.sha(parent/'tensors.npz')==record['source_tensor_sha256']
        assert inherited.sha(parent/'condition.pt')==record['condition_sha256']
        ctx['source_cells'][source]={n:dict(cell_path=str(parent),condition_identity=record['condition_identity']) for n in ('B0','B1')}
        packet=noise_by_id[source,noise]
        assert inherited.sha(Path(packet['path']))==packet['file_sha256']
        with np.load(packet['path'],allow_pickle=False) as payload:value=payload['float32'].copy()
    ctx['noise'][noise]=value;ctx['facts']['noise_files_sha256'][noise]=packet['file_sha256']
    return value,packet


def check():
    import numpy as np
    import torch
    import pi0242_train_infer_bridge as bridge
    card=inherited.read(ROOT/'eval_card.json');rows=plan()
    assert rows==card['rows'] and card['planned_queries']==1856 and card['query_cap']==1920
    _,old16,old_noise,old_packets,ctx,noises,conditions=context()
    assert len(noises)==256 and len(conditions)==640
    for row in [r for r in rows if r['point']=='G2000']:
        value,packet=bind_cell(row,ctx,old16,old_noise,old_packets,noises,conditions)
        assert value.shape==(1,40,132) and value.dtype==np.float32 and np.isfinite(value).all()
        with np.load(packet['path'],allow_pickle=False) as data:
            assert np.array_equal(data['bfloat16_bits'],torch.from_numpy(value).to(torch.bfloat16).view(torch.uint16).numpy())
        parent=ROOT/'conditions'/f"source-{row['episode_index']:06d}"
        policy=torch.load(parent/'condition.pt',weights_only=False,map_location='cpu')
        assert bridge._canonical_hash(bridge._leaf_inventory(policy))==conditions[row['source_episode_id']]['condition_identity']
        assert not any(any(word in leaf.lower() for word in bridge.FORBIDDEN_INPUT_TOKENS) for leaf in bridge._leaf_inventory(policy))
    inherited.write(ROOT/'eval_cpu_receipt.json',dict(status='PASS_FIXED_ROUTING_CONDITION_AND_NOISE_CPU',
          eval_card_sha256=inherited.sha(ROOT/'eval_card.json'),code_sha256=inherited.sha(Path(__file__)),
          fixed_cells=len(rows),baseline_cells=320,new_model_queries=0,new_optimizer_updates=0))
    print('PASS_FIXED_ROUTING_CONDITION_AND_NOISE_CPU',flush=True)


def run(point,view):
    import fcntl
    import numpy as np
    import torch
    import pi0247_d1_screen as d1
    import pi0242_train_infer_bridge as bridge
    import pi0243_staged_learning_to_sim as stage43
    import pi0228_train_calib_checkpoint_selection as selector
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    card=inherited.read(ROOT/'eval_card.json')
    assert inherited.sha(Path(__file__))==card['orchestrator_sha256'] and inherited.sha(ROOT/'noise/manifest.json')==card['noise_manifest_sha256']
    assert inherited.sha(ROOT/'cpu_codec_conditions.json')==card['codec_conditions_sha256']
    lock=(ROOT/'query.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=[r for r in card['rows'] if r['point']==point];assert len(rows) in (64,256,384)
    old,old16,old_noise,old_packets,ctx,noise_by_id,conditions=context()
    view=Path(view or card['models'][point]);files=ctx['p1'].verify_model_files(view)
    score={**card,'rows':rows,'point':point,'model_revision':files['revision'],'model_manifest_sha256':files['manifest_sha256']}
    directory=ROOT/'eval'/point;inherited.write(directory/'score_card.json',score);score_sha=inherited.sha(directory/'score_card.json')
    if (directory/'screen_report.json').exists():return
    from transformers import AutoModel
    model=AutoModel.from_pretrained(view,local_files_only=True,trust_remote_code=True).eval().to(device='cuda:0',dtype=torch.bfloat16)
    assert model.action_head.num_inference_timesteps==4
    counts=dict(queries=len((ROOT/'query_attempts.jsonl').read_text().splitlines()) if (ROOT/'query_attempts.jsonl').exists() else 0,
                heads=len((ROOT/'head_forwards.jsonl').read_text().splitlines()) if (ROOT/'head_forwards.jsonl').exists() else 0)
    original=bridge._stock_query;active={}
    def durable(path,row):
        with path.open('a') as stream:stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
    def count_head(module,args):
        assert counts['heads']<7680
        counts['heads']+=1;active['heads']+=1
        durable(ROOT/'head_forwards.jsonl',dict(head_ordinal=counts['heads'],query_ordinal=counts['queries'],point=point,**active))
    def query(*args,**kwargs):
        assert counts['queries']<1920
        counts['queries']+=1;active['heads']=0
        durable(ROOT/'query_attempts.jsonl',dict(query_ordinal=counts['queries'],point=point,model_revision=files['revision'],**active))
        handle=model.action_head.model.register_forward_pre_hook(count_head)
        try:return original(*args,**kwargs)
        finally:handle.remove()
    bridge._stock_query=query
    cells=[]
    try:
        for row in rows:
            source,noise=row['source_episode_id'],row['noise_name'];active.clear();active.update(source_episode_id=source,noise_name=noise)
            bind_cell(row,ctx,old16,old_noise,old_packets,noise_by_id,conditions)
            observation=directory/'observations'/f"source-{row['episode_index']:06d}"
            record=d1.run_one(args=SimpleNamespace(output_root=observation),card=score,card_sha=score_sha,context=ctx,
                       model_name=point,runtime={'id':'S4','steps':4,'inactive':'stock'},source=source,noise_name=noise,
                       model=model,model_files=files,bridge=bridge,selector=selector,stage43=stage43,gate=TrajectorySafetyGateV1())
            cells.append(dict(record,cohort=row['cohort'],phase=row['phase'],task=row['target_family'],
                         source_rollout_id=row['source_rollout_id'],old16=source in old16,source_cell=str(observation/'cells'/record['cell_key'])))
    finally:bridge._stock_query=original
    inherited.write(directory/'screen_report.json',dict(schema='PI0254_SCREEN_REPORT_V1',point=point,status='SCREEN_COMPLETE',
                   score_card_sha256=score_sha,model_revision=files['revision'],cells=cells,global_counts=counts,
                   deployment_selected_checkpoint='NONE'))


def cpu(point):
    import numpy as np
    import torch
    import pi0242_train_infer_bridge as bridge
    import pi0243_staged_learning_to_sim as stage43
    import pi0228_train_calib_checkpoint_selection as selector
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat,rot6d_rows_to_matrix,geodesic_angle
    _,old16,old_noise,old_packets,ctx,noise_by_id,conditions=context()
    report=inherited.read(ROOT/f'eval/{point}/screen_report.json');rows=plan();replayed=[]
    expected={(r['source_episode_id'],r['noise_name']):r for r in rows if r['point']==point}
    assert len(expected)==len(report['cells'])
    for row in report['cells']:
        key=row['source_episode_id'],row['noise_name'];definition=expected.pop(key)
        path=Path(row['source_cell']);record=inherited.read(path/'record.json')
        assert record['run_card_sha256']==report['score_card_sha256'] and record['model_revision']==report['model_revision']
        tensor=path/('tensors.npz' if record['status']=='VALID' else 'invalid_output.npz')
        assert inherited.sha(tensor)==record['tensor_sha256'] and inherited.sha(path/'attempt.json')==record['attempt_sha256']
        value,packet=bind_cell(definition,ctx,old16,old_noise,old_packets,noise_by_id,conditions)
        assert packet['file_sha256']==record['noise_file_sha256']
        if record['status']!='VALID':replayed.append(dict(row,status=record['status']));continue
        with np.load(tensor,allow_pickle=False) as data:
            assert np.array_equal(data['noise_source_float32'],value)
            assert np.array_equal(data['noise_initial'],torch.from_numpy(value).to(torch.bfloat16).float().numpy())
            assert data['trace_x'].shape==data['trace_v'].shape==(4,1,40,132)
            assert np.array_equal(data['trace_x'][0],data['noise_initial'])
            raw,state,expert=data['raw_native'],data['state'],data['expert']
            assert np.array_equal(raw,data['raw_scored']) and not record['m_path']['applicable']
            decoded=bridge._decode_physical(ctx['processor'],ctx['embodiment'],raw,state)
            assert np.max(np.abs(decoded-data['decoded']))<=1e-7
        metrics=stage43._screen_metrics(decoded[0].astype(np.float64),expert.astype(np.float64),None)
        for k,v in metrics.items():
            if isinstance(v,(float,int)) and k in record['metrics']:assert abs(float(v)-record['metrics'][k])<=1e-10
        q=matrix_to_quat(rot6d_rows_to_matrix(state[None,3:9]))[0]
        gate=TrajectorySafetyGateV1()
        model_gate=gate.evaluate(selector._build_chunk(decoded,chunk_id='pi0254-cpu'),now=0.,achieved_position=state[:3],achieved_orientation=q)
        expert_gate=gate.evaluate(selector._build_chunk(expert[None],chunk_id='pi0254-cpu-expert'),now=0.,achieved_position=state[:3],achieved_orientation=q)
        assert model_gate.decision.value==record['gate_decision']
        assert [i.code for i in model_gate.issues]==[i['code'] for i in record['gate']['issues']]
        residual=decoded[0,:,:3].astype(float)-expert[:,:3].astype(float);offset=residual.mean(0)
        rotation=geodesic_angle(rot6d_rows_to_matrix(decoded[0,:,3:9]),rot6d_rows_to_matrix(expert[:,3:9]))
        replayed.append(dict(row,expert_gate=expert_gate.as_dict(),expert_reject=expert_gate.decision.value=='REJECT',
                            model_only_reject=model_gate.decision.value=='REJECT' and expert_gate.decision.value!='REJECT',
                            mean_offset_m=float(np.linalg.norm(offset)),centered_shape_rmse_m=float(np.sqrt(np.mean(np.sum((residual-offset)**2,axis=1)))),
                            offset_xyz_m=offset.tolist(),first_xyz_error_m=float(np.linalg.norm(residual[0])),last_xyz_error_m=float(np.linalg.norm(residual[-1])),
                            first_so3_error_rad=float(rotation[0]),last_so3_error_rad=float(rotation[-1])))
    assert not expected
    inherited.write(ROOT/f'eval/{point}/cpu_replay.json',dict(status='PASS_INDEPENDENT_CPU_REPLAY',point=point,rows=replayed,
                   source_report_sha256=inherited.sha(ROOT/f'eval/{point}/screen_report.json'),new_model_queries=0))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--cpu-check',action='store_true');parser.add_argument('--point');parser.add_argument('--model-view');args=parser.parse_args()
    if args.prepare:return prepare()
    if args.cpu_check:return check()
    if args.cpu:return cpu(args.point)
    return run(args.point,args.model_view)


if __name__=='__main__':main()
