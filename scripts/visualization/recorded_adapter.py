"""Bound PI0257's finite cohort before opening any per-cell score report."""
import hashlib,json,shutil,sys,os,subprocess,time
from pathlib import Path
from eef_video_workflow import read,write,sha,DEFAULT_ROOT,scientific_key,admit_query,saved_path
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5')

def plan(root=DEFAULT_ROOT):
    selected=[]; sources=[]
    for cohort in ('O64','P64'):
        path=BASE/f'PI0254/design/{cohort}.json'; table=read(path)
        sources.append(dict(path=str(path),sha256=sha(path)))
        for family in 'abcd':
            candidates=[r for r in table['rows'] if r['target_family'].lower()==family]
            candidates.sort(key=lambda r:hashlib.sha256(f"PI0257_DISPLAY_V1|{cohort}|{r['source_episode_id']}".encode()).hexdigest())
            for row in candidates[:2]:
                assert row['split']=='TRAIN'
                selected.append(dict(cohort=cohort,selection_sha256=hashlib.sha256(f"PI0257_DISPLAY_V1|{cohort}|{row['source_episode_id']}".encode()).hexdigest(),**row))
    assert len(selected)==16
    matrix=dict(schema='PI0257_FIXED_DISPLAY_MATRIX_V1',selection='two per A-D stratum by SHA256(PI0257_DISPLAY_V1|cohort|full_source_id), no score inputs',metadata_sources=sources,windows=selected,
        models={'G2000':'efecea485ae77000599d19171b02491f8d514f42','U4608':'a90b3d3c751a51dff5ecfda1e9254c9f6d786664','FULL10000':'7bafb99f58f3ed9dea66e85fff8c8b79a7c9785d'},noises=['B0','B1'],logical_cells=96,query_attempt_cap=104,valid_query_cap=96,invalid_no_result_cap=8,head_cap=416)
    path=root/'delivery/display_matrix.json'
    if path.exists(): assert read(path)==matrix
    else: write(path,matrix)
    return matrix

def inspect(root=DEFAULT_ROOT):
    matrix=plan(root)
    for path in [BASE/'PI0252/screen/G2000/screen_report.json',BASE/'PI0253/screen/FULL10000/screen_report.json',BASE/'PI0254/eval/G2000/screen_report.json',BASE/'PI0254/eval/FULL10000/screen_report.json',BASE/'PI0254/eval/U576_4608/screen_report.json']:
        report=read(path); wanted={x['source_episode_id'] for x in matrix['windows']}
        selected=[r for r in report['cells'] if r['source_episode_id'] in wanted]
        print(json.dumps({'report':str(path),'top':{k:v for k,v in report.items() if not isinstance(v,(list,dict))},'selected_cells':len(selected),'sample':selected[0] if selected else None}),flush=True)
        if selected:
            import numpy as np
            cell=Path(selected[0]['source_cell']);print(json.dumps({'record':read(cell/'record.json')}),flush=True)
            with np.load(cell/'tensors.npz',allow_pickle=False) as z:print(json.dumps({'arrays':{k:[list(z[k].shape),str(z[k].dtype)] for k in z.files}}),flush=True)

def copy_bound(src,dst,expected=None):
    src,dst=Path(src),Path(dst); digest=sha(src)
    if expected: assert digest==expected,str(src)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists(): shutil.copyfile(src,dst)
    assert sha(dst)==digest
    return dict(source=str(src),path=str(dst),sha256=digest,bytes=dst.stat().st_size)

def stage(root=DEFAULT_ROOT):
    import numpy as np
    matrix=plan(root); index={}; bindings=[]
    reports=[('G2000','O64',BASE/'PI0252/screen/G2000'),('FULL10000','O64',BASE/'PI0253/screen/FULL10000'),
        ('G2000','P64',BASE/'PI0254/eval/G2000'),('FULL10000','P64',BASE/'PI0254/eval/FULL10000'),('U4608',None,BASE/'PI0254/eval/U576_4608')]
    for model,cohort,parent in reports:
        report=read(parent/'screen_report.json'); card=parent/'score_card.json'
        assert (report.get('hf_revision') or report.get('model_revision'))==matrix['models'][model]
        for r in report['cells']:
            if any(w['source_episode_id']==r['source_episode_id'] and (cohort is None or w['cohort']==cohort) for w in matrix['windows']):
                index[model,r['source_episode_id'],r['noise_name']]=(r,parent,card)
        bindings += [copy_bound(parent/name,root/'delivery/provenance'/f'{model}_{cohort or "all"}_{name}') for name in ('screen_report.json','score_card.json','cpu_replay.json')]
    cells=[]; inputs={}
    for w in matrix['windows']:
        source=w['source_episode_id']; wid=f"{w['cohort']}_{w['episode_index']:06d}"
        for noise in matrix['noises']:
            comparisons=[]
            for model in matrix['models']:
                identity=f'{wid}_{model}_{noise}'; item=dict(id=identity,window=wid,model=model,noise=noise,source_episode_id=source,cohort=w['cohort'])
                found=index.get((model,source,noise))
                origin='REUSED_EXACT_ARCHIVED'
                if not found:
                    own=root/'delivery/recorded/cells'/identity
                    if not (own/'record.json').exists():cells.append(dict(item,status='MISSING_ARCHIVED_OUTPUT'));continue
                    r=dict(read(own/'record.json'),source_cell=str(own));found=(r,own,root/'delivery/new_inference_card.json');origin='NEW_RECORDED_INPUT_INFERENCE'
                r,parent,card=found; src=Path(r['source_cell']); record=read(src/'record.json')
                assert record['model_revision']==matrix['models'][model] and record['run_card_sha256']==sha(card)
                assert record['status']=='VALID' and record['runtime']=='S4' and record['steps']==4 and not record['m_path']['applicable']
                assert record['condition_identity']==record['condition_identity_recomputed']
                dst=root/'delivery/recorded/cells'/identity
                for name,expected in [('record.json',None),('attempt.json',record['attempt_sha256']),('tensors.npz',record['tensor_sha256'])]: bindings.append(copy_bound(src/name,dst/name,expected))
                inp=Path(record['condition_source_cell']); idir=root/'delivery/recorded/inputs'/wid
                for name,expected in [('record.json',None),('condition.pt',record['condition_sha256']),('tensors.npz',record['source_tensor_sha256'])]: bindings.append(copy_bound(inp/name,idir/name,expected))
                inputs[wid]=str(idir)
                with np.load(dst/'tensors.npz',allow_pickle=False) as z:
                    noise_sha=hashlib.sha256(z['noise_source_float32'].tobytes()).hexdigest();assert noise_sha==record['noise_source_sha256']
                    assert np.array_equal(z['raw_native'],z['raw_scored']) and np.array_equal(z['trace_x'][0],z['noise_initial'])
                key=[record[k] for k in ('condition_identity','source_tensor_sha256','noise_source_sha256','runtime','steps','runner_sha256')]
                comparisons.append(key)
                cache_key=hashlib.sha256(json.dumps([matrix['models'][model],record['model_manifest_sha256'],*key],sort_keys=True).encode()).hexdigest()
                cells.append(dict(item,status=origin,path=str(dst),historical_record=record,cache_key=cache_key))
            assert comparisons and all(k==comparisons[0] for k in comparisons),'NON_PAIRED_CONTRACT: '+wid+' '+noise
    assert len(cells)==96
    result=dict(status='COMMON_CONTRACT_BOUND',matrix_sha256=sha(root/'delivery/display_matrix.json'),cells=cells,inputs=inputs,bindings=bindings,reused=sum(c['status']=='REUSED_EXACT_ARCHIVED' for c in cells),missing=sum(c['status']=='MISSING_ARCHIVED_OUTPUT' for c in cells))
    write(root/'delivery/completeness.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cells','inputs','bindings')}),flush=True)
    return result

def context():
    sys.path[:0]=[str(BASE/'PI0254/execution'),str(BASE/'PI0253/runtime')]
    import pi0254_eval as existing
    card=read(BASE/'PI0254/eval_card.json')
    assert sha(Path(existing.__file__))==card['orchestrator_sha256']
    import torch
    torch.set_num_threads(2)
    result=existing.context()
    return existing,result

def cpu(root=DEFAULT_ROOT):
    import numpy as np
    existing, (_,old16,old_noise,old_packets,ctx,noise_by_id,conditions)=context()
    import torch
    import pi0242_train_infer_bridge as bridge
    import pi0243_staged_learning_to_sim as scorer
    import pi0228_train_calib_checkpoint_selection as selector
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat,rot6d_rows_to_matrix
    complete=read(root/'delivery/completeness.json'); matrix=plan(root); evidence=[];paired={}
    model_bindings={}
    for model,path in [('G2000',BASE/'PI0254/bindings/G2000_FULL_STREAM_RECEIPT.json'),('FULL10000',BASE/'PI0254/bindings/FULL10000_FULL_STREAM_RECEIPT.json'),('U4608',BASE/'PI0254/models/U576_4608.binding.json')]:
        bound=read(path)
        tree=bound['tree_sha256'] if model!='U4608' else bound['full_content_readback']['source_tree_sha256']
        model_bindings[model]=dict(revision=matrix['models'][model],checkpoint_tree_sha256=tree,receipt=copy_bound(path,root/'delivery/provenance'/f'{model}_checkpoint_binding.json'))
    processor_hash=sha(BASE/'PI0250/inputs/frozen_v1/processor_base/processor_config.json')
    statistics_hash=sha(BASE/'PI0250/inputs/frozen_v1/processor_base/statistics.json')
    for name in ('processor_config.json','statistics.json'):
        copy_bound(BASE/'PI0250/inputs/frozen_v1/processor_base'/name,root/'delivery/provenance'/('C_T_'+name))
    decoder_hash=sha(Path(bridge.__file__))
    windows={w['source_episode_id']:w for w in matrix['windows']}
    for cell in complete['cells']:
        if 'path' not in cell:continue
        folder=saved_path(root/'delivery',cell['path']); r=read(folder/'record.json'); w=windows[cell['source_episode_id']]
        assert r['model_revision']==matrix['models'][cell['model']]
        pair_key=(cell['source_episode_id'],cell['noise'])
        contract=[r[k] for k in ('condition_identity','source_tensor_sha256','noise_source_sha256','runtime','steps','runner_sha256')]
        if pair_key in paired:assert paired[pair_key]==contract
        else:paired[pair_key]=contract
        cell['cache_key']=scientific_key(model_bindings[cell['model']]['checkpoint_tree_sha256'],processor_hash,statistics_hash,decoder_hash,cell['source_episode_id'],contract)
        value,packet=existing.bind_cell(dict(w,noise_name=cell['noise']),ctx,old16,old_noise,old_packets,noise_by_id,conditions)
        assert sha(folder/'tensors.npz')==r['tensor_sha256'] and r['noise_file_sha256']==packet['file_sha256']
        inp=saved_path(root/'delivery',complete['inputs'][cell['window']]);policy=torch.load(inp/'condition.pt',map_location='cpu',weights_only=False)
        assert bridge._canonical_hash(bridge._leaf_inventory(policy))==r['condition_identity']
        with np.load(folder/'tensors.npz',allow_pickle=False) as z:
            assert np.array_equal(z['noise_source_float32'],value) and np.array_equal(z['noise_initial'],torch.from_numpy(value).bfloat16().float().numpy())
            decoded=bridge._decode_physical(ctx['processor'],ctx['embodiment'],z['raw_native'],z['state'])
            assert np.max(np.abs(decoded-z['decoded']))<=1e-7
            metrics=scorer._screen_metrics(decoded[0].astype(float),z['expert'].astype(float),None)
            assert all(abs(v-r['metrics'][k])<=1e-10 for k,v in metrics.items())
            q=matrix_to_quat(rot6d_rows_to_matrix(z['state'][None,3:9]))[0]
            gate=TrajectorySafetyGateV1().evaluate(selector._build_chunk(decoded,chunk_id='pi0257-cpu'),now=0.,achieved_position=z['state'][:3],achieved_orientation=q)
            assert gate.decision.value==r['gate_decision'] and [x.code for x in gate.issues]==[x['code'] for x in r['gate']['issues']]
        evidence.append(dict(id=cell['id'],source_episode_id=cell['source_episode_id'],model=cell['model'],cohort=cell['cohort'],noise=cell['noise'],status=cell['status'],metrics=metrics,gate=gate.decision.value,metric_origin='INDEPENDENT_CPU_REPLAY_OF_SAVED_ARRAYS',new_physics=0))
    receipt=dict(status='PASS_DECODE_METRICS_GATE_INPUT_NOISE_CPU_REPLAY',cells=evidence,decoder_source_sha256=decoder_hash,statistics_sha256=statistics_hash,scorer_source_sha256=sha(Path(scorer.__file__)),processor_config_sha256=processor_hash,new_model_queries=0)
    write(root/'delivery/recorded_cpu_qa.json',receipt);print(json.dumps(dict(status=receipt['status'],cells=len(evidence))),flush=True)
    complete.update(missing=sum('path' not in c for c in complete['cells']),reused=sum(c['status']=='REUSED_EXACT_ARCHIVED' for c in complete['cells']),new=sum(c['status']=='NEW_RECORDED_INPUT_INFERENCE' for c in complete['cells']),model_bindings=model_bindings)
    write(root/'delivery/completeness.json',complete)

def infer(root=DEFAULT_ROOT,authorization=None):
    from eef_video_workflow import AUTH
    assert authorization==AUTH
    import fcntl
    lock=(root/'query.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    complete=read(root/'delivery/completeness.json');matrix=plan(root)
    assert complete['status']=='COMMON_CONTRACT_BOUND' and complete['matrix_sha256']==sha(root/'delivery/display_matrix.json')
    missing=[c for c in complete['cells'] if c['status']=='MISSING_ARCHIVED_OUTPUT']
    if not missing:print('NO_MISSING_CELLS_ZERO_NEW_CALLS');return
    assert all(c['model']=='FULL10000' and c['cohort']=='P64' and c['noise']=='B1' for c in missing) and len(missing)<=8
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    write(root/'delivery/inference_gpu_preflight.json',dict(observed=gpu,chosen_device=0,policy='one BF16 inference view; at least 18GiB free required; no other process modified',timestamp=time.time()))
    assert int(gpu.splitlines()[0].split(',')[2].strip())>=18*1024,'Insufficient verified free VRAM'
    existing,(_,old16,old_noise,old_packets,ctx,noise_by_id,conditions)=context()
    import torch
    import pi0247_d1_screen as d1
    import pi0242_train_infer_bridge as bridge
    import pi0243_staged_learning_to_sim as scorer
    import pi0228_train_calib_checkpoint_selection as selector
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from transformers import AutoModel
    from types import SimpleNamespace
    view=BASE/'PI0253/models/FULL10000/inference_view'
    files=ctx['p1'].verify_model_files(view)
    assert files['revision']==matrix['models']['FULL10000']
    restore=read(view/'pi0245_inference_restore.json');expected=read(BASE/'PI0254/bindings/FULL10000_FULL_STREAM_RECEIPT.json')
    assert restore['checkpoint_tree_sha256']==expected['tree_sha256']
    copy_bound(view/'pi0245_inference_restore.json',root/'delivery/provenance/FULL10000_inference_restore.json')
    card=dict(authorization=AUTH,matrix_sha256=sha(root/'delivery/display_matrix.json'),runner_sha256=sha(Path(d1.__file__)),model_revision=files['revision'],query_cap=104,head_cap=416)
    card_path=root/'delivery/new_inference_card.json'
    if card_path.exists():assert read(card_path)==card
    else:write(card_path,card)
    windows={w['source_episode_id']:w for w in matrix['windows']}
    ledger=root/'delivery/query_attempts.jsonl';heads=root/'delivery/head_calls.jsonl'
    def count(p):return len(p.read_text().splitlines()) if p.exists() else 0
    def append(p,r):
        with p.open('a') as f:f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
    model=AutoModel.from_pretrained(view,local_files_only=True,trust_remote_code=True).eval().to(device='cuda:0',dtype=torch.bfloat16)
    model.requires_grad_(False);assert model.action_head.num_inference_timesteps==4
    original=bridge._stock_query;active={}
    def hook(module,args):
        assert count(heads)<416;append(heads,dict(ordinal=count(heads)+1,cell=active['id']))
    def query(*args,**kwargs):
        admit_query(count(ledger),sum(c['status']=='NEW_RECORDED_INPUT_INFERENCE' for c in complete['cells']),count(heads))
        append(ledger,dict(ordinal=count(ledger)+1,cell=active['id'],revision=files['revision'],time=time.time()))
        h=model.action_head.model.register_forward_pre_hook(hook)
        try:
            with torch.inference_mode():return original(*args,**kwargs)
        finally:h.remove()
    bridge._stock_query=query
    try:
        for cell in missing:
            active['id']=cell['id'];w=windows[cell['source_episode_id']]
            existing.bind_cell(dict(w,noise_name=cell['noise']),ctx,old16,old_noise,old_packets,noise_by_id,conditions)
            ctx['source_cells'][cell['source_episode_id']][cell['noise']]['cell_path']=complete['inputs'][cell['window']]
            folder=root/'new_inference'/cell['id']
            r=d1.run_one(args=SimpleNamespace(output_root=folder),card=card,card_sha=sha(card_path),context=ctx,model_name='FULL10000',runtime=dict(id='S4',steps=4,inactive='stock'),source=cell['source_episode_id'],noise_name=cell['noise'],model=model,model_files=files,bridge=bridge,selector=selector,stage43=scorer,gate=TrajectorySafetyGateV1())
            assert r['status']=='VALID'
            original_folder=folder/'cells'/r['cell_key'];dst=root/'delivery/recorded/cells'/cell['id']
            for name in ('record.json','attempt.json','tensors.npz'):copy_bound(original_folder/name,dst/name)
            cell.update(status='NEW_RECORDED_INPUT_INFERENCE',path=str(dst),historical_record=r)
            write(root/'delivery/completeness.json',complete)
    finally:bridge._stock_query=original
    write(root/'delivery/query_counters.json',dict(query_attempts=count(ledger),head_calls=count(heads),valid_new_queries=sum(c['status']=='NEW_RECORDED_INPUT_INFERENCE' for c in complete['cells']),training=0,backward=0,optimizer=0,physics_steps=0,wbc_actor_calls=0))
    print('PASS_BOUNDED_MISSING_ONLY_INFERENCE',flush=True)

if __name__=='__main__':
    command=sys.argv[1] if len(sys.argv)>1 else 'stage'
    {'stage':stage,'cpu':cpu,'inspect':inspect}[command]()
