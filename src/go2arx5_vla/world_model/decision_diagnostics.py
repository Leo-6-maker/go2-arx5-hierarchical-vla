"""Frozen-model DEV diagnostics. No optimizer, reward fitting, or native integration."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from .storage import digest, write_json
from .paired_replay import read
from .replay import tensor_state, aligned, SCHEMA, SLICES


def pack(history, rows):
    if len(history)!=8 or len(rows)!=16:raise ValueError('Complete real burn8/predict16 branch required')
    if not np.array_equal(tensor_state(history[-1]['state_after']),tensor_state(rows[0]['state_before'])):
        raise ValueError('Common history and actual branch do not join')
    seq=history+rows
    for row in seq:aligned(row)
    context=seq[0]['context']
    return dict(state=np.stack([tensor_state(seq[0]['state_before'])]+[tensor_state(r['state_after'])for r in seq]),
        reference=np.stack([r['reference_compiled']for r in seq]),events=np.array([[r['state_after'][k]for k in SCHEMA['events']]for r in seq]),
        valid=np.ones(24,dtype=bool),done=np.array([r['terminal']for r in seq]),timeout=np.array([r['truncated']for r in seq]),
        clock=np.array([[r['t0']/32,r['native_progress']['point']*.05]for r in seq]),
        context=np.array([*context['goal_xyz'],context['table_top_m'],context['cube_edge_m']]),
        raw=np.stack([r['action_raw']for r in rows[:8]]))


def export_onset(root, manifest, output):
    root,output=Path(root),Path(output)
    if digest(manifest)!='005d3615e95b2d74c728201ad402f13d426ef9e2fc12bd6e1e46e72701d71ca1':
        raise ValueError('Exact original four-DEV probe manifest required')
    for entry in json.loads(Path(manifest).read_text()):
        if digest(root/entry['relative'])!=entry['sha256']:raise ValueError('Original DEV stream changed')
    report=json.loads((root/'RESULT.json').read_text())
    if report['protocol']!='WM02_S_DEV_ONSET_PROBE_V2':raise ValueError('Original DEV native probe required')
    examples=[];meta=[]
    for scene in report['scenes']:
        i=scene['scene']
        if i not in range(16,20):raise ValueError('Only original four diagnostic DEV parents')
        folder=root/f'scene-{i:02d}';history=read(folder/'COMMON_HISTORY.pkl.gz')['rows']
        # Order frozen before inference; names and original position are not model inputs.
        names=sorted(scene['candidates'],key=lambda n:hashlib.sha256(f'WM02_DECISION_V3/{i}/{n}'.encode()).digest())
        for name in names:
            rows=read(folder/(name+'.pkl.gz'))['rows'];examples.append(pack(history,rows));meta.append((i,name))
    if len(examples)!=20:raise ValueError('All twenty original DEV candidates retained')
    arrays={k:np.stack([e[k]for e in examples])for k in examples[0]}
    arrays.update(scene=np.array([x[0]for x in meta]),candidate=np.array([x[1]for x in meta]),
        partition=np.array(['DIAGNOSTIC_DEV']*20),anchor=np.array(['preclose']*20),
        purpose=np.array('WM02_PR349_FROZEN_DEV_DECISION_V1'),schema_json=np.array(json.dumps(SCHEMA)))
    output.mkdir(exist_ok=False)
    with (output/'DECISION.npz').open('xb')as f:np.savez_compressed(f,**arrays)
    result=dict(data_sha256=digest(output/'DECISION.npz'),original_stream_manifest_sha256=digest(manifest),
        source_native_sha=report['source_sha'],parents=[16,17,18,19],trajectories=20,new_physics=0,new_updates=0,
        candidate_order='SHA256 fixed per parent and candidate before inference; no name/index input',
        split='Original diagnostic DEV, never relabelled blind TEST')
    write_json(output/'AUDIT.json',result);return result


def frozen_model(binding, device='cpu'):
    import torch
    from .neural import WorldModel
    if digest(binding['path'])!=binding['sha256']:raise ValueError('Frozen checkpoint hash differs')
    c=torch.load(binding['path'],map_location='cpu',weights_only=True)
    # Registry/spec JSON carries sequences as lists; original checkpoint schema uses tuples.
    if (c['schema']!='WM02_ENGINEERING_CHECKPOINT_V1'
            or json.dumps(c['provenance'],sort_keys=True)!=json.dumps(binding['provenance'],sort_keys=True)
            or c['step']!=200):
        raise ValueError('Frozen endpoint/provenance differs')
    model=WorldModel(**c['model_config']);model.load_state_dict(c['model']);model.to(device).eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


def rule_scores(state,raw):
    """Deployable pre-close rule: close while retaining measured object/TCP alignment."""
    target_xy=state[SLICES['tcp_position']][:2]+raw[:,:,:2].sum(1)
    object_xy=state[SLICES['object_position']][:2]
    return -raw[:,-1,6]-10*np.linalg.norm(target_xy-object_xy,axis=-1)


def choices(predicted, truth, names, rule, scene):
    selected=int(np.argmax(predicted));random=int.from_bytes(hashlib.sha256(f'WM02_DECISION_RANDOM_V3/{scene}'.encode()).digest()[:4],'big')%len(names)
    out={}
    for policy,index in (('WM',selected),('same_observation_rule',int(np.argmax(rule))),('fixed_random',random)):
        out[policy]=dict(candidate=str(names[index]),actual_Q=float(truth[index]),regret=float(max(truth)-truth[index]))
    out['oracle_Q']=float(max(truth));out['nominal_Q']=float(truth[list(names).index('nominal')]);return out


def evaluate(spec, output):
    import torch
    from .contact_training import predict, errors, select
    from .neural_training import parameter_hash
    output=Path(output);start=time.monotonic();cpu=time.process_time()
    torch.set_num_threads(len(spec['cpus']))
    if digest(spec['replay'])!=spec['replay_sha256']:raise ValueError('Frozen decision replay differs')
    with np.load(spec['replay'],allow_pickle=False)as f:
        if str(f['purpose'])!='WM02_PR349_FROZEN_DEV_DECISION_V1':raise ValueError('Explicit original DEV decision replay required')
        b={k:torch.tensor(f[k],dtype=torch.float32)for k in ('state','reference','events','valid','done','timeout','clock','context','raw')}
        meta={k:f[k].copy()for k in ('scene','candidate')}
    models={name:frozen_model(binding)for name,binding in spec['models'].items()};result={}
    for name,model in models.items():
        pin=parameter_hash(model);blind=name=='no_future_action'
        with torch.no_grad():
            pred,event,_=predict(model,b['state'][:,:8],b['state'][:,8],b['reference'][:,:8],b['context'],b['clock'][:,:8],b['clock'][:,8],b['raw'],blind=blind)
            # Privileged diagnostic: actual current state at each future step, never the next label.
            hidden=model.burn(b['state'][:,:8],b['reference'][:,:8],b['context'],b['clock'][:,:8]);teacher_events=[]
            for i in range(8,24):
                ref=torch.zeros_like(b['reference'][:,i])if blind else b['reference'][:,i]
                _,e,_,hidden=model.step(b['state'][:,i],ref,b['context'],b['clock'][:,i],hidden);teacher_events.append(e.sigmoid())
            teacher_events=torch.stack(teacher_events,1)
        # Durable predictions precede future-label scoring and candidate comparisons.
        np.savez_compressed(output/(name+'_PREDICTIONS.npz'),state=pred.numpy(),events=event.numpy(),teacher_forced_events=teacher_events.numpy())
        rows=[];rank={};horizon_results={}
        for h in (8,16):
            horizon_results[str(h)]=errors(pred,event,b,h)
            horizon_results[str(h)]['teacher_forced_grasp_Brier']=float((teacher_events[:,:h,2]-b['events'][:,8:8+h,2]).square().mean())
            for scene in sorted(set(meta['scene'])):
                ids=np.flatnonzero(meta['scene']==scene);names=meta['candidate'][ids]
                rules=rule_scores(b['state'][ids[0],8].numpy(),b['raw'][ids].numpy())
                rank[f'{scene}/H{h}']=choices(event[ids,h-1,2].numpy(),b['events'][ids,8+h-1,2].numpy(),names,rules,int(scene))
                for j in ids:
                    rows.append(dict(scene=int(scene),candidate=str(meta['candidate'][j]),horizon=h,
                        actual_grasp=bool(b['events'][j,8+h-1,2]),predicted_grasp_probability=float(event[j,h-1,2]),
                        teacher_forced_grasp_probability=float(teacher_events[j,h-1,2]),
                        object_endpoint_error_m=float((pred[j,h-1,SLICES['object_position']]-b['state'][j,8+h,SLICES['object_position']]).norm()),
                        tcp_endpoint_error_m=float((pred[j,h-1,SLICES['tcp_position']]-b['state'][j,8+h,SLICES['tcp_position']]).norm())))
        if parameter_hash(model)!=pin:raise RuntimeError('Frozen evaluation changed model tensors')
        result[name]=dict(checkpoint_sha256=spec['models'][name]['sha256'],parameters_unchanged=True,horizons=horizon_results,candidates=rows,choices=rank)
    result=dict(status='FROZEN_DEV_ONSET_EVALUATED',execution_source_sha=spec['source_sha'],data_sha256=spec['replay_sha256'],models=result,
        WM_updates=0,Actor_updates=0,new_physics=0,CUDA_initialized=torch.cuda.is_initialized(),CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start,
        formal_reward=False,DEV_not_blind=True,teacher_forcing='Privileged actual current state/history diagnostic, not deployed multi-step forecast')
    write_json(output/'RESULT.json',result);return result
