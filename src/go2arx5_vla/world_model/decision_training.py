"""Versioned V5 numeric replay and fixed200 matched source dynamics, no Actor reward."""
from pathlib import Path
import json
import numpy as np
from .decision_native import KINDS
from .decision_diagnostics import pack,stable_choice,rule_scores
from .decision_collection import cards
from .paired_replay import read as read_rows
from .replay import SCHEMA,SLICES
from .storage import digest,write_json


def build(root,output):
    root=Path(root);output=Path(output);output.mkdir(exist_ok=False);examples=[];meta=[];denominator=[];streams=[]
    for i,card in enumerate(cards()):
        scene=root/f'scene-{i:02d}'
        if json.loads((scene/'construction/CARD.json').read_text())!=card:raise ValueError('Frozen V5 parent differs')
        result=json.loads((scene/'RESULT.json').read_text())
        for kind in KINDS:
            parent=scene/'branches'/kind;anchor=result['branches']['anchors'].get(kind,{})
            if parent.exists():history=read_rows(parent/'COMMON_HISTORY.pkl.gz')['rows']
            else:history=None
            names=sorted(anchor.get('candidates',{}))
            intended=['hold','nominal','left','right','grip_timing']if kind=='preclose'else['hold','toward_goal','away','left','right']if kind=='transport'else['hold','release','delay2','delay4','slow_open']
            for name in intended:
                path=parent/(name+'.pkl.gz');rows=read_rows(path)['rows']if path.exists()else[]
                status='COMPLETE16'if len(rows)==16 else'PARTIAL'if rows else'REJECT_OR_UNRUN'
                denominator.append(dict(scene=i,anchor=kind,candidate=name,status=status,actual_transitions=len(rows)))
                if status!='COMPLETE16':continue
                e=pack(history,rows);e['initial_event']=np.array([rows[0]['state_before'][k]for k in SCHEMA['events']]);examples.append(e)
                meta.append(dict(scene=i,anchor=kind,candidate=name,partition=card['ident_partition']));streams.append(dict(scene=i,anchor=kind,candidate=name,sha256=digest(path)))
    if not examples:raise ValueError('No real complete V5 branch')
    arrays={k:np.stack([x[k]for x in examples])for k in examples[0]};arrays.update({k:np.array([m[k]for m in meta])for k in meta[0]})
    arrays.update(purpose=np.array('WM02_DECISION_PAIRED_REPLAY_V5'),schema_json=np.array(json.dumps(SCHEMA)))
    with(output/'PAIRED.npz').open('xb')as f:np.savez_compressed(f,**arrays)
    result=dict(protocol='WM02_DECISION_PAIRED_REPLAY_V5',source_native_sha=json.loads((root/'RESULT.json').read_text())['source_sha'],
        data_sha256=digest(output/'PAIRED.npz'),intended_branches=600,complete16=len(examples),incomplete=600-len(examples),
        numeric_transitions=len(examples)*16,denominator=denominator,stream_pins=streams,
        new_physics=0,WM_updates=0,Actor_updates=0,holdout_outcomes='Not scored; numeric conversion only, no model selection',
        partial_policy='Retained in denominator, excluded from this fixed-complete-window diagnostic; never label padding as executed')
    write_json(output/'AUDIT.json',result);return result


def read(path,device,*,purpose='WM02_DECISION_PAIRED_REPLAY_V5'):
    import torch
    with np.load(path,allow_pickle=False)as f:
        if str(f['purpose'])!=purpose or json.loads(str(f['schema_json']))!=json.loads(json.dumps(SCHEMA)):
            raise ValueError('Explicit V5 real source numeric schema required')
        meta={k:f[k].copy()for k in ('scene','anchor','candidate','partition')}
        for p,ids in (('FIT',range(24)),('DIAGNOSTIC_DEV',range(24,32)),('ONCE_HOLDOUT',range(32,40))):
            if set(meta['scene'][meta['partition']==p])!=set(ids):raise ValueError('All intended whole parents must have real observations')
        data={k:torch.tensor(f[k],device=device,dtype=torch.float32)for k in ('state','reference','events','valid','done','timeout','clock','context','raw','initial_event')}
        if 'support'in f:data['support']=torch.tensor(f['support'],device=device,dtype=torch.float32)
    return data,meta


def quality(state,events,context,kind,*,support=None):
    if kind=='preclose':return events[...,2]
    if kind=='transport':return events[...,2]-10*(state[...,SLICES['object_position']][...,:2]-context[...,:2]).norm(dim=-1)
    if kind=='supported_release':
        if support is None:raise ValueError('Explicit native support label or declared conditional-score prior required; event3 is illegal_contact')
        return(1-events[...,2])*support
    raise ValueError('Declared mechanism only')


def evaluate(models,b,meta,folder):
    import torch
    from .contact_training import predict,errors,select,cv_prediction
    predictions=[];out={};folder=Path(folder)
    args=(b['state'][:,:8],b['state'][:,8],b['reference'][:,:8],b['context'],b['clock'][:,:8],b['clock'][:,8],b['raw'])
    with torch.no_grad():
        for model,blind in zip(models,(False,True)):
            predictions.append(predict(model,*args,blind=blind)[:2])
    np.savez_compressed(folder/'DECISION_PREDICTIONS.npz',conditioned=predictions[0][0].cpu().numpy(),conditioned_events=predictions[0][1].cpu().numpy(),
        blind=predictions[1][0].cpu().numpy(),blind_events=predictions[1][1].cpu().numpy(),raw=b['raw'].cpu().numpy())
    for arm,(pred,event)in zip(('action_conditioned','no_future_action'),predictions):
        decisions=[]
        for i in sorted(set(meta['scene'])):
            for kind in KINDS:
                ids=np.flatnonzero((meta['scene']==i)&(meta['anchor']==kind));ix=torch.tensor(ids,device=pred.device)
                initial=b['state'][ix[0],8];raw=b['raw'][ix].cpu().numpy()
                # Same-observation universal rule, no Teacher/event/stage input.
                obj=initial[SLICES['object_position']].cpu().numpy();tcp=initial[SLICES['tcp_position']].cpu().numpy();context=b['context'][ix[0]].cpu().numpy()
                if float(initial[SLICES['gripper_fraction']][0])>.8:rule=rule_scores(initial.cpu().numpy(),raw)
                elif np.linalg.norm(obj[:2]-context[:2])>.02:rule=-np.linalg.norm(obj[:2]+raw[:,:,:2].sum(1)-context[:2],axis=1)
                else:rule=raw[:,:,6].mean(1)
                for h in (8,16):
                    real=quality(b['state'][ix,8+h],b['events'][ix,8+h-1],b['context'][ix],kind,support=b['support'][ix,8+h-1]if kind=='supported_release'else None).cpu().numpy()
                    # No support head exists. Predicted release is a conditional
                    # non-grasp score at already supported, zero-EEF candidates;
                    # actual quality still requires recorded native support.
                    scores=quality(pred[ix,h-1],event[ix,h-1],b['context'][ix],kind,support=event.new_ones(len(ix))if kind=='supported_release'else None).cpu().numpy()
                    if arm=='no_future_action':scores=np.repeat(scores[0],len(scores))
                    chosen=stable_choice(scores,raw);ru=stable_choice(rule,raw)
                    import hashlib
                    random=int.from_bytes(hashlib.sha256(f'WM02_DECISION_V5_RANDOM/{i}/{kind}'.encode()).digest()[:4],'big')%len(ids)
                    decisions.append(dict(scene=int(i),anchor=kind,horizon=h,native_oracle_Q=float(max(real)),
                        selected=str(meta['candidate'][ids[chosen]]),selected_Q=float(real[chosen]),regret=float(max(real)-real[chosen]),
                        rule_Q=float(real[ru]),rule_regret=float(max(real)-real[ru]),random_Q=float(real[random]),random_regret=float(max(real)-real[random]),
                        candidate_names=meta['candidate'][ids].tolist(),predicted_Q=scores.tolist(),actual_Q=real.tolist(),
                        actual_grasp=b['events'][ix,8+h-1,2].cpu().tolist(),predicted_grasp=event[ix,h-1,2].cpu().tolist()))
        out[arm]=dict(horizons={str(h):errors(pred,event,b,h)for h in (1,8,16)},decisions=decisions)
    out['baselines']={str(h):{name:errors(p,None,b,h)for name,p in (('constant_velocity',cv_prediction(b['state'][:,8])),('state_persistence',b['state'][:,8:9].expand(-1,16,-1)))}for h in(1,8,16)}
    scope='DIAGNOSTIC_DEV8'if min(meta['scene'])<32 else'ONCE_HOLDOUT8'
    return dict(scope=scope+' new parents; no tuning or endpoint selection; correlated five candidates per anchor',models=out,
        mechanism_quality='Preclose grasp; transport grasp minus10*goalXYdistance; actual release nongrasp*native support. Release model score is conditional nongrasp, NOT a predicted native-support probability. Diagnostic only, no formal reward.',
        probability_distribution='Unresampled uniform generated-candidate protocol distribution; not calibrated deployment probability',formal_task_success=False,Actor_updates=0)
