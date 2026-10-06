"""Saved consumption, source commands and offline predictions; no model construction."""
import argparse, ast, csv, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from audit_existing import read, sha, write, table, RELEASE

def transitions(values):
    return [dict(time_s=float(i*.05),before=float(values[i-1]),after=float(values[i]))
            for i in np.flatnonzero(values[1:] != values[:-1])+1]

def audit(old, raw, root):
    out=root/'evidence/A0';out.mkdir(parents=True,exist_ok=True)
    provenance=read(root/'source_audit/SOURCE_BINDINGS.json')
    for name,row in provenance.items():assert sha(root/'source_audit'/name)==row['sha256']
    plan=read(old/'design/TRAIN_PLAN.json')['stream']
    actual=[json.loads(x) for x in (root/'source_audit/sample_audit.jsonl').read_text().splitlines()]
    assert len(actual)==len(plan)==16384
    proof={r['uid']:r for r in read(old/'design/PROCESSED_INPUTS.json')['rows']}
    card=read(old/'design/DATASET_CARD.json');rows={r['uid']:r for r in card['rows']}
    for index,(visit,used) in enumerate(zip(plan,actual)):
        assert all(used[k]==v for k,v in visit.items()) and used['micro_index']==index
        for name in ('condition_sha256','labels_masks_sha256'):assert used[name]==proof[used['uid']][name]
    new=plan[:8192];assert Counter(v['group'] for v in new)=={g:2048 for g in range(4)}
    counts=Counter(v['uid'] for v in actual);new_counts=Counter(v['uid'] for v in new)
    episodes=[e for e in card['episodes'] if e['split']=='TRAIN' and e['admitted']];assert len(episodes)==93
    command_bank={};episode_rows=[]
    for e in episodes:
        path=Path(e['path'])/'commands.npz'
        rr=next(r for r in card['rows'] if r['split']=='TRAIN' and r['seed']==e['seed'])
        assert sha(path)==rr['command_sha256']
        with np.load(path) as z:a=z['action'].copy();phase=z['phase'].copy()
        assert a.shape==(640,10) and np.isfinite(a).all() and set(np.unique(a[:,9]))=={0.,1.}
        command_bank[e['seed']]=(a,phase)
        own=[rows[uid] for uid in new_counts if rows[uid]['seed']==e['seed']]
        for r in own:assert r['anchor']+40<=len(a) and r['eligible'] and r['group']==phase[r['anchor']]
        episode_rows.append(dict(seed=e['seed'],command_sha256=sha(path),open_samples=int((a[:,9]==1).sum()),
            closed_samples=int((a[:,9]==0).sum()),transitions=transitions(a[:,9]),
            old_actual_visits=sum(counts[uid] for uid in counts if rows[uid]['seed']==e['seed']),
            new_visits=sum(new_counts[r['uid']] for r in own),new_unique_windows=len(own)))
    admitted={e['seed'] for e in episodes};classes=defaultdict(list)
    offline=read(old/'design/OFFLINE_CARD.json')['rows'];offline_rows=[]
    # Labels and prediction arrays are already fixed, including any gate failures.
    with (raw/'delivery/OFFLINE384.csv').open() as f:old_metrics={(r['point'],r['uid']):r for r in csv.DictReader(f)}
    for row in offline:
        uid=row['uid'];support=('ACTUALLY_CONSUMED_TRAIN' if counts[uid] else 'SAME_EPISODE_UNCONSUMED_TRAIN') if row['split']=='TRAIN' else 'VAL'
        if row['split']=='TRAIN':a=command_bank[row['seed']][0]
        else:
            path=Path(row['episode'])/'commands.npz';assert sha(path)==row['command_sha256']
            with np.load(path) as z:a=z['action'].copy()
        target=a[row['anchor']:row['anchor']+40,9]
        for point in ('BASE0','TASK8192'):
            folder=raw/'quality'/point/'OFFLINE'/uid
            with np.load(folder/'output.npz',allow_pickle=False) as z:
                decoded=z['decoded'][0];raw_action=z['raw'][0]
            g=decoded[:,9];assert np.allclose(g,(raw_action[:,9]+1)/2,rtol=0,atol=1e-7)
            m=old_metrics[point,uid]
            item=dict(uid=uid,point=point,split=row['split'],group=row['group'],support=support,old_visits=counts[uid],new_visits=new_counts[uid],
                      gate=m['gate'],XYZ_RMS_m=m['XYZ_RMS_m'],SO3_RMS_rad=m['SO3_RMS_rad'],
                      prefix8_XYZ_RMS_m=m['prefix8_XYZ_RMS_m'],prefix8_SO3_RMS_rad=m['prefix8_SO3_RMS_rad'])
            for label,n in [('full',40),('prefix8',8)]:
                err=np.abs(g[:n]-target[:n]);item[label+'_gripper_MAE']=float(err.mean())
                for kind,mask in [('OPEN',target[:n]>=.8),('CLOSE',target[:n]<=.2)]:
                    item[label+'_'+kind+'_count']=int(mask.sum());item[label+'_'+kind+'_MAE']=float(err[mask].mean()) if mask.any() else None
            target_closed=target<=.2;pred_closed=g<=.2;target_open=target>=.8;pred_open=g>=.8
            for name,mask in [('label_close',target_closed),('pred_close',pred_closed),('label_open',target_open),('pred_open',pred_open)]:
                edges=np.flatnonzero(~mask[:-1]&mask[1:])+1;item[name+'_transition_s']=json.dumps((edges*.05).tolist())
            item['output_sha256']=sha(folder/'output.npz');offline_rows.append(item)
            classes[point+'/'+support].append(uid)
    aliases=defaultdict(set)
    for uid in new_counts:aliases[proof[uid]['condition_sha256']].add(proof[uid]['labels_masks_sha256'])
    contradictory={k:sorted(v) for k,v in aliases.items() if len(v)>1}
    stat=read(root/'source_audit/statistics.json')['new_embodiment']['action']['gripper_position']
    assert stat['q01']==[0.] and stat['q99']==[1.]
    native=(root/'source_audit/native_head.py').read_text();tree=ast.parse(native)
    loss_node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='loss' for t in n.targets)
                   and 'action_loss.sum()' in ast.unparse(n))
    assert ast.unparse(loss_node.value)=='action_loss.sum() / (action_mask.sum() + 1e-06)'
    oldtrain=(root/'source_audit/train.py').read_text();g=next(n for n in ast.walk(ast.parse(oldtrain)) if isinstance(n,ast.FunctionDef) and n.name=='gripper_error')
    assert 'g[..., 9].detach().float()' in ast.unparse(g) and 'return physical(g, e, *a, **kw)' in ast.unparse(g)
    seeds=read(old/'design/EVALUATION_NOISE.json')['TRAIN_diagnostic_seeds'];assert len(seeds)==16
    fit=[]
    for seed in seeds:
        action,phase=command_bank[seed]
        for group in range(4):
            candidates=[rows[uid] for uid in new_counts if rows[uid]['seed']==seed and rows[uid]['group']==group]
            start=int(np.flatnonzero(phase==group)[0]);chosen=min(candidates,key=lambda r:(abs(r['anchor']-start),r['anchor']))
            fit.append(dict(chosen,new_visits=new_counts[chosen['uid']],old_actual_visits=counts[chosen['uid']],first_group_tick=start,
                condition_sha256=proof[chosen['uid']]['condition_sha256'],labels_masks_sha256=proof[chosen['uid']]['labels_masks_sha256']))
    assert len(fit)==64 and len({r['uid'] for r in fit})==64
    write(root/'design/TRAIN_PLAN.json',dict(release=RELEASE,source_plan_sha256=sha(old/'design/TRAIN_PLAN.json'),updates=4096,stream=new,
        arm_order=['CONTROL','GRIP'],noise_namespace='PI0277:TRAIN:27703',seed=27703,actual_noise_byte_comparison='REQUIRED_DURING_EXECUTION'))
    write(root/'design/FIT_CARD.json',dict(rows=fit,noises=2,policies=['TASK8192','CONTROL4096','GRIP4096'],
        selection='First group transition, closest eligible actually visited anchor; tie earlier; no model scores',noise_packets='PENDING_PREQUERY_FREEZE'))
    table(out/'OFFLINE_SUPPORT384.csv',offline_rows)
    write(out/'TRAIN_EXPOSURE.json',dict(status='PASS_ACTUAL16384_VISITS_AND_FROZEN8192_MATCHED_VISITS',episodes=episode_rows,
        old_unique_uids=len(counts),new_unique_uids=len(new_counts),new_group_counts=dict(Counter(v['group'] for v in new)),
        offline_support_counts={k:len(v) for k,v in classes.items()},exact_condition_different_label_hashes=contradictory,
        near_identical_input_aliasing='UNOBSERVED_NO_NEW_EMBEDDING_QUERY',sample_log_binding=provenance['sample_audit.jsonl'],
        masks='Existing qualified input proof retained; actual channel masks must be rechecked in cache admission',new_processor_calls=0))
    write(out/'GRIPPER_OBJECTIVE_AUDIT.json',dict(status='PASS_SOURCE_OBJECTIVE_ASYMMETRY_AND_SAVED_DECODE_BINDING',source_bindings=provenance,
        native_denominator=ast.unparse(loss_node.value),native_active_elements_for_full_batch1_H40=400,
        native_gripper_fraction_of_equal_active_weights='40/(400+1e-6); not 1/132',
        generated_gripper_old_logging_detached=True,native_gripper_supervision_present=True,indirect_generated_gripper_gradients_possible=True,
        saved384_inverse_affine_exact=True,command_inverse='(u+1)/2',action_statistics_q01=stat['q01'],action_statistics_q99=stat['q99'],
        original_data_admission_sha256=sha(old/'evidence/DATA_ADMISSION.json'),source_command_transitions_audited=93,
        execution_transport='A0 physical audit confirms native command equals 2*clip(decoded_fraction,0,1)-1; declared existing saturation only',
        full_A0='Combine AUDIT_EXISTING.json, TRAIN_EXPOSURE.json and this receipt; new loss CPU and real canary still required',
        new_model_queries=0,new_physics_steps=0,new_teacher_calls=0))
    print('PASS_EXPOSURE_AND_OBJECTIVE_AUDIT',json.dumps({k:len(v) for k,v in classes.items()}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();audit(a.old,a.raw,a.root)
