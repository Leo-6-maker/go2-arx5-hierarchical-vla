"""Read actual training journals; no model, tensor reconstruction or gradient calls."""
import argparse,collections,hashlib,json,math,statistics
from pathlib import Path

def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def audit(root,arm):
    t=root/arm/'main/training';complete=read(t/'SEGMENT_COMPLETE.json');steps=complete['actual_update'];n=2*steps
    assert complete['status']=='PASS_EXECUTED_SEGMENT' and complete['arm']==arm
    plan=read(root/'design/TRAIN_PLAN.json')['stream'][:n]
    proof={r['uid']:r for r in read(Path(read(root/'design/INPUT_BINDING.json')['old_root'])/'design/PROCESSED_INPUTS.json')['rows']}
    names=('sample_audit.jsonl','step_rng.jsonl','S4_audit.jsonl','gripper_objective.jsonl','full_loss.jsonl')
    sample,rng,s4,grip,full=[rows(t/name) for name in names]
    assert all(len(v)==n for v in (sample,rng,s4,grip,full))
    namespace=read(root/'design/TRAIN_PLAN.json')['noise_namespace']
    if '{unit}' not in namespace:namespace+=':{unit}'
    for i,(p,s,r,a,g,f) in enumerate(zip(plan,sample,rng,s4,grip,full)):
        assert all(s[k]==v for k,v in p.items()) and s['micro_index']==i
        assert s['condition_sha256']==proof[s['uid']]['condition_sha256']==r['condition_sha256']
        assert s['labels_masks_sha256']==proof[s['uid']]['labels_masks_sha256']
        assert r['unit']==s['unit'] and r['micro']==s['micro'] and r['mode_restored']
        seed=int.from_bytes(hashlib.sha256(namespace.format(unit=s['unit']).encode()).digest()[:8],'big')%2**31
        assert seed==r['seed']
        for k in ('native_epsilon','stock_initial_noise'):assert a[k]==r[k]
        assert a['native_tau']==r['native_tau'] and a['forward_index']==i+1
        assert a['rollout_steps']==a['rollout_head_evaluations']==a['checkpointed_euler_steps']==4
        assert not a['expert_action_in_condition'] and not a['active_action_modified']
        assert a['native_epsilon']['dtype']=='torch.bfloat16' and a['stock_vl_features']['dtype']=='torch.float32'
        assert len(a['euler_states'])==len(a['euler_velocities'])==4
        assert all(v['dtype']=='torch.float32' for v in a['euler_states']+a['euler_velocities'])
        assert all(math.isfinite(v) for v in (r['loss'],g['total'],g['weighted'],g['unweighted'],f['total']))
        assert math.isclose(g['weighted'],.1*g['unweighted'],rel_tol=1e-6,abs_tol=1e-6)
        assert math.isclose(g['total'],f['total']+g['weighted'],rel_tol=1e-6,abs_tol=1e-6)
        assert math.isclose(r['loss']*2,g['total'],rel_tol=1e-6,abs_tol=1e-6)
    gradients=rows(t/'update_gradient_norms.jsonl');timing=rows(t/'TIMINGS.jsonl')
    assert len(gradients)==len(timing)==steps
    assert [r['unit'] for r in gradients]==list(range(1,steps+1))
    assert all(math.isfinite(r['accumulated_norm']) and r['clip_max_norm']==1. and r['microbatches']==2 for r in gradients)
    scope=read(t/'PARAMETER_SCOPE.json');assert len(scope['trainable'])==471 and len(scope['frozen'])==559
    result=dict(status='PASS_ACTUAL_JOURNAL_INPUT_LOSS_S4_MODE_SCOPE_AUDIT',arm=arm,updates=steps,microbatches=n,
        inputs_and_labels='ALL_MATCH_FROZEN_PLAN_AND_PROOFS',actual_noise='ALL_OBSERVED_NATIVE_EPSILON_TAU_AND_S4_IDENTITIES_MATCH',
        raw_noise_regeneration='NOT_PERFORMED',mode_restored=n,S4_full_steps=4,loss_binding='OLD_PLUS_POINT1_GENERATED_GRIPPER',
        trainable=471,frozen=559,final_frozen_tensor_byte_comparison='UNOBSERVED_NOT_CLAIMED',
        phase_exposure=dict(collections.Counter(r['group'] for r in sample)),
        production=dict(median_update_s=statistics.median(r['completed_seconds'] for r in timing),median_loader_wait_s=statistics.median(r['loader_wait_seconds'] for r in timing),
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in timing),peak_process_tree_RSS_bytes=max(r['process_tree_RSS_bytes'] for r in timing)),
        journals=[dict(path=str(t/name),sha256=sha(t/name),rows=n) for name in names],
        quality='SEPARATE_ACTUAL_PHYSICAL_PANEL_REQUIRED',new_model_queries=0,new_gradient_queries=0,selected_checkpoint='NONE',deployment='NONE')
    target=root/'evidence'/f'{arm}_ACTUAL_TRAINING_AUDIT.json';tmp=target.with_suffix('.partial');tmp.write_text(json.dumps(result,sort_keys=True)+'\n');tmp.replace(target)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=('GRIP','FOCUS'),required=True);a=p.parse_args();audit(a.root,a.arm)
