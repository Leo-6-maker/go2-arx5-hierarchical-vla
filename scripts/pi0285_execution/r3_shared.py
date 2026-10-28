"""CPU-only frozen shared-stream audit, paired reporting and gated phase preparation."""
import argparse,json,time
from collections import Counter
from pathlib import Path
from contract import INSTRUCTIONS,SPLITS,sha
from data import immutable

CPU_STATUS='PASS_INDEPENDENT_CPU_CONTACT_SCORE_CLOCK_INSTALLATION_AND_PAIRED_INITIAL_INPUTS'


def ready(root):
    root=Path(root);data=root/'data'
    inputs=json.loads((data/'NATIVE_INPUTS_CHECK.json').read_text())
    assert inputs['status']=='PASS_ACTUAL_NATIVE_INPUTS_ONLY_NOT_CODEC_ADMISSION'
    for n,h in inputs['bindings'].items():assert sha(data/n)==h,n
    assert len(inputs['inputs'])==24 and len(inputs['two_worker_native_batch_hashes'])==6
    assert inputs['variable_language_native_padding'] and inputs['paired_initial_RGB_state_checked']
    assert inputs['leaked_policy_fields']==[] and inputs['TRAIN_statistics_only']
    assert all(v['instruction']==INSTRUCTIONS[v['task']] for v in inputs['inputs'])
    stats=json.loads((data/'STATISTICS_RELEASE.json').read_text())
    assert stats['DEV_or_confirmation_in_fit']==stats['old_demonstrations_in_fit']==0
    assert stats['tasks']==['T1','T2'] and stats['TRAIN_episodes']==128
    for n,h in stats['files'].items():assert sha(data/n)==h,n
    bank=json.loads((data/'SOURCE_BANK.json').read_text())
    assert bank['tasks']==['T1','T2']
    layouts={t:{e['layout'] for e in bank['admitted_episodes'] if e['task']==t and e['split']=='TRAIN'} for t in INSTRUCTIONS}
    assert len(layouts['T1']&layouts['T2'])>=48
    rows=[json.loads(x) for x in (data/'SHARED4096.jsonl').read_text().splitlines()]
    assert len(rows)==32768 and [r['index'] for r in rows]==list(range(32768))
    assert Counter(r['task'] for r in rows)=={'T1':16384,'T2':16384}
    for i,r in enumerate(rows):
        assert r['task']==('T1','T2')[i%2] and r['split']=='TRAIN'
        assert SPLITS['TRAIN'][0]<=r['layout']<=SPLITS['TRAIN'][1]
        assert r['instruction']==INSTRUCTIONS[r['task']] and r['source']=='expert'
        assert r['episode']==f"{r['task']}-TRAIN-{r['layout']}"
        assert r['uid']==f"{r['task']}-TRAIN-{r['layout']}-{r['anchor']}"
    value=dict(status='READY_CPU_SCHEMA_MASK_STREAM_NOT_TASK_CAPABILITY',owner='R3_WP7_SHARED_DELIVERY',state='READY',
               bindings={n:sha(data/n) for n in ('SHARED4096.jsonl','SOURCE_BANK.json','NATIVE_INPUTS_CHECK.json','STATISTICS_RELEASE.json','statistics.json','processor_config.json','embodiment_id.json')},
               rows=32768,task_counts={'T1':16384,'T2':16384},paired_TRAIN_layouts=len(layouts['T1']&layouts['T2']),
               reused_inputs=24,reused_native_batches=6,reused_workers=2,confirmation_in_stream=0,
               known_task_set_TRAIN_preprocessing=True,shared_tokenizer_collator_cache=True,
               prerequisites=['CONFIRM_A>=26/32','independent_exact_protected_full_Trainer_and_model','T2_DEV16_acquisition','measured_time_reserve_before_06:30'],
               task_capability='NOT_MEASURED_BY_READINESS',launch_allowed=False,
               consumed=dict(model_queries=0,optimizer_updates=0,physics=0),selected='NONE',deployment='NONE')
    return value


def checked_panel(root,point,split,tasks):
    root=Path(root);p=root/'source/evidence'/f'{point}_{split}.json';c=root/'audit'/f'{point}_{split}_CPU.json'
    v=json.loads(p.read_text());cpu=json.loads(c.read_text());size={'TRAIN':8,'DEV':16,'CONFIRM_A':32,'CONFIRM_B':32,'ACQUISITION':16}[split]
    assert cpu['status']==CPU_STATUS and cpu['source_sha256']==sha(p)
    assert len(v['rows'])==len(cpu['rows'])==size*len(tasks)
    expected=list(range(SPLITS['DEV' if split=='ACQUISITION' else split][0],SPLITS['DEV' if split=='ACQUISITION' else split][0]+size))
    assert len({(r['task'],r['layout']) for r in v['rows']})==len(v['rows'])
    counts={}
    for t in tasks:
        rr=[r for r in v['rows'] if r['task']==t];assert sorted(r['layout'] for r in rr)==expected
        assert all(r['point']==point and r['split']==split and r['instruction']==INSTRUCTIONS[t] and r['status'].startswith('VALID') for r in rr)
        counts[t]=sum(bool(r['persistent_success']) for r in rr)
        assert v['tasks'][t]==dict(requested=size,success=counts[t])
    assert [(r['task'],r['layout'],r['persistent_success']) for r in cpu['rows']]==[(r['task'],r['layout'],r['persistent_success']) for r in v['rows']]
    if len(tasks)==2:
        assert sorted(p['layout'] for p in cpu['pairs'])==expected
        assert all(p['same_initial_actual_RGB_state_and_cameras'] and all(p['same_initial_actual_RGB_state_and_cameras'].values()) for p in cpu['pairs'])
        counts['paired_both']=sum(all(next(r['persistent_success'] for r in v['rows'] if r['task']==t and r['layout']==s) for t in tasks) for s in expected)
        assert counts['paired_both']==v['paired_both_success']
    return counts,dict(source=str(p),source_sha256=sha(p),CPU=str(c),CPU_sha256=sha(c)),v,cpu


def paired(root,point,split,old_point):
    counts,proof,v,cpu=checked_panel(root,point,split,['T1','T2'])
    old,old_proof,_,_=checked_panel(root,old_point,split,['T1'])
    # ponytail: recombine already independently scored predicates; never run physics again.
    wrong={t:sum(r['score']['subpredicates'][('T2' if t=='T1' else 'T1')+'_hold']['hold_pass']==41 for r in cpu['rows'] if r['task']==t) for t in INSTRUCTIONS}
    return dict(status='ACTUAL_INDEPENDENT_CPU_PAIRED_REPORT',point=point,split=split,counts=counts,wrong_goal_hold=wrong,
                old_T1=dict(point=old_point,success=old['T1'],same_denominator=True,delta=counts['T1']-old['T1']),proofs=[proof,old_proof],selected='NONE',deployment='NONE')


def prepare(root,protected,export,audit,reserve):
    from r3_delivery import verify_protection
    root=Path(root);protected=Path(protected);v=verify_protection(protected)
    assert time.time()<1791498600,'SHARED cannot start after 2026-10-09 06:30 SGT'
    reserve=Path(reserve);r=json.loads(reserve.read_text())
    assert r['status']=='OWNER_MEASURED_SHARED_TRAIN_EVALUATE_AND_HANDOFF_RESERVE' and r['admitted']
    assert r['remaining_measured_seconds']>=r['required_seconds']>=4500
    assert time.time()<=r['valid_until_epoch']<=1791498600
    ready(root);point=v['point'];checked_panel(root,point,'ACQUISITION',['T2'])
    audit=Path(audit);a=json.loads(audit.read_text());export=Path(export);e=json.loads(export.read_text())
    assert a['status']=='PASS_ACTUAL_NATIVE_TRAINER_INPUT_PLAN_NOISE_LOSS_MODES_SCOPE_AND_COMPLETE_LOCAL_STATE'
    assert a['point']==point and a['export_sha256']==sha(export) and e['checkpoint']==a['checkpoint']
    assert a['entries']==v['full_Trainer']['entries']
    parent=Path(v['full_Trainer']['protected_root']);step=int(point.rsplit('-',1)[1])
    recovery=json.loads((parent/'PI0282_RECOVERY.json').read_text());assert recovery['cursor']==step*8
    oldcard=json.loads((root/'design'/(point.rsplit('-',1)[0]+'_CARD.json')).read_text())
    card=dict(oldcard,stage='SHARED4096',updates=4096,warmup=205,peak_LRs=[1e-4,1e-5],parent=str(parent),
              parent_files={x['path']:dict(bytes=x['bytes'],sha256=x['sha256']) for x in a['entries']},
              parent_export=str(export),parent_export_sha256=sha(export),parent_phase_step=step,
              parent_global_update=oldcard['parent_global_update']+step,parent_full_master_sha256=e['full_state_sha256'],plan_path=str(root/'data/SHARED4096.jsonl'),
              plan_sha256=sha(root/'data/SHARED4096.jsonl'),launch_allowed=True)
    immutable(root/'design/SHARED4096_CARD.json',card)
    return dict(status='READY_FULL_ADAM_SHARED_PHASE_CARD_NOT_LAUNCHED',card_sha256=sha(root/'design/SHARED4096_CARD.json'),protected_sha256=sha(protected),time_reserve_sha256=sha(reserve),model_queries=0,optimizer_updates=0,physics=0)


def self_check():
    import tempfile
    from r3_delivery import fixture_panels
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);fixture_panels(root,'SHARED4096-2048','DEV',['T1','T2']);fixture_panels(root,'PLACE4096-1024','DEV',['T1'])
        v=paired(root,'SHARED4096-2048','DEV','PLACE4096-1024');assert v['counts']=={'T1':16,'T2':16,'paired_both':16}
        p=root/'source/evidence/SHARED4096-2048_DEV.json';value=json.loads(p.read_text());value['rows'][0]['layout']=285300;p.write_text(json.dumps(value))
        c=root/'audit/SHARED4096-2048_DEV_CPU.json';cpu=json.loads(c.read_text());cpu['source_sha256']=sha(p);c.write_text(json.dumps(cpu))
        try:checked_panel(root,'SHARED4096-2048','DEV',['T1','T2'])
        except AssertionError:pass
        else:raise AssertionError('Confirmation layout accepted')
    return dict(status='PASS_CPU_PAIRED_FIXED_LAYOUTS_HASHES_DENOMINATORS_RETENTION',synthetic_fixtures=True,model_queries=0,physics=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('ready','paired','prepare','self-check'));p.add_argument('--root');p.add_argument('--point');p.add_argument('--split');p.add_argument('--old-point');p.add_argument('--protected');p.add_argument('--export');p.add_argument('--audit');p.add_argument('--reserve');p.add_argument('--output');a=p.parse_args()
    v=self_check() if a.mode=='self-check' else ready(a.root) if a.mode=='ready' else paired(a.root,a.point,a.split,a.old_point) if a.mode=='paired' else prepare(a.root,a.protected,a.export,a.audit,a.reserve)
    if a.output:immutable(a.output,v)
    print(json.dumps(v))
