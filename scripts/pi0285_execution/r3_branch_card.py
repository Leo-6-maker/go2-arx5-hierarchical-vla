"""CPU-only prospective R2 optional-phase cards; never launches or changes authority."""
import argparse,json,tempfile
from pathlib import Path
from contract import ROOT,sha,write
from data import immutable
from r2_branch import decide,KEYS,TOL
from r3_recovery import recovery_plan,dose_plan,FROZEN

AUDIT='PASS_ACTUAL_NATIVE_TRAINER_INPUT_PLAN_NOISE_LOSS_MODES_SCOPE_AND_COMPLETE_LOCAL_STATE'


def frozen_decision(root,path):
    root=Path(root);path=Path(path);v=json.loads(path.read_text())
    assert v['status']=='FROZEN_R2_SINGLE_BRANCH_DECISION_NOT_DATA_OR_TRAINING_ADMISSION'
    assert not (root/'qualification/CONFIRM_A_CANDIDATE.json').exists()
    assert v['branch'] in ('RECOVERY2048','DOSE2048')
    required={'audit/PLACE4096-2048_DIAG_CPU.json','audit/PLACE4096-4096_DIAG_CPU.json',
        'source/evidence/PLACE4096-4096_TRAIN.json','audit/PLACE4096-4096_TRAIN_CPU.json','qualification/PLACE4096-4096_ROUTE.json'}
    assert set(v['proofs'])==required,'Require the original selector complete proof set'
    for name,h in v['proofs'].items():assert sha(root/name)==h
    reports=[json.loads((root/'audit'/f'PLACE4096-{step}_DIAG_CPU.json').read_text()) for step in (2048,4096)]
    panel=json.loads((root/'source/evidence/PLACE4096-4096_TRAIN.json').read_text());assert len(panel['rows'])==8
    assert all(report['status']=='PASS_CPU_FIXED_DIAG_NATIVE_DECODE_NOISE_PREFIX_FULL_H40_AND_GATE' for report in reports)
    cpu=json.loads((root/'audit/PLACE4096-4096_TRAIN_CPU.json').read_text())
    assert cpu['status']=='PASS_INDEPENDENT_CPU_CONTACT_SCORE_CLOCK_INSTALLATION_AND_PAIRED_INITIAL_INPUTS'
    assert cpu['source_sha256']==sha(root/'source/evidence/PLACE4096-4096_TRAIN.json')
    assert json.loads((root/'qualification/PLACE4096-4096_ROUTE.json').read_text())['next_action']=='SELECT_R2_SINGLE_BRANCH_FROM_FROZEN_EVIDENCE'
    finish=0
    for row in panel['rows']:
        s=row['task_score'];h=s['history'];zone=s['subpredicates']['zone']
        finish+=bool(h['grasp'] and h['lift'] and h['horizontal_transfer'] and zone['first_true_s'] is not None and s['hold_pass']<41 and not s['persistent_success'])
    recomputed=decide(reports[0]['rows'],reports[1]['rows'],finish)
    assert all(v[k]==value for k,value in recomputed.items()),'Changed selector inputs or branch decision'
    return v


def parent_identity(root,audit_path,export_path):
    root=Path(root);audit_path=Path(audit_path);export_path=Path(export_path)
    a=json.loads(audit_path.read_text());e=json.loads(export_path.read_text());parent=Path(a['checkpoint'])
    assert a['status']==AUDIT and a['point']=='PLACE4096-4096'
    assert a['formal_microbatches']==8192 and a['chunks']==32768 and a['export_sha256']==sha(export_path)
    assert e['checkpoint']==str(parent) and e['full_state_sha256'] and e['full_FP32_master_state']
    entries={x['path']:dict(bytes=x['bytes'],sha256=x['sha256']) for x in a['entries']}
    assert len(entries)==len(a['entries']),'Duplicate parent inventory member'
    required=json.loads((parent/'PI0282_RECOVERY.json').read_text())
    assert required['cursor']==32768 and json.loads((parent/'trainer_state.json').read_text())['global_step']==4096
    assert set(required['required_files'])<=set(entries) and {'optimizer.pt','scheduler.pt','rng_state.pth'}<=set(entries)
    for name,item in entries.items():assert (parent/name).stat().st_size==item['bytes'] and sha(parent/name)==item['sha256']
    assert all(entries[n]['sha256']==h for n,h in e['model_files'].items())
    old=json.loads((root/'design/PLACE4096_CARD.json').read_text())
    assert old['stage']=='PLACE4096' and old['parent'] is None and old['parent_global_update']==0
    assert old['updates']==4096 and old['warmup']==205 and old['peak_LRs']==[1e-4,1e-5]
    assert a['plan_sha256']==sha(old['plan_path'])
    return old,dict(parent=str(parent),parent_files=entries,parent_export=str(export_path),parent_export_sha256=sha(export_path),
        parent_phase_step=4096,parent_global_update=4096,parent_full_master_sha256=e['full_state_sha256'],
        parent_audit_path=str(audit_path),parent_audit_sha256=sha(audit_path))


def planned_rows(root,decision,admission_path=None):
    root=Path(root)
    if decision['branch']=='DOSE2048':
        assert not (root/'recovery/A/COLLECTION_CLOCK.json').exists(),'No DOSE after attempted recovery collection'
        path=root/'data/PLACE4096.jsonl';original=[json.loads(line) for line in path.read_text().splitlines()]
        return dose_plan(original,decision),dict(expert_plan_path=str(path),expert_plan_sha256=sha(path))
    path=Path(admission_path or root/'recovery/A/ADMISSION.json');bank=json.loads(path.read_text())
    assert bank['status']=='PASS_ACTUAL_BRANCH_RESTORE_FRESH_CORRECTIONS_FIXED_CODEC' and bank['valid_branches']>=24 and bank['requested_branches']==32
    assert bank['processor']==FROZEN and len(set(bank['valid_slots']))==len(bank['valid_slots'])==bank['valid_branches']
    assert {r['episode'] for r in bank['rows']}==set(bank['valid_slots'])
    index=root/'data/WINDOW_INDEX.json';experts=json.loads(index.read_text())['rows']
    return recovery_plan(experts,bank),dict(recovery_admission_path=str(path),recovery_admission_sha256=sha(path),
        expert_window_index_path=str(index),expert_window_index_sha256=sha(index))


def prepare(root,decision_path,audit_path,export_path,out,admission_path=None):
    root=Path(root);out=Path(out);decision_path=Path(decision_path)
    decision=frozen_decision(root,decision_path);old,parent=parent_identity(root,audit_path,export_path)
    rows,source=planned_rows(root,decision,admission_path)
    assert len(rows)==16384 and [r['index'] for r in rows]==list(range(16384))
    path=out/'REPAIR_A2048.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    payload=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows)
    if path.exists():assert path.read_text()==payload
    else:path.write_text(payload)
    card=dict(old,**parent,**source,stage='REPAIR_A2048',branch=decision['branch'],updates=2048,warmup=103,peak_LRs=[5e-5,5e-6],
        plan_path=str(path),plan_sha256=sha(path),branch_decision_path=str(decision_path),branch_decision_sha256=sha(decision_path),
        launch_allowed=False,status='PREPARED_REAL_BOUND_BRANCH_CARD_OWNER_ADMISSION_PENDING',Adam='CARRY_EXACT_MOMENTS_AND_GLOBAL_STEPS',
        scheduler='NEW_COSINE2048_WARMUP103',phase_cursor=0,canary_status='NEW_PARENT_CARRY_SEMANTICS_NOT_COVERED_BY_PLACE_PARENT_NONE_PROOF')
    immutable(out/'REPAIR_A2048_CARD.prospective.json',card);return card


def validate_branch(root,card,rows):
    """Recommended authority hook, called only within the existing REPAIR_A2048 path."""
    root=Path(root);assert card['stage']=='REPAIR_A2048' and card['branch'] in ('RECOVERY2048','DOSE2048')
    path=Path(card['branch_decision_path']);assert sha(path)==card['branch_decision_sha256']
    decision=frozen_decision(root,path);assert card['branch']==decision['branch']
    _,parent=parent_identity(root,card['parent_audit_path'],card['parent_export'])
    assert all(card[k]==v for k,v in parent.items())
    expected,bindings=planned_rows(root,decision,card.get('recovery_admission_path'))
    assert rows==expected and all(card[k]==v for k,v in bindings.items())
    assert sum(r['source']=='corrected_suffix' for r in rows)==(8192 if card['branch']=='RECOVERY2048' else 0)


def self_check():
    def diag(scale):return [dict(uid=str(i),task='T1',split='TRAIN',phase=i//8,prefix={k:float(t*scale) for k,t in zip(KEYS,TOL)},gate=dict(decision='ACCEPT')) for i in range(32)]
    with tempfile.TemporaryDirectory(prefix='PI0285_SYNTHETIC_CARD_') as temporary:
        root=Path(temporary);cp=root/'checkpoint-4096';cp.mkdir()
        from data import plan
        experts=[dict(task='T1',split='TRAIN',layout=seed,episode=f'ep{seed}',phase=p,anchor=a,uid=f'{seed}:{p}:{a}',source='expert',
            instruction=__import__('contract').INSTRUCTIONS['T1'],event_first8=a==0,event_tail_only=a==1) for seed in range(285100,285108) for p in range(5) for a in range(4)]
        original=root/'data/PLACE4096.jsonl';original.parent.mkdir();original.write_text(''.join(json.dumps(v)+'\n' for v in plan(experts,['T1'],4096)))
        write(root/'design/PLACE4096_CARD.json',dict(stage='PLACE4096',parent=None,parent_global_update=0,updates=4096,warmup=205,peak_LRs=[1e-4,1e-5],plan_path=str(original),release='SYNTHETIC',canonical=0))
        write(cp/'PI0282_RECOVERY.json',dict(cursor=32768,required_files=['optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','model.safetensors']))
        write(cp/'trainer_state.json',dict(global_step=4096))
        for name in ('optimizer.pt','scheduler.pt','rng_state.pth','model.safetensors'):(cp/name).write_bytes(b'SYNTHETIC_NOT_REAL_TRAINER')
        export=root/'EXPORT.json';write(export,dict(checkpoint=str(cp),full_state_sha256='SYNTHETIC',full_FP32_master_state={'synthetic':{}},model_files={'model.safetensors':sha(cp/'model.safetensors')}))
        audit=root/'AUDIT.json';write(audit,dict(status=AUDIT,point='PLACE4096-4096',formal_microbatches=8192,chunks=32768,export_sha256=sha(export),checkpoint=str(cp),plan_sha256=sha(original),entries=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in cp.iterdir()]))
        for step,scale in ((2048,2),(4096,1.7)):write(root/'audit'/f'PLACE4096-{step}_DIAG_CPU.json',dict(status='PASS_CPU_FIXED_DIAG_NATIVE_DECODE_NOISE_PREFIX_FULL_H40_AND_GATE',rows=diag(scale)))
        write(root/'source/evidence/PLACE4096-4096_TRAIN.json',dict(rows=[dict(task_score=dict(history=dict(grasp=False,lift=False,horizontal_transfer=False),subpredicates=dict(zone=dict(first_true_s=None)),hold_pass=0,persistent_success=False)) for _ in range(8)]))
        write(root/'audit/PLACE4096-4096_TRAIN_CPU.json',dict(status='PASS_INDEPENDENT_CPU_CONTACT_SCORE_CLOCK_INSTALLATION_AND_PAIRED_INITIAL_INPUTS',source_sha256=sha(root/'source/evidence/PLACE4096-4096_TRAIN.json')))
        write(root/'qualification/PLACE4096-4096_ROUTE.json',dict(next_action='SELECT_R2_SINGLE_BRANCH_FROM_FROZEN_EVIDENCE'))
        proofpaths=['audit/PLACE4096-2048_DIAG_CPU.json','audit/PLACE4096-4096_DIAG_CPU.json','source/evidence/PLACE4096-4096_TRAIN.json','audit/PLACE4096-4096_TRAIN_CPU.json','qualification/PLACE4096-4096_ROUTE.json']
        decision=root/'qualification/R2_A_SINGLE_BRANCH.json';write(decision,dict(status='FROZEN_R2_SINGLE_BRANCH_DECISION_NOT_DATA_OR_TRAINING_ADMISSION',**decide(diag(2),diag(1.7),0),proofs={name:sha(root/name) for name in proofpaths}))
        card=prepare(root,decision,audit,export,root/'prospective');assert not card['launch_allowed'] and card['branch']=='DOSE2048'
        rows=[json.loads(line) for line in Path(card['plan_path']).read_text().splitlines()];validate_branch(root,card,rows)
        for mutation in (dict(card,branch='RECOVERY2048'),dict(card,parent_global_update=2048),dict(card,expert_plan_sha256='WRONG')):
            try:validate_branch(root,mutation,rows)
            except AssertionError:pass
            else:raise AssertionError('Wrong branch/parent/plan identity accepted')
        for step,scale in ((2048,2),(4096,.8)):write(root/'audit'/f'PLACE4096-{step}_DIAG_CPU.json',dict(status='PASS_CPU_FIXED_DIAG_NATIVE_DECODE_NOISE_PREFIX_FULL_H40_AND_GATE',rows=diag(scale)))
        write(decision,dict(status='FROZEN_R2_SINGLE_BRANCH_DECISION_NOT_DATA_OR_TRAINING_ADMISSION',**decide(diag(2),diag(.8),0),proofs={name:sha(root/name) for name in proofpaths}))
        write(root/'data/WINDOW_INDEX.json',dict(rows=experts))
        corrected=[dict(v,source='corrected_suffix',layout=seed,episode='correction'+str(seed),uid=f"correction{seed}:{v['phase']}:{v['anchor']}")
            for seed in range(285100,285124) for v in experts if v['layout']==285100]
        admission=root/'recovery/A/ADMISSION.json';write(admission,dict(status='PASS_ACTUAL_BRANCH_RESTORE_FRESH_CORRECTIONS_FIXED_CODEC',requested_branches=32,valid_branches=24,
            valid_slots=['correction'+str(seed) for seed in range(285100,285124)],processor=FROZEN,rows=corrected))
        recovery=prepare(root,decision,audit,export,root/'recovery-prospective',admission)
        validate_branch(root,recovery,[json.loads(line) for line in Path(recovery['plan_path']).read_text().splitlines()])
        assert recovery['branch']=='RECOVERY2048' and not recovery['launch_allowed']
        (cp/'optimizer.pt').write_bytes(b'PARTIAL')
        try:parent_identity(root,audit,export)
        except AssertionError:pass
        else:raise AssertionError('Changed parent accepted')
    return dict(status='PASS_SYNTHETIC_BRANCH_CARD_ONLY_NOT_NATIVE_TRAINER_OR_ADMISSION',synthetic_fixtures=True,model_queries=0,optimizer_updates=0,physics_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-check',action='store_true');p.add_argument('--root',default=str(ROOT));p.add_argument('--decision');p.add_argument('--audit');p.add_argument('--export');p.add_argument('--out');p.add_argument('--admission');p.add_argument('--report');a=p.parse_args()
    v=self_check() if a.self_check else prepare(a.root,a.decision,a.audit,a.export,a.out,a.admission)
    if a.report:write(a.report,v)
    print(json.dumps(v))
