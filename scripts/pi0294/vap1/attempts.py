"""Rehash known legal negative witnesses without admitting failed data.

Full successful data qualification and complete attempt accountability are
different predicates. Historical raw/export records are never rewritten.
"""
import copy
from pathlib import Path
import time

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.cr2.bank import closed_child, raw_file
from scripts.pi0294.cr2.runtime import bind_cpu, wait_priority
from scripts.pi0294.cr2.scope import sealed_file
from scripts.pi0294.fd1 import put, sha
from .plan import AUTH, bank_gate
from .scope import priority, validate

STATUS='PASS_REHASHED_ORIGINAL_LEGAL_PARTIAL_NEGATIVE'


def qualify(rows, cases, guard):
    result=copy.deepcopy(rows)
    for row in result:
        if row.get('evidence_qualified') is True:continue
        if row.get('status')!='PRESERVED_NEGATIVE_NOT_SUCCESS_DATA':continue
        guard();case=cases[row['id']];raw=Path(case['source_episode']);child=raw.parent.parent
        binding=read(child/'OUTPUT_BINDING.private.json');ledger=read(child/'RESOURCE_LEDGER.json')
        if (read(binding['lease'])['status']!='CLOSED' or ledger['status']!='TERMINAL_CAPTURE'
                or ledger['unknown_effects'] or read(child/'RUN_MANIFEST.json')['authority']['protocol_release']!=AUTH):
            raise PermissionError('VAP1_NEGATIVE_ONLY_SAME_AUTHORITY_CLOSED_CAPTURE')
        original=case['capture_receipt'];partial=raw/'PARTIAL_NEGATIVE_AUDIT.json'
        if not partial.is_file():continue
        capture=child/'evidence/audits'/row['id']/'CAPTURE_AUDIT.json'
        sealed_file(capture)
        if (read(capture)!=original or original['id']!=row['id']
                or original['audit_status']!='PASS_INDEPENDENT_PARTIAL_NEGATIVE_WITNESS'
                or original['raw_manifest_sha256']!=row['source_raw_manifest_sha256']
                or original['native_faults']!=0 or original['task_success'] is not False
                or row['admitted'] is not False or row['task_success'] is not False
                or row['native_faults']!=0 or row['unknown_effects']!=[]):
            raise PermissionError('VAP1_NEGATIVE_EXACT_ORIGINAL_CAPTURE_RECEIPT_AND_DENOMINATOR')
        sealed_file(partial,original['audit_sha256']);witness=read(partial)
        raw_file(raw/'FILE_MANIFEST.json',original['raw_manifest_sha256'])
        actual=read(raw/'RESULT.json');s1=read(raw/'S1_RESULT.json') if (raw/'S1_RESULT.json').exists() else {}
        if (witness.get('status')!='PASS_INDEPENDENT_PARTIAL_NEGATIVE_WITNESS'
                or witness.get('valid_scientific_negative_not_engineering_replacement') is not True
                or witness.get('manifest_sha256')!=original['raw_manifest_sha256']
                or witness.get('id')!=row['id'] or witness.get('complete') is not False
                or witness.get('admitted') is not False or witness.get('task_success') is not False
                or witness.get('rejected_prefix_live_ticks')!=0
                or actual.get('fixture') is not False or actual.get('model_calls')!=0
                or actual['physics_steps']!=witness['physics_steps'] or not 0<=actual['physics_steps']<16000
                or not any(s in (actual.get('error') or '') for s in
                    ('PRE_EXECUTION_SAFETY_REJECT','STATIC_QUALIFICATION_FAILED_NO_INTEGRATION'))
                or s1.get('live_native_failures',0)!=0 or s1.get('actions_modified',False)
                or s1.get('maximum_live_shadow_difference',0)>1e-10):
            raise PermissionError('VAP1_ONLY_PREVIOUSLY_AUDITED_SAFE_PARTIAL_NEGATIVE_NOT_ENGINEERING_FAILURE')
        for member in read(raw/'FILE_MANIFEST.json'):
            guard();name=Path(member['path']);p=raw/name
            if name.is_absolute() or '..' in name.parts or not p.resolve().is_relative_to(raw.resolve()):
                raise PermissionError('VAP1_NEGATIVE_RAW_MEMBER_ESCAPE')
            if p.stat().st_size!=member['bytes']:raise ValueError('VAP1_NEGATIVE_RAW_MEMBER_SIZE_CHANGED')
            raw_file(p,member['sha256'])
        movie=original['video'];video=child/movie['relative_path']
        if not video.resolve().is_relative_to(child.resolve()):raise PermissionError('VAP1_NEGATIVE_VIDEO_ESCAPE')
        if video.stat().st_size!=movie['bytes']:raise ValueError('VAP1_NEGATIVE_VIDEO_SIZE_CHANGED')
        raw_file(video,movie['sha256'])
        row['attempt_evidence']=dict(status=STATUS,partial_negative_audit_sha256=sha(partial),
            original_capture_audit_sha256=sha(capture),original_raw_manifest_sha256=original['raw_manifest_sha256'],
            complete_success_data=False,admitted=False,rejected_prefix_live_ticks=0,
            raw_and_video_rehashed=True,new_integrations=0,new_constructors=0)
    return result


def run(root):
    r=Path(root);m,inputs=validate(r,'PACK');work=read(r/'plans/WORK_PLAN.private.json')
    if work.get('operation')!='VAP1_REHASH_ATTEMPT_EVIDENCE':raise PermissionError('VAP1_EXACT_SAVED_ATTEMPT_AUDIT_OPERATION')
    lock=OwnerLock(r/'ATTEMPT_AUDIT.lock');began=time.process_time();ledger=read(r/'RESOURCE_LEDGER.json');failure=None
    assert ledger['status']=='RESERVED' and not ledger['unknown_effects'];bind_cpu(m)
    def guard(*unused):
        if (time.time()>=m['limits']['operation_deadline_epoch'] or time.process_time()-began>=m['limits']['CPU_s']-60
                or any((p/'STOP.json').exists() for p in (r,r.parent.parent))):
            raise PermissionError('VAP1_ATTEMPT_AUDIT_FINITE_CPU_CLOCK_OR_STOP')
    try:
        wait_priority(r,guard,priority);ledger['status']='RUNNING';write(r/'RESOURCE_LEDGER.json',ledger)
        rows=[];cases={}
        for source in work['closed_exports']:
            child=closed_child(r.parent.parent,source['root'],'EXPORT','TERMINAL_EXPORT',profile=work['bank_profile'])
            path=child/'evidence/PAIRED_EXPORT_RESULT.private.json';sealed_file(path,source['result_sha256'])
            rows.extend(read(path)['rows'])
            for case in read(child/'plans/WORK_PLAN.private.json')['cases']:
                if case['id'] in cases:raise PermissionError('VAP1_EXACT_FIRST_ATTEMPT_SET')
                cases[case['id']]=case
        if set(cases)!={x['id'] for x in rows}:raise PermissionError('VAP1_EXACT_NEGATIVE_AND_SUCCESS_DENOMINATOR')
        sealed_file(work['static_plan_path'],work['static_plan_sha256']);groups=read(work['static_plan_path'])['groups']
        actual=qualify(rows,cases,guard);gate=bank_gate(actual,groups,cumulative=work['bank_profile']=='VAP1_D48')
        result=dict(status='VAP1_ALL_ATTEMPTS_REHASHED_ORIGINAL_NEGATIVES_NOT_SUCCESS_DATA',
            rows=actual,gate=gate,HF_READY=False,new_physics=0,new_constructors=0,
            original_exports_modified=False,thresholds_changed=False)
        put(r/'evidence/ALL_ATTEMPT_EVIDENCE_RESULT.private.json',result);return result
    except BaseException as exc:failure=type(exc).__name__+': '+str(exc);raise
    finally:
        ledger.update(status='FAILED_PRESERVED' if failure else 'TERMINAL_PACK',CPU_s=time.process_time()-began,
            CPU_reserved_s=0,failure=failure);write(r/'RESOURCE_LEDGER.json',ledger);lock.close()
