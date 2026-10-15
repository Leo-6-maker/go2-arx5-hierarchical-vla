"""Rebind engineering-only source additions to sealed P1, without new queries."""
from pathlib import Path
import numpy as np
from .mps2_cpu import boundaries,first_input
from .mps2_geometry import packages
from .mps2_runtime import FOLDER,cards,require_adoption
from .task_capture_runtime import read,sha

def run(root,manifest):
    from scripts.pi0292.durable import write
    root=Path(root);p=root/FOLDER;require_adoption(root,manifest);old=read(p/'CPU_ADMISSION.json')
    checks=boundaries(root);identities=[]
    for r in read(p/'SOURCE_LOCK.json')['members'][:12]:
        if sha(Path(__file__).resolve().parents[2]/r['path'])!=r['sha256']:raise ValueError('MPS2_FROZEN_MEMBER_CHANGED')
    for entry in read(p/'CONDITIONS.json')['conditions']:
        if entry.get('phase')!='CPU_SAVED_INPUT':continue
        engine,state,mount=first_input(root,entry);card=next(c for c in cards() if c['task']==entry['task'] and c['duration_s']==12.)
        candidates=packages(engine,state,entry['task'],card);folder=p/'requests'/entry['id']/'0000';request=read(folder/'INPUT.private.json')
        for i,candidate in enumerate(candidates):
            for j,endpoint in enumerate(['approach','contact']):
                if not np.array_equal(candidate[endpoint],np.asarray(request['goals'][2*i+j]['pose'])):raise ValueError('MPS2_NEW_SOURCE_TARGETS_NOT_IDENTICAL')
        result=read(folder/'FINAL.private.json')
        if result['input_sha256']!=sha(folder/'INPUT.private.json'):raise ValueError('MPS2_FIRST_INPUT_CHANGED')
        identities.append(dict(condition=entry['id'],input_sha256=sha(folder/'INPUT.private.json'),first_result_sha256=sha(folder/'FINAL.private.json'),
            original_status=result['status'],new_candidate_queries=0))
    proof=dict(status='PASS_SAME_INPUT_AND_FIXED_CORE_NO_REQUERY',source_sha=manifest['code']['vla_commit'],
        original_CPU_sha256=sha(p/'CPU_ADMISSION.json'),original_CPU_source=old['source_sha'],original_checks_passed=old['checks_passed'],
        narrow_checks_reexecuted=len(checks),identities=identities,actual_search_counters=read(p/'SEARCH_LEDGER.json')['counters'],
        original_certificate_summary_deviation='P1 certificate IK_calls/RRT_calls defaulted to zero; authoritative durable slots preserve actual calls; source now populates call summaries. Original receipt retained.',
        new_IK_RRT_calls=0,new_physics=0,old_evidence_preserved=True)
    out=p/'CPU_EXECUTION_BINDING.json'
    if out.exists():
        previous=p/('CPU_BINDING_BEFORE_'+read(out)['source_sha']+'.json')
        if not previous.exists():write(previous,read(out))
    write(out,proof);return proof
