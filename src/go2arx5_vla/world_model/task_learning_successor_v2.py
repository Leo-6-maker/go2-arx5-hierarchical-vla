"""Move sealed endpoint decisions and bind same-generation continuation.

Only small decision metadata is transported. Original evidence bytes and the
owner's completed seal remain unchanged; model/optimizer state stays on Hopper.
This module performs no queue submission, model load or scientific selection.
"""
import base64,hashlib,json
from pathlib import Path
from . import task_learning_endpoint_decision_v2 as scientific
from . import task_learning_training_profile_v1 as training
from .task_learning_grasp_v1 import need
from .storage import digest

SCHEMA='TASK_LEARNING_PRIVATE_ENDPOINT_DECISION_TRANSFER_V2'
FILES=('ENDPOINT_DECISION.private.json','SCIENTIFIC_EVIDENCE.private.json',
       'DECISION_CONTRACT.json','NATIVE_POPULATION.private.json','OWNER_SEAL.private.json')


def checked(decision,evidence,contract,population,seal):
    from scripts.pi0294.e1.journal import digest as object_digest
    result=scientific.evaluate(evidence)
    need(decision.get('schema')=='TASK_LEARNING_ENDPOINT_DECISION_V1'
        and contract==scientific.contract()==evidence['contract']
        and decision.get('scientific_result')==result
        and decision.get('action')==result['action']
        and decision.get('allows_execution')==result['allows_execution']
        and decision.get('next_endpoint')==result['next_endpoint']
        and all(decision.get(k)==evidence.get(k)for k in ('generation_id','dataset_sha256',
            'checkpoint_sha256','global_step'))
        and evidence.get('TEST_results_read')is False and decision.get('TEST_results_read')is False,
        'UNCHANGED_RECOMPUTED_SEALED_ENDPOINT_DECISION')
    need(population.get('summary',{}).get('complete')is True
        and population.get('planned_conditions')==population.get('attempted_conditions')==80
        and population.get('unrun_conditions')==0
        and population.get('outcomes')==evidence['native_outcomes']
        and population.get('groups')==evidence['provenance']['actual_native_groups'],
        'ACTUAL_COMPLETE_POPULATION_IN_DECISION_TRANSPORT')
    need(seal.get('status')=='COMPLETE'and object_digest(seal.get('result'))==seal.get('result_sha256')
        and (seal.get('result')or{}).get('status')=='ACTUAL_COMPLETE_DEV_ENDPOINT_SCIENTIFIC_DECISION_SEALED'
        and seal['result'].get('scientific_result')==result
        and seal['result'].get('training_started')is False and seal['result'].get('TEST_opened')is False,
        'COMPLETED_ORIGINAL_OWNER_SCIENTIFIC_SEAL_NOT_A_PERMISSION_BOOLEAN')
    return result


def export_bundle(root,seal_path):
    """Read the original owner seal after its full-population decision finishes."""
    root=Path(root);seal_path=Path(seal_path)
    paths={n:root/n for n in FILES if n!='OWNER_SEAL.private.json'}
    paths['OWNER_SEAL.private.json']=seal_path
    values={};files={}
    for name,path in paths.items():
        need(path.resolve()==path and path.is_file()and not path.is_symlink()
            and path.stat().st_size<=16*1024**2,'BOUNDED_CANONICAL_DECISION_METADATA')
        raw=path.read_bytes();values[name]=json.loads(raw)
        files[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),body=base64.b64encode(raw).decode())
    doc=values['ENDPOINT_DECISION.private.json'];e=values['SCIENTIFIC_EVIDENCE.private.json'];seal=values['OWNER_SEAL.private.json']
    result=checked(doc,e,values['DECISION_CONTRACT.json'],values['NATIVE_POPULATION.private.json'],seal)
    need(doc['scientific_evidence']==scientific.descriptor(paths['SCIENTIFIC_EVIDENCE.private.json'])
        and doc['decision_contract']==scientific.descriptor(paths['DECISION_CONTRACT.json'])
        and e['provenance']['population']==scientific.descriptor(paths['NATIVE_POPULATION.private.json'])
        and seal['result']['decision']==scientific.descriptor(paths['ENDPOINT_DECISION.private.json']),
        'OWNER_SEAL_BINDS_THE_EXACT_DECISION_AND_EVIDENCE_FILES')
    need(seal.get('message_id')==root.name.upper()+'-SEAL','ORIGINAL_OWNER_DECISION_SEAL_KEY')
    return dict(schema=SCHEMA,files=files,scientific_result=result,
        original_decision=scientific.descriptor(paths['ENDPOINT_DECISION.private.json']),
        original_owner_seal=scientific.descriptor(seal_path),model_or_optimizer_bytes=0)


def unpack(bundle):
    need(bundle.get('schema')==SCHEMA and set(bundle.get('files',{}))==set(FILES)
        and bundle.get('model_or_optimizer_bytes')==0,'EXACT_SMALL_DECISION_BUNDLE')
    raw={};values={}
    for name,v in bundle['files'].items():
        need(type(v.get('bytes'))is int and 0<v['bytes']<=16*1024**2,'BOUNDED_DECISION_MEMBER_SIZE')
        data=base64.b64decode(v['body'],validate=True)
        need(len(data)==v['bytes']and hashlib.sha256(data).hexdigest()==v['sha256'],
            'UNCHANGED_PRIVATE_DECISION_TRANSPORT_BYTES')
        raw[name]=data;values[name]=json.loads(data)
    result=checked(*(values[n]for n in FILES))
    doc=values['ENDPOINT_DECISION.private.json'];e=values['SCIENTIFIC_EVIDENCE.private.json']
    need(doc['scientific_evidence']['sha256']==bundle['files']['SCIENTIFIC_EVIDENCE.private.json']['sha256']
        and doc['decision_contract']['sha256']==bundle['files']['DECISION_CONTRACT.json']['sha256']
        and e['provenance']['population']['sha256']==bundle['files']['NATIVE_POPULATION.private.json']['sha256'],
        'OWNER_SEALED_EVIDENCE_AND_POPULATION_BYTES_UNCHANGED')
    need(bundle.get('scientific_result')==result
        and bundle['original_decision']['sha256']==bundle['files']['ENDPOINT_DECISION.private.json']['sha256']
        and bundle['original_owner_seal']['sha256']==bundle['files']['OWNER_SEAL.private.json']['sha256']
        and values['OWNER_SEAL.private.json']['result']['decision']==bundle['original_decision'],
        'ORIGINAL_OWNER_DECISION_AND_TRANSPORT_IDENTITY')
    return raw,values,result


def localized_decision(bundle,root):
    """Change two location descriptors, preserving the actual scientific content."""
    raw,values,result=unpack(bundle);root=Path(root)
    need(root.is_absolute()and root.resolve()==root,'CANONICAL_OWN_DECISION_DESTINATION')
    doc=dict(values['ENDPOINT_DECISION.private.json'])
    for field,name in [('scientific_evidence','SCIENTIFIC_EVIDENCE.private.json'),
                       ('decision_contract','DECISION_CONTRACT.json')]:
        doc[field]=dict(path=str(root/name),sha256=bundle['files'][name]['sha256'])
    return raw,doc,result


def continuation_fields(prior,producer_result,decision_reference):
    """Keep the original generation and complete optimizer checkpoint, never FIT4."""
    doc=scientific.load(decision_reference);step=producer_result['global_step']
    p=prior['recipe'];next_endpoint=doc.get('next_endpoint')
    need(step==p['endpoint']and step in (2000,4000)and p['seed']==20261109
        and producer_result.get('status')=='CHECKPOINT_SEALED'
        and producer_result.get('generation_id')==prior['generation_id']
        and producer_result.get('profile')==training.PROFILE
        and next_endpoint==(4000 if step==2000 else 8000),
        'ACTUAL_FORMAL_PREDECESSOR_AND_NEXT_DECLARED_ENDPOINT')
    checkpoint=Path(producer_result['checkpoint']);root=Path(prior['root'])
    need(checkpoint.resolve()==checkpoint and checkpoint.is_relative_to(root/'training'),
        'ORIGINAL_OWN_FORMAL_OPTIMIZER_CHECKPOINT_STAYS_ON_HOPPER')
    pin=digest(checkpoint/'CHECKPOINT_COMPLETE.json');saved=json.loads((checkpoint/'CHECKPOINT_COMPLETE.json').read_bytes())
    need(pin==producer_result['restore']['checkpoint_sha256']==doc['checkpoint_sha256']
        and saved['global_step']==step and saved['cursor']==step*32
        and saved['generation_id']==prior['generation_id']and saved['recipe']==p,
        'EXACT_FULL_STATE_PREDECESSOR_NOT_DIAGNOSTIC_OR_PACKAGE_WEIGHTS')
    recipe=training.recipe(p['dataset_sha256'],p['statistics_sha256'],endpoint=next_endpoint,
        microbatch=p['microbatch'],seed=p['seed'])
    value=dict(recipe=recipe,start_step=step,resume=dict(path=str(checkpoint),sha256=pin),
        generation_id=prior['generation_id'],endpoint_decision=decision_reference)
    scientific.validate_continuation(doc,value)
    need(all(recipe[k]==p[k]for k in recipe if k!='endpoint'),
        'SAME_DATA_SCOPE_MICROBATCH_SEED_AND_CONTINUOUS_8000_SCHEDULE')
    return value


def failure_summary(evidence):
    """Describe complete observed outcomes, without inventing a hidden stage."""
    from collections import Counter
    counts=scientific.native_counts(evidence['native_panel'],evidence['native_outcomes'])
    need(counts['complete']is True,'COMPLETE_COMMON_DEV_BEFORE_FAILURE_DIAGNOSIS')
    rows=[];buckets={m:Counter()for m in ('B0','B1')}
    for row in evidence['native_outcomes']:
        score=row.get('score')or{}
        if row['hard_safety_stop']:kind='HARD_SAFETY_STOP'
        elif row['technical_valid']is not True:kind='TECHNICAL_FAILURE'
        elif row['reset_ready']is not True:kind='RESET_NOT_READY'
        elif row.get('policy_proposal_rejected'):kind='POLICY_PROPOSAL_REJECTED'
        elif row['task_success']:kind='COMPLETE_TASK_SUCCESS'
        elif score.get('standing_passes')is False:kind='STANDING_SCORE_NOT_MET'
        elif score.get('maximum_object_lift_m',0)<.020:kind='NO_20MM_LIFT_OBSERVED'
        elif score.get('longest_contiguous_grasp_lift_hold_s',0)<1.:kind='ONE_SECOND_HOLD_NOT_OBSERVED'
        else:kind='UNSUCCESSFUL_CAUSE_UNRESOLVED'
        buckets[row['method']][kind]+=1
        rows.append(dict(method=row['method'],layout=row['layout'],restart=row['restart'],
            native_run=row['native_run'],observed_outcome=kind,first_failure=row.get('first_failure'),
            actual_GR00T_queries=row.get('actual_GR00T_queries',0)))
    return dict(schema='TASK_LEARNING_COMPLETE_DEV_OBSERVED_FAILURES_V2',
        requested=80,attempted=len(rows),unrun=0,methods=counts['methods'],
        outcome_counts={m:dict(v)for m,v in buckets.items()},conditions=rows,
        mechanism_proven=False,
        limits=['No-lift alone cannot distinguish approach, closure and lift mechanisms.',
                'Use actual native traces and deployable observations for causal diagnosis.',
                'These outcomes are not Teacher labels or new training demonstrations.'],
        model_updates=0,new_native_integrations=0,TEST_results_read=False)
