"""Assemble the frozen scientific decision from actual transferred/native data.

The existing owner seals this once after closed DEV groups. No sampling, model
load, training or continuation launch occurs here; negative outcomes are kept.
"""
import json,os,subprocess,time
from pathlib import Path
from contextlib import closing
from . import task_learning_endpoint_decision_v2 as decision
from .task_learning_grasp_v1 import need
from .storage import digest


def delivery(root):
    root=Path(root);r=json.loads((root/'RESULT.private.json').read_bytes())
    lease=json.loads((root/'control/LEASE.json').read_bytes())
    need(root.resolve()==root and r.get('status')=='PASS_PRIVATE_TASK_POLICY_PACKAGE_TRANSFER_V2'
        and lease.get('status')=='CLOSED'and lease.get('worker_exit_code')==r.get('exit_code')==0
        and lease.get('result_sha256')==digest(root/'RESULT.private.json'),
        'ACTUAL_CLOSED0_PRIVATE_CANDIDATE_DELIVERY')
    # Large immutable model deltas were hashed on transfer and checked on Core
    # reconstruction. Recheck small producer evidence and retained probe arrays.
    files=r['all_delivered_files_sha256']
    for name,pin in files.items():
        if not name.startswith('producer/'):continue
        path=root/name
        need('..'not in Path(name).parts and path.resolve()==path and path.is_file()
            and digest(path)==pin,'SEALED_TRANSFERRED_ENDPOINT_EVIDENCE_'+name)
    p=root/'package/PACKAGE_MANIFEST.json';m=json.loads(p.read_bytes())
    original=root/'producer/RESULT.json';training=json.loads(original.read_bytes())
    spec=json.loads((root/'producer/EXECUTION_SPEC.json').read_bytes())
    cp=root/'producer/checkpoint/CHECKPOINT_COMPLETE.json';saved=json.loads(cp.read_bytes())
    need(digest(p)==r['package']['sha256']and r['package']['path']==str(root/'package')
        and digest(cp)==r['checkpoint_sha256']==m['checkpoint_sha256']
        and r['global_step']==m['global_step']==saved['global_step']==training['global_step']
        and r['generation_id']==m['generation_id']==saved['generation_id']==training['generation_id']
        and saved['recipe']==m['recipe']==spec['recipe']
        and m['recipe']['dataset_sha256']==r['dataset_sha256']
        and m['recipe']['statistics_sha256']==r['statistics_sha256'],
        'SAME_ACTUAL_ENDPOINT_CHECKPOINT_PACKAGE_RECIPE_AND_DATA')
    return dict(r,training_seed=m['recipe']['seed'])


def assemble(delivery_root,panel_reference,groups,*,previous_evidence=None):
    from .task_learning_student_population_v2 import aggregate
    r=delivery(delivery_root);root=Path(delivery_root)
    population=aggregate(panel_reference,groups);panel=decision.load(panel_reference)
    need(population['summary']['complete']is True,'PARTIAL_DEV_ONLY_CONTINUE_UNEXECUTED_OR_REPAIR')
    need(panel['policies']['B1']['package']==r['package']and panel['B1_global_step']==r['global_step'],
        'COMPLETE_NATIVE_DEV_OF_THE_ACTUAL_DELIVERED_CANDIDATE')
    step=r['global_step'];before={};after={}
    identity={k:r[k]for k in('generation_id','dataset_sha256','statistics_sha256')}
    for split,label in [('TRAIN','restored'),('DEV','restored-DEV')]:
        after[split]=decision.probe_summary(root/'producer'/label,split=split,global_step=step,**identity)
    e=dict(schema='TASK_LEARNING_ENDPOINT_SCIENTIFIC_EVIDENCE_V2',contract=decision.contract(),
        global_step=step,training_seed=r['training_seed'],**identity,checkpoint_sha256=r['checkpoint_sha256'],package_sha256=r['package']['sha256'],
        native_panel=panel,native_outcomes=population['outcomes'],open_after=after,
        provenance=dict(delivery=decision.descriptor(root/'RESULT.private.json'),
            delivery_lease=decision.descriptor(root/'control/LEASE.json'),panel=panel_reference,
            checkpoint=decision.descriptor(root/'producer/checkpoint/CHECKPOINT_COMPLETE.json'),
            actual_native_groups=population['groups']),TEST_results_read=False)
    if step==2000:
        need(previous_evidence is None,'FIRST2000_USES_OWN_STEP0_NOT_FIT4_OR_ANOTHER_GENERATION')
        for split,label in [('TRAIN','before'),('DEV','before-DEV')]:
            before[split]=decision.probe_summary(root/'producer'/label,split=split,global_step=0,**identity)
    else:
        need(previous_evidence is not None,'PREVIOUS_COMPLETE_CANDIDATE_EVIDENCE_REQUIRED')
        prior=decision.load(previous_evidence);prior_result=decision.evaluate(prior)
        spec=json.loads((root/'producer/EXECUTION_SPEC.json').read_bytes())
        need(prior.get('schema')=='TASK_LEARNING_ENDPOINT_SCIENTIFIC_EVIDENCE_V2'
            and all(prior[k]==r[k]for k in identity)
            and prior_result.get('allows_execution')is True and prior_result.get('next_endpoint')==step
            and prior['checkpoint_sha256']==spec['resume']['sha256'],
            'PREVIOUS_APPROVED_FULL_ENDPOINT_AND_EXACT_PRODUCER_RESUME')
        previous_delivery=Path(prior['provenance']['delivery']['path']).parent
        old=delivery(previous_delivery)
        need(old['checkpoint_sha256']==prior['checkpoint_sha256']
            and old['global_step']==prior['global_step'],'EXACT_PREVIOUS_DELIVERY_NOT_CURRENT_B0_BASELINE')
        for split,label in [('TRAIN','restored'),('DEV','restored-DEV')]:
            before[split]=decision.probe_summary(previous_delivery/'producer'/label,
                split=split,global_step=prior['global_step'],**identity)
        need(before==prior['open_after'],'PREVIOUS_MATCHED_PROBE_ARRAYS_STILL_SEALED')
        e.update(previous_native_panel=prior['native_panel'],previous_native_outcomes=prior['native_outcomes'])
        e['provenance']['previous_scientific_evidence']=previous_evidence
    e['open_before']=before
    return e,population,decision.evaluate(e)


def seal(request):
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.e1.journal import Journal
    from .task_learning_student_registration_v2 import actual_site
    panel=decision.load(request['panel']);native=decision.load(panel['registration']['native_base'])
    campaign=Path(native['campaign_root']);actual_site(campaign)
    root=Path(request['root']);delivery_root=Path(request['delivery_root'])
    r=delivery(delivery_root)
    need(root.parent==delivery_root.parent and root.parent.resolve()==root.parent
        and root.name==f'task-learning-endpoint-decision-step{r["global_step"]}-pkg{r["package"]["sha256"][:12]}',
        'ONE_SCIENTIFIC_DECISION_ROOT_PER_ACTUAL_CANDIDATE')
    for group in request['groups']:
        q=subprocess.run(['systemctl','--user','show','wm02-'+Path(group).name,
            '--property=MainPID,ActiveState,SubState'],capture_output=True,text=True,timeout=10,check=True)
        need('MainPID=0'in q.stdout,'ACTUAL_DEV_GROUP_PROCESS_TERMINAL_BEFORE_SCIENTIFIC_DECISION')
    frozen=decision.load(request['decision_contract'])
    need(frozen==decision.contract(),'EXACT_PUBLISHED_PREDECLARED_ENDPOINT_RULE')
    evidence,population,result=assemble(delivery_root,request['panel'],request['groups'],
        previous_evidence=request.get('previous_evidence'))
    j=Journal(campaign/'control/journal',campaign.name);key=root.name.upper()+'-SEAL'
    with closing(OwnerLock(campaign/'OWNER.lock')):
        need(not root.exists()and not j.path(key).exists(),'UNKNOWN_SCIENTIFIC_DECISION_NEVER_REWRITE')
        def effect():
            root.mkdir()
            write(root/'DECISION_CONTRACT.json',frozen);write(root/'NATIVE_POPULATION.private.json',population)
            evidence['provenance']['population']=decision.descriptor(root/'NATIVE_POPULATION.private.json')
            write(root/'SCIENTIFIC_EVIDENCE.private.json',evidence)
            doc=dict(schema='TASK_LEARNING_ENDPOINT_DECISION_V1',
                scientific_evidence=decision.descriptor(root/'SCIENTIFIC_EVIDENCE.private.json'),
                decision_contract=decision.descriptor(root/'DECISION_CONTRACT.json'),scientific_result=result,
                action=result['action'],allows_execution=result['allows_execution'],next_endpoint=result['next_endpoint'],
                **{k:r[k]for k in('generation_id','dataset_sha256','statistics_sha256','checkpoint_sha256')},
                global_step=r['global_step'],created_epoch=time.time(),TEST_results_read=False)
            write(root/'ENDPOINT_DECISION.private.json',doc)
            if result['action']=='FREEZE_FOR_SEED_CONFIRMATION_BEFORE_TEST'and r['training_seed']==20261109:
                write(root/'FIRST_SEED_GAIN_DECISION.private.json',dict(schema='TASK_LEARNING_FIRST_SEED_GAIN_DECISION_V1',
                    allows_execution=True,dataset_sha256=r['dataset_sha256'],first_seed=20261109,
                    first_seed_generation_id=r['generation_id'],allowed_confirmation_seeds=[20261110,20261111],
                    scientific_evidence=doc['scientific_evidence'],decision_contract=doc['decision_contract'],
                    scientific_result=result,TEST_results_read=False))
            return dict(status='ACTUAL_COMPLETE_DEV_ENDPOINT_SCIENTIFIC_DECISION_SEALED',
                decision=decision.descriptor(root/'ENDPOINT_DECISION.private.json'),scientific_result=result,
                training_started=False,TEST_opened=False)
        return j.once(key,dict(panel=request['panel'],checkpoint_sha256=r['checkpoint_sha256']),effect)
