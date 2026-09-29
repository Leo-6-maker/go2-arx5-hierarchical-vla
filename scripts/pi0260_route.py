"""Thin R1 evidence routing; no scheduler or background loop."""
import json

WORKFLOW='1f27d9c14ce0a91518930a8d8a2f6f0ffc2dd468'
OWNER='PI0260_SINGLE_OWNER_M'


def route(expected_hash, actual_hash, data, replay, helper=None, job=None,
          competing_owner=False, saved_result=False, all_outputs=False, presentation=False, arm_quality=None):
    if expected_hash!=actual_hash:return 'DENY_CHANGED_INPUT'
    if competing_owner:return 'DENY_DUPLICATE_OWNER'
    if job:return 'ADOPT_EXISTING_JOB'
    if data!='PASS':return 'DENY_DATA'
    if replay!='PASS':
        if helper and helper.get('hash')==actual_hash and helper.get('status')=='PASS':return 'REUSE_MATCHING_REPLAY'
        return 'RUN_SOURCE_SPECIFIC_REPLAY'
    if saved_result:return 'REUSE_CANONICAL_RESULT'
    if all_outputs:return 'COMPLETE'
    return 'ADMIT_NEXT_STAGE'


def test():
    cases=[
        ('no_helper',{},'ADMIT_NEXT_STAGE'),
        ('data_fail_helper_pass',dict(data='FAIL',helper=dict(hash='a',status='PASS')),'DENY_DATA'),
        ('generic_partial',dict(replay='PARTIAL'),'RUN_SOURCE_SPECIFIC_REPLAY'),
        ('matching_helper_once',dict(replay='NOT_RUN',helper=dict(hash='a',status='PASS')),'REUSE_MATCHING_REPLAY'),
        ('presentation_pending',dict(presentation=False),'ADMIT_NEXT_STAGE'),
        ('changed_source_hash',dict(actual_hash='b'),'DENY_CHANGED_INPUT'),
        ('duplicate_owner',dict(competing_owner=True),'DENY_DUPLICATE_OWNER'),
        ('canonical_reuse',dict(saved_result=True),'REUSE_CANONICAL_RESULT'),
        ('arm_quality_fail',dict(arm_quality='FAIL'),'ADMIT_NEXT_STAGE'),
        ('all_complete_no_helper',dict(all_outputs=True),'COMPLETE'),
        ('healthy_job',dict(job='verified-live-pid'),'ADOPT_EXISTING_JOB')]
    results=[]
    for name,changes,want in cases:
        args=dict(expected_hash='a',actual_hash='a',data='PASS',replay='PASS');args.update(changes)
        got=route(**args);assert got==want,(name,got,want);results.append(dict(case=name,result=got))
    return dict(status='PASS',workflow=WORKFLOW,cases=results,model_queries=0,optimizer_updates=0)


if __name__=='__main__':print(json.dumps(test(),indent=2))
