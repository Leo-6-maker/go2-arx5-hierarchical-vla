"""Read-only validation of original zero-step engineering evidence, never task success."""
import json
from pathlib import Path
from scripts.pi0294.native_dataset import sha

def zero_step(root,identifier,*,source_sha):
    root=Path(root);folder=root/'episodes'/identifier
    read=lambda p:json.loads(p.read_bytes())
    result=read(folder/'RESULT.json');adj=read(folder/'PV1_ADJUDICATION.json')
    receipt=root/'evidence/PV1_AUDIT_RECEIPTS'/(identifier+'.json');audit=read(receipt)
    def need(ok,why):
        if not ok:raise PermissionError('V5_CD_ENGINEERING_'+why)
    need(adj['status']=='ENGINEERING_INCOMPLETE'and adj['pipeline_trustworthy']is False
        and result['status']=='FAILED_PRESERVED'and result['source_sha']==source_sha
        and result['id']==adj['attempt_id']==identifier
        and result['physics_steps']==result['model_calls']==adj['main_integrations']==0
        and adj['S1_rejections']==adj['live_native_failures']==0
        and result['physics_starts']==1 and result['rgb_pairs']==1
        and adj['result_sha256']==sha(folder/'RESULT.json'),'EXACT_ORIGINAL_ZERO_STEP_RESULT')
    need(audit['status']=='PASS_ZERO_INTEGRATION_ENGINEERING_EVIDENCE'
        and audit['attempt_id']==identifier and audit['execution_source_sha']==source_sha
        and audit['adjudication']==adj and audit['errors']==[]
        and audit['task_result_available']is False and audit['scientific_policy_outcome_valid']is False
        and audit['physical_integrations']==0 and audit['score']is None
        and audit['checks']and all(v is True for v in audit['checks'].values()),'REAL_ORIGINAL_AUDIT_NOT_MISSING_CAUSAL_AS_SUCCESS')
    manifest=read(folder/'FILE_MANIFEST.json')
    need(all((folder/x['path']).is_file()and (folder/x['path']).stat().st_size==x['bytes']
        and sha(folder/x['path'])==x['sha256']for x in manifest),'ALL_ORIGINAL_RAW_BYTES')
    need(all(not list((folder/'relay'/kind).glob('*.json'))for kind in('intents','commands','acks'))
        and read(folder/'samples.json')==read(folder/'references.json')==[]
        and not(folder/'SCORE.json').exists()and not(folder/'physics.npz').exists(),'NO_COMMAND_OR_PHYSICS_OR_TASK_SCORE')
    return dict(status='CONFIRMED_ORIGINAL_ZERO_STEP_ENGINEERING_NOT_VALID_CONDITION',
        attempt_id=identifier,result_sha256=sha(folder/'RESULT.json'),adjudication_sha256=sha(folder/'PV1_ADJUDICATION.json'),
        original_audit_sha256=sha(receipt),valid_conditions=0,new_query_or_physics=0)
