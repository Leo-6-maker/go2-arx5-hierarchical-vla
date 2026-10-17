"""Account for engineering attempts stopped before the first physical step."""
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import sha


def audit(root,identifier,video_writer,*,adjudication_name='E1_ADJUDICATION.json'):
    if adjudication_name not in ['E1_ADJUDICATION.json','MP1_ADJUDICATION.json']:
        raise PermissionError('EXACT_E1_OR_MP1_ADJUDICATION_ONLY')
    root=Path(root);folder=root/'episodes'/identifier;out=root/'evidence/zero-step-audits'/identifier
    if out.exists():return read(out/'AUDIT.private.json')
    out.mkdir(parents=True,mode=0o700);checks={};errors=[];value=dict(attempt_id=identifier)
    def check(name,condition):
        checks[name]=bool(condition)
        if not condition:errors.append(name)
    try:
        a=read(folder/adjudication_name);result=read(folder/'RESULT.json');worker=read(folder/'WORKER_RECEIPT.json');card=read(folder/'CARD.json')
        check('actual_attempt_identity',a['attempt_id']==card['id']==result['id']==worker['id']==identifier)
        check('actual_engineering_not_policy_outcome',a['status']=='ENGINEERING_INCOMPLETE' and a['pipeline_trustworthy'] is False)
        check('actual_native_step_and_command_counts_zero',result['physics_steps']==worker['physics_steps']==a['main_integrations']==0)
        check('adjudication_original_result_sha',a['result_sha256']==sha(folder/'RESULT.json'))
        check('actual_source_matches_worker',result['source_sha']==worker['source_sha'])
        manifest=read(folder/'FILE_MANIFEST.json')
        check('all_existing_raw_manifest_members_exact',all((folder/x['path']).is_file() and (folder/x['path']).stat().st_size==x['bytes'] and sha(folder/x['path'])==x['sha256'] for x in manifest))
        check('no_submitted_supervision_or_reference_labels',read(folder/'samples.json')==[] and read(folder/'references.json')==[])
        check('no_command_intent_command_or_ACK',all(not list((folder/'relay'/kind).glob('*.json')) for kind in ['intents','commands','acks']))
        with np.load(folder/'highrate.npz',allow_pickle=False) as z:check('actual_highrate_empty',z['rows'].shape[0]==0)
        s1=read(folder/'S1_RESULT.json');check('no_S1_prefix_or_native_violation',s1['shadow_invocations']==s1['pre_execution_rejections']==s1['live_native_failures']==0)
        paths=[sorted((folder/name).glob('*.png')) for name in ['body_rgb','wrist_rgb']]
        check('exact_one_initial_paired_RGB',len(paths[0])==len(paths[1])==worker['rgb_pairs']==1 and paths[0][0].name==paths[1][0].name)
        for i,files in enumerate(paths):
            with Image.open(files[0]) as image:check('initial_RGB_'+str(i),image.mode=='RGB' and image.size==(256,256))
        check('no_physics_or_task_score_fabricated',not (folder/'physics.npz').exists() and not (folder/'SCORE.json').exists())
        prior=root/'evidence/audits'/identifier/'AUDIT.private.json'
        movie=video_writer(folder,out/'actual-initial-body-wrist.mp4',1)
        movie.update(logical_id=('MP1_ZERO_STEP/' if adjudication_name=='MP1_ADJUDICATION.json' else 'E1_ZERO_STEP/')+identifier,relative_path=str((out/'actual-initial-body-wrist.mp4').relative_to(root)),native_duration_s=0.,
            initial_frame_only=True,task_trajectory_video=False)
        value.update(status='PASS_ZERO_INTEGRATION_ENGINEERING_EVIDENCE' if not errors else 'FAIL_ZERO_INTEGRATION_EVIDENCE',
            adjudication=a,score=None,video=movie,actual_native_safety_flag_rows=0,physical_integrations=0,
            model_autonomous_evaluation=True,execution_source_sha=result['source_sha'],
            task_result_available=False,scientific_policy_outcome_valid=False,
            prior_incomplete_audit_sha256=sha(prior) if prior.exists() else None,
            prior_incomplete_audit_preserved=True,missing_physics_not_synthesized=True)
    except Exception as exc:value.update(status='ZERO_INTEGRATION_AUDIT_INCOMPLETE',error=type(exc).__name__+': '+str(exc))
    value.update(checks=checks,errors=errors);write(out/'AUDIT.private.json',value);return value
