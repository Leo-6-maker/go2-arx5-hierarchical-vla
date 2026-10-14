"""One finite synthetic CPU contract check, never an actual qualification."""
from copy import deepcopy
from pathlib import Path
import argparse
import sys
if not __package__:
    sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parents[2]/'src')]
    __package__='scripts.pi0294'
from scripts.pi0292.durable import write
from .continuation import RUN,PARENT,AUTHORITY,GO_COMMENT,T0,LIMITS,MODE_AUTHORITY,validate_authority,validate_mode,sha


def check(directory):
    root=Path(directory);root.mkdir(exist_ok=False)
    m=dict(schema='RUN_MANIFEST_V1',run_uuid=RUN,pi='PI0294',owner='EXISTING_LOCAL_PC',
        authority=dict(protocol_ref=AUTHORITY,go_comment=GO_COMMENT,approval='DIRECT_HUMAN_R3_MODE_SEPARATED_PACKAGE'),
        parent_run_uuid=PARENT,limits=LIMITS,mode_authority=MODE_AUTHORITY)
    validate_authority(m,T0+1)
    wrong=deepcopy(m);wrong['mode_authority']['OFFLINE_TRAIN_ALLOWED_WITH_P1_D1_UNQUALIFIED']=False
    try:validate_authority(wrong,T0+1)
    except PermissionError:pass
    else:raise AssertionError('unadopted mode accepted')
    parent=root/PARENT;checkpoint=parent/'training/main/checkpoint-512'
    write(root/'PARENT_BINDING.private.json',dict(root=str(parent)))
    proof=root/'LOCAL_READY.fixture.json'
    write(proof,dict(status='LOCAL_CHECKPOINT_READY',global_step=512,consumed_samples=16384,
        full_member_SHA_verified=True,source_directory=str(checkpoint),fixture=True))
    request=dict(stage='main',mode='main',allowed_modes=['main'],max_steps=4096,stop_at=2048,
        resume=str(checkpoint),local_M512_ready_relative=proof.name,local_M512_ready_sha256=sha(proof))
    validate_mode(root,request)
    try:validate_mode(root,dict(request,stop_at=4096))
    except PermissionError:pass
    else:raise AssertionError('M4096 accepted')
    write(root/'S1.fixture.json',dict(run_uuid=RUN,status='S1_TEACHER_QUALIFIED',
        teacher_complete_successes=3,teacher_rejects=0,live_native_failures=0,clone_contexts_passed=8,fixture=True))
    try:validate_mode(root,dict(stage='resident',S1_qualification_relative='S1.fixture.json',S1_qualification_sha256=sha(root/'S1.fixture.json')))
    except PermissionError:pass
    else:raise AssertionError('Teacher3 allows live')
    result=dict(status='PASS_R3_MODE_CPU_CONTRACT_ONLY',fixture=True,offline_not_waiting_P1_D1=True,
                M4096_rejected=True,unqualified_live_rejected=True,model_physics_optimizer=0)
    write(root/'RESULT.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    print(check(p.parse_args().output))
