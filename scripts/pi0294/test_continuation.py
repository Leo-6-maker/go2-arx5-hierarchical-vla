"""CPU fixtures prove boundary rejection, not a model or physics result."""
import copy
import tempfile
import unittest
import uuid
from pathlib import Path

from scripts.pi0292.durable import write
from . import continuation as c
from .continuation_session import validate_stage
from .native_dataset import sha
from .relay_transport import model_registry
from .test_model_registry import ModelRegistryTests


class ContinuationTests(unittest.TestCase):
    def manifest(self):
        return dict(schema='RUN_MANIFEST_V1',run_uuid=c.RUN,pi='PI0294',owner='EXISTING_LOCAL_PC',
            parent_run_uuid=c.PARENT,authority=dict(protocol_ref=c.AUTHORITY,go_comment=c.GO_COMMENT,
            approval='DIRECT_HUMAN_R3_MODE_SEPARATED_PACKAGE'),limits=copy.deepcopy(c.LIMITS),
            mode_authority=copy.deepcopy(c.MODE_AUTHORITY))

    def test_exact_new_go_and_no_reset(self):
        value=self.manifest()
        c.validate_authority(value,c.T0+1)
        broken=copy.deepcopy(value);broken['authority']['approval']='USER_CHAT_COMPLETE_SECTION5_PACKAGE'
        with self.assertRaises(PermissionError):c.validate_authority(broken,c.T0+1)
        for key in c.MODE_AUTHORITY:
            broken=copy.deepcopy(value);broken['mode_authority'][key]=False
            with self.assertRaises(PermissionError):c.validate_authority(broken,c.T0+1)
        for key,changed in [('run_uuid',c.PARENT),('parent_run_uuid','other'),('owner','another')]:
            broken=copy.deepcopy(value);broken[key]=changed
            with self.assertRaises(PermissionError):c.validate_authority(broken,c.T0+1)
        for key in c.LIMITS:
            broken=copy.deepcopy(value);broken['limits'][key]=True if key=='reset_on_resume' else c.LIMITS[key]+1
            with self.assertRaises(PermissionError):c.validate_authority(broken,c.T0+1)
        with self.assertRaises(PermissionError):c.validate_authority(value,c.CUTOFF)
        c.validate_authority(value,c.CUTOFF,closeout=True)
        with self.assertRaises(PermissionError):c.validate_authority(value,c.DEADLINE,closeout=True)

    def allocation(self):
        now=c.T0+100
        request=dict(stage='continuation',submit_count=1,walltime_s=21600,prior_GPU_allocated_s=0,
            request_nonce=str(uuid.uuid4()),numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',
            microbatch=2,workers=0,allowed_peer_ids=[],predicted_remaining_peak_bytes=80*c.GIB)
        job=dict(job_state='R',Job_Owner='own@hopper',project=c.PROJECT,exec_host='compute/0',
                 Resource_List=dict(walltime='06:00:00',ngpus=1))
        project=dict(observed_epoch=now-10,balance_su=1,project=dict(name=c.PROJECT,active=True,
            users=['own'],cluster=['hopper-m-02'],start_date='2026-10-01',end_date='2026-11-01'))
        return dict(now=now,user='own',job_id='1',jobs={'1':job},hostname='compute',
            quota=dict(user='own',report_epoch=now,hard_bytes=200*c.GIB,used_bytes=90*c.GIB),
            project_receipt=project,request=request)

    def test_native_job_project_quota_and_budget_gates(self):
        value=self.allocation();c.validate_allocation(**value)
        for key,new in [('submit_count',4),('walltime_s',28800),('prior_GPU_allocated_s',7201),
                         ('microbatch',4),('workers',2),('stage','night-final'),('stage','checkpoint')]:
            broken=copy.deepcopy(value);broken['request'][key]=new
            with self.assertRaises(PermissionError):c.validate_allocation(**broken)
        for key,new in [('hostname','hopper-l-01'),('job_id',None),('now',c.CUTOFF)]:
            broken=copy.deepcopy(value);broken[key]=new
            with self.assertRaises(PermissionError):c.validate_allocation(**broken)
        broken=copy.deepcopy(value);broken['quota']['used_bytes']=101*c.GIB
        with self.assertRaises(PermissionError):c.validate_allocation(**broken)
        broken=copy.deepcopy(value);broken['project_receipt']['project']['active']=False
        with self.assertRaises(PermissionError):c.validate_allocation(**broken)

    def test_resume_only_and_no_new_scheduler(self):
        value=self.allocation();value['request'].update(stage='main',max_steps=4096,stop_at=2048,resume='/local/M512',walltime_s=14400)
        value['jobs']['1']['Resource_List']['walltime']='04:00:00'
        c.validate_allocation(**value)
        for change in (dict(resume=None),dict(max_steps=6144),dict(stop_at=512)):
            broken=copy.deepcopy(value);broken['request'].update(change)
            with self.assertRaises(PermissionError):c.validate_allocation(**broken)

    def test_cpu_queue_is_eval_only_and_incremental(self):
        m=self.manifest()
        a=dict(run_uuid=c.RUN,T0_epoch=c.T0,cutoff_epoch=c.CUTOFF,deadline_epoch=c.DEADLINE,
            lease_expires_epoch=c.DEADLINE,allowed_cpus=list(range(8)),
            limits=dict(physics_starts=32,physics_steps=512000,rgb_pairs=20512,cpu_s=86400,output_bytes=16*c.GIB))
        p=dict(kind='MODEL_EVALUATION',not_formal_data=True,episodes=[dict(card=dict(relay_request_relative='sealed.json'))])
        c.validate_cpu_admission(m,a,p,1,c.T0+1)
        c.validate_cpu_admission(m,a,p,2,c.T0+1)
        for changes in (dict(kind='COLLECTION'),dict(not_formal_data=False)):
            with self.assertRaises(PermissionError):c.validate_cpu_admission(m,a,dict(p,**changes),1,c.T0+1)
        for k in a['limits']:
            bad=copy.deepcopy(a);bad['limits'][k]+=1
            with self.assertRaises(PermissionError):c.validate_cpu_admission(m,bad,p,1,c.T0+1)
        with self.assertRaises(PermissionError):c.validate_cpu_admission(m,a,p,4,c.T0+1)

    def test_same_job_initial_auth_can_age_not_refresh_per_stage(self):
        value=self.allocation();value['now']+=2000
        with self.assertRaises(PermissionError):c.validate_allocation(**value)
        # File/process parent proofs are separately checked by admit().
        value['initial']=dict(observed_epoch=c.T0+100)
        value['quota']['report_epoch']=value['now']
        c.validate_allocation(**value)

    def test_archive_blocked_is_not_local_ready_or_foreign_transfer(self):
        args=ModelRegistryTests().fixture()
        args.update(run_uuid=c.RUN,now=c.T0+100,deadline_epoch=c.T0+2000,continuation_manifest=self.manifest())
        args['panel']['run_uuid']=c.PARENT
        args['delivery'].update(status='LOCAL_CHECKPOINT_READY',run_uuid=c.RUN,
            consumer_host_role='hopper-neural',inventory_sha256=args['inventory_sha256'],full_member_SHA_verified=True)
        del args['delivery']['revision'];del args['delivery']['prefix']
        value=model_registry(**args)
        self.assertEqual(value['checkpoint_identity']['HF_revision'],'NOT_APPLICABLE_LOCAL_CONSUMER')
        self.assertEqual(value['checkpoint_identity']['relay_identity']['model_revision'],'d'*40)
        for changes in (dict(status='REMOTE_ARCHIVE_BLOCKED'),dict(full_member_SHA_verified=False),
                        dict(consumer_host_role='corelab'),dict(inventory_sha256='9'*64)):
            broken=copy.deepcopy(args);broken['delivery'].update(changes)
            with self.assertRaises(PermissionError):model_registry(**broken)

    def test_directive_cannot_reopen_smoke_or_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'child.json'
            parent=dict(root=folder,source_sha='a'*40,run_manifest_sha256='b'*64,
                code_seal_sha256='c'*64,bindings='private',bindings_sha256='d'*64,
                walltime_s=21600,submit_count=1,prior_GPU_allocated_s=0,
                numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',microbatch=2,workers=0,
                request_nonce=str(uuid.uuid4()),project_admission_sha256='e'*64,allowed_peer_ids=[])
            child=dict(parent,stage='resident',mode='resident',allowed_modes=['resident'])
            proof=root/'S1_TEACHER_QUALIFIED.json'
            write(proof,dict(run_uuid=c.RUN,status='S1_TEACHER_QUALIFIED',teacher_complete_successes=4,
                teacher_rejects=0,live_native_failures=0,clone_contexts_passed=8))
            child.update(S1_qualification_relative=proof.name,S1_qualification_sha256=sha(proof))
            write(path,child)
            stage=dict(run_uuid=c.RUN,job_id='1',sequence=0,mode='resident',request_relative='child.json',request_sha256=sha(path))
            self.assertEqual(validate_stage(root,parent,stage,job_id='1',sequence=0),path)
            for changed in (dict(mode='checkpoint'),dict(mode='smoke'),dict(sequence=1),dict(request_relative='../child.json')):
                broken=dict(stage,**changed)
                with self.assertRaises(PermissionError):validate_stage(root,parent,broken,job_id='1',sequence=0)
            ready=dict(run_uuid=c.RUN,status='S1_TEACHER_QUALIFIED',teacher_complete_successes=4,
                teacher_rejects=0,live_native_failures=0,clone_contexts_passed=8)
            for key,value in (('teacher_complete_successes',3),('teacher_rejects',1),
                              ('live_native_failures',1),('clone_contexts_passed',7)):
                write(proof,dict(ready,**{key:value}))
                child['S1_qualification_sha256']=sha(proof);write(path,child)
                stage['request_sha256']=sha(path)
                with self.assertRaises(PermissionError):validate_stage(root,parent,stage,job_id='1',sequence=0)
            write(root/'PARENT_BINDING.private.json',dict(root=str(root/c.PARENT)))
            child.update(stage='main',mode='main',allowed_modes=['main'],max_steps=4096,stop_at=2048,resume='/local/M512')
            write(path,child);stage.update(mode='main',request_sha256=sha(path))
            with self.assertRaises(PermissionError):validate_stage(root,parent,stage,job_id='1',sequence=0)


if __name__ == '__main__':
    unittest.main()
