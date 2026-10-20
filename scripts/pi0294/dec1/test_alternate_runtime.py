"""CPU-only event/control and admission fixtures, never a GPU or a model."""
import base64
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from scripts.pi0294.pv1.pbs_owner import publish_release_record
from scripts.pi0294.mp1.probe_compare import PRECISION
from scripts.pi0294.e0_scope import DATA
from . import alternate_runtime as runtime
from .authority import AUTHORITY,PARENT_AUTHORITY,PARENT,R0,HARD_CLOSE
from .alternate_accounting import VERSION,digest
from .test_alternate_accounting import fixture as accounting_fixture,NOW


def put(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():p.chmod(0o600)
    p.write_text(json.dumps(v,indent=2)+'\n');p.chmod(0o400)
    return runtime.sha(p)


class EventTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name)
        self.request=accounting_fixture()[1];self.allocation=dict(job_id='1469999.stdct-mgmt-02',request_id=self.request['request_id'])

    def make(self):return runtime.Events(self.root,self.request,self.allocation)

    def test_intent_precedes_completion_and_final_summary_counts_actual_draws(self):
        with patch.object(runtime,'control',return_value={}):
            events=self.make();events('MODEL_LOAD_BEGIN',{});events('MODEL_LOAD_COMPLETE',{})
            events('QUERY_BEGIN',dict(index=0,seed=29400))
            value=runtime.event_summary(self.root,self.request,self.allocation['job_id'])
            self.assertEqual(value['unknown_query_intents'],[0]);self.assertEqual(value['query_attempts'],1)
            path=self.root/'query-00.npz';path.write_bytes(b'opaque-CPU-fixture-not-model-output')
            events('QUERY_COMPLETE',dict(index=0,seed=29400,arrays_sha256=runtime.sha(path),inference_precision=PRECISION))
            events('FINAL',dict(status='fixture'))
            final=runtime.event_summary(self.root,self.request,self.allocation['job_id'])
            self.assertEqual(final['completed_queries'],1);self.assertEqual(final['unknown_query_intents'],[])
            self.assertTrue(final['journal_final']);self.assertTrue(final['saved_observation_not_closed_loop'])
            for p in (self.root/'events').glob('*.json'):self.assertEqual(p.stat().st_mode&0o777,0o400)

    def test_killed_or_missing_completion_remains_unknown_not_a_free_query(self):
        with patch.object(runtime,'control',return_value={}):
            events=self.make();events('MODEL_LOAD_BEGIN',{});events('MODEL_LOAD_COMPLETE',{})
            events('QUERY_BEGIN',dict(index=0,seed=29400));events('FINAL',dict(status='killed'))
        value=runtime.event_summary(self.root,self.request,self.allocation['job_id'])
        self.assertEqual(value['unknown_query_intents'],[0]);self.assertEqual(value['completed_queries'],0)

    def test_known_failure_is_terminal_not_resampled(self):
        with patch.object(runtime,'control',return_value={}):
            events=self.make();events('MODEL_LOAD_BEGIN',{});events('MODEL_LOAD_COMPLETE',{})
            events('QUERY_BEGIN',dict(index=0,seed=29400))
        with patch.object(runtime,'control',side_effect=PermissionError('owner stopped')):
            events('QUERY_FAILURE',dict(index=0,seed=29400,error_type='RuntimeError'));events('FINAL',{})
        value=runtime.event_summary(self.root,self.request,self.allocation['job_id'])
        self.assertEqual(value['failed_queries'],1);self.assertEqual(value['unknown_query_intents'],[])
        with self.assertRaises(FileExistsError):self.make()

    def test_no_stale_owner_or_wrong_precision_completion_can_qualify(self):
        with patch.object(runtime,'control',side_effect=PermissionError('stale')):
            events=self.make()
            with self.assertRaises(PermissionError):events('MODEL_LOAD_BEGIN',{})
        self.assertEqual(list((self.root/'events').iterdir()),[])
        with patch.object(runtime,'control',return_value={}):
            events('MODEL_LOAD_BEGIN',{});events('MODEL_LOAD_COMPLETE',{});events('QUERY_BEGIN',dict(index=0,seed=29400))
            path=self.root/'query-00.npz';path.write_bytes(b'fixture')
            with self.assertRaisesRegex(PermissionError,'PRECISION'):
                events('QUERY_COMPLETE',dict(index=0,seed=29400,arrays_sha256=runtime.sha(path),
                    inference_precision=dict(PRECISION,cuda_matmul_allow_tf32=True)))
        self.assertEqual(runtime.event_summary(self.root,self.request,self.allocation['job_id'])['unknown_query_intents'],[0])

    def test_control_binds_same_job_fresh_project_and_real_current_owner(self):
        root=self.root;r=self.request;job=self.allocation['job_id']
        r['owner_session']='one'
        put(root/'RUN_MANIFEST.json',dict(CPU_fixture=True))
        put(root/'COLD_INPUTS_READY.json',dict(request_sha256=digest(r),status='PASS_CPU_MODEL_COSMOS_SOURCE_RGB_AND_ALIAS_BYTES_NO_MODEL'))
        bound=dict(job_id=job,request_id=r['request_id'],request_sha256=digest(r),submission_nonce=r['submission_nonce'],
            global_submit_count=12,reserved_GPU_s=1860,reserved_queries=17,other_live_GPU_jobs=[],
            run_manifest_sha256=runtime.sha(root/'RUN_MANIFEST.json'),cold_input_audit_sha256=runtime.sha(root/'COLD_INPUTS_READY.json'))
        put(root/'GLOBAL_ALLOCATION_BOUND.json',bound)
        release=dict(job_id=job,request_id=r['request_id'],request_sha256=digest(r),submission_nonce=r['submission_nonce'],
            global_bound_sha256=runtime.sha(root/'GLOBAL_ALLOCATION_BOUND.json'),fresh_Vanda_project_admission=True,observed_epoch=NOW)
        put(root/'START_COMPATIBILITY.json',release)
        beat=dict(request_id=r['request_id'],job_id=job,source_sha=r['source_sha'],owner_session='one',
            original_parent_STOP=False,global_unknown_effects=[],Hopper_owned_jobs=[],Vanda_other_owned_jobs=[],observed_epoch=NOW)
        r['owner_session']='one'
        # Canonical request digest changes when the owner binding is added.
        bound['request_sha256']=release['request_sha256']=digest(r)
        put(root/'GLOBAL_ALLOCATION_BOUND.json',bound);release['global_bound_sha256']=runtime.sha(root/'GLOBAL_ALLOCATION_BOUND.json')
        put(root/'START_COMPATIBILITY.json',release)
        path=root/'control/heartbeats/00000.json';put(path,beat)
        allocation=dict(job_id=job,observed_epoch=NOW-1,allocation_end_epoch=NOW+1000)
        self.assertEqual(runtime.control(root,r,allocation,now=NOW),beat)
        for change in (lambda b:b.update(observed_epoch=NOW-121),lambda b:b.update(job_id='foreign'),
                       lambda b:b.update(Hopper_owned_jobs=['another']),lambda b:b.update(original_parent_STOP=True)):
            b=deepcopy(beat);change(b);put(path,b)
            with self.assertRaises(PermissionError):runtime.control(root,r,allocation,now=NOW)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name)/'dec1-vanda-compat-01';self.root.mkdir()
        root=self.root;r=accounting_fixture()[1];r['owner_session']='one';r['run_uuid']=root.name;self.request=r
        self.files={n:'a'*64 for n in ('config.json','statistics.json','model.safetensors.index.json',
            'processor/embodiment_id.json','processor/processor_config.json','processor/statistics.json',
            'model-00001-of-00003.safetensors','model-00002-of-00003.safetensors','model-00003-of-00003.safetensors')}
        self.files['statistics.json']=R0
        canonical=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
        identity=dict(files=self.files,checkpoint_manifest_sha256=PARENT,statistics_sha256=R0,global_step=2048,
            relay_identity=dict(profile='B1',model_revision=canonical(self.files)[:40],source_sha=r['source_sha'],
                config_sha256=self.files['config.json'],processor_sha256=canonical({n:h for n,h in self.files.items() if n.startswith('processor/')}),statistics_sha256=R0))
        stream=io.BytesIO();Image.new('RGB',(256,256)).save(stream,format='PNG');image=stream.getvalue()
        saved=dict(provenance=dict(dataset_manifest_sha256=DATA,statistics_sha256=R0,split='TRAIN'),
            payload=dict(language='fixture task',state_eef10=[0]*10,frame='PI0291_MUJOCO_WORLD',tcp='ARX5_NATIVE_SITE_x5_ee',
                images=[dict(camera=c,png_base64=base64.b64encode(image).decode(),sha256=hashlib.sha256(image).hexdigest())
                        for c in ('external_rgb','wrist_rgb')]))
        consumer=dict(inference_files_sha256=self.files,status='PASS_FULL_CONSUMER_BYTES_AND_REUSED_FROZEN_SHARD_NO_MODEL')
        values={'REQUEST.json':r,'POLICY_IDENTITY.json':identity,'ASSET_CONSUMPTION.json':consumer,
            'HF_READY.json':dict(fixture=True),'PROBE_OBSERVATION.private.json':saved,
            'ASSET_PROGRESS.json':dict(checkpoint='registered-input')}
        for n,v in values.items():put(root/n,v)
        self.lease=root.parent/'lease.json';put(self.lease,dict(status='ACTIVE',run_uuid=root.name,
            authority_release=AUTHORITY,original_T0_epoch=1792348196,deadline_epoch=HARD_CLOSE))
        self.registry=root.parent/'registry.json';put(self.registry,dict(entries=[dict(role='one',path=str(root))]))
        self.m=dict(dec1_qualification_version=VERSION,dec1_request_id=r['request_id'],
            authority=dict(protocol_ref=AUTHORITY),parent_authority_release=PARENT_AUTHORITY,
            campaign_run_uuid='20261018T182956Z-localpc-pv1',original_T0_epoch=1792348196,output_root=str(root),
            run_uuid=root.name,owner_uid=os.getuid(),owner_user='e1192548',deadline_epoch=HARD_CLOSE,
            registered_runs_root=str(root.parent),lease=str(self.lease),
            limits=dict(ngpus=1,walltime_s=1800,PBS_submissions=1,queries=17,active_cpu=4,robot_physics_steps=0,optimizer_steps=0),
            code=dict(vla_commit='a'*40,capsule_relative='capsule/code'),
            sealed_files={str(root/n):runtime.sha(root/n) for n in values},checkpoint='registered-input',
            asset_progress=str(root/'ASSET_PROGRESS.json'),identity=str(root/'POLICY_IDENTITY.json'),
            observation=str(root/'PROBE_OBSERVATION.private.json'),stage0_result=str(root/'SAME_JOB_HARDWARE.json'))
        self.seal()

    def seal(self):
        root=self.root;put(root/'RUN_MANIFEST.json',self.m)
        put(root/'OUTPUT_BINDING.private.json',dict(run_manifest_sha256=runtime.sha(root/'RUN_MANIFEST.json'),
            uid=os.getuid(),lease=str(self.lease),registry=str(self.registry),role='one'))

    def call(self):
        with patch.object(runtime,'CONSUMER',runtime.sha(self.root/'ASSET_CONSUMPTION.json')), \
             patch.object(runtime,'READY',runtime.sha(self.root/'HF_READY.json')), \
             patch.object(runtime,'__file__',str(self.root/'capsule/code/scripts/pi0294/dec1/alternate_runtime.py')), \
             patch('scripts.operations.vanda_inference_probe.validate',return_value=self.m), \
             patch.object(runtime.time,'time',return_value=NOW):
            return runtime.validate(self.root)

    def test_exact_scope_CPU_admission_uses_current_CORR_not_historical_probe(self):
        m,r=self.call();self.assertEqual(m,self.m);self.assertEqual(r,self.request)
        self.assertFalse((self.root/'NATIVE_PROCESS.json').exists())

    def test_changed_lease_identity_scope_source_and_saved_split_fail_before_GPU(self):
        for change in (lambda m:m.update(original_T0_epoch=NOW),
                lambda m:m['limits'].update(optimizer_steps=1),
                lambda m:m.update(stage0_result='old-node-PASS'),
                lambda m:m.update(identity='old-M2048-identity')):
            original=deepcopy(self.m);change(self.m);self.seal()
            with self.assertRaises(PermissionError):self.call()
            self.m=original
        self.seal();saved=runtime.read(self.root/'PROBE_OBSERVATION.private.json');saved['provenance']['split']='DEV'
        put(self.root/'PROBE_OBSERVATION.private.json',saved)
        self.m['sealed_files'][str(self.root/'PROBE_OBSERVATION.private.json')]=runtime.sha(self.root/'PROBE_OBSERVATION.private.json');self.seal()
        with self.assertRaisesRegex(PermissionError,'TRAIN_INPUT'):self.call()

    def test_native_R_requires_actual_owner_nonce_resources_and_compute_host(self):
        m=self.m;r=self.request
        native=dict(job_state='R',Job_Owner='e1192548@login',project='CFP04-CF-007',Job_Name='pi0294_dec1_a40',
            Resource_List=dict(ngpus=1,ncpus=4,walltime='00:30:00'),stime=NOW-20,
            Variable_List=dict(DEC1_REQUEST_ID=r['request_id'],DEC1_NONCE=r['submission_nonce']))
        def query(*a,**kw):return SimpleNamespace(stdout=json.dumps(dict(Jobs={'1469999.stdct-mgmt-02':native})))
        with patch.dict(os.environ,{'PBS_JOBID':'1469999.stdct-mgmt-02'}), \
             patch.object(runtime.socket,'gethostname',return_value='GN-A40-010'), \
             patch.object(runtime.time,'time',return_value=NOW):
            proof=runtime.native_running(m,r,query=query)
            self.assertEqual(proof['allocation_end_epoch'],NOW-20+1800)
            for change in (lambda:native.update(job_state='Q'),lambda:native['Resource_List'].update(ngpus=2),
                           lambda:native['Variable_List'].update(DEC1_NONCE='foreign')):
                original=deepcopy(native);change()
                with self.assertRaises(PermissionError):runtime.native_running(m,r,query=query)
                native.clear();native.update(original)


class ColdChecks(unittest.TestCase):
    def test_hashes_pinned_HF_blob_links_without_model_and_refuses_escape(self):
        for escape in (False,True):
            with self.subTest(escape=escape),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'run';root.mkdir();cache=Path(tmp)/'repo'
                snapshot=cache/'snapshots/fixed';snapshot.mkdir(parents=True)
                blob=cache/'blobs/member';blob.parent.mkdir();blob.write_bytes(b'cosmos fixture')
                actual=Path(tmp)/'outside' if escape else blob
                if escape:actual.write_bytes(blob.read_bytes())
                (snapshot/'config.json').symlink_to(actual)
                checkpoint=Path(tmp)/'inference';checkpoint.mkdir();(checkpoint/'weight').write_bytes(b'fixture')
                put(root/'POLICY_IDENTITY.json',dict(files={'weight':runtime.sha(checkpoint/'weight')}))
                put(root/'observation.json',dict(fixture=True))
                put(root/'COLD_CPU_PREPAYMENT.json',dict(authority_release=AUTHORITY,CPU_upper_s=300,
                    inside_original_and_DEC1_GPU_site_CPU_caps=True,campaign_run_uuid='20261018T182956Z-localpc-pv1'))
                m=dict(checkpoint=str(checkpoint),bindings=dict(cosmos=str(snapshot)),
                    cosmos_files={'config.json':runtime.sha(blob)},observation=str(root/'observation.json'),
                    sealed_files={str(root/'COLD_CPU_PREPAYMENT.json'):'fixture'})
                r=accounting_fixture()[1]
                with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}), \
                     patch.object(runtime,'validate',return_value=(m,r)), \
                     patch.object(runtime.os,'sched_setaffinity'),patch.object(runtime.resource,'setrlimit'), \
                     patch.object(runtime.signal,'signal'),patch.object(runtime.signal,'alarm'), \
                     patch('scripts.pi0294.local_backbone.local_model_name') as alias:
                    os.environ.pop('PBS_JOBID',None)
                    if escape:
                        with self.assertRaisesRegex(PermissionError,'COSMOS_MEMBER'):runtime.cold_check(root)
                        self.assertTrue((root/'COLD_INPUT_FAILURE.json').exists());alias.assert_not_called()
                    else:
                        value=runtime.cold_check(root)
                        self.assertTrue(value['status'].startswith('PASS_CPU_'));alias.assert_called_once()
                        self.assertEqual(runtime.sealed(root/'COLD_INPUTS_READY.json')['new_models'],0)
                    with self.assertRaisesRegex(PermissionError,'NO_REPEATED'):runtime.cold_check(root)


if __name__=='__main__':unittest.main()
