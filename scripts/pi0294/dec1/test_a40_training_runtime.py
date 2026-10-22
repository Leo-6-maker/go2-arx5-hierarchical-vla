"""Platform routing and phase contracts; synthetic CPU-only, never model calls."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.pi0292.durable import write
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1 import training
from scripts.pi0294.dec1 import a40_training_context as c
from scripts.pi0294.dec1 import a40_training_runtime as runtime
from .a40_allocation import STATUS
from .test_a40_allocation import fixture,future_inputs
from scripts.pi0294.dec1 import a40_future,future_budget


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);write(path,value);path.chmod(0o400)


class RuntimeTests(unittest.TestCase):
    def test_storage_failure_is_caught_before_torch_or_model_load_with_committed_step_in_final(self):
        from scripts.pi0294.dec1 import checkpoint_storage
        from scripts.pi0294 import numerics
        for committed in (0,1024):
            with self.subTest(committed=committed),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);directory=root/'pbs/alloc-01';request=directory/'REQUEST.json'
                r=dict(expected_native_start_step=0,CPU_upper_s=1000,source_sha='a'*40)
                put(request,r)
                ledger=SimpleNamespace(initialize=lambda:None,guard=lambda:None,
                    summary=lambda:dict(last_native_step=committed,optimizer_updates_committed=committed))
                with (patch.object(c,'validate_root',return_value=({},r)),
                      patch.object(c,'consumer_records',return_value={}),
                      patch.object(runtime,'allocation',return_value=dict(job_id='fixture',request_sha256='b'*64)),
                      patch.object(runtime.os,'sched_setaffinity'),patch.object(runtime.os,'sched_getaffinity',return_value={0,1,2,3}),
                      patch.object(runtime.resource,'setrlimit'),patch.object(runtime,'Ledger',return_value=ledger),
                      patch.object(checkpoint_storage,'prewrite_headroom',side_effect=PermissionError('FIXTURE_STORAGE')) as prewrite,
                      patch.object(numerics,'prepare_environment') as torch_environment):
                    with self.assertRaisesRegex(PermissionError,'FIXTURE_STORAGE'):runtime.run(root,request)
                    prewrite.assert_called_once_with(root);torch_environment.assert_not_called()
                final=json.loads((directory/'PBS_PROCESS_FINAL.json').read_bytes())
                self.assertEqual(final['phase'],'ACTUAL_NATIVE_SOURCE_PREMODEL')
                self.assertEqual(final['global_step'],committed)
                self.assertEqual(final['last_returned_training_step'],0)
                self.assertFalse((root/'MODEL_LOAD_INTENTS').exists())

    def test_before_probes_recomputes_real_whole_block_and_binds_ready_bytes(self):
        for mutation in (None,'stale_budget','wrong_ledger','wrong_job','changed_plan'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);directory=root/'pbs/alloc-13';directory.mkdir(parents=True)
                f=fixture();inputs=future_inputs(f);now=f['now'];allocation=dict(job_id=f['job_id'],request_sha256=hashlib.sha256(inputs['request_bytes']).hexdigest())
                put(directory/'REQUEST.json',json.loads(inputs['request_bytes']))
                (directory/'REQUEST.json').chmod(0o600);(directory/'REQUEST.json').write_bytes(inputs['request_bytes']);(directory/'REQUEST.json').chmod(0o400)
                plan='plans/resource.json';put(root/plan,json.loads(inputs['plan_bytes']))
                # Preserve the original serialized bytes just as the native owner does.
                (root/plan).chmod(0o600);(root/plan).write_bytes(inputs['plan_bytes']);(root/plan).chmod(0o400)
                put(root/'plans/VAP1_ADOPTION_AND_DELTA_BUDGET.json',json.loads(inputs['amendment_bytes']))
                put(root/'RUN_MANIFEST.json',dict(execution=dict(sealed_inputs={plan:sha(root/plan)})))
                with patch.object(future_budget,'VAP1_ADOPTION',hashlib.sha256((root/'plans/VAP1_ADOPTION_AND_DELTA_BUDGET.json').read_bytes()).hexdigest()):
                    inputs['amendment_bytes']=(root/'plans/VAP1_ADOPTION_AND_DELTA_BUDGET.json').read_bytes()
                    expected=a40_future.recompute(**inputs)
                    def owner_release(unused):
                        material={k+'_base64':base64.b64encode(inputs[k+'_bytes']).decode()
                            for k in ('ledger','DEC1_ledger','snapshot','progress')}
                        if mutation=='wrong_ledger':material['ledger_base64']=base64.b64encode(b'{}').decode()
                        put(directory/'budget.private.json',material)
                        proof=deepcopy(expected)
                        if mutation=='stale_budget':proof['observed_epoch']=now-121
                        put(directory/'A40_PRETRAIN_RELEASE.json',dict(job_id='other' if mutation=='wrong_job' else f['job_id'],
                            request_sha256=allocation['request_sha256'],submission_nonce=json.loads(inputs['request_bytes'])['submission_nonce'],
                            whole_block_stop_step=4096,observed_epoch=now,ready_sha256=sha(directory/'A40_PRETRAIN_READY.json'),
                            material_relative=str((directory/'budget.private.json').relative_to(root)),
                            material_sha256=sha(directory/'budget.private.json'),plan_relative=plan,recomputed_budget=proof))
                        if mutation=='changed_plan':put(root/plan,dict(forged=True))
                    with patch.object(c,'require_training',return_value=directory),patch.object(c.time,'time',return_value=now):
                        if mutation is None:
                            result=c.authorize_phase(root,directory,allocation,lambda:None,'BEFORE_PROBES',sleep=owner_release)
                            admitted=json.loads((directory/'A40_FIRST_UPDATE_ADMISSION.json').read_text())
                            self.assertEqual(admitted['recomputed_budget'],expected)
                            self.assertFalse(result['new_allocation_or_budget'])
                        else:
                            with self.assertRaises((PermissionError,KeyError)):
                                c.authorize_phase(root,directory,allocation,lambda:None,'BEFORE_PROBES',sleep=owner_release)

    def test_A40_is_not_routed_through_or_labeled_as_H200_qualification(self):
        with (patch.object(c,'require_training',return_value='A40_DIRECTORY') as v,
             patch.object(training,'current_directory',side_effect=AssertionError('H200_PATH'))):
            self.assertEqual(training.native_training_directory('root',dict(status=STATUS)),'A40_DIRECTORY')
            v.assert_called_once()
        with self.assertRaises(PermissionError):training.native_training_directory('root',dict(status='UNKNOWN'))

    def test_after_probes_reuses_whole_job_release_not_a_second_first_update_handshake(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'pbs/alloc-13';directory.mkdir(parents=True)
            request=json.loads(fixture()['request_bytes']);put(directory/'REQUEST.json',request)
            release=dict(job_id='17.stdct-mgmt-02',request_sha256='a'*64,submission_nonce=request['submission_nonce'],
                whole_block_stop_step=4096,observed_epoch=1)
            put(directory/'A40_PRETRAIN_RELEASE.json',release)
            put(directory/'A40_FIRST_UPDATE_ADMISSION.json',dict(release_sha256=sha(directory/'A40_PRETRAIN_RELEASE.json'),request_sha256='a'*64))
            allocation=dict(job_id=release['job_id'],request_sha256='a'*64)
            folder=root/'evidence/A_CONTROL-1024/TRAIN_PROBES';folder.mkdir(parents=True)
            put(folder/'RESULT.json',dict(fixture=True))
            with (patch.object(c,'require_training',return_value=directory),patch.object(c,'probe_receipt',return_value=dict(records=[])),
                 patch.object(c.time,'time',return_value=10000)):
                result=c.authorize_phase(root,directory,allocation,lambda:None,'AFTER_PROBES')
                self.assertFalse(result['new_allocation_or_budget'])
                self.assertFalse((directory/'A40_PRETRAIN_READY.json').exists())
            release['whole_block_stop_step']=1024;put(directory/'A40_PRETRAIN_RELEASE.json',release)
            with patch.object(c,'require_training',return_value=directory),patch.object(c,'probe_receipt',return_value=dict(records=[])):
                with self.assertRaises(PermissionError):c.authorize_phase(root,directory,allocation,lambda:None,'AFTER_PROBES')

    def test_native_bootstrap_rejects_login_and_wrong_request_before_models(self):
        f=fixture();r=json.loads(f['request_bytes'])
        with (patch.dict(runtime.os.environ,dict(PBS_JOBID=f['job_id'],USER='fixture')),
             patch.object(runtime.socket,'gethostname',return_value='stdct-login-01')):
            with self.assertRaises(PermissionError):runtime.native_job(r)

    def test_native_compute_rechecks_own_scheduler_record_not_owner_label(self):
        for mutation in (None,'nonce','project','repeat','CPU','owner','node'):
            f=fixture();r=json.loads(f['request_bytes']);job=deepcopy(f['job'])
            if mutation=='nonce':job['Variable_List']['PV1_SUBMISSION_NONCE']='other'
            elif mutation=='project':job['project']='other'
            elif mutation=='repeat':job['run_count']=2
            elif mutation=='CPU':job['Resource_List']['ncpus']=12
            elif mutation=='owner':job['Job_Owner']='other@node'
            elif mutation=='node':job['exec_host']='gn-a40-other/0'
            process=SimpleNamespace(stdout=json.dumps(dict(Jobs={f['job_id']:job})))
            with (self.subTest(mutation=mutation),patch.dict(runtime.os.environ,dict(PBS_JOBID=f['job_id'],USER='fixture')),
                  patch.object(runtime.socket,'gethostname',return_value=f['hostname']),
                  patch.object(runtime.subprocess,'run',return_value=process)):
                if mutation is None:self.assertEqual(runtime.native_job(r)[0],f['job_id'])
                else:
                    with self.assertRaises(PermissionError):runtime.native_job(r)

    def test_fixed_probe_uses_initialized_original_dataset_and_freezes_actual_arrays(self):
        from scripts.pi0294.pv1 import neural,probes
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);events=[]
            dataset=SimpleNamespace(initialize=lambda:events.append('initialize'))
            def actual_probe(r,label,instance,data,ledger,guard):
                self.assertEqual(events,['initialize']);self.assertIs(data,dataset)
                folder=root/'evidence'/label/'TRAIN_PROBES';folder.mkdir(parents=True)
                put(folder/'RESULT.json',dict(fixture=True));(folder/'arrays.npz').write_bytes(b'CPU_FIXTURE')
                events.append('probes');return dict(fixture=True)
            with patch.object(neural,'train_dataset',return_value=dataset),patch.object(probes,'run',side_effect=actual_probe):
                self.assertEqual(runtime.fixed_probes(root,'A_CONTROL',None,None,lambda:None),dict(fixture=True))
            self.assertEqual(events,['initialize','probes'])
            self.assertEqual((root/'evidence/A_CONTROL-1024/TRAIN_PROBES/arrays.npz').stat().st_mode&0o777,0o400)

    def test_first32_are_actual_counted_steps_and_cost_miss_does_not_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'pbs/alloc-13';directory.mkdir(parents=True)
            put(root/'plans/resource.json',dict(seconds_per_update_upper=2.))
            put(directory/'A40_PRETRAIN_RELEASE.json',dict(plan_relative='plans/resource.json'))
            put(root/'RUN_MANIFEST.json',dict(execution=dict(sealed_inputs={'plans/resource.json':sha(root/'plans/resource.json')})))
            put(directory/'A40_FIRST_UPDATE_ADMISSION.json',dict(release_sha256=sha(directory/'A40_PRETRAIN_RELEASE.json')))
            with patch.object(c,'require_training',return_value=directory):
                result=c.first32(root,directory,dict(job_id='17.stdct-mgmt-02',request_sha256='a'*64),
                    [3.]*32,native_loss=.1,peak_bytes=100)
                self.assertFalse(result['continue_same_state']);self.assertEqual(result['actual_A_CONTROL_updates'],32)
                self.assertEqual(result['additional_updates'],0)


if __name__=='__main__':unittest.main()
