import os
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1.neural_scope import inside,validate
from scripts.pi0294.pv1.native_admission import admit
from scripts.pi0294.pv1 import neural_scope
from scripts.pi0294.pv1.scope import SCOPE,PROTOCOL
from scripts.pi0294.pv1.decision import AUTHORITY
from scripts.pi0294.pv1.recipe import PARENT
from scripts.pi0294.pv1.dataset import POPULATION
from scripts.pi0294.e1.scope import DATA,STATS
from scripts.pi0294.fd1 import sha
from scripts.pi0294.test_pv1_dataset import manifest as bank_fixture


def prepared_fixture(tmp):
    parent=Path(tmp)/'campaign';root=parent/'children/policy-01';code=root/'capsules/fixture/code'
    def put(p,value):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
    now=time.time()
    scope=json.loads((Path(__file__).resolve().parents[2]/SCOPE).read_text())
    put(code/SCOPE,scope);put(code/PROTOCOL,dict(nonexecuted_protocol_fixture=True))
    put(code/'scripts/pi0294/pv1/neural_scope.py',dict(nonexecuted_source_fixture=True))
    frozen={}
    for i in range(15):
        name='fixture-frozen-'+str(i)+'.py';put(code/name,dict(fixture=True));frozen[name]=sha(code/name)
    seal=put(code.parent/'SOURCE_SEAL.json',dict(source_sha='f'*40,
        files={str(p.relative_to(code)):sha(p) for p in code.rglob('*') if p.is_file()}))
    t0=now-100
    a=dict(authority_release=AUTHORITY,run_uuid='ORIGINAL_CAMPAIGN',bounds=scope['new_campaign_budget'],T0_epoch=t0,
        stop_new_jobs_epoch=t0+144*3600,stop_new_training_epoch=t0+156*3600,
        stop_new_physics_epoch=t0+160*3600,hard_closeout_epoch=t0+168*3600)
    put(parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json',a)
    lease=put(parent/'leases/child.json',dict(status='ACTIVE',run_uuid=root.name,owner='EXISTING_LOCAL_PC',
        authority_release=AUTHORITY,deadline_epoch=now+3600))
    pl=put(parent/'leases/parent.json',dict(status='ACTIVE_CONTROL_PLANE_ONLY',authority_release=AUTHORITY))
    put(parent/'OUTPUT_BINDING.private.json',dict(lease=str(pl),run_uuid=a['run_uuid']))
    role=neural_scope.ROLE+'__'+root.name
    registry=put(parent/'REGISTRY.json',dict(entries=[dict(role=role,path=str(root),run_uuid=root.name)]))
    put(root/'OUTPUT_BINDING.private.json',dict(uid=os.getuid(),role=role,run_root=str(root),
        registered_parent=str(root.parent),lease=str(lease),registry=str(registry)))
    old=parent/'inputs/old';new=parent/'inputs/new';correction=parent/'inputs/C1';checkpoint=parent/'inputs/CORR'
    for folder in (old,new,correction):put(folder/'statistics.json',dict(nonexecuted_statistics_fixture=True))
    put(old/'DATASET_MANIFEST.json',dict(nonexecuted_old_data_fixture=True))
    put(new/'DATASET_MANIFEST.json',bank_fixture())
    put(new/'READY.json',dict(status='SEALED_FULL_SUCCESS_OFFICIAL_CODEC_ADMITTED',dataset_manifest_sha256=sha(new/'DATASET_MANIFEST.json')))
    put(correction/'DATASET_MANIFEST.json',dict(nonexecuted_C1_fixture=True))
    put(checkpoint/'CHECKPOINT_COMPLETE.json',dict(nonexecuted_parent_fixture=True))
    put(root/'NEURAL_BINDINGS.private.json',dict(dataset=str(old)))
    put(root/'plans/PV1_TRAIN_INPUTS.private.json',dict(new_dataset=str(new),C1_dataset=str(correction),CORR_parent_checkpoint=str(checkpoint)))
    put(root/'PV1_STAGE_BINDING.json',dict(schema='PI0294_PV1_ADMITTED_SHARED_POLICY_STAGE_V1',
        authority_release=AUTHORITY,campaign_run_uuid=a['run_uuid'],population_plan_sha256=POPULATION,
        Teacher_takeover=False,normalizer_refit=False,evaluation_data_for_training=False,route='CLEARANCE_PAIR',
        new_bank_manifest_sha256=sha(new/'DATASET_MANIFEST.json'),new_bank_ready_sha256=sha(new/'READY.json'),
        C1_manifest_sha256=sha(correction/'DATASET_MANIFEST.json')))
    m=dict(schema='PI0294_PV1_NEURAL_CHILD_RUN_MANIFEST_V1',scope=neural_scope.ROLE,
        owner='EXISTING_LOCAL_PC',run_uuid=root.name,code=dict(capsule_relative=str(code.relative_to(root)),
            source_seal_relative=str(seal.relative_to(root)),source_seal_sha256=sha(seal),vla_commit='f'*40,
            frozen_numerical_files=frozen,official_gr00t_commit='51d4c89f72fda44cbf77285c6a8114b52676b8a1'),
        execution=dict(profile='hopper-neural',uid=os.getuid(),private_path_role=role,
            output_binding_sha256=sha(root/'OUTPUT_BINDING.private.json'),control_hostnames=[os.uname().nodename],
            sealed_inputs={n:sha(root/n) for n in ['NEURAL_BINDINGS.private.json','plans/PV1_TRAIN_INPUTS.private.json','PV1_STAGE_BINDING.json']}),
        authority=dict(protocol_ref=AUTHORITY,scope_sha256=sha(code/SCOPE),protocol_sha256=sha(code/PROTOCOL)),
        limits=dict(PV1_T0_epoch=a['T0_epoch'],PV1_hard_close_epoch=a['hard_closeout_epoch'],operation_deadline_epoch=now+3600))
    put(root/'RUN_MANIFEST.json',m);worker=put(code.parent/'WORKER_RUN_MANIFEST.json',m);worker.chmod(0o400)
    def digest(p):
        if Path(p).name=='statistics.json':return STATS
        if Path(p)==old/'DATASET_MANIFEST.json':return DATA
        if Path(p)==checkpoint/'CHECKPOINT_COMPLETE.json':return PARENT
        return sha(p)
    return root,code,worker,digest,new


class NativeScopeContracts(unittest.TestCase):
    def test_complete_metadata_fixture_is_control_only_and_mutation_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,code,worker,digest,new=prepared_fixture(tmp)
            with patch.dict(os.environ,{'PV1_RUN_MANIFEST':str(worker),'CUDA_VISIBLE_DEVICES':''},clear=True),\
                    patch.object(neural_scope,'__file__',str(code/'scripts/pi0294/pv1/neural_scope.py')),\
                    patch.object(neural_scope,'sha',side_effect=digest):
                self.assertEqual(validate(root)[0]['run_uuid'],root.name)
                # A fake metadata entry never supplies a native scheduler allocation.
                with self.assertRaises(PermissionError):admit(root,root/'pbs/alloc-01/REQUEST.json')
                (new/'READY.json').write_text('{}')
                with self.assertRaises(PermissionError):validate(root)

    def test_login_is_rejected_before_data_or_scheduler_or_model(self):
        with patch.dict(os.environ,{},clear=True),patch('scripts.pi0294.pv1.native_admission.validate') as root:
            with self.assertRaisesRegex(PermissionError,'NO_MODEL_ON_HOPPER_LOGIN'):
                admit('/nonexistent','/nonexistent')
            root.assert_not_called()

    def test_unregistered_schema_rejected_before_neural_input_imports(self):
        with tempfile.TemporaryDirectory() as tmp,patch('scripts.pi0294.pv1.neural_scope.read',return_value=dict(schema='OLD_NOT_PV1')):
            with self.assertRaisesRegex(PermissionError,'COLD_HOPPER_CHILD'):
                validate(tmp)

    def test_no_source_or_request_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('/outside','../outside','pbs/../../outside'):
                with self.assertRaises(PermissionError):inside(tmp,name)
            outside=Path(tmp).parent/'OUTSIDE_PV1_FIXTURE'
            (Path(tmp)/'alias').symlink_to(outside)
            with self.assertRaises(PermissionError):inside(tmp,'alias')
