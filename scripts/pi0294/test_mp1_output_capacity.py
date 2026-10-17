import copy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.pi0294.mp1 import output_capacity as capacity,pbs_owner
from scripts.pi0294.native_dataset import sha


class OutputCapacity(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'parent/children/source-pair-01';self.root.mkdir(parents=True)
        (self.root/'pbs').mkdir()
        self.put(self.root.parent.parent/'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json',dict(T0_epoch=10,new_submission_and_candidate_cutoff_epoch=time.time()+3600))
        self.put(self.root/'MP1_STAGE_BINDING.json',dict(schema='MP1_SOURCE_SHARED_STAGE_V1'))
        code=self.root/'capsules'/capacity.RUNTIME/'code';self.put(code/'scripts/frozen.json',dict(frozen=True))
        seal=code.parent/'SOURCE_SEAL.json';self.put(seal,dict(source_sha=capacity.RUNTIME,files={'scripts/frozen.json':sha(code/'scripts/frozen.json')}))
        self.put(self.root/'plans/input.json',dict(unchanged=True))
        self.put(self.root/'plans/OLD_E1_BINDING.private.json',dict(closed=True))
        self.manifest=dict(schema='RUN_MANIFEST_V1',run_uuid=self.root.name,owner='EXISTING_LOCAL_PC',scope='MP1_SOURCE_SHARED_STUDY',
            limits={'GPU':43200},interfaces={'stats':'fixed'},data={'dataset':'v3.0'},authority={'protocol_ref':capacity.AUTHORITY},
            code=dict(vla_commit=capacity.RUNTIME,capsule_relative=str(code.relative_to(self.root)),source_seal_relative=str(seal.relative_to(self.root)),
                source_seal_sha256=sha(seal),official_gr00t_commit='fixed',unchanged_numerical_files={'scripts/frozen.json':sha(code/'scripts/frozen.json')}),
            execution=dict(profile='hopper-neural',uid=os.getuid(),hostnames=['fixture'],sealed_inputs={'plans/input.json':sha(self.root/'plans/input.json')}))
        self.put(self.root/'RUN_MANIFEST.json',self.manifest)
        self.management=copy.deepcopy(self.manifest);self.management['code']['vla_commit']='new-management'
        self.evidence=[]
        for branch,step in [('S-BC',512),('S-BC',2048),('S-CORR',512)]:
            folder=self.root/'training'/branch/('checkpoint-'+str(step));self.put(folder/'state.json',{'saved':True})
            file=folder/'CHECKPOINT_COMPLETE.json';self.put(file,dict(global_step=step,full_resume=True,files={'state.json':sha(folder/'state.json')}))
            self.evidence.append(dict(branch=branch,step=step,manifest_sha256=sha(file),logical_member_bytes=(folder/'state.json').stat().st_size))
        self.binding=self.root/'plans/OUTPUT_ESTIMATE.json'
        self.value=dict(schema='MP1_EVIDENCED_SOURCE_OUTPUT_ESTIMATE_REBINDING_V1',authority_release=capacity.AUTHORITY,parent_run='parent',source_child=self.root.name,
            original_T0_epoch=10,old_estimate_bytes=140*capacity.GIB,new_estimate_bytes=180*capacity.GIB,protocol_peak_limit_bytes=400*capacity.GIB,
            new_scientific_budget=False,scientific_inputs_changed=False,deletion_required=False,launch_source_sha=capacity.RUNTIME,
            original_run_manifest_sha256=sha(self.root/'RUN_MANIFEST.json'),checkpoint_size_evidence=self.evidence,
            maximum_total_complete_checkpoint_slots=7,noncheckpoint_margin_bytes=2*capacity.GIB)
        self.rebind()

    def put(self,p,v):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))

    def rebind(self):
        if self.binding.exists():self.binding.chmod(0o600)
        self.put(self.binding,self.value);self.binding.chmod(0o400)

    def check(self):return capacity.checked_estimate(self.root,self.binding,self.management)

    def test_retains_original_launch_not_management_source(self):
        upper,launch=self.check();self.assertEqual(upper,180*capacity.GIB);self.assertEqual(launch,self.manifest)
        self.assertNotEqual(launch['code']['vla_commit'],self.management['code']['vla_commit'])

    def test_no_budget_input_or_deletion_change(self):
        for field in ['new_scientific_budget','scientific_inputs_changed','deletion_required']:
            for value in [True,0,None]:
                with self.subTest(field=field,value=value):
                    self.value[field]=value;self.rebind()
                    with self.assertRaises(PermissionError):self.check()
            self.value[field]=False

    def test_fixed_authority_runtime_clock_and_caps(self):
        for field,value in [('authority_release','other'),('launch_source_sha','other'),('original_T0_epoch',11),('old_estimate_bytes',141*capacity.GIB),
                ('protocol_peak_limit_bytes',401*capacity.GIB),('new_estimate_bytes',401*capacity.GIB),('new_estimate_bytes',140*capacity.GIB),
                ('new_estimate_bytes',True),('maximum_total_complete_checkpoint_slots',8),('noncheckpoint_margin_bytes',0)]:
            with self.subTest(field=field,value=value):
                original=self.value[field];self.value[field]=value;self.rebind()
                with self.assertRaises(PermissionError):self.check()
                self.value[field]=original

    def test_management_cannot_change_science_or_limits(self):
        for field in ['limits','data','execution','authority','scope','interfaces']:
            with self.subTest(field=field):
                old=self.management[field];self.management[field]='changed'
                with self.assertRaises(PermissionError):self.check()
                self.management[field]=old

    def test_management_cannot_change_frozen_code_or_upstream(self):
        self.management['code']['official_gr00t_commit']='other'
        with self.assertRaises(PermissionError):self.check()

    def test_binding_must_be_readonly_not_symlink_and_under_plans(self):
        self.binding.chmod(0o600)
        with self.assertRaises(PermissionError):self.check()
        self.binding.chmod(0o400);other=self.root/'OTHER.json';other.symlink_to(self.binding)
        with self.assertRaises(PermissionError):capacity.checked_estimate(self.root,other,self.management)
        other.unlink();other.write_bytes(self.binding.read_bytes());other.chmod(0o400)
        with self.assertRaises(PermissionError):capacity.checked_estimate(self.root,other,self.management)

    def test_original_source_members_and_sealed_inputs_checked(self):
        for p in [self.root/'plans/input.json',self.root/'capsules'/capacity.RUNTIME/'code/scripts/frozen.json']:
            with self.subTest(path=p.name):
                original=p.read_bytes();p.write_bytes(b'changed')
                with self.assertRaises(PermissionError):self.check()
                p.write_bytes(original)

    def test_manifest_identity_not_an_intended_version(self):
        p=self.root/'RUN_MANIFEST.json';self.put(p,dict(self.manifest,status='changed'))
        with self.assertRaises(PermissionError):self.check()

    def test_requires_all_three_fixed_complete_states_and_unique_evidence(self):
        original=copy.deepcopy(self.value['checkpoint_size_evidence'])
        for evidence in [original[:-1],original+[original[0]],[]]:
            with self.subTest(evidence=evidence):
                self.value['checkpoint_size_evidence']=evidence;self.rebind()
                with self.assertRaises(PermissionError):self.check()

    def test_rejects_changed_size_and_nonfullstate(self):
        self.evidence[0]['logical_member_bytes']+=1;self.rebind()
        with self.assertRaises(PermissionError):self.check()
        self.evidence[0]['logical_member_bytes']-=1
        p=self.root/'training/S-BC/checkpoint-512/CHECKPOINT_COMPLETE.json';v=json.loads(p.read_text());v['full_resume']=False;self.put(p,v)
        self.evidence[0]['manifest_sha256']=sha(p);self.rebind()
        with self.assertRaises(PermissionError):self.check()

    def test_projection_must_cover_seven_existing_shape_states(self):
        self.value['noncheckpoint_margin_bytes']=180*capacity.GIB;self.rebind()
        with self.assertRaises(PermissionError):self.check()

    def prepare(self,rebinding=None,memory_gb=225):
        self.put(self.root/'HOPPER_RESOURCE_LEDGER.json',dict(PBS_submissions=[],GPU_reserved_s=0,GPU_allocated_s=0))
        self.put(self.root/'NEURAL_BINDINGS.private.json',dict(python='/fixture/python'))
        with patch.object(pbs_owner,'validate_runtime',return_value=self.management),patch.object(pbs_owner,'old_scope_closed'),patch.object(pbs_owner,'no_overlap'),\
                patch.object(pbs_owner,'personal_quota',return_value=dict(report_epoch=time.time(),hard_bytes=999*capacity.GIB,used_bytes=0)),\
                patch.object(pbs_owner,'project_proof',return_value={'fresh':True}),patch.object(pbs_owner.subprocess,'check_output',return_value='0 fixture'):
            return pbs_owner.prepare(self.root,output_rebinding=rebinding,memory_gb=memory_gb)

    def test_prepare_rebound_estimate_keeps_original_launch_and_seals_binding(self):
        result=self.prepare(self.binding);d=self.root/'pbs/alloc-01';request=json.loads((d/'REQUEST.json').read_text())
        self.assertEqual(result['submit_count'],1);self.assertEqual(request['source_sha'],capacity.RUNTIME)
        self.assertEqual(request['management_source_sha'],'new-management');self.assertTrue(request['scientific_launch_source_unchanged'])
        self.assertEqual(request['sealed_inputs']['plans/OUTPUT_ESTIMATE.json'],sha(self.binding))
        self.assertIn(str(self.root/self.manifest['code']['capsule_relative']),(d/'job.pbs').read_text())
        self.assertEqual(request['additional_output_upper_bytes'],180*capacity.GIB)

    def test_default_prepare_uses_existing_estimate_and_manifest(self):
        self.prepare();request=json.loads((self.root/'pbs/alloc-01/REQUEST.json').read_text())
        self.assertEqual(request['source_sha'],'new-management');self.assertEqual(request['additional_output_upper_bytes'],140*capacity.GIB)
        self.assertNotIn('management_source_sha',request);self.assertNotIn('output_estimate_binding_relative',request)
        self.assertIn('mem=225gb',(self.root/'pbs/alloc-01/job.pbs').read_text())

    def test_reuse_existing_128GB_profile_without_changing_scientific_launch(self):
        self.prepare(self.binding,128)
        self.assertIn('mem=128gb',(self.root/'pbs/alloc-01/job.pbs').read_text())
        self.assertEqual(json.loads((self.root/'pbs/alloc-01/REQUEST.json').read_text())['source_sha'],capacity.RUNTIME)

    def test_reject_unqualified_memory_profiles_before_preparation(self):
        for memory in [0,64,256,True]:
            with self.subTest(memory=memory):
                with self.assertRaises(PermissionError):self.prepare(memory_gb=memory)
                self.assertFalse((self.root/'pbs/alloc-01').exists())


if __name__=='__main__':unittest.main()
