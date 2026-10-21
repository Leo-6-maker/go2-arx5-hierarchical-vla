"""Synthetic filesystem provenance, never a model or native data qualification."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from . import a40_training_context as c
from . import a40_allocation as a
from .test_a40_allocation import fixture
from .test_a40_training_runtime import put


def root_fixture(base):
    r=json.loads(fixture()['request_bytes']);root=base/'campaign/children'/r['run_uuid'];root.mkdir(parents=True)
    inputs={name:{} for name in (c.CONSUMERS,'NEURAL_BINDINGS.private.json','plans/PV1_TRAIN_INPUTS.private.json',
        'plans/PRIMARY_LEARNING_RECIPES.json','plans/FROZEN_SOURCE_PAIR_RECIPE.json',
        'plans/INITIAL_CORR_PARAMETER_INVENTORY.json','plans/PV1_FIXED_TRAIN_PROBES.json',
        'plans/PROBE_OBSERVATION.private.json','POLICY_IDENTITY.json','ASSET_CONSUMPTION.json',
        'plans/VAP1_ADOPTION_AND_DELTA_BUDGET.json')}
    inputs['NEURAL_BINDINGS.private.json']=dict(source=str(base/'official'))
    put(base/'environment.json',dict(fixture=True))
    r['python']=sys.executable
    inputs['plans/A40_ENVIRONMENT_BINDING.private.json']=dict(schema='VAP1_A40_REUSED_ENVIRONMENT_BINDING_V1',
        profile=c.PROFILE,old_probe_profile_scope_changed=False,python=sys.executable,venv_prefix=sys.prefix,
        python_binary_sha256=sha(sys.executable),environment_registry=str(base/'environment.json'),
        environment_registry_sha256=sha(base/'environment.json'),official_source=str(base/'official'),
        official_commit='51d4c89f72fda44cbf77285c6a8114b52676b8a1')
    inputs['plans/PUBLICATION_AND_CI.json']=dict(source_sha=r['source_sha'],CI_head_sha=r['source_sha'],
        CI_conclusion='success',CI_status='completed',files_exact_readback=True)
    inputs['plans/VAP1_DUAL_PLATFORM_CONCURRENCY_ADOPTION.json']=dict(fixture=True)
    for name,value in inputs.items():put(root/name,value)
    put(root.parent.parent/'authority/VAP1/VAP1_DUAL_PLATFORM_CONCURRENCY_ADOPTION.json',dict(fixture=True))
    m=dict(schema='PI0294_PV1_NEURAL_CHILD_RUN_MANIFEST_V1',platform_training_version=a.VERSION,
        owner='EXISTING_LOCAL_PC',owner_uid=os.getuid(),owner_user='e1192548',run_uuid=root.name,
        output_root=str(root),registered_runs_root=str(base),
        execution=dict(profile=c.PROFILE,sealed_inputs={n:sha(root/n) for n in inputs}),
        code=dict(vla_commit=r['source_sha'],capsule_relative='capsule/code'),
        authority=dict(protocol_ref=a.PARENT_AUTHORITY,amendments=[a.AUTHORITY],training_admission_version=a.SCIENCE_VERSION))
    put(root/'RUN_MANIFEST.json',m)
    put(base/'lease.json',dict(status='ACTIVE',run_uuid=root.name,authority_release=a.PARENT_AUTHORITY,
        original_T0_epoch=a.T0,deadline_epoch=a.HARD_CLOSE))
    put(base/'registry.json',dict(entries=[dict(role='fixture',path=str(root))]))
    put(root/'OUTPUT_BINDING.private.json',dict(run_root=str(root),uid=os.getuid(),
        run_manifest_sha256=sha(root/'RUN_MANIFEST.json'),lease=str(base/'lease.json'),registry=str(base/'registry.json'),role='fixture'))
    r['run_manifest_sha256']=sha(root/'RUN_MANIFEST.json');r['sealed_inputs']=m['execution']['sealed_inputs']
    request=root/'pbs/alloc-13/REQUEST.json';put(request,r)
    return root,request


def consumers_fixture(base):
    from scripts.pi0294.e1.scope import DATA
    from .source_data_original import C1_MANIFEST
    root=base/'trainer';root.mkdir();records={};bindings={};inputs={}
    put(root/'RUN_MANIFEST.json',dict(registered_runs_root=str(base)))
    for kind in ('new139','original126_C1'):
        child=base/kind;child.mkdir();role=kind
        manifest=dict(run_uuid=child.name,owner='EXISTING_LOCAL_PC',owner_uid=os.getuid(),
            authority_release=a.AUTHORITY,original_T0_epoch=a.T0,
            source_data_profile='PV1_139' if kind=='new139' else 'ORIGINAL126_C1')
        result=dict(statistics_sha256=a.R0,full_member_hashes_verified=True,GPU=0,PBS=0,models=0,
            new_optimizer_updates=0,new_physics=0)
        if kind=='new139':
            result.update(status='PASS_VANDA_CPU_FIXED_PV1_139_TRAIN_FULL_MEMBER_AND_READER',dataset_manifest_sha256=a.BANK)
            dataset=child/'datasets/FROZEN_TRAIN_BANK';put(dataset/'DATASET_MANIFEST.json',{})
            inputs.update(new_dataset=str(dataset),new_raw_bindings=str(child/'RAW_BINDINGS.private.json'))
        else:
            result.update(status='PASS_VANDA_CPU_ORIGINAL126_AND_C1_FIXED_MEMBERS_AND_OFFICIAL_READER',
                original_manifest_sha256=DATA,C1_manifest_sha256=C1_MANIFEST,original_TRAIN_parents=126,C1_student_parents=4,DEV_raw_fetched=0)
            dataset=child/'datasets/original';put(dataset/'DATASET_MANIFEST.json',{})
            put(child/'datasets/C1/DATASET_MANIFEST.json',{})
            bindings.update(dataset=str(dataset),raw_bindings=str(child/'ORIGINAL_RAW_BINDINGS.private.json'))
            inputs.update(C1_dataset=str(child/'datasets/C1'),C1_raw_bindings=str(child/'C1_RAW_BINDINGS.private.json'))
        put(child/'RUN_MANIFEST.json',manifest);put(child/'RESULT.private.json',result)
        put(child/'lease.json',dict(status='CLOSED',run_uuid=child.name,owner='EXISTING_LOCAL_PC',
            authority_release=a.AUTHORITY,result_sha256=sha(child/'RESULT.private.json')))
        put(child/'registry.json',dict(entries=[dict(role=role,path=str(child),result_sha256=sha(child/'RESULT.private.json'))]))
        put(child/'OUTPUT_BINDING.private.json',dict(uid=os.getuid(),run_root=str(child),
            run_manifest_sha256=sha(child/'RUN_MANIFEST.json'),lease=str(child/'lease.json'),registry=str(child/'registry.json'),role=role))
        records[kind]=dict(root=str(child),manifest_sha256=sha(child/'RUN_MANIFEST.json'),result_sha256=sha(child/'RESULT.private.json'),lease_sha256=sha(child/'lease.json'))
    put(root/c.CONSUMERS,records);put(root/'NEURAL_BINDINGS.private.json',bindings);put(root/'plans/PV1_TRAIN_INPUTS.private.json',inputs)
    return root,records


class ContextTests(unittest.TestCase):
    def test_root_requires_real_profile_sealed_inputs_environment_and_source(self):
        for mutation in (None,'profile','lease','grant','mode','CI','request'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                base=Path(tmp);root,request=root_fixture(base)
                if mutation=='profile':
                    v=read(root/'RUN_MANIFEST.json');v['execution']['profile']='vanda-neural-probe';put(root/'RUN_MANIFEST.json',v)
                elif mutation=='lease':
                    v=read(base/'lease.json');v['status']='CLOSED';put(base/'lease.json',v)
                elif mutation=='grant':put(root.parent.parent/'authority/VAP1/VAP1_DUAL_PLATFORM_CONCURRENCY_ADOPTION.json',dict(different=True))
                elif mutation=='mode':(root/'ASSET_CONSUMPTION.json').chmod(0o600)
                elif mutation=='CI':put(root/'plans/PUBLICATION_AND_CI.json',dict(CI_conclusion='success',source_sha='other'))
                elif mutation=='request':
                    v=read(request);v['source_sha']='f'*40;put(request,v)
                with (patch.object(c.time,'time',return_value=fixture()['now']),patch.object(c,'grant'),
                     patch.object(c,'admitted',return_value=True),
                     patch.object(c,'__file__',str(root/'capsule/code/scripts/pi0294/dec1/a40_training_context.py'))):
                    if mutation is None:self.assertEqual(c.validate_root(root,request)[1]['run_uuid'],root.name)
                    else:
                        with self.assertRaises(PermissionError):c.validate_root(root,request)

    def test_sealed_rejects_mutable_file_symlink_and_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);put(root/'good.json',dict(fixture=True))
            self.assertEqual(c.sealed(root,'good.json'),dict(fixture=True))
            (root/'link.json').symlink_to(root/'good.json');(root/'good.json').chmod(0o600)
            for name in ('link.json','good.json','../outside.json',str(root/'good.json')):
                with self.subTest(name=name),self.assertRaises(PermissionError):c.sealed(root,name)

    def test_consumers_need_closed_actual_receipts_not_directories_or_wrong_lease(self):
        for mutation in (None,'lease','result','registry','manifest','path'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root,records=consumers_fixture(Path(tmp));child=Path(records['new139']['root'])
                if mutation=='lease':
                    v=read(child/'lease.json');v['run_uuid']='another';put(child/'lease.json',v)
                elif mutation=='result':put(child/'RESULT.private.json',dict(status='PASS'))
                elif mutation=='registry':
                    v=read(child/'registry.json');v['entries'][0]['path']=str(root);put(child/'registry.json',v)
                elif mutation=='manifest':put(child/'RUN_MANIFEST.json',dict(authority_release='other'))
                elif mutation=='path':
                    v=read(root/'plans/PV1_TRAIN_INPUTS.private.json');v['new_dataset']=str(root);put(root/'plans/PV1_TRAIN_INPUTS.private.json',v)
                real_sha=c.sha
                def member_sha(path):
                    path=Path(path)
                    if path.name=='DATASET_MANIFEST.json':
                        from scripts.pi0294.e1.scope import DATA
                        from .source_data_original import C1_MANIFEST
                        return a.BANK if path.parent.name=='FROZEN_TRAIN_BANK' else DATA if path.parent.name=='original' else C1_MANIFEST
                    return real_sha(path)
                with patch.object(c,'sha',side_effect=member_sha):
                    if mutation is None:self.assertEqual(set(c.verify_consumers(root)),set(records))
                    else:
                        with self.assertRaises(PermissionError):c.verify_consumers(root)


if __name__=='__main__':unittest.main()
