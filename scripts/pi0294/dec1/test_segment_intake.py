"""Tiny synthetic checkpoint/probe files, no models, allocation or tensors."""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1 import training
from . import segment_intake as intake,a40_training_context as context,checkpoint_storage as storage
from . import segmented_training as seg


def put(path,value,readonly=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o600)
    path.write_text(json.dumps(value));path.chmod(0o400 if readonly else 0o600);return path


@contextmanager
def fixture(arm='A_CONTROL'):
    with tempfile.TemporaryDirectory() as tmp:
        source=Path(tmp)/'source';target=Path(tmp)/'target';source.mkdir();target.mkdir();(target/'control').mkdir()
        recipe=dict(arm=arm,seed=20261019)
        data=SimpleNamespace(manifest_sha='a'*64,
            make_sampler=lambda dataset,stop:SimpleNamespace(state_dict=lambda n:dict(cursor=n,stop=stop,seed=20261019)))
        job='77.stdct-mgmt-02';source_sha='b'*40;checkpoints={}
        stats=intake.sha(put(source/'stats.fixture',dict(fixture='NO_REAL_DATA')))
        for step in (1024,2048):
            cp=source/'training'/arm/f'checkpoint-{step}';cp.mkdir(parents=True)
            for name in ('config.json','optimizer.pt','scheduler.pt','rng_state.pth','model.safetensors','processor/config.json'):
                put(cp/name,dict(fixture=name))
            put(cp/'statistics.json',dict(fixture='NO_REAL_DATA'))
            put(cp/'trainer_state.json',dict(global_step=step))
            put(cp/'SAMPLER_STATE.json',dict(recipe=recipe,manifest_sha256=data.manifest_sha,statistics_sha256=stats,
                sampler=data.make_sampler(data,4096*32).state_dict(step*32)))
            c=dict(global_step=step,full_resume=True,origin_job_id=job,origin_source_sha=source_sha,
                files={p.relative_to(cp).as_posix():intake.sha(p) for p in cp.rglob('*') if p.is_file()})
            checkpoints[step]=intake.sha(put(cp/'CHECKPOINT_COMPLETE.json',c))
        plan=dict(windows=[dict(id=f'probe-{i}') for i in range(12)])
        put(source/'plans/PV1_FIXED_TRAIN_PROBES.json',plan)
        folder=source/'evidence'/(arm+'-1024')/'TRAIN_PROBES';folder.mkdir(parents=True)
        records=[]
        for row in plan['windows']:
            p=folder/(row['id']+'.npz');p.write_bytes(b'FIXTURE_NOT_NPZ_OR_MODEL_DATA');p.chmod(0o400)
            records.append(dict(id=row['id'],arrays_sha256=intake.sha(p)))
        result=dict(status='ACTUAL_FIXED_TRAIN_PROBES_NOT_TASK_SUCCESS',complete=True,records=records,
            new_physics=0,physical_success_gate=False,checkpoint_manifest_sha256=checkpoints[1024],
            probe_plan_sha256=intake.sha(source/'plans/PV1_FIXED_TRAIN_PROBES.json'))
        probe_sha=intake.sha(put(folder/'RESULT.json',result))
        m=dict(code=dict(vla_commit=source_sha),authority=dict(segmented_authorization_release=seg.RELEASE))
        manifest_sha=intake.sha(put(source/'RUN_MANIFEST.json',m))
        effects=dict(job_id=job,request_id='75e12289-7b99-4f46-bd69-bbe64ff9cad0',request_sha256='c'*64,
            source_sha=source_sha,arm=arm,last_native_step=2048,recipe_sha256=intake.digest(recipe),
            pending_updates=[],unknown_query_intents=[],query_attempts=29,effect_journal_complete=True,
            start_native_step=32 if arm=='A_CONTROL' else 0,optimizer_updates_committed=2016 if arm=='A_CONTROL' else 2048)
        terminal=dict(status='CLOSED_SEGMENT_FULL_STATE_PRESERVED',job_id=job,recoverable_full_state=True,
            global_step=2048,settlement=dict(unknown_effects=[]),effects=effects)
        terminal_sha=intake.sha(put(source/'TERMINAL_RECONCILIATION.json',terminal))
        lease_sha=intake.sha(put(source/'lease.json',dict(status='CLOSED',run_uuid=source.name,terminal_receipt_sha256=terminal_sha)))
        put(source/'OUTPUT_BINDING.private.json',dict(run_root=str(source),uid=os.getuid(),run_manifest_sha256=manifest_sha,
            lease=str(source/'lease.json')))
        binding=dict(root=str(source),arm=arm,job_id=job,source_sha=source_sha,request_id=effects['request_id'],
            request_sha256=effects['request_sha256'],run_manifest_sha256=manifest_sha,terminal_sha256=terminal_sha,
            lease_sha256=lease_sha,checkpoint1024_sha256=checkpoints[1024],checkpoint2048_sha256=checkpoints[2048],probe_result_sha256=probe_sha)
        with patch.object(training,'STATS',stats),patch.object(intake,'R0',stats),patch.object(intake,'RECIPE',intake.digest(recipe)):
            yield source,target,binding,recipe,data


def cold_inputs(source,target,binding,proof):
    plan='plans/PV1_FIXED_TRAIN_PROBES.json'
    target.joinpath('plans').mkdir(exist_ok=True)
    shutil.copyfile(source/plan,target/plan);target.joinpath(plan).chmod(0o400)
    put(target/intake.BINDING,binding);put(target/intake.PROOF,proof)
    result=intake.read(source/'evidence'/(binding['arm']+'-1024')/'TRAIN_PROBES/RESULT.json')
    put(target/'plans/COMPLETED_1024_TRAIN_PROBES.json',result)
    put(target/'RUN_MANIFEST.json',dict(authority=dict(segmented_authorization_release=seg.RELEASE),
        execution=dict(sealed_inputs={n:intake.sha(target/n) for n in (intake.BINDING,intake.PROOF)})))


class SegmentIntakeTests(unittest.TestCase):
    def test_both_arms_audit_full_states_and_all_twelve_original_arrays_without_mutation(self):
        for arm in ('A_CONTROL','B_COVERAGE'):
            with self.subTest(arm=arm),fixture(arm) as (s,t,b,recipe,data):
                before={p:intake.sha(p) for p in s.rglob('*') if p.is_file()}
                proof=intake.audit_closed_source(b,recipe=recipe,dataset=data)
                self.assertEqual(proof['actual_consumed_cursor'],65536)
                self.assertEqual(len(proof['probe_files']),13)
                self.assertEqual(set(proof['checkpoint_members']),{'1024','2048'})
                self.assertEqual(before,{p:intake.sha(p) for p in s.rglob('*') if p.is_file()})

    def test_corrupt_full_optimizer_probe_array_and_wrong_arm_never_qualify(self):
        for kind in ('optimizer','probe','arm','terminal','lease'):
            with self.subTest(kind=kind),fixture() as (s,t,b,recipe,data):
                if kind=='optimizer':p=s/'training/A_CONTROL/checkpoint-2048/optimizer.pt'
                elif kind=='probe':p=s/'evidence/A_CONTROL-1024/TRAIN_PROBES/probe-0.npz'
                elif kind=='arm':recipe['arm']='B_COVERAGE';p=None
                elif kind=='terminal':p=s/'TERMINAL_RECONCILIATION.json'
                else:p=s/'lease.json'
                if p:p.chmod(0o600);p.write_text('DAMAGED')
                with self.assertRaises((PermissionError,ValueError)):intake.audit_closed_source(b,recipe=recipe,dataset=data)

    def test_missing_scientific1024_state_cannot_be_replaced_by2048_weights(self):
        with fixture() as (s,t,b,recipe,data):
            (s/'training/A_CONTROL/checkpoint-1024/optimizer.pt').unlink()
            with self.assertRaises(PermissionError):intake.audit_closed_source(b,recipe=recipe,dataset=data)

    def test_cold_copy_stays_full_resume_and_does_not_rerun_the_probe_draws(self):
        with fixture() as (s,t,b,recipe,data):
            proof=intake.audit_closed_source(b,recipe=recipe,dataset=data);cold_inputs(s,t,b,proof)
            shutil.copytree(s/'training',t/'training');shutil.copytree(s/'evidence',t/'evidence')
            result=intake.verify_copy(t,recipe=recipe,dataset=data)
            self.assertEqual(result['global_step'],2048)
            self.assertEqual(result['new_GPU_PBS_models_queries_updates_physics'],0)
            p=t/'training/A_CONTROL/checkpoint-2048/rng_state.pth';p.chmod(0o600);p.write_text('CHANGED')
            with self.assertRaises(ValueError):intake.verify_copy(t,recipe=recipe,dataset=data)

    def test_native_copy_uses_owned_full_members_and_one_exact_frozen_parent_link(self):
        with fixture() as (s,t,b,recipe,data):
            proof=intake.audit_closed_source(b,recipe=recipe,dataset=data);cold_inputs(s,t,b,proof)
            parent=t.parent/'parent';parent.mkdir();name='model.safetensors'
            shutil.copyfile(s/'training/A_CONTROL/checkpoint-2048'/name,parent/name)
            original=(parent/name).stat();frozen_sha=intake.sha(parent/name)
            put(t/storage.POLICY,dict(members={name:frozen_sha}))
            put(t/'plans/PV1_TRAIN_INPUTS.private.json',dict(CORR_parent_checkpoint=str(parent)))
            directory=t/'pbs/alloc-17';put(directory/'REQUEST.json',dict(training_block=dict(arm='A_CONTROL'),additional_output_upper_bytes=70000000000))
            with patch.dict(os.environ,dict(PBS_JOBID='fixture-not-real-allocation')),\
                    patch.object(context,'current_directory',return_value=directory),\
                    patch.object(intake,'limits',return_value=(2048,4096)),\
                    patch.object(storage,'_parent_files',return_value={name:frozen_sha}):
                result=intake.stage(t,recipe=recipe,dataset=data,guard=lambda:None,allocation={})
                self.assertEqual(result['global_step'],2048)
                for step in (1024,2048):
                    self.assertEqual((t/'training/A_CONTROL'/f'checkpoint-{step}'/name).stat().st_ino,original.st_ino)
                with self.assertRaises(PermissionError):intake.stage(t,recipe=recipe,dataset=data,guard=lambda:None,allocation={})
            self.assertEqual((parent/name).stat().st_mode,original.st_mode)

    def test_live_or_unknown_effect_source_rejected_even_with_refreshed_metadata_hashes(self):
        for kind in ('live','unknown','wrong_job','step'):
            with self.subTest(kind=kind),fixture() as (s,t,b,recipe,data):
                terminal=intake.read(s/'TERMINAL_RECONCILIATION.json');lease=intake.read(s/'lease.json')
                if kind=='live':lease['status']='ACTIVE'
                elif kind=='unknown':terminal['effects']['pending_updates']=[dict(local_step_before=2048)]
                elif kind=='wrong_job':terminal['effects']['job_id']='88.stdct-mgmt-02'
                else:terminal['global_step']=1024
                b['terminal_sha256']=intake.sha(put(s/'TERMINAL_RECONCILIATION.json',terminal))
                lease['terminal_receipt_sha256']=b['terminal_sha256'];b['lease_sha256']=intake.sha(put(s/'lease.json',lease))
                with self.assertRaises(PermissionError):intake.audit_closed_source(b,recipe=recipe,dataset=data)
