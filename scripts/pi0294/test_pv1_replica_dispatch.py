"""CPU-only evidence fixtures; no actual DEV, model, allocation or training."""
from contextlib import contextmanager,ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,patch

from scripts.pi0294.pv1 import neural,baseline
from scripts.pi0294.pv1.decision import AUTHORITY
from scripts.pi0294.pv1.evaluation_plan import POPULATION_FILE,development
from scripts.pi0294.pv1.recipe import PARENT,primary_recipe
from scripts.pi0294.test_pv1_baseline import fixtures,raw
from scripts.pi0294.test_pv1_evaluation_plan import groups
from scripts.pi0294.test_pv1_recipe import old_recipe
from scripts.pi0294.test_mp1_transport_fixture import put


def candidate_evidence(candidate,counts,checkpoint):
    rows=fixtures();seen=dict(T1=0,T2=0)
    for item in rows:
        row=json.loads(item['adjudication']);task=row['task'];seen[task]+=1
        row['attempt_id']=row['attempt_id'].replace('B0_CORR',candidate);row['model']=candidate
        success=seen[task]<=counts[task]
        row.update(status='SUCCESS' if success else 'VALID_TASK_FAIL',
            failure_stage='COMPLETE_SUCCESS' if success else 'OBJECT_TASK_RESULT')
        result=json.loads(item['result']);result['id']=row['attempt_id'];item['result']=raw(result)
        row['result_sha256']=hashlib.sha256(item['result']).hexdigest();item['adjudication']=raw(row)
        audit=json.loads(item['audit']);audit.update(attempt_id=row['attempt_id'],adjudication=row,
            score=dict(complete_32s=True,clock_valid=True,safety_valid=True,persistent_success=success))
        item['audit']=raw(audit)
        ready=json.loads(item['model_ready']);ready['label']=ready['label'].replace('B0_CORR',candidate)
        ready['identity'].update(label=candidate,global_step=2048 if candidate=='B0_CORR' else 4096,
            checkpoint_manifest_sha256=checkpoint);item['model_ready']=raw(ready)
        chain=json.loads(item['causal_chain']);chain.update(attempt_id=row['attempt_id'],
            adjudication_sha256=hashlib.sha256(item['adjudication']).hexdigest(),identity=ready['identity'])
        item['causal_chain']=raw(chain)
    return rows


@contextmanager
def sealed_selection(counts=None):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);files={};population=raw(dict(groups=groups()))
        population_sha=hashlib.sha256(population).hexdigest()
        def store(name,value):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():p.chmod(0o600)
            p.write_bytes(raw(value));p.chmod(0o400);files[name]=neural.sha(p)
        store(POPULATION_FILE,json.loads(population))
        selection=dict(schema='PI0294_PV1_P2_REPLICA_SELECTION_V1',authority_release=AUTHORITY,
            local_recipe_stage='P2',population_plan_sha256=population_sha,route='COVERAGE_FIRST',
            new_bank_manifest_sha256='b'*64,LOCKBOX_opened=False,replica_seed=20261020,
            replica_steps=4096,winner='B_COVERAGE',candidates={})
        for candidate,count,checkpoint in [('B0_CORR',16,PARENT),('A_CONTROL',18,'a'*64),('B_COVERAGE',20,'c'*64)]:
            evidence=[]
            successes=counts.get(candidate,dict(T1=count,T2=count)) if counts else dict(T1=count,T2=count)
            for i,item in enumerate(candidate_evidence(candidate,successes,checkpoint)):
                paths={}
                for kind,blob in item.items():
                    name=f'plans/replica-evidence/{candidate}/{i}/{kind}.json'
                    store(name,json.loads(blob));paths[kind]=name
                evidence.append(paths)
            selection['candidates'][candidate]=dict(checkpoint_manifest_sha256=checkpoint,
                global_step=2048 if candidate=='B0_CORR' else 4096,evidence=evidence)
        name='plans/PV1_P4_REPLICA_SELECTION.json';store(name,selection)
        stage=dict(route='COVERAGE_FIRST',new_bank_manifest_sha256='b'*64)
        def input_(name):
            if Path(name).is_absolute() or '..' in Path(name).parts or files.get(name)!=neural.sha(root/name):
                raise PermissionError('CPU_FIXTURE_SEAL_CHANGED')
            return json.loads((root/name).read_text())
        with patch.object(neural,'POPULATION',population_sha),patch.object(baseline,'POPULATION',population_sha):
            yield root,selection,stage,input_,store


class ReplicaAdmission(unittest.TestCase):
    def test_probe_only_admission_requires_actual_charged_1024_and_separate_namespace(self):
        for replica in (False,True):
            with self.subTest(replica=replica),tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
                root=Path(tmp);population=raw(dict(groups=groups()));population_sha=hashlib.sha256(population).hexdigest()
                stack.enter_context(patch.object(neural,'POPULATION',population_sha))
                stack.enter_context(patch.object(baseline,'POPULATION',population_sha))
                stack.enter_context(patch.object(neural,'replica_selection',return_value='B_COVERAGE'))
                admission=baseline.admission(population,route='COVERAGE_FIRST',new_bank_manifest_sha256='b'*64,evidence=fixtures())
                arm='B_COVERAGE';key=('REPLICA_' if replica else '')+arm
                block=dict(arm=arm,replica=replica,probe_only=True,stop_after_step=1024,
                    resume_checkpoint_relative=f'training/{key}/checkpoint-1024')
                values={'PV1_STAGE_BINDING.json':dict(route='COVERAGE_FIRST',new_bank_manifest_sha256='b'*64),
                    POPULATION_FILE:json.loads(population),'plans/PV1_B0_DEV_ADMISSION.json':admission,
                    'plans/SOURCE_EVALUATION_PLAN.json':development(groups(),route='COVERAGE_FIRST',
                        candidate=key,new_bank_manifest_sha256='b'*64),
                    'plans/PRIOR_EVALUATION_OUTCOMES.json':[],'plans/PRIMARY_TRAIN_BLOCK.json':block,
                    'plans/INITIAL_CORR_PARAMETER_INVENTORY.json':{},'plans/PV1_FIXED_TRAIN_PROBES.json':{},
                    'plans/FROZEN_SOURCE_PAIR_RECIPE.json':old_recipe(),
                    'plans/PRIMARY_LEARNING_RECIPES.json':dict(recipes={key:primary_recipe(old_recipe(),
                        'COVERAGE_FIRST',arm,replica=replica)})}
                def bind():
                    for name,v in values.items():
                        p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
                        if p.exists():p.chmod(0o600)
                        p.write_bytes(raw(v));p.chmod(0o400)
                    put(root/'RUN_MANIFEST.json',dict(execution=dict(sealed_inputs={n:neural.sha(root/n) for n in values})))
                bind();put(root/block['resume_checkpoint_relative']/'CHECKPOINT_COMPLETE.json',dict(global_step=1024,full_resume=True))
                value=dict(pending_updates=[],unknown_effects=[],training_runs={key:dict(last_native_step=1024)})
                ledger=root/'ledger.json';put(ledger,value)
                stack.enter_context(patch.object(neural,'Ledger',return_value=SimpleNamespace(path=ledger)))
                request=dict(stage='P4_REPLICA' if replica else 'P2_PRIMARY',training_block=block)
                neural.stage_work(root,request)
                for change in [dict(pending_updates=[dict(unknown=True)]),dict(unknown_effects=['unknown']),
                        dict(training_runs={key:dict(last_native_step=1023)})]:
                    put(ledger,dict(value,**change))
                    with self.assertRaisesRegex(PermissionError,'EXACT_CHARGED_1024'):neural.stage_work(root,request)
                put(ledger,value)
                completed=root/'evidence'/(key+'-1024')/'TRAIN_PROBES/RESULT.json'
                put(completed,dict(complete=True))
                with self.assertRaisesRegex(PermissionError,'NO_REMAINING_1024_PROBES'):neural.stage_work(root,request)
                completed.unlink()
                if replica:
                    block['resume_checkpoint_relative']='training/B_COVERAGE/checkpoint-1024';bind()
                    with self.assertRaisesRegex(PermissionError,'OWN_NATIVE_PRIMARY_FULL_RESUME'):neural.stage_work(root,request)
                block['probe_only']=False;block['resume_checkpoint_relative']=f'training/{key}/checkpoint-1024'
                block['stop_after_step']=4096;bind()
                with self.assertRaises(FileNotFoundError):neural.stage_work(root,request)

    def test_actual_complete_byte_bound_pair_selects_only_predeclared_winner(self):
        with sealed_selection() as (root,selection,stage,input_,store):
            self.assertEqual(neural.replica_selection(root,input_,stage),'B_COVERAGE')
            selection['winner']='A_CONTROL';store('plans/PV1_P4_REPLICA_SELECTION.json',selection)
            with self.assertRaisesRegex(PermissionError,'PREDECLARED_WINNER'):
                neural.replica_selection(root,input_,stage)

    def test_replication_gate_preserves_task_minimum_baseline_retention_and_net_gain(self):
        for counts in [dict(A_CONTROL=dict(T1=17,T2=24),B_COVERAGE=dict(T1=24,T2=17)),
                dict(B0_CORR=dict(T1=20,T2=20)),
                dict(B0_CORR=dict(T1=24,T2=16),A_CONTROL=dict(T1=18,T2=18),B_COVERAGE=dict(T1=20,T2=20))]:
            with self.subTest(counts=counts),sealed_selection(counts) as (root,selection,stage,input_,store):
                with self.assertRaisesRegex(PermissionError,'PREDECLARED_WINNER'):
                    neural.replica_selection(root,input_,stage)

    def test_failed_top_rank_does_not_silently_fall_back_to_lower_rank(self):
        counts=dict(B0_CORR=dict(T1=24,T2=0),A_CONTROL=dict(T1=22,T2=23),
            B_COVERAGE=dict(T1=23,T2=21))
        with sealed_selection(counts) as (root,selection,stage,input_,store):
            for winner in ('A_CONTROL','B_COVERAGE'):
                selection['winner']=winner;store('plans/PV1_P4_REPLICA_SELECTION.json',selection)
                with self.assertRaisesRegex(PermissionError,'PREDECLARED_WINNER'):
                    neural.replica_selection(root,input_,stage)

    def test_replica_dev_requires_exact_actual_trained_second_seed_not_relabelled_primary(self):
        with sealed_selection() as (root,selection,stage,input_,store),ExitStack() as stack:
            key='REPLICA_B_COVERAGE';recipe=primary_recipe(old_recipe(),'COVERAGE_FIRST','B_COVERAGE',replica=True)
            plan=development(groups(),route='COVERAGE_FIRST',candidate=key,new_bank_manifest_sha256='b'*64)
            checkpoint=root/'training'/key/'checkpoint-4096'
            put(checkpoint/'SAMPLER_STATE.json',dict(recipe=recipe))
            put(checkpoint/'CHECKPOINT_COMPLETE.json',dict(global_step=4096,full_resume=True,
                files={'SAMPLER_STATE.json':neural.sha(checkpoint/'SAMPLER_STATE.json')}))
            additions={'PV1_STAGE_BINDING.json':stage,'plans/SOURCE_EVALUATION_PLAN.json':plan,
                'plans/PRIOR_EVALUATION_OUTCOMES.json':[],'plans/INITIAL_CORR_PARAMETER_INVENTORY.json':{},
                'plans/PRIMARY_LEARNING_RECIPES.json':dict(recipes={key:recipe}),
                'plans/FROZEN_SOURCE_PAIR_RECIPE.json':old_recipe(),
                'plans/EVALUATION_CHECKPOINT.private.json':dict(candidate=key,global_step=4096,
                    checkpoint=str(checkpoint),checkpoint_manifest_sha256=neural.sha(checkpoint/'CHECKPOINT_COMPLETE.json'))}
            for name,v in additions.items():store(name,v)
            sealed={str(p.relative_to(root)):neural.sha(p) for p in (root/'plans').rglob('*.json')}
            sealed['PV1_STAGE_BINDING.json']=neural.sha(root/'PV1_STAGE_BINDING.json')
            put(root/'RUN_MANIFEST.json',dict(execution=dict(sealed_inputs=sealed)))
            value=dict(pending_updates=[],unknown_effects=[],training_runs={key:dict(last_native_step=4096)})
            ledger=root/'ledger.json';put(ledger,value)
            stack.enter_context(patch.object(neural,'Ledger',return_value=SimpleNamespace(path=ledger)))
            stack.enter_context(patch.object(neural,'session_spec',side_effect=lambda root,label:plan['panels'][label]))
            request=dict(stage='P4_DEV',evaluation_candidate=key)
            neural.stage_work(root,request)
            value['training_runs'][key]['last_native_step']=4095;put(ledger,value)
            with self.assertRaisesRegex(PermissionError,'CHARGED_4096'):neural.stage_work(root,request)
            value['training_runs'][key]['last_native_step']=4096;put(ledger,value)
            recipe['seed']=20261019;store('plans/PRIMARY_LEARNING_RECIPES.json',dict(recipes={key:recipe}))
            sealed['plans/PRIMARY_LEARNING_RECIPES.json']=neural.sha(root/'plans/PRIMARY_LEARNING_RECIPES.json')
            put(root/'RUN_MANIFEST.json',dict(execution=dict(sealed_inputs=sealed)))
            with self.assertRaisesRegex(PermissionError,'PREFROZEN_PRIMARY_RECIPE'):neural.stage_work(root,request)

    def test_reported_success_counts_cannot_replace_missing_actual_audit(self):
        with sealed_selection() as (root,selection,stage,input_,store):
            selection['candidates']['B_COVERAGE']['successes']=dict(T1=24,T2=24)
            selection['candidates']['B_COVERAGE']['evidence'].pop()
            store('plans/PV1_P4_REPLICA_SELECTION.json',selection)
            with self.assertRaisesRegex(PermissionError,'INCOMPLETE_DEV48'):
                neural.replica_selection(root,input_,stage)

    def test_lockbox_p3_and_wrong_replica_seed_fail_closed(self):
        for change,error in [(dict(LOCKBOX_opened=True),'UNOPENED_LOCKBOX'),
                (dict(local_recipe_stage='P3'),'P3_REPLICATION_2048_NOT_IMPLEMENTED'),
                (dict(replica_seed=20261019),'UNOPENED_LOCKBOX')]:
            with self.subTest(change=change),sealed_selection() as (root,selection,stage,input_,store):
                selection.update(change);store('plans/PV1_P4_REPLICA_SELECTION.json',selection)
                with self.assertRaisesRegex(PermissionError,error):neural.replica_selection(root,input_,stage)

    def test_generalized_dev_admission_keeps_candidate_identity_and_all_b0_checks(self):
        population=raw(dict(groups=groups()));checkpoint='c'*64
        with patch.object(baseline,'POPULATION',hashlib.sha256(population).hexdigest()):
            evidence=candidate_evidence('REPLICA_B_COVERAGE',dict(T1=18,T2=19),checkpoint)
            result=baseline.admission(population,route='COVERAGE_FIRST',new_bank_manifest_sha256='b'*64,
                evidence=evidence,candidate='REPLICA_B_COVERAGE',checkpoint_manifest_sha256=checkpoint)
            self.assertEqual(result['successes'],dict(T1=18,T2=19))
            with self.assertRaises(PermissionError):
                baseline.admission(population,route='COVERAGE_FIRST',new_bank_manifest_sha256='b'*64,
                    evidence=evidence,candidate='B_COVERAGE',checkpoint_manifest_sha256=checkpoint)

    def test_probe_only_dispatch_never_calls_trainer_and_keeps_replica_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp)/'campaign/children/cold';directory=root/'pbs/alloc-04';directory.mkdir(parents=True)
            key='REPLICA_B_COVERAGE';checkpoint=root/'training'/key/'checkpoint-1024'
            block=dict(arm='B_COVERAGE',replica=True,probe_only=True,stop_after_step=1024,
                resume_checkpoint_relative=str(checkpoint.relative_to(root)))
            request=dict(stage='P4_REPLICA',training_block=block,source_sha='a'*40,
                request_id='fixture-request',numerical_mode='CPU_FIXTURE_NOT_GPU')
            put(directory/'REQUEST.json',request);put(root/'plans/PRIMARY_TRAIN_BLOCK.json',block)
            put(root.parent.parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json',dict(stop_new_training_epoch=10**12))
            recipe=primary_recipe(old_recipe(),'COVERAGE_FIRST','B_COVERAGE',replica=True)
            put(root/'plans/PRIMARY_LEARNING_RECIPES.json',dict(recipes={key:recipe}))
            put(root/'plans/FROZEN_SOURCE_PAIR_RECIPE.json',old_recipe())
            ledger=MagicMock();ledger.path=root/'ledger.json';put(ledger.path,dict(CPU_s=0))
            allocation=dict(job_id='CPU_FIXTURE',hostname='CPU_FIXTURE',allocation_end_epoch=10**12)
            instance=SimpleNamespace(identity=dict(global_step=1024));dataset=MagicMock()
            fake_torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:1,get_device_name=lambda i:'H200'),
                set_num_threads=lambda n:None)
            stack.enter_context(patch.dict(sys.modules,{'torch':fake_torch}))
            stack.enter_context(patch.dict(neural.os.environ,{'CUDA_VISIBLE_DEVICES':'CPU_FIXTURE'}))
            replacements=dict(admit=lambda *a:allocation,Ledger=lambda *a:ledger,stage_work=lambda *a:None,
                prepare_environment=lambda *a:None,configure_torch=lambda *a:None,read_live_owner=lambda *a:None,
                qualify_transport=lambda *a:{},consume=lambda *a:{},parent_checkpoint=lambda *a:root/'CORR',
                require_current=lambda *a:{},train_dataset=lambda *a:dataset,
                checkpoint_identity=lambda p,*a:dict(global_step=1024,checkpoint_manifest_sha256='c'*64))
            for name,replacement in replacements.items():stack.enter_context(patch.object(neural,name,replacement))
            stack.enter_context(patch.object(neural.os,'sched_setaffinity'))
            stack.enter_context(patch.object(neural.resource,'setrlimit'))
            stack.enter_context(patch.object(neural.model,'load',return_value=instance))
            stack.enter_context(patch.object(neural.model,'unload'))
            checked=stack.enter_context(patch.object(neural,'checked_resume',return_value=1024))
            trainer=stack.enter_context(patch.object(neural,'train'))
            probes=stack.enter_context(patch.object(neural,'probes',return_value=dict(complete=True)))
            result=neural.run(root,directory/'REQUEST.json')
            trainer.assert_not_called();checked.assert_called_once_with(root,key,checkpoint,recipe,dataset)
            self.assertEqual(probes.call_args.args[1],key+'-1024')
            self.assertEqual(result['status'],'PV1_EXACT_1024_PROBES_RECOVERED_ZERO_OPTIMIZER_UPDATES')
