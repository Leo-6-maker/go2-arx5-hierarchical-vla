"""No-model regressions for the separately bounded original NI16 work."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch

import numpy as np

from scripts.pi0294.mp1 import endpoint_anchor_ni as ni
from scripts.pi0294.mp1 import n1_limits as n1
from scripts.pi0294.mp1.failure_config import ATTEMPTS,selected_sequences
from scripts.pi0294.mp1.probe_compare import PRECISION


class EndpointAnchorNITests(unittest.TestCase):
    def setUp(self):
        repo=Path(__file__).resolve().parents[2];reports=repo/'reports/PI0294/20261016T191445Z-localpc-mp1'
        self.card=json.loads((repo/'reports/PI0294/design/N1_CURRENT_ANCHOR_ENDPOINT_NI16_V1.json').read_text())
        self.gate=json.loads((reports/'N1_EIGHT_NATIVE_HISTORY_COPY_ACTUAL.json').read_text())
        self.matrix=json.loads((reports/'BEHAVIOR_AND_FAILURE_MATRIX.json').read_text())
        states={r['id']:r for r in self.gate['rows']};core=[];native=[];files={}
        for parent_number,identifier in enumerate(ATTEMPTS):
            for sequence in selected_sequences(identifier):
                name=identifier+'-S'+str(sequence).zfill(3);key=str(uuid.uuid5(uuid.NAMESPACE_URL,name))
                fingerprint=hashlib.sha256(name.encode()).hexdigest();parts=identifier.split('-')
                core.append(dict(id=name,task=parts[-1],scene_group='PI0294/TRAIN/'+'/'.join(parts[-3:-1]),
                    episode_uuid=str(uuid.uuid5(uuid.NAMESPACE_URL,identifier)),query_uuid=key,sequence=sequence,
                    observation_sha256=states[name]['observation_sha256'],observation_fingerprint=fingerprint))
                row=dict(id=name,query_uuid=key,sequence=sequence,observation_fingerprint=fingerprint,
                    checkpoint_manifest_sha256=ni.BASELINE,actual_precision=dict(PRECISION),noise_seed=sequence,
                    actual_native_noise_fingerprint=dict(shape=[1,40,132],dtype='torch.float32',sha256=fingerprint),
                    receipt_sha256='a'*64,arrays_sha256='b'*64)
                native.append(row);base='evidence/S-BC-512-TRAIN4-ALLOCATION-01/resident/receipts/'+key
                files[base+'.json']=row['receipt_sha256'];files[base+'.npz']=row['arrays_sha256']
        self.join=dict(status='PASS_EIGHT_CURRENT_SBC512_ACTIONS_ORIGINAL_INPUT_NOISE_AND_NATIVE_ARRAY_BINDINGS',
            new_queries=0,new_physics=0,core=dict(rows=core),hopper=dict(rows=native,checkpoint_manifest_sha256=ni.BASELINE))
        self.plan=dict(schema=ni.SCHEMA,authority_release=n1.RELEASE,run_uuid='source-pair-01',allocation_number=10,
            new_queries_max=16,new_training_updates=0,new_physics=0,prior_generation_queries=2390,
            prior_formal_updates=4096,N1_adoption_sha256=n1.ADOPTED_DELTA_SHA,N1_T0_epoch=1792264081,native_receipt_files=files)

    def validate(self):return ni.validate_values(self.plan,self.card,self.gate,self.join,self.matrix)

    def test_exact_eight_old_inputs_two_fixed_endpoints_and_noise_bindings(self):
        value=self.validate();self.assertEqual(len(value['rows']),8);self.assertEqual(len(value['native']),8)
        self.assertEqual(set(value['native']),ni.selected_ids());self.assertEqual(self.card['maximum_each_draw_wall_s'],60)

    def test_cannot_add_budget_reset_clock_change_model_or_skip_actual_qualification(self):
        for obj,key,value in [(self.plan,'allocation_number',11),(self.plan,'new_queries_max',17),
                (self.plan,'prior_generation_queries',0),(self.plan,'N1_T0_epoch',0),
                (self.card,'maximum_allocated_H200_s',28800),(self.card,'maximum_new_PBS',2),
                (self.card,'maximum_each_draw_wall_s',120),(self.card,'models',['S-CORR-2048']),
                (self.card,'actual_new_same_job600s80cycle_qualification_required',False),
                (self.card,'output_is_closed_loop_or_task_success',True)]:
            previous=deepcopy(obj[key]);obj[key]=value
            with self.subTest(key=key),self.assertRaises(PermissionError):self.validate()
            obj[key]=previous

    def test_gate_must_be_real_closed_strict_eight_route_not_old_failed_gate(self):
        for key,value in [('full_eight_anchor_gate_passed',False),('new_lease','ACTIVE'),
                ('comparison_absolute_tolerance',1e-9),('result_sha256','0'*64),('rows',self.gate['rows'][:-1])]:
            previous=deepcopy(self.gate[key]);self.gate[key]=value
            with self.subTest(key=key),self.assertRaises(PermissionError):self.validate()
            self.gate[key]=previous

    def test_no_dev_substitution_changed_noise_precision_or_duplicate_inputs(self):
        row=self.join['core']['rows'][0];other=self.join['hopper']['rows'][0]
        cases=[(row,'scene_group','PI0294/DEV/F0/000'),(row,'sequence',999),
            (other,'actual_native_noise_fingerprint',dict(shape=[1,40,10],dtype='torch.float32',sha256='0'*64)),
            (other,'actual_precision',dict(PRECISION,mode='TRAINING_TF32')),
            (other,'observation_fingerprint','0'*64)]
        for obj,key,value in cases:
            previous=deepcopy(obj[key]);obj[key]=value
            with self.subTest(key=key),self.assertRaises(PermissionError):self.validate()
            obj[key]=previous

    def test_matrix_requires_all24_independent_endpoint_audits_and_terminal_old_cells(self):
        original=deepcopy(self.matrix)
        cell=next(c for c in self.matrix['cells'] if '-2048-' in c['model']);cell['valid_policy_outcome']=False
        with self.assertRaises(PermissionError):self.validate()
        self.matrix=deepcopy(original);self.matrix['cells'][0]['status']='PENDING_OR_IN_PROGRESS'
        with self.assertRaises(PermissionError):self.validate()
        self.matrix=deepcopy(original);row=next(r for r in self.matrix['actual_attempts'] if '-2048-' in r['model'] and r['qualified_policy_outcome'])
        row['cross_host_chain_verified']=False
        with self.assertRaises(PermissionError):self.validate()

    def test_unbound_legacy_scope_has_no_implicit_NI_work(self):
        self.assertEqual(ni.binding_files(dict(target_allocation_number=9)),[])
        for boundary in [dict(target_allocation_number=9,endpoint_probe_binding_relative=ni.PLAN),
                dict(target_allocation_number=10,endpoint_probe_binding_relative='../wrong')]:
            with self.assertRaises(PermissionError):ni.binding_files(boundary)
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(ni.closed_draws(folder),0)
            (Path(folder)/'evidence'/ni.LABEL).mkdir(parents=True)
            with self.assertRaises(ni.MutationOutcomeUnknown):ni.closed_draws(folder)

    def test_NI_boundary_cannot_replace_the_actual_source09_outcomes_with_a_summary(self):
        outcomes=[]
        for row in self.matrix['actual_attempts']:
            a={k:row[k] for k in ['attempt_id','status','failure_stage','model','task','split','scene_group',
                'duration_s','main_integrations','RGB_pairs','S1_rejections','live_native_failures','teacher_takeover']}
            a.update(pipeline_trustworthy=row['adjudication_pipeline_trustworthy'],result_sha256=row['raw_result_sha256']);outcomes.append(a)
        past=dict(GPU_allocated_s=42088,generation_queries=2390,optimizer_updates=4096,pending_updates=0,
            GPU_reserved_s=0,PBS_submissions=[{}]*9)
        with patch.object(ni,'read',return_value=self.matrix):
            ni.checked_source_boundary('/mock',past,outcomes)
            for key,bad in [('GPU_allocated_s',0),('generation_queries',0),('pending_updates',1),('GPU_reserved_s',7200)]:
                with self.subTest(key=key),self.assertRaises(PermissionError):ni.checked_source_boundary('/mock',dict(past,**{key:bad}),outcomes)
            changed=deepcopy(outcomes);changed[-1]['main_integrations']+=200
            with self.assertRaises(PermissionError):ni.checked_source_boundary('/mock',past,changed)

    def test_all_five_cold_inputs_are_immutable_hash_bound_and_not_symlinks(self):
        from scripts.pi0294.native_dataset import sha
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            values={ni.CARD:self.card,ni.GATE:self.gate,ni.JOIN:self.join,ni.MATRIX:self.matrix}
            def put(relative,value):
                p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));p.chmod(0o400)
            for relative,value in values.items():put(relative,value)
            plan=dict(self.plan,binding_file_sha256={relative:sha(root/relative) for relative in values});put(ni.PLAN,plan)
            m=dict(execution=dict(sealed_inputs={relative:sha(root/relative) for relative in ni.FILES}))
            q=dict(target_allocation_number=10,endpoint_probe_binding_relative=ni.PLAN)
            self.assertEqual(len(ni.validate_binding(root,q,m)['rows']),8)
            self.assertEqual(set(ni.binding_files(q)),set(ni.FILES))
            p=root/ni.GATE;p.chmod(0o600)
            with self.assertRaises(PermissionError):ni.validate_binding(root,q,m)
            p.chmod(0o400);m['execution']['sealed_inputs'][ni.GATE]='0'*64
            with self.assertRaises(PermissionError):ni.validate_binding(root,q,m)

    def test_native_request_rejects_NI_walltime_prior_cost_or_capacity_mismatch_before_qsub(self):
        from scripts.pi0294.mp1 import allocation
        from scripts.pi0294.mp1.n1_boundary import paths
        from scripts.pi0294.native_dataset import sha
        with tempfile.TemporaryDirectory() as folder:
            parent=Path(folder)/'parent';root=parent/'children/source';(root/'plans').mkdir(parents=True)
            def put(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            a=dict(T0_epoch=100,new_submission_and_candidate_cutoff_epoch=200,new_experiment_cutoff_epoch=300)
            put(parent/'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json',a);put(root/'RUN_MANIFEST.json',{})
            lease=parent/'oldlease.json';put(lease,dict(status='CLOSED',run_uuid='20261016T150228Z-localpc-e1'))
            put(root/'plans/OLD_E1_BINDING.private.json',dict(lease=str(lease)))
            put(root/paths(10)[0],dict(endpoint_probe_binding_relative=ni.PLAN))
            request=dict(run_uuid='source',source_sha='fixture',run_manifest_sha256=sha(root/'RUN_MANIFEST.json'),
                mode='MP1_FIXED_SOURCE_PAIR',numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',submit_count=10,
                walltime_s=7200,prior_GPU_allocated_s=42088,additional_output_upper_bytes=8*1024**3,created_epoch=190,sealed_inputs={})
            ledger=dict(GPU_reserved_s=7200,PBS_submissions=[{}]*10,GPU_allocated_s=42088)
            manifest=dict(code=dict(vla_commit='fixture'),execution=dict(n1_authority_binding={}))
            bound=dict(mode='MP1_FIXED_SOURCE_PAIR',PBS=12,GPU_s=129600,job_wall_s_max=7200)
            with patch('scripts.pi0294.mp1.n1_limits.clock',return_value=a),patch.object(allocation,'limits',return_value=bound), \
                    patch.object(ni,'validate_binding',return_value={}):
                self.assertEqual(allocation.request_contract(root,request,manifest,ledger,250),a)
                for key,bad in [('walltime_s',7201),('prior_GPU_allocated_s',42087),('additional_output_upper_bytes',9*1024**3)]:
                    changed=dict(request,**{key:bad})
                    with self.subTest(key=key),self.assertRaises(PermissionError):
                        allocation.request_contract(root,changed,manifest,ledger,250)

    def test_bounded_runner_draws_only16_and_noise_failure_stops_without_resampling(self):
        for wrong_noise in [False,True]:
            with self.subTest(wrong_noise=wrong_noise),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);(root/'evidence').mkdir();ledger=type('Ledger',(),dict(path=root/'ledger.json'))()
                counters=dict(generation_queries=2390,optimizer_updates=4096,pending_updates=[]);draws=[];loads=[];unloads=[]
                binding=self.validate()
                def fixed(root,label,identifier,instance,payload,seed,target,ledger):
                    counters['generation_queries']+=1;draws.append((label,identifier))
                    noise=dict(binding['native'][identifier]['actual_native_noise_fingerprint'])
                    if wrong_noise:noise['sha256']='0'*64
                    return np.zeros((40,10)),dict(noise=noise,inference_precision=dict(PRECISION),arrays_sha256='c'*64)
                def digest(path):return ni.CHECKPOINTS[path.parent.parent.name+'-2048']
                with patch.object(ni,'current',return_value=binding),patch.object(ni,'read',side_effect=lambda p:dict(counters) if p==ledger.path else {}), \
                        patch.object(ni,'sha',side_effect=digest),patch.object(ni,'original_payload',return_value={}), \
                        patch.object(ni.model,'fixed_draw',side_effect=fixed), \
                        patch('scripts.pi0294.mp1.pre_model_qualification.require',return_value=dict(status='MOCK_ONLY_NOT_NATIVE')):
                    invoke=lambda:ni.run(root,dict(submit_count=10,walltime_s=7200),ledger,lambda:None,
                        lambda checkpoint,label,trainable:loads.append((label,trainable)) or {},lambda:unloads.append(True),lambda:object())
                    if wrong_noise:
                        with self.assertRaises(PermissionError):invoke()
                        self.assertEqual(len(draws),1);self.assertEqual(len(loads),1);self.assertEqual(len(unloads),1)
                        self.assertEqual(counters['generation_queries'],2391)
                        self.assertFalse((root/'evidence'/ni.LABEL/'RESULT.json').exists())
                    else:
                        result=invoke();self.assertEqual(result['new_neural_queries'],16)
                        self.assertEqual(len(draws),16);self.assertEqual(len(set(draws)),16)
                        self.assertEqual(len(loads),2);self.assertTrue(all(not trainable for _,trainable in loads))
                        self.assertEqual(len(unloads),2);self.assertEqual(counters['generation_queries'],2406)
                        self.assertEqual(result['new_optimizer_updates'],0);self.assertEqual(result['new_physics'],0)

    def test_closed_NI_arrays_join_completed_one_shot_journals_and_reject_extra_intents(self):
        from scripts.pi0294.e1.journal import Journal
        from scripts.pi0294.mp1.n1_boundary import paths
        from scripts.pi0294.native_dataset import sha
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);binding=self.validate();records=[]
            def put(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            put(root/'RUN_MANIFEST.json',{});put(root/paths(10)[0],{})
            journal=Journal(root/'reservations/probe-generations',root.name)
            for name in ni.CHECKPOINTS:
                checkpoint=root/'training'/name.removesuffix('-2048')/'checkpoint-2048'/'CHECKPOINT_COMPLETE.json'
                files={'config.json':'f'*64};put(checkpoint,dict(files=files,global_step=2048,full_resume=True));weights=ni.digest(files)
                for row in binding['rows']:
                    native=binding['native'][row['id']];target=root/'evidence'/ni.LABEL/name/(row['id']+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(target,physical_H40=np.zeros((40,10)),normalized=np.zeros((1,40,132)),
                        noise_seed=np.asarray(native['noise_seed'],dtype=np.uint64),
                        noise_fingerprint_json=np.asarray(json.dumps(native['actual_native_noise_fingerprint'])),
                        inference_precision_json=np.asarray(json.dumps(PRECISION)),observation_fingerprint=np.asarray(native['observation_fingerprint']),
                        checkpoint_files_fingerprint=np.asarray(weights))
                    meta=dict(noise=native['actual_native_noise_fingerprint'],inference_precision=dict(PRECISION),arrays_sha256=sha(target))
                    label=ni.LABEL+'-'+name;args=dict(label=label,id=row['id'],seed=native['noise_seed'],
                        observation_fingerprint=native['observation_fingerprint'],checkpoint_files_fingerprint=weights,precision=PRECISION['mode'])
                    journal.once('probe-'+label+'-'+row['id'],args,lambda:meta)
                    records.append(dict(id=row['id'],model=name,arrays_sha256=sha(target),noise=native['actual_native_noise_fingerprint'],
                        noise_seed=native['noise_seed'],original_query_uuid=native['query_uuid'],observation_fingerprint=native['observation_fingerprint'],
                        checkpoint_manifest_sha256=ni.CHECKPOINTS[name],new_query=True,live_or_shadow_integrations=0,inference_precision=dict(PRECISION)))
            result=root/'evidence'/ni.LABEL/'RESULT.json'
            put(result,dict(status='COMPLETE_SIXTEEN_SAME_INPUT_NOISE_ENDPOINT_DRAWS_NOT_POLICY_EVALUATION',records=records,
                new_neural_queries=16,new_optimizer_updates=0,new_physics=0,closed_loop_or_task_success_claimed=False))
            def digest(path):
                return ni.CHECKPOINTS[path.parent.parent.name+'-2048'] if path.name=='CHECKPOINT_COMPLETE.json' else sha(path)
            with patch.object(ni,'validate_binding',return_value=binding),patch.object(ni,'sha',side_effect=digest):
                self.assertEqual(ni.closed_draws(root),16)
                extra=journal.path('probe-'+ni.LABEL+'-UNDECLARED');put(extra,dict(status='INTENT'))
                with self.assertRaises(ni.MutationOutcomeUnknown):ni.closed_draws(root)
                extra.unlink();target=result.parent/records[0]['model']/(records[0]['id']+'.npz')
                target.write_bytes(b'corrupt')
                with self.assertRaises(PermissionError):ni.closed_draws(root)


if __name__=='__main__':unittest.main()
