import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest

from .e1.dataset import SmallSampler,select_small8
from .e1.journal import Journal,MutationOutcomeUnknown,process_is_same
from .e1.trainer import learning_rate_multiplier


def die_after_intent(folder):
    import os
    Journal(folder,'fixture').once('query',{'sequence':0},lambda:os._exit(17))


class E1Tests(unittest.TestCase):
    def test_actual_exposure_discloses_E1_T1_only_correction_confound(self):
        from .e1.exposure import summarize
        rows=[]
        for c in range(64):
            correction=c%32>=16;task='T1' if correction or c%2==0 else 'T2'
            rows.append(dict(cursor=c,split='TRAIN',task=task,origin='E0_STATE_AGGREGATION_ONLY' if correction else 'new',episode_id='segment' if correction else task))
        value=summarize(rows,64,'M2-CORR')
        self.assertEqual(value['task_samples'],dict(T1=48,T2=16));self.assertEqual(value['origin_samples'],dict(original=32,C0=32))
        self.assertTrue(value['matched_total_updates_not_matched_task_exposure']);self.assertFalse(value['causal_correction_gain_claim_supported'])
        with self.assertRaises(ValueError):summarize(rows+rows[:1],64,'M2-CORR')
        with self.assertRaises(ValueError):summarize(rows[:-1],64,'M2-CORR')
        rows[0]['origin']='NEW_C1'
        with self.assertRaises(PermissionError):summarize(rows,64,'M2-CORR')

    def test_exposure_BC_exact_consumption_ignores_only_prefetch(self):
        from .e1.exposure import summarize
        rows=[dict(cursor=c,split='TRAIN',task='T1' if c%2==0 else 'T2',origin='new',episode_id=str(c%8)) for c in range(96)]
        value=summarize(rows,64,'M2-BC')
        self.assertEqual(value['actual_consumed_samples'],64)
        self.assertEqual(value['task_samples'],dict(T1=32,T2=32))
        self.assertEqual(value['origin_samples'],dict(original=64,C0=0))
        self.assertEqual(value['correction_segments_observed'],0)
        rows[0]['origin']='E0_STATE_AGGREGATION_ONLY';rows[0]['task']='T2'
        with self.assertRaises(PermissionError):summarize(rows,64,'M2-BC')

    def test_zero_step_attempt_has_evidence_but_no_fabricated_task_score(self):
        import numpy as np
        from PIL import Image
        from .e1.zero_step_audit import audit
        from .e1.analysis import audited_outcome
        from .native_dataset import sha
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'episodes/fixture';folder.mkdir(parents=True)
            result=dict(id='fixture',physics_steps=0,source_sha='fixture')
            def put(name,v):(folder/name).write_text(json.dumps(v))
            put('RESULT.json',result);put('WORKER_RECEIPT.json',dict(result,rgb_pairs=1));put('CARD.json',dict(id='fixture'))
            decision=dict(attempt_id='fixture',status='ENGINEERING_INCOMPLETE',pipeline_trustworthy=False,main_integrations=0,result_sha256=sha(folder/'RESULT.json'),duration_s=0)
            put('E1_ADJUDICATION.json',decision);put('samples.json',[]);put('references.json',[])
            put('S1_RESULT.json',dict(shadow_invocations=0,pre_execution_rejections=0,live_native_failures=0))
            np.savez(folder/'highrate.npz',rows=np.empty((0,45)))
            for name in ['body_rgb','wrist_rgb']:
                (folder/name).mkdir();Image.new('RGB',(256,256),(1,2,3)).save(folder/name/'0000.png')
            put('FILE_MANIFEST.json',[])
            def movie(*args):return dict(actual_paired_frames=1,decoded_frames=1,padding_or_interpolation=False)
            value=audit(root,'fixture',movie)
            self.assertEqual(value['status'],'PASS_ZERO_INTEGRATION_ENGINEERING_EVIDENCE');self.assertIsNone(value['score'])
            self.assertFalse(value['scientific_policy_outcome_valid']);self.assertTrue(value['video']['initial_frame_only'])
            self.assertFalse(audited_outcome(decision,value,autonomous=True)['pipeline_trustworthy'])
            self.assertFalse((folder/'physics.npz').exists());self.assertFalse((folder/'SCORE.json').exists())

    def test_NFS_direct_open_only_reads_last_actual_query_responses(self):
        from .e1.nfs_visibility import refresh_completed_responses
        import uuid
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'outbox').mkdir();ids=[str(uuid.uuid4()) for _ in range(3)]
            (root/'QUERY_STATE.json').write_text(json.dumps(dict(queries=[dict(query_uuid=i) for i in ids])))
            for i in ids:(root/'outbox'/(i+'.json')).write_text('{}')
            self.assertEqual(refresh_completed_responses(root),2)
            (root/'outbox'/(ids[-1]+'.json')).unlink()
            self.assertEqual(refresh_completed_responses(root),1)
            (root/'QUERY_STATE.json').write_text(json.dumps(dict(queries=[dict(query_uuid='../escape')])))
            with self.assertRaises(ValueError):refresh_completed_responses(root)

    def test_visibility_refresh_only_stats_existing_regular_response_files(self):
        from .e1.nfs_visibility import refresh
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'one.json').write_text('{}');(root/'ignore.npz').write_text('not touched');(root/'alias.json').symlink_to(root/'one.json')
            before={p.name:p.read_bytes() for p in root.iterdir() if not p.is_symlink()}
            self.assertEqual(refresh(root),1)
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir() if not p.is_symlink()})

    def test_full_behavior_matrix_does_not_promote_two_T1_audits_to_complete_comparison(self):
        from .e1.behavior_matrix import build
        source=json.loads(Path('reports/PI0294/20261016T150228Z-localpc-e1/M1_1024_COMPLETE_TRAIN4_AND_T1_AUDITS.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'evidence').mkdir();(root/'episodes').mkdir()
            audits=source['independent_T1_actual_evidence_audits']
            (root/'evidence/MAIN_EVIDENCE_AUDIT_INDEX.private.json').write_text(json.dumps(dict(rows=audits)))
            for audit in audits:
                row=audit['adjudication'];folder=root/'episodes'/row['attempt_id'];folder.mkdir()
                (folder/'E1_ADJUDICATION.json').write_text(json.dumps(row))
                (folder/'CARD.json').write_text(json.dumps(dict(id=row['attempt_id'],task=row['task'],scene_group=row['scene_group'],split='TRAIN')))
            value=build(root)
            self.assertEqual(value['status'],'PARTIAL_WITH_EXPLICIT_EVIDENCE_GAPS')
            self.assertEqual(value['fixed_terminal_models']['M1-1024-CANONICAL']['T1']['task_successes'],2)
            self.assertEqual(value['fixed_terminal_models']['M2-CORR']['T1']['valid_policy_outcomes'],0)
            self.assertFalse(value['probe_S1_available']);self.assertEqual(value['selected'],'NONE')
            self.assertEqual(len(value['evidence_sha256']),5)
            self.assertEqual(len(value['predeclared_condition_matrix']),12)
            self.assertEqual(sum(r['status']=='NOT_RUN' for r in value['predeclared_condition_matrix']),10)
            row=dict(audits[0]['adjudication'],attempt_id='zero',model='M2-BC-REMAINDER-03',task='T2',duration_s=0.,main_integrations=0,
                status='ENGINEERING_INCOMPLETE',pipeline_trustworthy=False,failure_stage='ENGINEERING_OR_UNKNOWN')
            folder=root/'episodes/zero';folder.mkdir()
            (folder/'E1_ADJUDICATION.json').write_text(json.dumps(row))
            (folder/'CARD.json').write_text(json.dumps(dict(id='zero',task='T2',scene_group=row['scene_group'],split='TRAIN')))
            zero=dict(attempt_id='zero',adjudication=row,status='PASS_ZERO_INTEGRATION_ENGINEERING_EVIDENCE',errors=[],checks={'fixture':True},
                physical_integrations=0,score=None,video={'initial_frame_only':True},scientific_policy_outcome_valid=False)
            (root/'evidence/MAIN_EVIDENCE_AUDIT_INDEX.private.json').write_text(json.dumps(dict(rows=audits+[zero])))
            value=build(root)
            self.assertEqual(value['audit_gaps'],[])
            cell=next(r for r in value['predeclared_condition_matrix'] if 'zero' in r['actual_attempt_ids'])
            self.assertEqual(cell['status'],'ENGINEERING_INCOMPLETE');self.assertFalse(cell['valid_policy_outcome'])
            self.assertEqual(cell['evidence_accounted_attempts'],1)

    def test_file_RPC_write_timeout_covers_full_pipe_without_resending(self):
        import os,time
        from .e1.stdio import write_frame
        reader,writer=os.pipe()
        try:
            os.set_blocking(writer,False);count=0
            while True:
                try:count+=os.write(writer,b'x'*4096)
                except BlockingIOError:break
            with self.assertRaises(TimeoutError):write_frame(writer,b'unique-request\n',time.monotonic()+.02)
            self.assertFalse(os.get_blocking(writer));os.set_blocking(reader,False);actual=b''
            while True:
                try:actual+=os.read(reader,65536)
                except BlockingIOError:break
            self.assertEqual(actual,b'x'*count)
        finally:os.close(reader);os.close(writer)

    def test_file_RPC_partial_writes_deliver_one_exact_frame(self):
        import os,time,threading
        from .e1.stdio import write_frame
        reader,writer=os.pipe();chunks=[];payload=b'one-request:'+b'x'*131072+b'\n'
        def receive():
            while True:
                part=os.read(reader,997)
                if not part:break
                chunks.append(part)
        worker=threading.Thread(target=receive);worker.start()
        try:
            write_frame(writer,payload,time.monotonic()+5);self.assertTrue(os.get_blocking(writer))
        finally:os.close(writer);worker.join(timeout=5);os.close(reader)
        self.assertFalse(worker.is_alive());self.assertEqual(b''.join(chunks),payload)

    def test_probe_summary_never_hides_nonfinite_or_short_window_in_episode_mean(self):
        from .e1.analysis import errors
        def row(identifier,values):return dict(episode_id=identifier,errors=dict(pointwise={k:list(values) for k in ['XYZ_euclidean_m','SO3_rad','gripper_absolute']}))
        good=row('one',[1.]*40)
        self.assertEqual(errors([good])['H40']['XYZ_euclidean_m']['episode_macro_mean'],1.)
        for bad in [[2.]*39,[float('nan')]*40,[float('inf')]*40,[None]*40,[True]*40]:
            output=errors([good,row('two',bad)])['H40']['XYZ_euclidean_m']
            self.assertEqual(output['episodes'],2);self.assertEqual(output['finite_complete_episodes'],1)
            self.assertIsNone(output['episode_macro_mean']);self.assertIsNone(output['worst_episode_mean'])
        self.assertEqual(errors([row('one',[1.]*8)])['prefix8']['XYZ_euclidean_m']['episode_macro_mean'],1.)
        self.assertIsNone(errors([row('one',[1.]*8)])['H40']['XYZ_euclidean_m']['episode_macro_mean'])

    def test_actual_expert_ack_uses_saved_tick_fields_and_full_clock(self):
        import numpy as np
        from .e1.collection_export import check_ack_progress
        rows=np.zeros((200,45));rows[:,0]=np.arange(5001,5201)*.002
        ack=dict(tick_before=5000,tick_after=5200)
        arrays=dict(rows=rows,tick_before=np.array(5000),tick_after=np.array(5200))
        check_ack_progress(arrays,ack)
        for bad in [dict(arrays,tick_after=np.array(5199)),dict(arrays,rows=rows[:-1])]:
            with self.assertRaises(ValueError):check_ack_progress(bad,ack)
        rows[100,0]+=.002
        with self.assertRaises(ValueError):check_ack_progress(arrays,ack)

    def test_checkpoint_archive_labels_do_not_promote_partial_states(self):
        from .e1.checkpoints import checkpoint_spec
        self.assertEqual(checkpoint_spec('M1-256'),('M1',256,False))
        self.assertEqual(checkpoint_spec('M2-CORR-PARTIAL-127'),('M2-CORR',127,True))
        for label in ['M2-CORR-PARTIAL-0','M2-BC-PARTIAL-512','M1-PARTIAL-12','../M2-BC','M2-BC-PARTIAL-01']:
            with self.assertRaises(ValueError):checkpoint_spec(label)

    def test_fixed_probe_completed_draw_is_readonly_and_unknown_not_resampled(self):
        import numpy as np
        from types import SimpleNamespace as NS
        from .e1.model import fixed_draw
        counts=[];charges=[]
        def generate(request,seed):
            counts.append(seed);return dict(physical_H40=np.zeros((40,10)),normalized=np.zeros((1,40,132)),noise=dict(sha256='fixture'),inference_precision=dict(cuda_matmul_allow_tf32=False))
        instance=NS(identity=dict(files={'fixture':'digest'}),generate=generate);ledger=NS(charge=lambda key,n:charges.append(n))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);target=root/'fixed.npz';request=dict(snapshot_uuid='one',state_eef10=[0.]*10)
            first=fixed_draw(root,'fixture','one',instance,request,7,target,ledger)
            second=fixed_draw(root,'fixture','one',instance,dict(request,snapshot_uuid='new-transport-uuid'),7,target,ledger)
            np.testing.assert_array_equal(first[0],second[0]);self.assertEqual(counts,[7]);self.assertEqual(charges,[1])
            with self.assertRaises(ValueError):fixed_draw(root,'fixture','one',instance,request,8,target,ledger)
            def lost(request,seed):raise ConnectionError('unknown draw')
            instance.generate=lost
            with self.assertRaises(ConnectionError):fixed_draw(root,'fixture','two',instance,request,9,root/'unknown.npz',ledger)
            with self.assertRaises(MutationOutcomeUnknown):fixed_draw(root,'fixture','two',instance,request,9,root/'unknown.npz',ledger)
            self.assertEqual(charges,[1,1])

    def test_remainder_never_repeats_valid_failure_or_success(self):
        from .e1.remaining import panel
        base=dict(model='M2-BC',scene_group='PI0294/TRAIN/F0/000',task='T1',status='VALID_TASK_FAIL',pipeline_trustworthy=True,
            failure_stage='OBJECT_TASK_RESULT',live_native_failures=0,attempt_id='actual-negative')
        rows=[base,dict(base,task='T2',status='ENGINEERING_INCOMPLETE',pipeline_trustworthy=False,attempt_id='interrupted')]
        value=panel(rows,'M2-BC');self.assertEqual(len(value['complete']),1);self.assertEqual(len(value['remaining']),3)
        self.assertEqual(value['new_engineering_recoveries'],1)
        rows.append(dict(base,model='M2-BC-REMAINDER-03',status='SUCCESS',attempt_id='forbidden-repeat'))
        with self.assertRaises(PermissionError):panel(rows,'M2-BC')
        self.assertEqual(panel(rows,'M2-BC',source_stopped=True)['remaining'],[])
        with self.assertRaises(PermissionError):panel([],'M1-256')

    def test_partial_resume_requires_full_hashes_exact_cursor_and_no_unknown_update(self):
        import hashlib
        from unittest.mock import patch
        from .e1.train import checked_resume
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);branch=root/'training/M2-BC';checkpoint=branch/'checkpoint-12';checkpoint.mkdir(parents=True);(branch/'session-start').mkdir()
            recipe=dict(seed=20261016,max_steps=512)
            (branch/'session-start/RECIPE.json').write_text(json.dumps(recipe))
            for name in ['optimizer.pt','scheduler.pt','rng_state.pth','statistics.json']:(checkpoint/name).write_bytes(b'own-fixture-not-loaded')
            (checkpoint/'trainer_state.json').write_text(json.dumps(dict(global_step=12)))
            (checkpoint/'SAMPLER_STATE.json').write_text(json.dumps(dict(recipe=recipe,sampler=dict(cursor=384))))
            files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir()}
            (checkpoint/'CHECKPOINT_COMPLETE.json').write_text(json.dumps(dict(global_step=12,full_resume=True,files=files)))
            with patch('scripts.pi0294.e1.train.STATS',files['statistics.json']):
                ledger=dict(optimizer_updates=1036,pending_updates=0)
                self.assertEqual(checked_resume(root,'M2-BC',checkpoint,recipe,ledger),12)
                with self.assertRaises(PermissionError):checked_resume(root,'M2-BC',checkpoint,recipe,dict(ledger,pending_updates=1))
                with self.assertRaises(PermissionError):checked_resume(root,'M2-BC',checkpoint,recipe,dict(ledger,optimizer_updates=1035))
                (checkpoint/'optimizer.pt').write_bytes(b'changed')
                with self.assertRaises(ValueError):checked_resume(root,'M2-BC',checkpoint,recipe,ledger)

    def test_behavior_summary_excludes_collection_and_scope_deviation(self):
        from .e1.analysis import autonomous_summary
        base=dict(model='fixture',task='T1',status='SUCCESS',audit_verified=True,pipeline_trustworthy=True,persistent_success=True,native_safety_flag_rows=0,native_duration_s=32.,failure_stage='NONE',model_autonomous_evaluation=True,excluded_scope_deviation=False)
        rows=[dict(base,scene_group='PI0294/TRAIN/F0/000'),dict(base,scene_group='PI0294/TRAIN/F1/000')]
        self.assertTrue(autonomous_summary(rows)['fixture']['T1']['source_task_competence_observed'])
        rows[-1]['model_autonomous_evaluation']=False
        self.assertFalse(autonomous_summary(rows)['fixture']['T1']['source_task_competence_observed'])
        rows[-1]['model_autonomous_evaluation']=True;rows[-1]['excluded_scope_deviation']=True
        self.assertEqual(autonomous_summary(rows)['fixture']['T1']['conditions_attempted'],1)

    def test_final_summary_counts_conditions_not_replacement_attempts(self):
        from .e1.analysis import autonomous_summary,canonical_model
        base=dict(model='M2-BC',task='T1',status='SUCCESS',audit_verified=True,pipeline_trustworthy=True,persistent_success=True,
            native_safety_flag_rows=0,native_duration_s=32.,failure_stage='COMPLETE_SUCCESS',model_autonomous_evaluation=True,excluded_scope_deviation=False)
        broken=dict(base,scene_group='PI0294/TRAIN/F0/000',status='ENGINEERING_INCOMPLETE',pipeline_trustworthy=False,persistent_success=False,failure_stage='ENGINEERING_OR_UNKNOWN')
        fixed=dict(base,scene_group='PI0294/TRAIN/F0/000',model='M2-BC-REMAINDER-03')
        second=dict(base,scene_group='PI0294/TRAIN/F1/000')
        result=autonomous_summary([broken,fixed,second])['M2-BC']['T1']
        self.assertEqual((result['conditions_attempted'],result['actual_attempts'],result['task_successes']),(2,3,2))
        self.assertTrue(result['complete_predeclared_panel']);self.assertEqual(result['unverified_or_untrustworthy_attempts'],1)
        for candidates in [[dict(broken,status='VALID_TASK_FAIL',pipeline_trustworthy=True),fixed],[fixed,dict(fixed)]]:
            with self.assertRaises(PermissionError):autonomous_summary(candidates)
        result=autonomous_summary([broken,dict(broken,scene_group='PI0294/TRAIN/F1/000')])['M2-BC']['T1']
        self.assertTrue(result['all_conditions_attempted']);self.assertFalse(result['complete_predeclared_panel'])
        self.assertEqual(canonical_model('M1-256-CANONICAL'),'M1-256-CANONICAL')
        self.assertEqual(canonical_model('OTHER-REMAINDER-03'),'OTHER-REMAINDER-03')

    def test_missing_audit_cannot_support_success_or_grant_engineering_retry(self):
        from .e1.analysis import audited_outcome,autonomous_summary
        row=dict(model='M2-BC',task='T1',attempt_id='fixture',status='SUCCESS',pipeline_trustworthy=True,duration_s=32.,
            scene_group='PI0294/TRAIN/F0/000',failure_stage='COMPLETE_SUCCESS')
        value=audited_outcome(row,None,autonomous=True)
        self.assertFalse(value['pipeline_trustworthy']);self.assertEqual(value['status'],'SUCCESS')
        self.assertTrue(value['adjudication_pipeline_trustworthy']);self.assertIsNone(value['persistent_success'])
        self.assertEqual(autonomous_summary([value])['M2-BC']['T1']['task_successes'],0)
        with self.assertRaises(ValueError):audited_outcome(row,dict(adjudication=dict(row,attempt_id='wrong')),autonomous=True)

    def test_probe_s1_accept_is_not_mistaken_for_native_PASS(self):
        from contextlib import contextmanager
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        import numpy as np
        from .e1.probe_s1 import check
        from go2arx5_vla.task_data.codec import pose10
        class Ledger:
            def guard(self):pass
            @contextmanager
            def steps(self,*args):yield dict(actual=0)
        state=pose10(np.eye(4),0.);action=np.repeat(state[None],40,axis=0)
        engine=NS(pose=lambda name:np.eye(4),data=NS(qpos=np.zeros(2),time=0.),finger_q=[0,1],steps=0)
        row=dict(id='fixture',task='T1',payload=dict(state_eef10=state.tolist()))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'evidence/PROBE_S1_ROWS').mkdir(parents=True)
            with patch('scripts.pi0294.e1.probe_s1.gate_action',return_value=dict(decision='ACCEPT')),patch('scripts.pi0294.e1.probe_s1.predict',return_value=(dict(decision='PASS'),[])) as native:
                value=check(root,root,'fixture','on_student',row,engine,Ledger(),action,'approach')
                self.assertTrue(value['prefix_executable']);self.assertEqual(native.call_count,1)
            with patch('scripts.pi0294.e1.probe_s1.gate_action',return_value=dict(decision='REJECT')),patch('scripts.pi0294.e1.probe_s1.predict') as native:
                value=check(root,root,'fixture2','on_student',row,engine,Ledger(),action,'approach')
                self.assertFalse(value['prefix_executable']);native.assert_not_called()

    def test_only_engineering_incomplete_256_condition_can_recover(self):
        from .e1.driver import recoverable_256_conditions
        rows=[dict(scene_group='PI0294/TRAIN/'+f+'/000',task=t,status='VALID_TASK_FAIL',pipeline_trustworthy=True,failure_stage='S1_PRE_EXECUTION_REJECT',live_native_failures=0) for f in ['F0','F1'] for t in ['T1','T2']]
        with self.assertRaises(PermissionError):recoverable_256_conditions(rows)
        rows[-1].update(status='ENGINEERING_INCOMPLETE',pipeline_trustworthy=False)
        self.assertEqual(recoverable_256_conditions(rows),{('PI0294/TRAIN/F1/000','T2')})
        rows[-1]['live_native_failures']=1
        with self.assertRaises(PermissionError):recoverable_256_conditions(rows)

    def test_control_file_read_retries_visibility_only_and_is_bounded(self):
        import errno
        from unittest.mock import patch
        from .e1.control_read import read
        with patch('scripts.pi0294.e1.control_read.time.sleep'),patch('scripts.pi0294.e1.control_read.original_read',side_effect=[FileNotFoundError(errno.ENOENT,'NFS'),{'ok':True}]) as source:
            self.assertEqual(read('fixture'),{'ok':True});self.assertEqual(source.call_count,2)
        with patch('scripts.pi0294.e1.control_read.time.sleep'),patch('scripts.pi0294.e1.control_read.original_read',side_effect=FileNotFoundError(errno.ENOENT,'NFS')) as source:
            with self.assertRaises(FileNotFoundError):read('fixture')
            self.assertEqual(source.call_count,6)
        with patch('scripts.pi0294.e1.control_read.original_read',side_effect=ValueError('malformed')) as source:
            with self.assertRaises(ValueError):read('fixture')
            self.assertEqual(source.call_count,1)

    def test_original_inference_precision_restored_even_after_failure(self):
        from types import SimpleNamespace as NS
        from .e1.model import original_inference_precision
        for initial in [True,False]:
            torch=NS(backends=NS(cuda=NS(matmul=NS(allow_tf32=initial))))
            with self.assertRaises(RuntimeError):
                with original_inference_precision(torch) as proof:
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    self.assertEqual(proof['training_flag_on_entry'],initial)
                    raise RuntimeError('fixture')
            self.assertEqual(torch.backends.cuda.matmul.allow_tf32,initial)

    def test_correction_window_dedup_binds_state_actual_H40_and_RGB(self):
        import numpy as np
        from .e1.correction import window_key
        state=np.zeros(10);actions=np.zeros((40,10));rgb=['a'*64,'b'*64]
        key=window_key(state,actions,rgb)
        self.assertEqual(key,window_key(state.copy(),actions.copy(),list(rgb)))
        actions[-1,-1]=1
        self.assertNotEqual(key,window_key(state,actions,rgb))

    def test_mixed_sampler_exact_half_and_resume_no_unqualified_task(self):
        from .e1.mixture import MixedSampler
        class Dataset:pass
        d=Dataset();d.original_manifest_sha='original';d.manifest_sha='combined';d.rows=[]
        for row in select_small8(self.manifest()):d.rows.extend((row,a) for a in range(5))
        d.small_rows=len(d.rows)
        for i in range(2):
            row=dict(id='correction-'+str(i),task='T1',split='TRAIN',origin='E0_STATE_AGGREGATION_ONLY')
            d.rows.extend((row,a) for a in range(7))
        s=MixedSampler(d,stop=128);pairs=list(s)
        for start in range(0,128,32):
            ids=[i for c,i in pairs[start:start+32]]
            original=[d.rows[i][0]['task'] for i in ids if i<d.small_rows]
            corrected=[d.rows[i][0]['task'] for i in ids if i>=d.small_rows]
            self.assertEqual(original.count('T1'),8);self.assertEqual(original.count('T2'),8)
            self.assertEqual(corrected,['T1']*16)
        self.assertEqual(list(MixedSampler(d,start=19,stop=128)),pairs[19:])
        self.assertEqual(s.state_dict(32)['cursor'],32)

    def test_intervention_switch_is_one_way_and_never_executes_rejected_student(self):
        import numpy as np
        from .e1.intervention import InterventionTeacher
        from scripts.pi0292.relay_protocol import sealed
        from go2arx5_vla.task_data.codec import pose10
        class Value:pass
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'evidence').mkdir();folder=root/'episode';(folder/'relay/observations').mkdir(parents=True)
            (root/'evidence/ORACLE_RECOVERY_RESULTS.json').write_text(json.dumps(dict(tasks=dict(T1=dict(status='RECOVERY_QUALIFIED_FOR_E1'),T2=dict(status='NOT_QUALIFIED')))))
            student=Value();student.teacher=Value();student.pending={'command':{'tick':0}};student.acknowledged=0
            student.finish=lambda status,error:None
            engine=Value();engine.steps=0;state=dict(tcp=np.eye(4),gripper_fraction=1.)
            packet=sealed(dict(payload=dict(state_eef10=pose10(state['tcp'],1.).tolist())))
            (folder/'relay/observations/000.json').write_text(json.dumps(packet))
            with self.assertRaises(PermissionError):InterventionTeacher(student,engine,root,folder,dict(task='T2'))
            t=InterventionTeacher(student,engine,root,folder,dict(task='T1'))
            self.assertTrue(t.intervene(state,dict(decision='REJECT')))
            self.assertIsNone(student.pending);self.assertEqual(engine.steps,0)
            self.assertFalse(t.intervene(state,dict(decision='REJECT')))
            t.finish('PARTIAL',None)
            receipt=json.loads((folder/'INTERVENTION_RESULT.json').read_text())
            self.assertEqual(receipt['switches'],1);self.assertEqual(receipt['data_class'],'EXPERT_ASSISTED_DATA')
            self.assertEqual(receipt['expert_executed_commands'],0)

    def test_fixed_cosine_warmup_and_floor(self):
        self.assertEqual(learning_rate_multiplier(0),0.)
        self.assertEqual(learning_rate_multiplier(32),1.)
        self.assertAlmostEqual(learning_rate_multiplier(1024),.1)
        values=[learning_rate_multiplier(s) for s in range(32,1025)]
        self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))

    def test_atomic_numeric_payload_complete_before_visibility(self):
        import numpy as np
        from .e1.artifacts import atomic_npz
        with tempfile.TemporaryDirectory() as d:
            p=Path(d,'payload.npz');atomic_npz(p,x=np.arange(12).reshape(4,3))
            with np.load(p,allow_pickle=False) as z:np.testing.assert_array_equal(z['x'],np.arange(12).reshape(4,3))
            self.assertFalse(list(Path(d).glob('*.pending.*')))

    def test_completed_same_id_read_does_not_repeat_operation(self):
        with tempfile.TemporaryDirectory() as d:
            j=Journal(d,'fixture');calls=[]
            self.assertEqual(j.once('q',{'x':1},lambda:calls.append(1) or {'ok':True}),{'ok':True})
            self.assertEqual(j.once('q',{'x':1},lambda:calls.append(2)),{'ok':True})
            self.assertEqual(calls,[1])
            with self.assertRaises(ValueError):j.completed('q',{'x':2})

    def test_real_client_process_exit_preserves_unknown_intent(self):
        with tempfile.TemporaryDirectory() as d:
            p=multiprocessing.get_context('spawn').Process(target=die_after_intent,args=(d,));p.start();p.join(10)
            self.assertFalse(p.is_alive());self.assertEqual(p.exitcode,17)
            with self.assertRaises(MutationOutcomeUnknown):Journal(d,'fixture').once('query',{'sequence':0},lambda:self.fail('resent'))

    def test_disconnect_after_remote_effect_does_not_resend(self):
        with tempfile.TemporaryDirectory() as d:
            effects=[];j=Journal(d,'fixture')
            def disconnected():effects.append('remote-effect');raise ConnectionError('response lost')
            with self.assertRaises(ConnectionError):j.once('install',{'prefix':0},disconnected)
            with self.assertRaises(MutationOutcomeUnknown):j.once('install',{'prefix':0},disconnected)
            self.assertEqual(effects,['remote-effect'])

    def test_partial_temporary_response_not_visible(self):
        with tempfile.TemporaryDirectory() as d:
            j=Journal(d,'fixture');Path(d,'q.json.pending.123').write_text('{"status":')
            self.assertIsNone(j.completed('q',{'x':1}))
            j.once('q',{'x':1},lambda:{'ok':True})
            self.assertEqual(j.completed('q',{'x':1}),{'ok':True})
            p=Path(d,'q.json');v=json.loads(p.read_text());v['result']={'ok':False};p.write_text(json.dumps(v))
            with self.assertRaises(ValueError):j.completed('q',{'x':1})

    def test_proc_identity_with_spaces_and_zombie(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d,'42');p.mkdir();f=p/'stat'
            fields=['S']+['0']*18+['123']+['0']*5
            f.write_text('42 (a name with spaces) '+' '.join(fields))
            self.assertTrue(process_is_same(dict(pid=42,proc_start='123'),d))
            self.assertFalse(process_is_same(dict(pid=42,proc_start='124'),d))
            fields[0]='Z';f.write_text('42 (x) '+' '.join(fields))
            self.assertFalse(process_is_same(dict(pid=42,proc_start='123'),d))

    def manifest(self):
        rows=[]
        for split in ['TRAIN','DEV']:
            for f in ['F0','F1']:
                for n in ['000','001','002']:
                    for t in ['T1','T2']:
                        rows.append(dict(id=f'PI0294-{split}-{f}-{n}-{t}',split=split,origin='new',task=t,family=f,windows=601))
        return dict(episodes=rows)

    def test_small8_original_deterministic_complete_pairs_no_dev(self):
        m=self.manifest();m['episodes']=[r for r in m['episodes'] if r['id']!='PI0294-TRAIN-F0-001-T1']
        rows=select_small8(m)
        self.assertEqual(len(rows),8)
        self.assertTrue(all(r['split']=='TRAIN' for r in rows))
        self.assertIn('PI0294-TRAIN-F0-002-T1',[r['id'] for r in rows])
        self.assertNotIn('PI0294-TRAIN-F0-001-T2',[r['id'] for r in rows])
        m['episodes']=[r for r in m['episodes'] if r['id']!='PI0294-TRAIN-F1-000-T1']
        with self.assertRaises(ValueError):select_small8(m)

    def test_small_sampler_task_uniform_without_stage_weighting(self):
        class Dataset:pass
        d=Dataset();d.manifest_sha='fixture';d.rows=[]
        for row in select_small8(self.manifest()):
            d.rows.extend((row,a) for a in range(7))
        s=SmallSampler(d,stop=320)
        pairs=list(s);self.assertEqual(len(pairs),320)
        self.assertEqual([d.rows[i][0]['task'] for c,i in pairs],['T1','T2']*160)
        self.assertEqual(list(SmallSampler(d,start=17,stop=320)),pairs[17:])
        self.assertEqual(s.state_dict(32)['cursor'],32)
        with self.assertRaises(ValueError):s.state_dict(321)


if __name__=='__main__':unittest.main()
