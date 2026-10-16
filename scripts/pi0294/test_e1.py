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
    def test_behavior_summary_excludes_collection_and_scope_deviation(self):
        from .e1.analysis import autonomous_summary
        base=dict(model='fixture',task='T1',pipeline_trustworthy=True,persistent_success=True,native_safety_flag_rows=0,native_duration_s=32.,failure_stage='NONE',model_autonomous_evaluation=True,excluded_scope_deviation=False)
        rows=[dict(base,scene_group='PI0294/TRAIN/F0/000'),dict(base,scene_group='PI0294/TRAIN/F1/000')]
        self.assertTrue(autonomous_summary(rows)['fixture']['T1']['source_task_competence_observed'])
        rows[-1]['model_autonomous_evaluation']=False
        self.assertFalse(autonomous_summary(rows)['fixture']['T1']['source_task_competence_observed'])
        rows[-1]['model_autonomous_evaluation']=True;rows[-1]['excluded_scope_deviation']=True
        self.assertEqual(autonomous_summary(rows)['fixture']['T1']['conditions_attempted'],1)

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
