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
