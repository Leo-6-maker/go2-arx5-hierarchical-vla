"""Ordering and sealed-scope tests only; no CUDA, models, physics or PBS."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import long_training as long
from . import future_budget as budget
from .source_admission import VERSION
from .test_future_budget import fixtures, amendment, raw, NOW


class LongTrainingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        policy = Path(__file__).resolve().parents[3]/'reports/PI0294/design/VAP1_LONG_ALLOCATION_SCHEDULING_20261021.json'
        p = self.root/long.POLICY; p.parent.mkdir(); p.write_bytes(policy.read_bytes()); p.chmod(0o400)
        self.manifest = dict(execution=dict(sealed_inputs={long.POLICY:long.POLICY_SHA}))
        self.request = dict(scientific_admission_version=VERSION, stage='P2_PRIMARY',
            sealed_inputs={long.POLICY:long.POLICY_SHA}, training_block=dict(
                arm='A_CONTROL', replica=False, stop_after_step=4096,
                resume_checkpoint_relative=None, same_allocation_1024_probes=True))

    def put(self, name, value):
        p = self.root/name; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value)); return p

    def probe_fixture(self):
        checkpoint = self.put('training/A_CONTROL/checkpoint-1024/CHECKPOINT_COMPLETE.json',
            dict(global_step=1024, full_resume=True))
        plan = self.put('plans/PV1_FIXED_TRAIN_PROBES.json', dict(windows=[dict(id=str(i)) for i in range(12)]))
        return self.put('evidence/A_CONTROL-1024/TRAIN_PROBES/RESULT.json', dict(
            status='ACTUAL_FIXED_TRAIN_PROBES_NOT_TASK_SUCCESS', complete=True,
            records=[dict(id=str(i)) for i in range(12)], new_physics=0, physical_success_gate=False,
            checkpoint_manifest_sha256=long.sha(checkpoint), probe_plan_sha256=long.sha(plan)))

    def test_new_long_scope_is_explicit_and_legacy_is_unchanged(self):
        self.assertTrue(long.admitted(self.root, self.manifest, self.request))
        legacy=deepcopy(self.request); del legacy['training_block']['same_allocation_1024_probes']
        self.assertFalse(long.admitted(self.root, self.manifest, legacy))
        for changes in [dict(same_allocation_1024_probes=False), dict(replica=True),
                        dict(arm='A_4MM'), dict(stop_after_step=1024)]:
            bad=deepcopy(self.request); bad['training_block'].update(changes)
            with self.subTest(changes=changes), self.assertRaises(PermissionError):
                long.admitted(self.root, self.manifest, bad)
        bad=deepcopy(self.request); bad['stage']='P4_REPLICA'
        with self.assertRaises(PermissionError):long.admitted(self.root,self.manifest,bad)
        (self.root/long.POLICY).chmod(0o600)
        with self.assertRaises(PermissionError):long.admitted(self.root,self.manifest,self.request)

    def test_phase_identity_and_real_fixed_probes_before_update_1025(self):
        long.phase(self.root,self.manifest,self.request,name='BEFORE_PROBES',stop=1024,resume_relative=None)
        with self.assertRaises(FileNotFoundError):
            long.phase(self.root,self.manifest,self.request,name='AFTER_PROBES',stop=4096,
                resume_relative='training/A_CONTROL/checkpoint-1024')
        result=self.probe_fixture()
        long.phase(self.root,self.manifest,self.request,name='AFTER_PROBES',stop=4096,
            resume_relative='training/A_CONTROL/checkpoint-1024')
        for change in [dict(complete=False), dict(new_physics=1), dict(physical_success_gate=True),
                       dict(records=[dict(id='duplicate')]*12), dict(checkpoint_manifest_sha256='f'*64)]:
            original=result.read_bytes();v=json.loads(original);v.update(change);result.write_text(json.dumps(v))
            with self.subTest(change=change),self.assertRaises(PermissionError):long.probe_receipt(self.root,'A_CONTROL')
            result.write_bytes(original)
        with self.assertRaises(PermissionError):
            long.phase(self.root,self.manifest,self.request,name='AFTER_PROBES',stop=4096,
                resume_relative='training/B_COVERAGE/checkpoint-1024')

    def test_single_allocation_training_probe_full_state_resume_order(self):
        events=[]
        def train(**kwargs):
            events.append(('train',kwargs));return dict(global_step=kwargs['stop'])
        result,records=long.execute(start_step=0,train=train,probe=lambda:events.append('probe') or dict(complete=True),
            unload=lambda:events.append('unload'),load=lambda p,trainable:events.append(('load',p,trainable)),
            guard=lambda:events.append('guard'),checkpoint='OWN_CHECKPOINT_1024')
        self.assertEqual(result['global_step'],4096);self.assertEqual(len(records),3)
        self.assertEqual(events,[('train',dict(stop=1024,phase='BEFORE_PROBES',checkpoint=None)),
            'unload','guard',('load','OWN_CHECKPOINT_1024',False),'probe','unload','guard',
            ('load','OWN_CHECKPOINT_1024',True),
            ('train',dict(stop=4096,phase='AFTER_PROBES',checkpoint='OWN_CHECKPOINT_1024'))])

    def test_partial_or_probe_failure_never_repeats_or_continues_training(self):
        for partial in (True,False):
            events=[]
            def train(**kwargs):
                events.append(kwargs['stop']);return dict(global_step=300 if partial else 1024)
            def probe():raise PermissionError('probe unknown effect preserved')
            args=dict(start_step=0,train=train,probe=probe,unload=lambda:None,
                load=lambda *a,**k:None,guard=lambda:None,checkpoint='OWN')
            if partial:self.assertEqual(long.execute(**args)[0]['global_step'],300)
            else:
                with self.assertRaises(PermissionError):long.execute(**args)
            self.assertEqual(events,[1024])

    def test_long_allocation_forecast_counts_two_submissions_not_four(self):
        s,p,q=fixtures();q['long_allocation_policy_sha256']=long.POLICY_SHA
        jobs=[]
        for arm in p['candidates']:
            row=deepcopy(next(j for j in q['training_jobs'] if j['arm']==arm))
            row.update(stop_step=4096,same_allocation_1024_probes=True,walltime_upper_s=25200)
            row['resources'].update(H200_s=25260,optimizer_updates=4096,Hopper_CPU_s=100000,engine_starts=3)
            jobs.append(row)
        q['training_jobs']=jobs;data=raw(amendment())
        with patch.object(budget,'VAP1_ADOPTION',hashlib.sha256(data).hexdigest()):
            result=budget.forecast(snapshot=s,progress_bytes=raw(p),plan=q,now_epoch=NOW,amendment_bytes=data)
            self.assertTrue(result['admissible_forecast'])
            self.assertEqual(result['future_required_additional']['PBS_submissions'],5)
            self.assertEqual(result['future_required_additional']['optimizer_updates'],8192)
            self.assertEqual(result['future_required_additional']['neural_queries'],24+7040)
            q['training_jobs'][0]['resources']['neural_queries']=0
            with self.assertRaises(PermissionError):
                budget.forecast(snapshot=s,progress_bytes=raw(p),plan=q,now_epoch=NOW,amendment_bytes=data)


if __name__=='__main__':unittest.main()
