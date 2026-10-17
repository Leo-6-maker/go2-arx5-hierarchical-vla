from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .mp1 import fixed_probes as probes
from .mp1.probe_compare import PRECISION
from .native_dataset import sha
from .test_mp1_behavior_matrix import plan,receipt

model=probes.model
MutationOutcomeUnknown=probes.MutationOutcomeUnknown


class FixedProbeResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'source-pair-01';(self.root/'plans').mkdir(parents=True)
        self.label='S-CORR-2048-TRAIN4'
        self.state=np.asarray([0.,0.,.5,1.,0.,0.,0.,1.,0.,.5])
        self.target=np.tile(self.state,(40,1))
        windows=[dict(id='expert-'+str(i),episode_id='episode',task='T1' if i<48 else 'T2',anchor=i,split='TRAIN',noise_seed=i+1,
            state_sha256=hashlib.sha256(self.state.tobytes()).hexdigest(),absolute_H40_sha256=hashlib.sha256(self.target.tobytes()).hexdigest()) for i in range(96)]
        self.expert=dict(TRAIN_only=True,count=96,windows=windows)
        images=[np.zeros((4,4,3),dtype=np.uint8),np.ones((4,4,3),dtype=np.uint8)]
        self.students=dict(states=[dict(id='student-TRAIN-'+str(i),payload=model.payload(self.state,images,'fixed'),noise_seed=1000+i,tick=200*i) for i in range(12)])
        self.put(self.root/'plans/PROBE_MANIFEST.json',self.expert)
        self.put(self.root/'plans/STUDENT_PROBES.private.json',self.students)
        self.dataset=SimpleNamespace(rows=[(dict(id='episode'),i) for i in range(96)],raw=lambda i:(self.state,self.target,images,'fixed'))
        self.calls=[]
        def generate(payload,seed):
            self.calls.append(seed)
            return dict(physical_H40=self.target.copy(),normalized=np.zeros((40,10)),noise=dict(seed=seed),inference_precision=PRECISION)
        self.instance=SimpleNamespace(identity=dict(files={'config.json':'a'*64}),processor=None,generate=generate)
        self.costs=[];self.ledger=SimpleNamespace(charge=lambda key,count:self.costs.append((key,count)))
        self.folder=self.root/'evidence'/self.label/'probes'

    def put(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')

    def run_complete(self):
        with patch.object(model,'expert_decode',side_effect=lambda processor,state,target:(target,{})):
            return probes.complete(self.root,self.label,self.instance,self.dataset,self.ledger,lambda:None)

    def test_known_complete_probe_set_reuses_every_draw_without_model_call(self):
        self.assertEqual(probes.status(self.root,self.label,self.instance.identity['files'])['new_queries_required'],108)
        result=self.run_complete();self.assertEqual(result['new_queries'],108)
        before={p.name:sha(p) for p in self.folder.iterdir()}
        self.assertTrue(probes.status(self.root,self.label,self.instance.identity['files'])['complete'])
        self.instance.generate=lambda *a: (_ for _ in ()).throw(AssertionError('NO_REPEATED_GENERATION'))
        self.assertEqual(self.run_complete()['new_queries'],0)
        self.assertEqual(before,{p.name:sha(p) for p in self.folder.iterdir()})
        self.assertEqual(len(self.calls),108);self.assertEqual(sum(n for _,n in self.costs),108)

    def test_complete_draws_with_missing_aggregate_are_rebuilt_not_resampled(self):
        self.run_complete();(self.folder/'ON_STUDENT_RESULTS.private.json').unlink()
        state=probes.status(self.root,self.label,self.instance.identity['files'])
        self.assertFalse(state['complete']);self.assertEqual(state['new_queries_required'],0)
        self.assertEqual(self.run_complete()['new_queries'],0)
        self.assertEqual(len(self.calls),108)

    def test_partial_known_unstarted_tail_only_consumes_missing_draws(self):
        self.run_complete()
        for row in self.students['states'][-2:]:
            (self.folder/(row['id']+'.npz')).unlink()
            (self.root/'reservations/probe-generations'/('probe-'+self.label+'-'+row['id']+'.json')).unlink()
        (self.folder/'ON_STUDENT_RESULTS.private.json').unlink()
        self.assertEqual(probes.status(self.root,self.label,self.instance.identity['files'])['new_queries_required'],2)
        self.assertEqual(self.run_complete()['new_queries'],2)
        self.assertEqual(self.calls[-2:],[1010,1011]);self.assertEqual(len(self.calls),110)

    def test_unknown_intent_or_orphan_array_never_regenerates(self):
        self.folder.mkdir(parents=True)
        self.put(self.root/'reservations/probe-generations'/('probe-'+self.label+'-expert-0.json'),dict(status='INTENT'))
        with self.assertRaises(MutationOutcomeUnknown):self.run_complete()
        self.assertEqual(self.calls,[])

    def test_wrong_model_and_changed_saved_result_fail_before_generation(self):
        self.run_complete();self.calls.clear()
        other={'config.json':'b'*64}
        with self.assertRaises(ValueError):probes.status(self.root,self.label,other)
        path=self.folder/'ON_EXPERT_RESULTS.json';value=json.loads(path.read_text());value['records'][0]['seed']+=1;self.put(path,value)
        with self.assertRaises(ValueError):self.run_complete()
        self.assertEqual(self.calls,[])

    def test_zero_success_complete_panels_skip_load_but_folder_only_does_not(self):
        spec=plan()['panels'][self.label]
        rows=[receipt(c,'VALID_TASK_FAIL') for c in spec['cards']]
        self.folder.mkdir(parents=True)
        work=probes.panel_work(self.root,self.label,self.instance.identity['files'],rows,spec)
        self.assertFalse(work['skip_before_model_load'])
        self.run_complete()
        work=probes.panel_work(self.root,self.label,self.instance.identity['files'],rows,spec)
        self.assertTrue(work['skip_before_model_load']);self.assertEqual(work['remaining'],[])
        self.assertEqual(work['probes']['new_queries_required'],0)
        dev=plan()['panels']['S-CORR-2048-DEV8']
        work=probes.panel_work(self.root,'S-CORR-2048-DEV8',self.instance.identity['files'],rows,dev)
        self.assertFalse(work['skip_before_model_load']);self.assertEqual(len(work['remaining']),8)

    def test_fresh_run_matches_unchanged_original_probe_primitive(self):
        self.run_complete()
        original=Path(self.tmp.name)/'original/source-pair-01'
        self.put(original/'plans/PROBE_MANIFEST.json',self.expert)
        self.put(original/'plans/STUDENT_PROBES.private.json',self.students)
        with patch.object(model,'expert_decode',side_effect=lambda processor,state,target:(target,{})):
            model.probes(original,self.label,self.instance,self.dataset,self.ledger,lambda:None)
        old=original/'evidence'/self.label/'probes'
        for name in ['ON_EXPERT_RESULTS.json','ON_STUDENT_RESULTS.private.json']:
            self.assertEqual(json.loads((old/name).read_text()),json.loads((self.folder/name).read_text()))
        self.assertEqual({p.name:sha(p) for p in old.glob('*.npz')},{p.name:sha(p) for p in self.folder.glob('*.npz')})


if __name__=='__main__':unittest.main()
