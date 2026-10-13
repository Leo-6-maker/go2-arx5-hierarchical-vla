"""CPU-only sampling/allocation/cache contracts, not neural qualification."""
import unittest
from types import SimpleNamespace

import numpy as np

from .native_dataset import ByteLRU
from .sampling import BalancedSampler, PHASES
from .neural import recipe_for


def fixture():
    rows=[]
    for task,origin in [('T1','legacy'),('T1','new'),('T2','new')]:
        for family in ('F0','F1'):
            for episode in range(2):
                record=dict(id=f'{task}-{origin}-{family}-{episode}',task=task,origin=origin,family=family,
                            stage_anchors={p:list(range(i*2,i*2+2)) for i,p in enumerate(PHASES)})
                rows.extend((record,anchor) for anchor in range(10))
    return SimpleNamespace(rows=rows,manifest_sha='UNIT_FIXTURE_ONLY')


class NeuralContracts(unittest.TestCase):
    def test_task_and_old_exposure(self):
        data=fixture();sampler=BalancedSampler(data,stop=600)
        samples=[data.rows[i][0] for _,i in sampler]
        self.assertEqual(sum(r['task']=='T1' for r in samples),300)
        self.assertEqual(sum(r['origin']=='legacy' for r in samples),100)

    def test_resume_cursor_independent_of_prefetch(self):
        data=fixture()
        full=list(BalancedSampler(data,stop=320))
        restored=list(BalancedSampler(data,start=96,stop=320))
        self.assertEqual(full[96:],restored)

    def test_empty_phase_explicit_renormalization(self):
        data=fixture()
        for row,anchor in data.rows:
            row['stage_anchors']={p:[] for p in PHASES}
            row['stage_anchors']['hold']=list(range(10))
        sampler=BalancedSampler(data,stop=32)
        self.assertTrue(sampler.empty)
        self.assertEqual(len(list(sampler)),32)

    def test_formal_single_task_rejected(self):
        data=fixture();data.rows=[r for r in data.rows if r[0]['task']=='T1']
        with self.assertRaises(ValueError):
            BalancedSampler(data)

    def test_lru_bounded(self):
        cache=ByteLRU(8)
        cache.put('a',np.ones(4,dtype=np.uint8),4)
        cache.put('b',np.ones(5,dtype=np.uint8),5)
        self.assertIsNone(cache.get('a'))
        self.assertLessEqual(cache.bytes,8)
        cache.put('big',None,9)
        self.assertLessEqual(cache.bytes,8)

    def test_recipe_real_effective_batch(self):
        for micro in (2,4,8):
            for workers in (0,2,4):
                recipe=recipe_for('main',micro,workers)
                self.assertEqual(recipe['per_device_train_batch_size']*recipe['gradient_accumulation_steps'],32)
                self.assertEqual(recipe['action_loss'],'UNCHANGED_NATIVE_MASKED_FLOW_MATCHING')
                self.assertFalse(recipe['tune_llm'])
                self.assertFalse(recipe['tune_visual'])
                self.assertTrue(recipe['bf16'])


if __name__=='__main__':
    unittest.main()
