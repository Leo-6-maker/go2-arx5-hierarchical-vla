"""Opaque synthetic checkpoint files only; no tensors or pickle are loaded."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.pi0292 import test_checkpoint_capacity as original_fixture
from scripts.pi0292.durable import read,write
from .native_dataset import sha
from .sampling import BalancedSampler,PHASES
from .checkpoint_delivery import inventory,verify_remote,capacity,GIB


class CheckpointDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture=original_fixture.TestCheckpointInventory();self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.dataset=Path(self.temp.name)
        (self.dataset/'statistics.json').write_bytes((self.fixture.root/'statistics.json').read_bytes())
        self.manifest=dict(statistics_sha256=sha(self.dataset/'statistics.json'),episodes=[
            dict(id='old',split='TRAIN',windows=601)])
        self.bind('engineering')

    def bind(self,kind):
        step=4 if kind=='engineering' else 512
        recipe=dict(kind='smoke' if kind=='engineering' else 'main',max_steps=4 if kind=='engineering' else 6144,
            effective_batch=32,seed=29400)
        if kind=='main':
            self.manifest.update(schema='PI0294_FROZEN_MULTITASK_NATIVE_V1',frozen=True)
            self.manifest['episodes']=[dict(id=task+origin,task=task,origin=origin,family='F0',split='TRAIN',windows=601,
                stage_anchors={phase:list(range(601)) if phase=='hold' else [] for phase in PHASES})
                for task,origin in [('T1','legacy'),('T1','new'),('T2','new')]]
        write(self.dataset/'DATASET_MANIFEST.json',self.manifest)
        data=SimpleNamespace(manifest_sha=sha(self.dataset/'DATASET_MANIFEST.json'),
            rows=[(row,a) for row in self.manifest['episodes'] for a in range(601)])
        sampler=BalancedSampler(data,stop=recipe['max_steps']*32,smoke=kind=='engineering').state_dict(step*32)
        self.fixture.put('trainer_state.json',dict(global_step=step))
        self.fixture.put('SAMPLER_STATE.json',dict(recipe=recipe,sampler=sampler,
            statistics_sha256=self.fixture.stats,manifest_sha256=data.manifest_sha,
            worker_trace=dict(status='PASS_ACTUAL_CONSUMED_WORKER_IDENTITIES',start=0,stop=step*32,
                count=step*32,prefetch_excluded=True)))
        self.fixture.put('CHECKPOINT_COMPLETE.json',dict(global_step=step,
            status='FULL_STATE_SAVED_MODEL_ARCHIVE_HASH_PENDING',required_files={
                n:sha(self.fixture.root/n) for n in ('optimizer.pt','scheduler.pt','rng_state.pth',
                    'trainer_state.json','statistics.json','SAMPLER_STATE.json')}))

    def test_complete_engineering_bytes_not_a_policy_or_resume_pass(self):
        result=inventory(self.fixture.root,self.dataset,expected_step=4,kind='engineering')
        self.assertFalse(result['model_loaded']);self.assertFalse(result['optimizer_pickle_loaded'])
        self.assertFalse(result['policy_candidate']);self.assertFalse(result['selection_qualified'])
        self.assertEqual(result['selected'],'NONE')

    def test_main_M512_is_development_not_selected_candidate(self):
        self.bind('main')
        result=inventory(self.fixture.root,self.dataset,expected_step=512,kind='main')
        self.assertEqual(result['consumed_samples'],512*32)
        self.assertFalse(result['policy_candidate'])
        self.assertFalse(result['selection_qualified'])

    def test_saved_sampler_cannot_substitute_prefetch_for_consumed_cursor(self):
        value=read(self.fixture.root/'SAMPLER_STATE.json');value['sampler']['cursor']+=2
        self.fixture.put('SAMPLER_STATE.json',value)
        complete=read(self.fixture.root/'CHECKPOINT_COMPLETE.json')
        complete['required_files']['SAMPLER_STATE.json']=sha(self.fixture.root/'SAMPLER_STATE.json')
        self.fixture.put('CHECKPOINT_COMPLETE.json',complete)
        with self.assertRaisesRegex(ValueError,'CONSUMED_SAMPLE_CURSOR_CHANGED'):
            inventory(self.fixture.root,self.dataset,expected_step=4,kind='engineering')

    def test_bank_identity_or_kind_cannot_be_relabelled(self):
        with self.assertRaises(ValueError):inventory(self.fixture.root,self.dataset,expected_step=4,kind='main')
        with self.assertRaises(ValueError):inventory(self.fixture.root,self.dataset,expected_step=512,kind='engineering')
        self.manifest['changed']='not an equivalent version';write(self.dataset/'DATASET_MANIFEST.json',self.manifest)
        with self.assertRaises(ValueError):inventory(self.fixture.root,self.dataset,expected_step=4,kind='engineering')

    def test_fixed_remote_file_set_bytes_and_lfs_sha_required(self):
        files={'model.safetensors':dict(bytes=123,sha256='a'*64)}
        remote=SimpleNamespace(path='prefix/model.safetensors',size=123,lfs=SimpleNamespace(sha256='a'*64))
        api=SimpleNamespace(list_repo_tree=lambda *a,**kw:[remote])
        def deny(*a,**kw):raise AssertionError('Large LFS artifact must not be downloaded again')
        verify_remote(api,'private/repo','prefix','b'*40,files,deny,None)
        for key,value in [('size',124),('lfs',SimpleNamespace(sha256='c'*64)),('path','prefix/extra')]:
            with self.subTest(key=key):
                original=getattr(remote,key);setattr(remote,key,value)
                with self.assertRaises(ValueError):verify_remote(api,'private/repo','prefix','b'*40,files,deny,None)
                setattr(remote,key,original)

    def test_cpu_delivery_cannot_expand_peak_or_duration(self):
        request=dict(max_owned_bytes=180*GIB,growth_reserved_bytes=32*GIB,
            initial_owned_bytes=100*GIB,deadline_epoch=100)
        capacity(request,110*GIB,20*GIB,99)
        for value,owned,free,now in [(dict(request,max_owned_bytes=181*GIB),110*GIB,20*GIB,99),
            (dict(request,growth_reserved_bytes=33*GIB),110*GIB,20*GIB,99),
            (request,133*GIB,20*GIB,99),(request,110*GIB,9*GIB,99),
            (request,110*GIB,20*GIB,100)]:
            with self.subTest(owned=owned,free=free,now=now):
                with self.assertRaises(PermissionError):capacity(value,owned,free,now)


if __name__=='__main__':unittest.main()
