"""Synthetic CPU acceptance; not physics or native model qualification."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.iteration.fixture import run_fixture
from scripts.iteration.plan import collection_plan, hopper_plan
from scripts.iteration.queue import Coordinator
from scripts.iteration.rgb_cache import VerifiedRGBLRU
from scripts.pi0292.durable import write


def card():
    return dict(status='NOT_AUTHORIZED',workers=2,cores_per_worker=1,max_concurrent_cores=4,
        episodes=[dict(id='FIXTURE_'+str(n),split='TRAIN' if n<3 else 'DEV',seed=900+n) for n in range(4)],
        per_episode=dict(physics_steps=16000,rgb_pairs=641,cpu_s=900,output_bytes=1048576),
        limits=dict(physics_steps=64000,rgb_pairs=2564,cpu_s=3600,output_bytes=4194304))


class Preparation(unittest.TestCase):
    def test_immutable_no_go_plan(self):
        original=card();c=copy.deepcopy(original);p=collection_plan(c)
        self.assertEqual(c,original);self.assertFalse(p['authorizes_execution'])
        self.assertIn('NOT_IMPLEMENTED',p['live_adapter'])
        self.assertEqual(p['workers']*p['cores_per_worker']+p['audit_workers']+p['coordinator_cores'],4)

    def test_bad_ids_splits_threads_or_limits(self):
        for modify in [lambda c:c['episodes'].append(c['episodes'][0]),
            lambda c:c['episodes'][0].update(id='../escape'),
            lambda c:c['episodes'][0].update(split='CONFIRM'),
            lambda c:c.update(cores_per_worker=2),lambda c:c.update(workers=4),
            lambda c:c['limits'].update(cpu_s=1),lambda c:c.update(status='GO'),
            lambda c:c['per_episode'].update(cpu_s=float('nan'))]:
            with self.subTest(modify=modify):
                c=card();modify(c)
                with self.assertRaises(ValueError):collection_plan(c)

    def test_reserved_inflight_no_retry_or_restart(self):
        with tempfile.TemporaryDirectory() as root:
            p=collection_plan(card());q=Coordinator(root,p,fixture=True)
            try:
                q.reserve('FIXTURE_0');q.reserve('FIXTURE_1')
                for identifier in ('FIXTURE_0','FIXTURE_2','NOT_REGISTERED'):
                    with self.assertRaises(ValueError):q.reserve(identifier)
                q.unknown('FIXTURE_0')
                self.assertEqual(len(q.value['reservations']),2)
                with self.assertRaises(ValueError):q.finish()
                with self.assertRaises(ValueError):q.reserve('FIXTURE_2')
            finally:q.close()
            with self.assertRaises(FileExistsError):Coordinator(root,p,fixture=True)
            with self.assertRaises(ValueError):Coordinator(root,p)

    def test_failed_receipt_and_overrun_charge_actual_stop(self):
        with tempfile.TemporaryDirectory() as root:
            q=Coordinator(root,collection_plan(card()),fixture=True)
            try:
                t=q.reserve('FIXTURE_0')
                receipt=dict(id=t['id'],fixture=True,status='FAIL',physics_steps=0,rgb_pairs=0,model_calls=0,cpu_s=901,output_bytes=3)
                write(Path(t['output'])/'WORKER_RECEIPT.json',receipt)
                q.complete(receipt)
                self.assertEqual(q.value['actual']['cpu_s'],901)
                self.assertEqual(q.value['status'],'FIXTURE_STOP_NO_AUTOMATIC_RETRY')
                self.assertFalse(q.value['reservations'])
                with self.assertRaises(ValueError):q.complete(receipt)
            finally:q.close()

    def test_spawn_two_workers_isolated_durable_results(self):
        with tempfile.TemporaryDirectory() as root:
            result=run_fixture(root,card())
            self.assertEqual(result['status'],'FIXTURE_TERMINAL_ZERO_PHYSICS')
            self.assertEqual(len(result['records']),4)
            self.assertEqual(len({r['pid'] for r in result['records']}),2)
            self.assertEqual(result['actual']['physics_steps'],0)
            self.assertEqual(result['model_calls'],0)
            self.assertFalse(result['reservations'])
            for record in result['records']:
                packet=json.loads((Path(root)/'episodes'/record['id']/'SYNTHETIC.json').read_text())
                self.assertEqual(packet['id'],record['id'])

    def test_one_episode_does_not_wait_for_unused_second_worker(self):
        with tempfile.TemporaryDirectory() as root:
            c=card();c['episodes']=c['episodes'][:1]
            result=run_fixture(root,c)
            self.assertEqual(result['status'],'FIXTURE_TERMINAL_ZERO_PHYSICS')
            self.assertEqual(len(result['records']),1)

    def test_output_symlink_escape_keeps_reservation_unchanged(self):
        with tempfile.TemporaryDirectory() as root,tempfile.TemporaryDirectory() as other:
            q=Coordinator(root,collection_plan(card()),fixture=True)
            try:
                (Path(root)/'episodes').symlink_to(other,target_is_directory=True)
                with self.assertRaisesRegex(ValueError,'OUTPUT_PATH_ESCAPE'):q.reserve('FIXTURE_0')
                self.assertFalse(q.value['reservations']);self.assertFalse(list(Path(other).iterdir()))
            finally:q.close()

    def test_hopper_plan_pinned_cpu_only(self):
        identity={k:'a'*64 for k in ('dataset_manifest_sha256','statistics_sha256','processor_sha256','source_seal_sha256')}
        identity.update(hf_revision='b'*40,optimization_split='TRAIN')
        result=hopper_plan(identity)
        self.assertEqual([c['num_workers'] for c in result['cases']],[0,2,4])
        self.assertFalse(result['authorizes_execution']);self.assertEqual(result['model_calls'],0)
        identity['optimization_split']='DEV'
        with self.assertRaises(ValueError):hopper_plan(identity)


class Cache(unittest.TestCase):
    def test_hit_copy_bytes_eviction_and_oversize_no_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);size=256*256*3
            for n in range(3):Image.fromarray(np.full((256,256,3),n,dtype=np.uint8)).save(root/f'{n}.png')
            hashes={n:hashlib.sha256((root/f'{n}.png').read_bytes()).hexdigest() for n in range(3)}
            cache=VerifiedRGBLRU(root,size*2)
            first=cache.get('0.png',hashes[0]);first[:]=255
            np.testing.assert_array_equal(cache.get('0.png',hashes[0]),np.zeros((256,256,3),dtype=np.uint8))
            self.assertEqual(cache.hits,1)
            cache.get('1.png',hashes[1]);cache.get('2.png',hashes[2])
            self.assertEqual(cache.bytes,size*2);self.assertEqual(len(cache.rows),2)
            misses=cache.misses;cache.get('0.png',hashes[0]);self.assertEqual(cache.misses,misses+1)
            small=VerifiedRGBLRU(root,size-1);small.get('0.png',hashes[0]);self.assertEqual(small.bytes,0)

    def test_mutation_hash_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);Image.new('RGB',(256,256)).save(root/'image.png')
            digest=hashlib.sha256((root/'image.png').read_bytes()).hexdigest()
            cache=VerifiedRGBLRU(root,256*256*3);cache.get('image.png',digest)
            Image.new('RGB',(256,256),color='red').save(root/'image.png')
            with self.assertRaisesRegex(ValueError,'IDENTITY_CHANGED'):cache.get('image.png',digest)
            with self.assertRaisesRegex(ValueError,'HASH_CHANGED'):cache.get('image.png','a'*64)
            (root/'escape').symlink_to('/etc/hosts')
            with self.assertRaisesRegex(ValueError,'PATH_ESCAPE'):cache.get('escape','a'*64)


if __name__=='__main__':unittest.main()
