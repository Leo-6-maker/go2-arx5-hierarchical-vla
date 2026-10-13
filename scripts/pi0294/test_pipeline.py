"""CPU-only rejection and accounting tests. No MuJoCo integration or model."""
import copy
import os
from pathlib import Path
import socket
import tempfile
import unittest

from scripts.iteration.queue import LiveCoordinator, digest, value_digest
from scripts.pi0292.durable import read, write


class QueueTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.clock = 1791901600
        self.source = self.root / 'source'
        self.source.mkdir()
        write(self.source / 'code.json', {'immutable': True})
        write(self.root / 'RUN_MANIFEST.json', dict(pi='PI0294', run_uuid='cpu-test',
            authority={'protocol_ref': 'f' * 40}))
        write(self.root / 'SOURCE_SEAL.json', dict(source_sha='f' * 40,
            files={'code.json': digest(self.source / 'code.json')}))
        self.plan = dict(fixture=False, authorizes_execution=True,
            episodes=[dict(id=i, card=dict(id=i, split='TRAIN', task='T1')) for i in ('a', 'b', 'c')])
        write(self.root / 'plan.json', self.plan)
        self.admission = dict(schema='PI0294_LIVE_ADMISSION_V1', fixture=False,
            status='LIVE_ADMITTED_ACTUAL_RUNTIME', hostname=socket.gethostname(),
            profile='4080-simulation', root=str(self.root), run_uuid='cpu-test',
            plan_sha256=digest(self.root / 'plan.json'), run_manifest_sha256=digest(self.root / 'RUN_MANIFEST.json'),
            source_seal_sha256=digest(self.root / 'SOURCE_SEAL.json'), source_sha='f' * 40,
            source_root=str(self.source), asset_root=str(self.source), T0_epoch=1791901497,
            cutoff_epoch=1791939600, deadline_epoch=1791943200, lease_expires_epoch=1791943200,
            qualified_workers=2, allowed_cpus=sorted(os.sched_getaffinity(0))[:10],
            memory_limit_bytes=8 * 1024**3, free_space_margin_bytes=0,
            limits=dict(physics_starts=400, physics_steps=6400000, rgb_pairs=256400,
                cpu_s=576000, output_bytes=192 * 1024**3),
            prior_actual=dict(physics_starts=0, physics_steps=0, rgb_pairs=0, cpu_s=0, output_bytes=0))
        self.admit()

    def tearDown(self):
        self.temp.cleanup()

    def admit(self):
        write(self.root / 'admission.json', self.admission)

    def owner(self, **kwargs):
        return LiveCoordinator(self.root, self.root / 'plan.json', self.root / 'admission.json',
            now=lambda: self.clock, **kwargs)

    def receipt(self, ticket, **changes):
        folder = Path(ticket['output'])
        folder.mkdir()
        value = dict(id=ticket['id'], card_sha256=ticket['card_sha256'], fixture=False,
            status='VALID_COMPLETE', model_calls=0, physics_starts=1, physics_steps=16000,
            rgb_pairs=641, cpu_s=1., output_bytes=100, memory_bytes=100)
        value.update(changes)
        write(folder / 'WORKER_RECEIPT.json', value)
        return value

    def test_disjoint_atomic_reservations_and_duplicate_rejection(self):
        owner = self.owner()
        try:
            a, b = owner.reserve('a'), owner.reserve('b')
            self.assertFalse(set(a['cpus']) & set(b['cpus']))
            self.assertFalse(Path(a['output']).exists())
            self.assertEqual(read(owner.path)['reservations']['a'], a)
            with self.assertRaisesRegex(ValueError, 'FIRST_ATTEMPT'):
                owner.reserve('a')
            with self.assertRaisesRegex(ValueError, 'SLOTS'):
                owner.reserve('c')
            owner.complete(self.receipt(a))
            self.assertEqual(owner.value['actual']['physics_starts'], 1)
            self.assertEqual(owner.reserve('c')['cpus'], a['cpus'])
        finally:
            owner.close()

    def test_crash_retains_reservation_and_forbids_stage_boundary(self):
        owner = self.owner()
        owner.reserve('a')
        owner.unknown('a')
        owner.close()
        self.assertIn('a', read(self.root / 'LIVE_QUEUE.json')['reservations'])
        with self.assertRaisesRegex(ValueError, 'TERMINAL_STAGE'):
            self.owner(stage_boundary=True)

    def test_fixture_cannot_be_labeled_live(self):
        self.admission['fixture'] = True
        self.admit()
        with self.assertRaisesRegex(ValueError, 'FIXTURE'):
            self.owner()

    def test_cumulative_terminal_stage_cannot_reset_costs_or_ids(self):
        self.plan['episodes'] = self.plan['episodes'][:1]
        write(self.root / 'plan.json', self.plan)
        self.admission['plan_sha256'] = digest(self.root / 'plan.json')
        self.admit()
        owner = self.owner()
        owner.complete(self.receipt(owner.reserve('a')))
        owner.finish()
        actual = copy.deepcopy(owner.value['actual'])
        owner.close()
        self.plan['episodes'] = [dict(id='d', card=dict(id='d', split='TRAIN', task='T2'))]
        write(self.root / 'plan.json', self.plan)
        self.admission['plan_sha256'] = digest(self.root / 'plan.json')
        self.admit()
        with self.assertRaisesRegex(ValueError, 'CUMULATIVE'):
            self.owner(stage_boundary=True)
        self.admission['prior_actual'] = actual
        self.admit()
        owner = self.owner(stage_boundary=True)
        try:
            owner.complete(self.receipt(owner.reserve('d')))
            self.assertEqual(owner.finish()['actual']['physics_starts'], 2)
        finally:
            owner.close()

    def test_production4_needs_actual_parallel_qualification(self):
        self.admission['allowed_cpus'] = sorted(os.sched_getaffinity(0))[:16]
        self.admit()
        with self.assertRaisesRegex(ValueError, 'NOT_QUALIFIED'):
            self.owner(workers=4)
        self.plan['kind'] = 'PARALLEL_QUALIFICATION'
        for row in self.plan['episodes']:
            row['card']['split'] = 'DIAGNOSTIC'
        write(self.root / 'plan.json', self.plan)
        self.admission.update(diagnostic_workers=4, plan_sha256=digest(self.root / 'plan.json'))
        self.admit()
        owner = self.owner(workers=4)
        owner.close()

    def test_budget_reserves_full_upper_bound_before_start(self):
        self.admission['limits']['physics_starts'] = 1
        self.admit()
        owner = self.owner()
        try:
            owner.reserve('a')
            with self.assertRaisesRegex(ValueError, 'BUDGET'):
                owner.reserve('b')
            self.assertEqual(owner.value['actual']['physics_starts'], 0)
        finally:
            owner.close()

    def test_worker_receipt_rejects_synthetic_and_preserves_upper_bound(self):
        owner = self.owner()
        try:
            ticket = owner.reserve('a')
            receipt = self.receipt(ticket, fixture=True)
            with self.assertRaisesRegex(ValueError, 'PHYSICS_ONLY'):
                owner.complete(receipt)
            self.assertIn('a', owner.value['reservations'])
        finally:
            owner.close()


if __name__ == '__main__':
    unittest.main()
