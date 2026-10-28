"""CPU-only rejection and accounting tests. No MuJoCo integration or model."""
import copy
import os
import pickle
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from scripts.iteration.queue import LiveCoordinator, digest, value_digest
from scripts.pi0292.durable import read, write


class QueueTest(unittest.TestCase):
    def setUp(self):
        self.affinity = patch('scripts.iteration.queue.os.sched_getaffinity', return_value=set(range(16)))
        self.affinity.start()
        self.addCleanup(self.affinity.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.clock = 1791901600
        self.source = self.root / 'source'
        self.source.mkdir()
        write(self.source / 'code.json', {'immutable': True})
        write(self.root / 'RUN_MANIFEST.json', dict(pi='PI0294', run_uuid='cpu-test',
            authority={'protocol_ref': 'f' * 40}, code={'capsule_relative': 'source'}))
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

    def test_cpu_sidecar_is_sealed_without_changing_collector_source(self):
        sidecar=self.root/'CPU_MANIFEST.json'
        value=dict(execution=dict(profile='4080-cpu-data'),code=dict(vla_commit='b'*40))
        write(sidecar,value)
        owner=self.owner()
        try:
            with patch('scripts.pi0294.run_admission.validate',return_value=value) as validate:
                ticket=owner.reserve_auxiliary('new-cpu-code',cpu_count=1,timeout_s=30,output_bytes=100,
                    memory_bytes=100,manifest_path=sidecar)
            validate.assert_called_once_with(self.root,profile='4080-cpu-data',manifest_path=sidecar)
            self.assertEqual(ticket['source_sha'],'b'*40)
            self.assertEqual(ticket['manifest_relative'],'CPU_MANIFEST.json')
            self.assertEqual(ticket['run_manifest_sha256'],digest(sidecar))
            self.assertEqual(owner.value['source_sha'],'f'*40)
            self.assertEqual(owner.value['actual'],self.admission['prior_actual'])
        finally:owner.close()

    def test_auxiliary_cannot_claim_neural_or_foreign_manifest(self):
        owner=self.owner()
        try:
            sidecar=self.root/'BAD_MANIFEST.json'
            write(sidecar,dict(execution=dict(profile='hopper-neural')))
            with self.assertRaisesRegex(ValueError,'NEURAL_PROFILE'):
                owner.reserve_auxiliary('bad-neural',cpu_count=1,timeout_s=30,output_bytes=100,
                    memory_bytes=100,manifest_path=sidecar)
            with self.assertRaisesRegex(ValueError,'REGISTERED_RUN'):
                owner.reserve_auxiliary('bad-path',cpu_count=1,timeout_s=30,output_bytes=100,
                    memory_bytes=100,manifest_path=self.root.parent/'outside.json')
            self.assertFalse(owner.value['auxiliary'])
        finally:owner.close()

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

    def test_auxiliary_cores_are_disjoint_from_coordinator_and_workers(self):
        self.admission['memory_limit_bytes'] = 12 * 1024**3
        self.admit()
        owner = self.owner()
        try:
            a, b = owner.reserve('a'), owner.reserve('b')
            cpu = owner.reserve_auxiliary('audit-a', cpu_count=2, timeout_s=30,
                                          output_bytes=100, memory_bytes=100)
            self.assertEqual(cpu['cpus'], [7, 8])
            self.assertFalse(set(cpu['cpus']) & set(a['cpus'] + b['cpus'] + [6]))
            self.assertEqual(read(self.root / 'reservations' / 'aux-audit-a.json'), cpu)
            with self.assertRaisesRegex(ValueError, 'SAFE_AUXILIARY'):
                owner.reserve_auxiliary('../escape', cpu_count=1, timeout_s=1, output_bytes=1)
        finally:
            owner.close()

    def test_too_few_actual_allowed_cores_rejected(self):
        with patch('scripts.iteration.queue.os.sched_getaffinity', return_value={0, 1}):
            with self.assertRaisesRegex(ValueError, 'ACTUAL_ALLOWED_CPU'):
                self.owner()

    def test_admission_cannot_point_at_another_capsule(self):
        other = self.root / 'other'
        other.mkdir()
        write(other / 'code.json', {'immutable': True})
        self.admission['source_root'] = str(other)
        self.admit()
        with self.assertRaisesRegex(ValueError, 'DIFFERS_FROM_MANIFEST_CAPSULE'):
            self.owner()

    def test_source_advance_requires_terminal_evidence_and_preserves_all_costs(self):
        self.plan['episodes'] = self.plan['episodes'][:1]
        write(self.root / 'plan.json', self.plan)
        self.admission['plan_sha256'] = digest(self.root / 'plan.json')
        self.admit()
        owner = self.owner()
        owner.complete(self.receipt(owner.reserve('a')))
        owner.finish()
        actual, old_queue_hash = copy.deepcopy(owner.value['actual']), digest(owner.path)
        owner.close()
        self.plan['episodes'] = [dict(id='d', card=dict(id='d', split='TRAIN', task='T2'))]
        write(self.root / 'plan.json', self.plan)
        self.admission.update(plan_sha256=digest(self.root / 'plan.json'), prior_actual=actual,
                              source_sha='e' * 40)
        write(self.root / 'SOURCE_SEAL.json', dict(source_sha='e' * 40,
            files={'code.json': digest(self.source / 'code.json')}))
        self.admission['source_seal_sha256'] = digest(self.root / 'SOURCE_SEAL.json')
        self.admit()
        with self.assertRaisesRegex(ValueError, 'SOURCE_BOUNDARY_RECEIPT_REQUIRED'):
            self.owner(stage_boundary=True)
        proof = self.root / 'SOURCE_BOUNDARY.json'
        write(proof, dict(status='VERIFIED_TERMINAL_SOURCE_ADVANCE', run_uuid='cpu-test',
            old_source_sha='f' * 40, new_source_sha='e' * 40,
            old_queue_sha256=old_queue_hash, no_active_consumers=True))
        self.admission.update(source_boundary_receipt_path=str(proof),
                              source_boundary_receipt_sha256=digest(proof))
        self.admit()
        owner = self.owner(stage_boundary=True)
        try:
            self.assertEqual(owner.value['actual'], actual)
            self.assertEqual(len(owner.value['records']), 1)
            self.assertEqual(owner.value['stages'][0]['source_sha'], 'f' * 40)
            self.assertEqual(owner.value['stages'][1]['source_sha'], 'e' * 40)
        finally:
            owner.close()

    def test_terminal_cpu_owner_preserves_first_attempts_and_accounts_only_auxiliary(self):
        self.plan['episodes'] = self.plan['episodes'][:1]
        write(self.root / 'plan.json', self.plan)
        self.admission['plan_sha256'] = digest(self.root / 'plan.json')
        self.admit()
        owner = self.owner()
        owner.complete(self.receipt(owner.reserve('a')))
        owner.finish()
        self.admission['prior_actual'] = copy.deepcopy(owner.value['actual'])
        owner.close()
        self.admit()
        owner = LiveCoordinator.open_terminal(self.root, self.root / 'plan.json', self.root / 'admission.json',
            now=lambda: self.clock)
        try:
            owner.reserve_auxiliary('bank-v3_0', cpu_count=2, timeout_s=30, output_bytes=100, memory_bytes=100)
            owner.complete_auxiliary('bank-v3_0', cpu_s=2., output_bytes=30)
            self.assertEqual(owner.value['actual']['physics_starts'], 1)
            self.assertEqual(owner.value['actual']['cpu_s'], 3.)
            self.assertEqual(len(owner.value['records']), 1)
            with self.assertRaisesRegex(ValueError, 'AUXILIARY_ALREADY'):
                owner.reserve_auxiliary('bank-v3_0', cpu_count=2, timeout_s=30, output_bytes=100)
            self.assertEqual(owner.finish()['status'], 'LIVE_TERMINAL_ALL_FIRST_ATTEMPTS')
        finally:
            owner.close()


class PairedAdmissionTest(unittest.TestCase):
    def test_runpy_launcher_uses_a_picklable_importable_worker(self):
        import runpy
        import importlib
        namespace = runpy.run_module('scripts.pi0294.dispatcher', run_name='cpu_launch_contract')
        worker = importlib.import_module('scripts.pi0294.dispatcher')._spawn_worker
        restored = pickle.loads(pickle.dumps(worker))
        self.assertIs(restored, worker)
        self.assertEqual(namespace['dispatch'].__globals__['__name__'], 'cpu_launch_contract')

    def test_unterminated_first_attempt_cannot_be_admitted(self):
        from scripts.pi0294.bank import new_admission
        base = dict(seed=1, split='TRAIN', family='F0', group_id='one', initial_state_sha256='same')
        cards = [dict(base, id='a', task='T1', task_id='PICK', instruction='pick'),
                 dict(base, id='b', task='T2', task_id='PUSH', instruction='push')]
        with self.assertRaisesRegex(ValueError, 'MUST_BE_TERMINAL'):
            new_admission('.', {'cards': cards}, {'records': [], 'reservations': {}})

    def test_language_pair_must_share_initial_state_and_split(self):
        from scripts.pi0294.bank import new_admission
        base = dict(seed=1, split='TRAIN', family='F0', group_id='one', initial_state_sha256='same')
        cards = [dict(base, id='a', task='T1', task_id='PICK', instruction='pick'),
                 dict(base, id='b', task='T2', task_id='PUSH', instruction='push', seed=2)]
        with self.assertRaisesRegex(ValueError, 'BEYOND_LANGUAGE'):
            new_admission('.', {'cards': cards}, {'records': [], 'reservations': {}})
        cards[1]['seed'] = 1
        cards[1]['split'] = 'DEV'
        with self.assertRaisesRegex(ValueError, 'INITIAL_STATE_AND_SPLIT'):
            new_admission('.', {'cards': cards}, {'records': [], 'reservations': {}})

    def test_stage_labels_are_offline_and_unknown_buckets_are_explicit(self):
        from scripts.pi0294.bank import stages
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / 'samples.json', [dict(privileged={'teacher_stage': s}) for s in
                                         ('approach', 'contact', 'push', 'detach', 'hold', 'unexpected')])
            buckets, unknown = stages(root)
            self.assertEqual(buckets, dict(approach=[0], contact=[1], transport=[2], withdraw=[3], hold=[4]))
            self.assertEqual(unknown, {'unexpected': [5]})

    def test_push_rejects_a_between_frame_lift_and_exact_300ms_grasp(self):
        import numpy as np
        from scripts.pi0294.audit import independent_push_score
        poses = np.broadcast_to(np.eye(4), (641, 4, 4)).copy()
        poses[:, 0, 3] = np.minimum(np.arange(641) / 320, 1) * .16
        poses[:, 2, 3] = .52
        tcp = poses.copy()
        tcp[:, 2, 3] += .15
        micro = np.zeros((16000, 5))
        micro[:8000, 0] = 1
        micro[:, 3] = .52
        data = dict(time_s=np.arange(641) / 20, object_pose=poses, tcp_world=tcp,
            object_linear_velocity=np.zeros((641, 3)), object_angular_velocity=np.zeros((641, 3)),
            robot_object_contact=np.arange(641) < 320, contact_audit=micro,
            contact_time_s=np.arange(1, 16001) * .002)
        for key in ('object_present', 'contact_valid', 'gripper_state_valid', 'safety_valid'):
            data[key] = np.ones(641, dtype=bool)
        for key in ('native_grasp', 'left_object_contact', 'right_object_contact', 'native_abort',
                    'illegal_contact', 'joint_limit_penetration', 'reset', 'takeover'):
            data[key] = np.zeros(641, dtype=bool)
        card = dict(goal_xy=[.16, 0])
        result = independent_push_score(data, card, True, .5)
        self.assertTrue(result['persistent_success'])
        micro[21, 3] = .526
        self.assertFalse(independent_push_score(data, card, True, .5)['persistent_success'])
        micro[21, 3] = .52
        micro[:150, 1] = 1
        result = independent_push_score(data, card, True, .5)
        self.assertFalse(result['persistent_success'])
        self.assertAlmostEqual(result['maximum_bilateral_grasp_s'], .3)


if __name__ == '__main__':
    unittest.main()
