from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.pi0292.durable import write
from scripts.pi0294.mp1 import n1_limits as n1
from scripts.pi0294.native_dataset import sha

REPO = Path(__file__).resolve().parents[2]


class N1LimitsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        parent = Path(self.tmp.name) / n1.PARENT_RUN
        self.root = parent / 'children/source-pair-01'
        self.root.mkdir(parents=True)
        budget = json.loads((REPO / 'reports/PI0294/design/MAINLINE_ACCELERATION_MP1_20261017.json').read_text())['incremental_budget']
        self.original = dict(authority_release=n1.PARENT_AUTHORITY, run_uuid=n1.PARENT_RUN,
                             T0_epoch=n1.ORIGINAL_T0, incremental_budget=budget,
                             new_submission_and_candidate_cutoff_epoch=1792696485,
                             new_experiment_cutoff_epoch=1792754085, hard_closeout_epoch=1792782885)
        self.card = json.loads((REPO / list(n1.BUNDLE)[2]).read_text())
        self.delta = dict(schema='MP1_N1_ADOPTION_DELTA_V1', id='MP1_N1_SINGLE_AUTHORITY_ADOPTION_01',
                          authority_release=n1.RELEASE, parent_authority_release=n1.PARENT_AUTHORITY,
                          parent_run_uuid=n1.PARENT_RUN, source_child='source-pair-01',
                          owner='EXISTING_SINGLE_LOCAL_PC', original_T0_epoch=n1.ORIGINAL_T0,
                          bundle_sha256=n1.BUNDLE, reset_on_resume=False, new_competing_owner=False,
                          original_birth_manifests_cards_and_costs_preserved=True, old_scope_costs_reclassified=False,
                          autonomy_started_epoch=1792264081, publication_epoch=1792261823,
                          new_submission_and_candidate_cutoff_epoch=1792372081,
                          new_experiment_cutoff_epoch=1792386481, hard_closeout_epoch=1792393681)
        write(parent / 'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json', self.original)
        for target, source in zip(n1.FILES[:3], n1.BUNDLE):
            p = self.root / target
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((REPO / source).read_bytes())
            p.chmod(0o400)
        write(self.root / n1.DELTA, self.delta)
        (self.root / n1.DELTA).chmod(0o400)
        self.manifest = dict(execution=dict(sealed_inputs={p:sha(self.root / p) for p in n1.FILES},
            n1_authority_binding=dict(release=n1.RELEASE, adoption_delta_sha256=sha(self.root / n1.DELTA))))
        write(self.root / 'RUN_MANIFEST.json', self.manifest)

    def test_effective_cumulative_limits_without_mutating_birth(self):
        original = deepcopy(self.original)
        v = n1.resolve(self.root)
        self.assertEqual(v['source']['PBS'], 12)
        self.assertEqual(v['source']['GPU_s'], 129600)
        self.assertEqual(v['parent_H200_s'], 80 * 3600)
        self.assertEqual(v['parent']['H200_allocated_GPU_hours'], 80)
        self.assertEqual(v['parent']['Hopper_new_GPU_PBS_submissions'], 24)
        self.assertEqual(v['source']['CPU_s'], dict(core=648000, hopper=518400))
        self.assertEqual(v['unchanged']['source_formal_updates'], 4096)
        self.assertEqual(v['original_T0_epoch'], 1792178085)
        self.assertEqual(self.original, original)

    def test_no_manifest_adoption_does_not_hot_apply_sidecars(self):
        self.manifest['execution'].pop('n1_authority_binding')
        self.assertIsNone(n1.resolve(self.root, self.manifest))

    def test_all_contiguous_slots_and_rejected_unbound_sixth(self):
        for number in range(1,13):
            label = 'S-CORR-2048-TRAIN4-ALLOCATION-' + str(number).zfill(2)
            self.assertEqual(n1.allocation_number(label,12)[1], number)
        for label in ['P-ALLOCATION-00','P-ALLOCATION-13','P-ALLOCATION-6','P-ALLOCATION-٠٦']:
            with self.subTest(label=label), self.assertRaises(PermissionError):
                n1.allocation_number(label,12)
        with self.assertRaises(PermissionError):
            n1.allocation_number('P-ALLOCATION-06')

    def test_wrong_identity_reset_or_broader_clock_rejected(self):
        for key, value in [('source_child','other'), ('parent_run_uuid','other'),
                           ('original_T0_epoch',0), ('reset_on_resume',True),
                           ('new_competing_owner',True), ('hard_closeout_epoch',1792393682),
                           ('autonomy_started_epoch',1792264082)]:
            delta = dict(self.delta, **{key:value})
            with self.subTest(key=key), self.assertRaises(PermissionError):
                n1.validated_delta(delta,self.original)

    def test_changed_card_delta_permissions_and_seal_rejected(self):
        for relative in n1.FILES:
            path = self.root / relative
            original = path.read_bytes()
            path.chmod(0o600)
            with self.subTest(path=relative), self.assertRaises(PermissionError):
                n1.resolve(self.root)
            path.write_bytes(original + b' ')
            path.chmod(0o400)
            with self.assertRaises(PermissionError):n1.resolve(self.root)
            path.chmod(0o600);path.write_bytes(original);path.chmod(0o400)

    def test_prior_cost_and_reservation_not_a_new_empty_pool(self):
        n1.require_remaining(24119,14400,20000,129600)
        with self.assertRaises(PermissionError):n1.require_remaining(120000,9600,1,129600)
        with self.assertRaises(ValueError):n1.require_remaining(True,0,1,129600)


if __name__ == '__main__':unittest.main()
