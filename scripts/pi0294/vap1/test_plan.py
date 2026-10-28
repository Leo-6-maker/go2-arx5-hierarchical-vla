import unittest

from scripts.pi0294.cr2 import plan as old
from . import plan


class VAP1PlanTests(unittest.TestCase):
    def groups(self):
        return [plan.candidate(*s, 0) for s in plan.slots()]

    def rows(self):
        return [dict(id=c['id'], task_success=True, admitted=True, evidence_qualified=True,
                     native_faults=0, unknown_effects=[], R0_PASS_to_R1_FAIL=False)
                for c in plan.cards(self.groups())]

    def test_new_population_is_deterministic_not_old_CR2(self):
        groups = self.groups()
        self.assertEqual(len(groups), 24)
        self.assertEqual(len({g['seed'] for g in groups}), 24)
        for slot in plan.slots():
            for index in range(32):
                g = plan.candidate(*slot, index)
                self.assertEqual(g, plan.candidate(*slot, index))
                self.assertNotEqual(g['source_xy'], old.candidate(*slot[1:], index)['source_xy'])
                with self.assertRaises(PermissionError):
                    old.validate_group(g)

    def test_all_cards_sealed_and_batch_order(self):
        cards = plan.cards(self.groups())
        self.assertEqual(len({c['id'] for c in cards}), 48)
        self.assertTrue(all(c['replicate'] == 0 for c in cards[:24]))
        self.assertTrue(all(c['replicate'] == 1 for c in cards[24:]))
        with self.assertRaises(PermissionError):
            plan.cards(self.groups()[:-1])

    def test_actual_gates_and_no_incomplete_or_duplicate_pass(self):
        groups, rows = self.groups(), self.rows()
        self.assertEqual(plan.bank_gate(rows[:24], groups)['status'], 'PASS_VAP1_DATA_BANK_GATE')
        self.assertEqual(plan.bank_gate(rows, groups, cumulative=True)['status'], 'PASS_VAP1_DATA_BANK_GATE')
        self.assertEqual(plan.bank_gate(rows[:23], groups)['status'], 'NOT_READY')
        for bad in (rows, rows[:24] + rows[:1]):
            with self.assertRaises(PermissionError): plan.bank_gate(bad, groups)
        rows[0]['R0_PASS_to_R1_FAIL'] = True
        self.assertEqual(plan.bank_gate(rows[:24], groups)['status'], 'NOT_READY')

    def test_failures_preserved_without_redraw(self):
        groups, rows = self.groups(), self.rows()
        for i in (0, 2, 4): rows[i].update(task_success=False, admitted=False)
        self.assertEqual(plan.bank_gate(rows[:24], groups)['status'], 'NOT_READY')
        for r in rows: r.pop('unknown_effects')
        self.assertEqual(plan.bank_gate(rows, groups, cumulative=True)['status'], 'NOT_READY')


if __name__ == '__main__': unittest.main()
