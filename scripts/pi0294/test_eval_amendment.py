"""Finite resource boundary checks; zero neural queries and physics."""
import copy
import unittest
from . import continuation as c
from .test_continuation import ContinuationTests


class EvaluationAmendmentTests(unittest.TestCase):
    def test_original_and_amended_boundaries(self):
        value = ContinuationTests().allocation()
        c.validate_allocation(**value)
        peer = copy.deepcopy(value['jobs']['1'])
        peer['project'] = 'OTHER_PROJECT'
        value['jobs'].update({'2': peer, '3': copy.deepcopy(peer)})
        with self.assertRaises(PermissionError):
            c.validate_allocation(**value)
        value['request'].update(submit_count=3, walltime_s=7200,
            prior_GPU_allocated_s=7148, evaluation_amendment_sha256=c.EVAL_AMENDMENT_SHA)
        value['jobs']['1']['Resource_List']['walltime'] = '02:00:00'
        c.validate_allocation(**value)
        for changes in (dict(stage='main'), dict(submit_count=4), dict(walltime_s=7201),
                        dict(prior_GPU_allocated_s=14401), dict(evaluation_amendment_sha256='0'*64)):
            bad = copy.deepcopy(value)
            bad['request'].update(changes)
            with self.assertRaises(PermissionError):
                c.validate_allocation(**bad)
        for changes in (dict(Job_Owner='foreign@host'), dict(Resource_List=dict(ngpus=2))):
            bad = copy.deepcopy(value)
            bad['jobs']['2'].update(changes)
            with self.assertRaises(PermissionError):
                c.validate_allocation(**bad)
        value['jobs']['4'] = copy.deepcopy(peer)
        with self.assertRaises(PermissionError):
            c.validate_allocation(**value)


if __name__ == '__main__':
    unittest.main()
