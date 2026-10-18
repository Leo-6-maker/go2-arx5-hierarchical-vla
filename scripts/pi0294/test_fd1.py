"""CPU fixtures only: FD1 labels cannot promote diagnostics into task success."""
import ast
import importlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.pi0294 import fd1
from scripts.pi0294.fd1_shadow import geometry_category


class FD1Tests(unittest.TestCase):
    def test_task_phase_uses_events_not_teacher_approach(self):
        self.assertEqual(fd1.actual_phase('T2',True,True,False,False,True,False,False),
                         'OBJECT_TRANSPORT_CONTACT_ACTIVE')
        self.assertEqual(fd1.actual_phase('T2',True,True,False,False,True,True,False),
                         'POST_CONTACT_DISPLACED_NOW_SEPARATED')
        self.assertEqual(fd1.actual_phase('T2',False,False,False,False,True,True,False),
                         'NO_RECORDED_OBJECT_CONTACT_OR_MOTION')

    def test_terminal_event_is_not_success(self):
        self.assertEqual(fd1.actual_phase('T1',True,True,True,True,False,False,False),
                         'GRASP_OR_CARRY_NOT_RELEASED')
        self.assertNotIn('SUCCESS',fd1.actual_phase('T2',True,True,True,False,True,True,True))

    def test_first_time_uses_actual_grain(self):
        self.assertEqual(fd1.first_time(np.array([.002,.004,.006]),[False,True,True]),.004)
        self.assertIsNone(fd1.first_time(np.arange(3),[False]*3))

    def test_missing_geom_is_not_inferred(self):
        self.assertEqual(geometry_category(dict(predicted_native_safety=['ILLEGAL_CONTACT_OR_PENETRATION'],illegal_pairs=[])), 'UNPROVEN')
        for safety,expected in [('TCP_VELOCITY','TCP_velocity'),('JOINT_LIMIT','joint_limit')]:
            self.assertEqual(geometry_category(dict(predicted_native_safety=[safety],illegal_pairs=[])),expected)

    def test_geometry_rule_not_four_mm_permission(self):
        for pair,rule,expected in [(['left_pad','table'],'ARM_TABLE_OR_FLOOR','table-pad'),
                                  (['link4','table'],'ARM_TABLE_OR_FLOOR','arm-table-other'),
                                  (['cube_geom','other'],'PENETRATION_STOP','penetration-other')]:
            self.assertEqual(geometry_category(dict(predicted_native_safety=['ILLEGAL_CONTACT_OR_PENETRATION'],
                illegal_pairs=[dict(geom_names=pair,original_rule=rule,signed_distance_m=-.00001)])),expected)

    def test_no_import_time_physics(self):
        for name in ['fd1','fd1_shadow']:
            module=importlib.import_module('scripts.pi0294.'+name)
            tree=ast.parse(Path(module.__file__).read_text())
            for node in tree.body:
                if isinstance(node,ast.Import):
                    self.assertFalse(any(a.name.startswith(('mujoco','torch','transformers')) for a in node.names))
                if isinstance(node,ast.ImportFrom):
                    self.assertNotIn(node.module,['scripts.pi0294.scene','scripts.pi0291_arx5_data.engine'])

    def test_transitive_readonly_imports_exist_without_engine(self):
        for name in ['scripts.iteration.rgb_cache','scripts.pi0294.e1.initial_history',
                     'scripts.pi0294.e1.collision']:
            self.assertIsNotNone(importlib.import_module(name))

    def test_exclusive_artifact_writes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'receipt.json'
            fd1.put(p,dict(old=True))
            with self.assertRaises(FileExistsError):
                fd1.put(p,dict(old=False))
            self.assertTrue(fd1.read(p)['old'])
            fd1.update(p,dict(new=True))
            self.assertEqual(fd1.read(p),dict(new=True))

    def test_identical_sealed_plan_reused_without_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'plan.json'
            fd1.seal_or_verify(p,dict(ids=['fixed']))
            before=p.stat().st_ino,p.read_bytes()
            fd1.seal_or_verify(p,dict(ids=['fixed']))
            self.assertEqual(before,(p.stat().st_ino,p.read_bytes()))
            with self.assertRaisesRegex(ValueError,'IDENTITY_CHANGED'):
                fd1.seal_or_verify(p,dict(ids=['replacement']))


if __name__=='__main__':
    unittest.main()
