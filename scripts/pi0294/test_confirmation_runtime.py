from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock

import numpy as np

from scripts.pi0292.durable import write
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.mp1 import confirmation_config as c,confirmation_static as s
from scripts.pi0294.mp1 import core


class ConfirmationRuntimeTests(unittest.TestCase):
    def test_exact_card_and_delegation_do_not_add_budget(self):
        repo=Path(__file__).resolve().parents[2]
        p=repo/'reports/PI0294/design/SOURCE_CONFIRM_V1_FINITE_SCOPE.json'
        self.assertEqual(sha(p),c.CARD_SHA)
        v=json.loads(p.read_text());self.assertEqual(c.CAPS['main_starts'],v['maximum_all_engine_starts'])
        self.assertEqual(c.CAPS['generation_queries'],6400);self.assertEqual(c.CAPS['optimizer_updates'],0)
        self.assertLessEqual(42833+57600+21600,129600);self.assertEqual(10+v['maximum_new_Hopper_PBS'],12)
        self.assertFalse(v['confirmation_to_training']);self.assertFalse(v['WBC_or_S1_changes'])

    def test_source_native_history_must_be_closed_exact_source10(self):
        base=dict(GPU_allocated_s=42833,PBS_submissions=[{}]*10,generation_queries=2406,optimizer_updates=4096,
            GPU_reserved_s=0,pending_updates=[])
        with patch.object(c,'native_history') as native:
            c.checked_prior(base,[]);native.assert_called_once()
            for key,value in [('GPU_allocated_s',0),('generation_queries',2390),('optimizer_updates',0),
                    ('GPU_reserved_s',1),('pending_updates',[{}]),('PBS_submissions',[{}]*9)]:
                with self.subTest(key=key):
                    changed=dict(base,**{key:value})
                    with self.assertRaises(PermissionError):c.checked_prior(changed,[])

    def test_confirmation_native_request_rejects_extra_slot_or_spend(self):
        request=dict(submit_count=1,walltime_s=28800,mode=c.MODE,additional_output_upper_bytes=8*1024**3,prior_GPU_allocated_s=0)
        old=dict(optimizer_updates=0,pending_updates=[],generation_queries=0)
        bound=dict(source=dict(GPU_s=129600,PBS=12,generation_queries=12000))
        with patch.object(c,'authority',return_value=bound),patch.object(c,'read',return_value=old), \
                patch.object(c,'validate_plan'),patch.object(c,'prior_closure'):
            c.request_contract(Path('/not-used'),request,dict(GPU_reserved_s=28800))
            for key,value in [('submit_count',3),('walltime_s',28801),('additional_output_upper_bytes',1),
                    ('mode','MP1_FIXED_SOURCE_PAIR'),('prior_GPU_allocated_s',40000)]:
                with self.subTest(key=key):
                    with self.assertRaises(PermissionError):c.request_contract(Path('/not-used'),dict(request,**{key:value}),dict(GPU_reserved_s=28800))

    def test_prior_closure_never_resets_or_replays_unknown_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);relative=c.closure_path(1)
            value=dict(schema='MP1_CONFIRM_PRIOR_ALLOCATION_EFFECT_CLOSURE_V1',run_uuid=root.name,prior_allocations=0,
                prior_GPU_s=0,prior_generation_queries=0,all_recorded_processes_terminal=True,unknown_effects=[],
                core=dict(status='PASS_ALL_CLOSED_CORE_EFFECTS_RECORDED',rows=[]),
                hopper=dict(status='PASS_ALL_NATIVE_DRAWS_CLASSIFIED_AND_ARRAYS_VERIFIED',cumulative_generation_queries=0),
                rows=[],new_physics=0,new_queries=0)
            p=root/relative;p.parent.mkdir();write(p,value);p.chmod(0o400)
            request=dict(submit_count=1,prior_GPU_allocated_s=0,sealed_inputs={relative:sha(p)},
                confirmation_prior_closure_relative=relative,confirmation_prior_closure_sha256=sha(p))
            c.prior_closure(root,request,dict(generation_queries=0))
            for key,x in [('unknown_effects',['unresolved']),('all_recorded_processes_terminal',False),('new_queries',1)]:
                with self.subTest(key=key):
                    with patch.object(c,'read',return_value=dict(value,**{key:x})):
                        with self.assertRaises(PermissionError):c.prior_closure(root,request,dict(generation_queries=0))

    def test_static_inspection_saves_actual_pairs_but_never_integrates(self):
        group=c.plan.original_group('F0',0,0)
        result=dict(status='STATIC_PASS_PENDING_LIVE_CONTACT',integration_steps=0)
        engine=Mock();engine.steps=0;engine.data.time=0.;engine.xml='<original />'
        engine.render.return_value=({k:np.zeros((256,256,3),dtype=np.uint8) for k in ['body_rgb','wrist_rgb']},{})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ledger=Mock()
            with patch.object(s,'read',return_value=dict(asset_root='fixture-only')),patch.object(s,'make_engine',return_value=engine), \
                    patch.object(s,'qualify_engine',return_value=dict(result)):
                value=s.inspect(root,group,'T1',ledger,root/'out',lambda:None)
            ledger.charge.assert_called_once_with('main_starts',1);engine.step.assert_not_called();engine.close.assert_called_once()
            self.assertEqual(set(value['actual_static_images']),{'body_rgb','wrist_rgb'})
            self.assertFalse(value['Teacher_success_rollout']);self.assertFalse(value['new_positive_labels'])
            for row in value['actual_static_images'].values():self.assertEqual(sha(root/row['path']),row['sha256'])

    def test_core_cannot_load_candidate_or_reselect_after_policy_open(self):
        with patch.object(c,'read',return_value=dict(execution=dict(profile='4080-simulation'))):
            with self.assertRaises(PermissionError):c.candidate_checkpoint(Path('/not-used'))
        with patch.object(s,'validate_runtime',return_value=dict(execution=dict(confirmation_phase='POLICY_FROZEN'))):
            with self.assertRaises(PermissionError):s.run(Path('/not-used'))

    def test_policy_core_rejects_static_only_phase_before_any_resource_or_engine(self):
        with patch.object(core,'validate_runtime',return_value={}),patch.object(core,'read',return_value=dict(schema=c.SCHEMA)), \
                patch.object(c,'validate_plan',side_effect=PermissionError('STATIC_ONLY_NOT_POLICY')),patch.object(core,'Ledger') as ledger:
            with self.assertRaisesRegex(PermissionError,'STATIC_ONLY_NOT_POLICY'):
                core.run(Path('/not-used'),'not-opened.json')
            ledger.assert_not_called()


if __name__=='__main__':unittest.main()
