from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from .mp1 import anchor_dynamic as dynamic
from .mp1 import anchor_dynamic_config as config
from .native_dataset import sha
from .test_mp1_n1_limits import N1LimitsTests


class AnchorDynamicTests(unittest.TestCase):
    def setUp(self):
        self.fixture=N1LimitsTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.parent=self.fixture.root.parent.parent;self.root=self.fixture.root.parent/config.CHILD
        self.root.mkdir();self.stage=dict(schema=config.SCHEMA,diagnostic_created_epoch=1792264091,
            diagnostic_deadline_epoch=1792267691,diagnostic_CPU_ids=[3])
        self.put('MP1_STAGE_BINDING.json',self.stage)
        repo=Path(__file__).resolve().parents[2]
        self.card=json.loads((repo/'reports/PI0294/design/N1_ANCHOR_DYNAMIC_PARITY_V1.json').read_text())
        self.put(config.CARD,self.card,True);self.put(config.INPUTS,{},True)
        digest=sha(self.parent/config.n1.FIRST_ADOPTION)
        self.manifest=dict(execution=dict(anchor_dynamic_n1_binding=dict(release=config.n1.RELEASE,
            adoption_delta_sha256=digest),sealed_inputs={p:sha(self.root/p) for p in [config.CARD,config.INPUTS]}))
        self.clock=1792264092
        self.delta_patch=patch.object(config.n1,'ADOPTED_DELTA_SHA',digest);self.delta_patch.start();self.addCleanup(self.delta_patch.stop)
        self.env=patch.dict(os.environ,CUDA_VISIBLE_DEVICES='');self.env.start();self.addCleanup(self.env.stop)

    def put(self,relative,value,immutable=False):
        p=self.root/relative;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():p.chmod(0o600)
        p.write_text(json.dumps(value)+'\n')
        if immutable:p.chmod(0o400)

    def reseal_card(self):
        self.put(config.CARD,self.card,True)
        self.manifest['execution']['sealed_inputs'][config.CARD]=sha(self.root/config.CARD)

    def test_separately_bound_cpu_simulation_window_keeps_original_first_adoption(self):
        value=config.validate_window(self.root,self.manifest,self.clock)
        self.assertEqual(value['CPU_ids'],[3]);self.assertEqual(value['N1_T0_epoch'],1792264081)
        self.assertEqual(value['deadline_epoch'],self.stage['diagnostic_deadline_epoch'])
        self.assertEqual(config.CAPS['live_and_recovery_integrations'],0)
        self.assertEqual(config.CAPS['generation_queries'],0)

    def test_card_cannot_expand_shadow_neural_or_safety_threshold(self):
        original=deepcopy(self.card)
        for field,value in [('comparison_absolute_tolerance',1e-9),('maximum_new_GPU_PBS',1),
                            ('Teacher_queries_or_new_positive_labels',True)]:
            self.card=deepcopy(original);self.card[field]=value;self.reseal_card()
            with self.subTest(field=field),self.assertRaises(PermissionError):
                config.validate_window(self.root,self.manifest,self.clock)
        self.card=deepcopy(original);self.card['caps']['generation_queries']=1;self.reseal_card()
        with self.assertRaises(PermissionError):config.validate_window(self.root,self.manifest,self.clock)

    def test_scope_cannot_extend_window_use_cuda_or_multiple_cpus(self):
        self.stage['diagnostic_deadline_epoch']+=1;self.put('MP1_STAGE_BINDING.json',self.stage)
        with self.assertRaises(PermissionError):config.validate_window(self.root,self.manifest,self.clock)
        self.stage['diagnostic_deadline_epoch']-=1;self.stage['diagnostic_CPU_ids']=[3,4]
        self.put('MP1_STAGE_BINDING.json',self.stage)
        with self.assertRaises(PermissionError):config.validate_window(self.root,self.manifest,self.clock)
        self.stage['diagnostic_CPU_ids']=[3];self.put('MP1_STAGE_BINDING.json',self.stage)
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0'),self.assertRaises(PermissionError):
            config.validate_window(self.root,self.manifest,self.clock)

    def test_existing_original_parity_tolerance_and_finite_shape_are_strict(self):
        self.assertEqual(dynamic.compare_trace(np.zeros((2,3)),np.zeros((2,3)),'same'),0)
        for bad in [np.ones((2,3))*2e-10,np.zeros((1,3)),np.full((2,3),np.nan)]:
            with self.subTest(shape=bad.shape),self.assertRaises(ValueError):
                dynamic.compare_trace(bad,np.zeros((2,3)),'bad')
        self.assertEqual(dynamic.TOLERANCE,1e-10)

    def test_mismatch_records_exact_maximum_and_first_failed_coordinate(self):
        actual=np.zeros((3,4));expected=actual.copy();actual[0,2]=2e-10;actual[2,3]=-5e-9
        with self.assertRaises(dynamic.TraceMismatch) as raised:dynamic.compare_trace(actual,expected,'native')
        d=raised.exception.mismatch
        self.assertEqual((d['maximum_at_row'],d['maximum_at_column']),(2,3))
        self.assertEqual((d['first_outside_tolerance_row'],d['first_outside_tolerance_column']),(0,2))
        self.assertEqual(d['actual_at_maximum'],-5e-9);self.assertEqual(d['expected_at_maximum'],0)
        self.assertEqual(d['absolute_tolerance'],1e-10)

    def test_preserved_arrays_are_finite_hashable_immutable_and_not_overwritten(self):
        p=self.root/'trace.npz';value=np.array([[1.,2.],[3.,4.]])
        dynamic.save_arrays(p,observed_rows=value)
        with np.load(p,allow_pickle=False) as z:np.testing.assert_array_equal(z['observed_rows'],value)
        self.assertEqual(p.stat().st_mode&0o222,0);self.assertEqual(len(sha(p)),64)
        with self.assertRaises(FileExistsError):dynamic.save_arrays(p,observed_rows=value)

    def test_trace_child_is_one_exact_failed_anchor_not_full_gate_or_eight_replay(self):
        old=self.root;self.root=old.parent/config.TRACE_CHILD;old.rename(self.root)
        self.stage.update(schema=config.TRACE_SCHEMA,diagnostic_deadline_epoch=self.stage['diagnostic_created_epoch']+600)
        self.put('MP1_STAGE_BINDING.json',self.stage)
        repo=Path(__file__).resolve().parents[2]
        self.card=json.loads((repo/'reports/PI0294/design/N1_ANCHOR_FAILURE_TRACE_V1.json').read_text())
        self.put(config.TRACE_CARD,self.card,True)
        self.manifest['execution']['sealed_inputs']={p:sha(self.root/p) for p in [config.TRACE_CARD,config.INPUTS]}
        value=config.validate_window(self.root,self.manifest,self.clock)
        spec=config.specification(self.stage)
        self.assertEqual(value['deadline_epoch'],self.stage['diagnostic_created_epoch']+600)
        self.assertEqual(spec['caps']['main_starts'],1);self.assertEqual(spec['caps']['shadow_integrations'],200)
        self.assertEqual(spec['CPU_core_s']['core'],600);self.assertTrue(spec['trace_only'])
        for name,value in [('only_state',config.FAILED_STATE+'-OTHER'),('single_state_pass_can_qualify_all_eight',True)]:
            original=self.card[name];self.card[name]=value;self.put(config.TRACE_CARD,self.card,True)
            self.manifest['execution']['sealed_inputs'][config.TRACE_CARD]=sha(self.root/config.TRACE_CARD)
            with self.subTest(name=name),self.assertRaises(PermissionError):config.validate_window(self.root,self.manifest,self.clock)
            self.card[name]=original
        with self.assertRaises(PermissionError):config.specification(dict(schema='UNBOUND_NEXT_SCOPE'))

    def test_mujoco_state_api_projects_saved_live_state_without_integration(self):
        calls=[]
        class Data:
            def __init__(self,model):self.ctrl=np.array([4.]);self.xfrc_applied=np.zeros((1,6))
        def set_state(model,data,state,kind):
            calls.append(state.copy());data.time=state[0];data.qpos=state[1:2];data.qvel=state[2:3]
        mj=SimpleNamespace(mjtState=SimpleNamespace(mjSTATE_INTEGRATION=7),
            mj_stateSize=lambda *a:3,MjData=Data,mj_setState=set_state)
        engine=SimpleNamespace(model=object(),ids=dict(tool=0))
        rows=np.array([[1.,2.,3.,99.,98.,97.,0.,200.]])
        result=dynamic.project_observed_to_highrate(engine,rows,mj=mj)
        np.testing.assert_array_equal(result,np.array([[1.,2.,3.,4.,0.,0.,0.,0.,0.,0.,0.]]))
        self.assertEqual(len(calls),1);np.testing.assert_array_equal(calls[0],rows[0,:3])
        bad=rows.copy();bad[0,-2]=2
        with self.assertRaises(ValueError):dynamic.project_observed_to_highrate(engine,bad,mj=mj)

    def test_input_hash_path_and_original_source_cpu_pool_are_not_bypassed(self):
        p=self.root/'owned';p.write_text('owned');digest=sha(p)
        self.assertEqual(dynamic.checked_member(self.root,'owned',digest),p)
        for relative,digest_value in [('../other',digest),('owned','0'*64)]:
            with self.subTest(relative=relative),self.assertRaises(ValueError):
                dynamic.checked_member(self.root,relative,digest_value)
        (self.root/'linked').symlink_to(p)
        with self.assertRaises(ValueError):dynamic.checked_member(self.root,'linked',digest)
        records=self.root/'worker_processes';records.mkdir()
        proc_start=Path('/proc/self/stat').read_text().split()[21]
        (records/'active.json').write_text(json.dumps(dict(pid=os.getpid(),proc_start=proc_start)))
        with patch.object(dynamic.os,'sched_getaffinity',side_effect=lambda pid:{0,1,2,3} if pid==0 else {0,1,2}):
            dynamic.require_disjoint_cpu(self.root,[3])
            with self.assertRaises(PermissionError):dynamic.require_disjoint_cpu(self.root,[1])


if __name__=='__main__':unittest.main()
