import copy
import json
import tempfile
import unittest
from pathlib import Path
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.pilot2_contact_model import planned_openings
from go2arx5_vla.world_model.pilot2_supported import stage_for, GOALS
from go2arx5_vla.world_model.pilot2_static_contact_parallel import validate_single, validate_batch, SCHEMA, COMMISSION, PRODUCTION
from go2arx5_vla.world_model.target_parallel import recorded_split


class StaticContactParallel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.binary = self.root/'binary'; self.binary.write_bytes(b'SYNTHETIC_NATIVE_IDENTITY_NOT_NATIVE_QUALIFICATION')
        self.model = self.root/'model'; self.model.write_bytes(b'SYNTHETIC_SCENE')
        self.observer = self.root/'observer'; self.observer.write_bytes(b'SYNTHETIC_ORIGINAL_SAFETY')
        source = Path(__file__).resolve().parents[2]
        self.ep = json.loads((source/'experiments/wm02/pilot2_supported_collection_v1.json').read_text())
        self.ep.update(parent='SYNTHETIC_PARENT', scene_id='SYNTHETIC_SCENE', split='TRAIN',
            close_start_atom=56, command_program='CLOSE_KEEP', task_goal=GOALS[0])
        self.prior = self.root/'single'; (self.prior/'native-01').mkdir(parents=True); (self.prior/'control').mkdir()
        self.paths = dict(native=self.prior/'native-01/RESULT.json', episode=self.prior/'EPISODE_MANIFEST.private.json',
            score=self.prior/'native-01/SUPPORTED_CLOSE_RESULT.private.json', spec=self.prior/'EXECUTION_SPEC.json')
        self.old_spec = dict(source_sha='SYNTHETIC_SOURCE', contact_forecast_observer=dict(models=['SYNTHETIC_FROZEN_MODEL']))
        self.native = dict(status='NATIVE_STAGE_PASSED', source_sha='SYNTHETIC_SOURCE', shutdown_clean=True,
            physics_stops=[], actual_target_integrations=6703, native_binary_sha256=digest(self.binary),
            scene_sha256=digest(self.model), observer_sha256=digest(self.observer))
        self.score = dict(first_blocker=None, same_native_process=True, physics_resets=0, state_teleports=0,
            actual_task_atoms=168, goal_success=False, samples=[dict(native_start_step=2503+25*i,
            native_end_step=2528+25*i, stage=stage_for(i, self.ep), executed_reference_EEF10=[float(v)])
                for i, v in enumerate(planned_openings(self.ep))])
        self.spec = dict(native_binary=str(self.binary), model_path=str(self.model), observer_library=str(self.observer),
            contact_forecast_observer=copy.deepcopy(self.old_spec['contact_forecast_observer']))
        self.refresh()
    def tearDown(self): self.tmp.cleanup()
    def refresh(self):
        for k, v in [('native',self.native),('episode',self.ep),('score',self.score),('spec',self.old_spec)]:
            self.paths[k].write_text(json.dumps(v))
        self.spec['single_static_qualification'] = dict(path=str(self.paths['native']),
            **{k+'_sha256':digest(p) for k,p in self.paths.items()})
        (self.prior/'control/LEASE.json').write_text(json.dumps(dict(status='CLOSED', worker_exit_code=0,
            spec_sha256=digest(self.paths['spec']))))

    def test_only_complete_single_program_qualifies_without_success_filter(self):
        validate_single(self.spec,self.ep)
        for patch in [dict(close_start_atom=64),dict(parent='OTHER'),dict(task_goal=GOALS[1])]:
            with self.subTest(patch=patch), self.assertRaises(PermissionError):validate_single(self.spec,dict(self.ep,**patch))
        (self.prior/'control/LEASE.json').write_text(json.dumps(dict(status='ACTIVE')))
        with self.assertRaises(PermissionError):validate_single(self.spec,self.ep)

    def test_changed_evidence_or_forecast_observer_or_executor_cannot_qualify(self):
        self.paths['score'].write_text('{}')
        with self.assertRaises(PermissionError):validate_single(self.spec,self.ep)
        self.refresh();self.spec['contact_forecast_observer']['models']=['NEW_MODEL']
        with self.assertRaises(PermissionError):validate_single(self.spec,self.ep)
        self.spec['contact_forecast_observer']=copy.deepcopy(self.old_spec['contact_forecast_observer'])
        self.binary.write_bytes(b'OTHER_EXECUTOR')
        with self.assertRaises(PermissionError):validate_single(self.spec,self.ep)

    def test_interrupted_or_substituted_command_program_cannot_be_parallel_proof(self):
        self.score['samples'][80]['stage']='EARLY_RELEASE';self.refresh()
        with self.assertRaises(PermissionError):validate_single(self.spec,self.ep)
        self.score['samples'][80]['stage']=stage_for(80,self.ep)
        self.score['samples'][80]['executed_reference_EEF10']=[1.];self.refresh()
        with self.assertRaises(PermissionError):validate_single(self.spec,self.ep)
        self.score['samples'].pop();self.refresh()
        with self.assertRaises(PermissionError):validate_single(self.spec,self.ep)

    def test_batch_requires_actual_same_executor_duration_then_static_full_duration(self):
        prior=self.root/'dual';(prior/'batch-01').mkdir(parents=True);(prior/'control').mkdir()
        proof=prior/'batch-01/RESULT.json';proof.write_text('{}')
        batch=dict(native_binary=str(self.binary),observer_library=str(self.observer),pilot1_binding='SYNTHETIC',
            pilot1_binding_sha256='SYNTHETIC',pilot2_plan_sha='SYNTHETIC',purpose=COMMISSION)
        old=dict(batch,schema='WM02_PILOT2_TWO_COVERAGE_BATCH_V1',purpose='COVERAGE_DUAL_DURATION_COMMISSION',native_runs_root=str(self.root))
        def write_prior():
            (prior/'EXECUTION_SPEC.json').write_text(json.dumps(old))
            (prior/'control/LEASE.json').write_text(json.dumps(dict(status='CLOSED',worker_exit_code=0,
                spec_sha256=digest(prior/'EXECUTION_SPEC.json'))))
        write_prior()
        children=[];value=dict(children=[])
        for i,(program,goal) in enumerate(zip(['CLOSE_KEEP','CLOSE_RELEASE'],GOALS)):
            ep=self.root/f'episode{i}.json';ep.write_text(json.dumps(dict(self.ep,command_program=program,task_goal=goal)))
            children.append(dict(episode_manifest=str(ep)));r=self.root/f'child{i}';(r/'native-01').mkdir(parents=True)
            n=dict(self.native,actual_target_integrations=4103);(r/'native-01/RESULT.json').write_text(json.dumps(n));value['children'].append(dict(run=r.name,ticks=4103))
        validate_batch(batch,children,proof,value)
        self.assertEqual(recorded_split(dict(schema=SCHEMA,purpose=COMMISSION),{}),'QUALIFICATION')
        with self.assertRaises(PermissionError):validate_batch(dict(batch,purpose=PRODUCTION),children,proof,value)
        old.update(schema=SCHEMA,purpose=COMMISSION);write_prior()
        with self.assertRaises(PermissionError):validate_batch(dict(batch,purpose=PRODUCTION),children,proof,value)
        for row in value['children']:
            row['ticks']=6703;(self.root/row['run']/'native-01/RESULT.json').write_text(json.dumps(self.native))
        validate_batch(dict(batch,purpose=PRODUCTION),children,proof,value)
        self.observer.write_bytes(b'OTHER_SAFETY_WRAPPER')
        with self.assertRaises(PermissionError):validate_batch(dict(batch,purpose=PRODUCTION),children,proof,value)

    def test_layout_routes_static_batches_through_single_and_full_duration_checks(self):
        from unittest.mock import patch
        from go2arx5_vla.world_model.target_parallel import validate_layout
        runs=self.root/'runs';runs.mkdir();batch_root=runs/'batch';batch_root.mkdir();children=[];slots=[]
        previous=self.root/'previous';previous.write_text('{}')
        for i in range(2):
            childroot=runs/str(i);childroot.mkdir()
            child=dict(registered_root=str(childroot),parallel_batch_spec=str(batch_root/'EXECUTION_SPEC.json'),
                campaign_root=str(self.root),physical_owner_lock='SYNTHETIC_OWNER',source_sha='SYNTHETIC',
                ROS_DOMAIN_ID=220+i,cpus=list(range(i*4,i*4+4)),phase='collect-target-pilot2-supported',max_ticks=6703,
                pilot1_binding='SYNTHETIC_BINDING',pilot1_binding_sha256='SYNTHETIC_PIN',pilot2_plan_sha='SYNTHETIC_PLAN',
                previous_stage=str(previous),supported_static_parallel_schema=SCHEMA)
            sp=childroot/'SPEC.json';sp.write_text(json.dumps(child));children.append(child)
            slots.append(dict(spec=str(sp),spec_sha256=digest(sp)))
        proof=self.root/'proof';proof.write_text(json.dumps(dict(status='TWO_ENVIRONMENT_NATIVE_PILOT_PASSED',
            children=[dict(ticks=4103,shutdown_clean=True,exit_code=0)]*2)))
        batch=dict(schema=SCHEMA,purpose=COMMISSION,slots=slots,native_runs_root=str(runs),registered_root=str(batch_root),
            campaign_root=str(self.root),physical_owner_lock='SYNTHETIC_OWNER',source_sha='SYNTHETIC',cpus=list(range(8)),
            pilot1_binding='SYNTHETIC_BINDING',pilot1_binding_sha256='SYNTHETIC_PIN',pilot2_plan_sha='SYNTHETIC_PLAN',
            parallel_qualification=str(proof),parallel_qualification_sha256=digest(proof))
        # File-backed single and batch evidence is checked in the tests above;
        # here verify that the existing owner/layout path cannot skip either.
        with patch('go2arx5_vla.world_model.pilot2_supported.validate',return_value=6703) as single, \
                patch('go2arx5_vla.world_model.pilot2_static_contact_parallel.validate_batch') as prior:
            validate_layout(batch,children);self.assertEqual(single.call_count,2);prior.assert_called_once()
            prior.side_effect=PermissionError('NO_ACTUAL_STATIC_DURATION')
            with self.assertRaisesRegex(PermissionError,'NO_ACTUAL_STATIC_DURATION'):validate_layout(batch,children)
            prior.side_effect=None
            with self.assertRaises(PermissionError):validate_layout(batch,[children[0],dict(children[1],supported_static_parallel_schema='OTHER')])


if __name__ == '__main__': unittest.main()
