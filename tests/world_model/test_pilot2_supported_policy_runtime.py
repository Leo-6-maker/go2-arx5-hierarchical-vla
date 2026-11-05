import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from tests.world_model.test_pilot2_supported_decision import supported
from go2arx5_vla.world_model.pilot2_supported_policy_runtime import choose,contract,SCHEMA
from go2arx5_vla.world_model.pilot2_supported_policy_runtime import validate_single_qualification
from go2arx5_vla.world_model.storage import digest


class PausedSupportedPolicy(unittest.TestCase):
    def episode(self,goal='SUPPORTED_GRASP_HOLD',method='GOAL_FSM'):
        e=supported('DEV','Q','CLOSE_KEEP')['episode']
        return dict(e,schema=SCHEMA,purpose='SUPPORTED_POLICY_EVALUATION',
            command_program='SELECT_KEEP_RELEASE_AFTER_SUPPORTED_HOLD',policy_decision_atom=112,
            method=method,task_goal=goal)

    def test_goal_rule_chooses_before_integration_using_only_actual_prefix(self):
        for goal,program in [('SUPPORTED_GRASP_HOLD','CLOSE_KEEP'),('RELEASE_FROM_SUPPORTED_GRASP','CLOSE_RELEASE')]:
            with tempfile.TemporaryDirectory() as tmp:
                rows=supported('DEV','Q','CLOSE_KEEP')['samples'][:112];events=[]
                a=SimpleNamespace(output=Path(tmp),spec=dict(source_sha='SYNTHETIC_FIXTURE'),
                    observe=lambda:dict(step=5303),command_event=events.append)
                e=choose(a,self.episode(goal),rows);self.assertEqual(e['command_program'],program)
                d=json.loads((Path(tmp)/'SUPPORTED_POLICY_DECISION.private.json').read_text())
                self.assertTrue(d['decision_prefix_eligible']);self.assertFalse(d['future_response_observed'])
                self.assertEqual(events[0]['phase'],'supported-policy-choice-before-install')

    def test_failed_hold_prefix_opens_and_is_marked_ineligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=supported('DEV','Q','CLOSE_KEEP')['samples'][:112]
            rows[-1]['native_contact_events'][-1]['native_grasp']=False
            a=SimpleNamespace(output=Path(tmp),spec=dict(source_sha='SYNTHETIC_FIXTURE'),
                observe=lambda:dict(step=5303),command_event=lambda e:None)
            result=choose(a,self.episode(),rows);self.assertEqual(result['command_program'],'CLOSE_RELEASE')
            d=json.loads((Path(tmp)/'SUPPORTED_POLICY_DECISION.private.json').read_text())
            self.assertFalse(d['decision_prefix_eligible'])

    def test_unqualified_timing_and_unknown_policy_rejected(self):
        for change in [dict(close_start_atom=64),dict(policy_decision_atom=120),dict(method='UNQUALIFIED')]:
            with self.assertRaises(PermissionError):contract(dict(self.episode(),**change))

    def test_parallel_requires_matching_actual_single_not_a_task_success_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);native=root/'native';native.mkdir();e=self.episode()
            model=root/'model';model.write_bytes(b'SYNTHETIC_MODEL');binary=root/'binary';binary.write_bytes(b'SYNTHETIC_NATIVE')
            p=native/'RESULT.json';p.write_text(json.dumps(dict(status='NATIVE_STAGE_PASSED',shutdown_clean=True,
                physics_stops=[],actual_target_integrations=6703,scene_sha256=digest(model),native_binary_sha256=digest(binary))))
            ep=root/'EPISODE_MANIFEST.private.json';ep.write_text(json.dumps(e));decision=native/'SUPPORTED_POLICY_DECISION.private.json'
            d=dict(native_step=5303,future_response_observed=False,decision_prefix_eligible=True,
                policy_sha256=None,goal_success=False);decision.write_text(json.dumps(d))
            s=dict(model_path=str(model),native_binary=str(binary),single_policy_qualification=dict(
                path=str(p),sha256=digest(p),decision_sha256=digest(decision),episode_sha256=digest(ep)))
            validate_single_qualification(s,e)
            with self.assertRaises(PermissionError):validate_single_qualification(s,dict(e,method='CONSTANT_KEEP'))
            d['decision_prefix_eligible']=False;decision.write_text(json.dumps(d));s['single_policy_qualification']['decision_sha256']=digest(decision)
            with self.assertRaises(PermissionError):validate_single_qualification(s,e)

    def test_supported_two_slot_commission_and_full_duration_production_are_separate(self):
        from unittest.mock import patch
        from go2arx5_vla.world_model.target_parallel import validate_layout
        from go2arx5_vla.world_model.pilot2_supported import GOALS
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runs=root/'runs';runs.mkdir();batchroot=runs/'batch';batchroot.mkdir();children=[];slots=[]
            for i,goal in enumerate(GOALS):
                childroot=runs/str(i);childroot.mkdir();ep=childroot/'EPISODE.json'
                e=dict(self.episode(goal),parent='P',split='TRAIN');ep.write_text(json.dumps(e))
                previous=root/('prior'+str(i));previous.write_text('{}')
                child=dict(registered_root=str(childroot),parallel_batch_spec=str(batchroot/'EXECUTION_SPEC.json'),
                    campaign_root=str(root),physical_owner_lock='SYNTHETIC_OWNER_PATH',source_sha='SYNTHETIC',
                    ROS_DOMAIN_ID=220+i,cpus=list(range(i*4,i*4+4)),phase='collect-target-pilot2-supported',max_ticks=6703,
                    pilot1_binding='SYNTHETIC_BINDING',pilot1_binding_sha256='SYNTHETIC_SHA',
                    previous_stage=str(previous),episode_manifest=str(ep),pilot2_plan_sha='SYNTHETIC_PLAN',
                    supported_policy_parallel_schema='WM02_PILOT2_TWO_SUPPORTED_POLICY_BATCH_V1')
                sp=childroot/'SPEC.json';sp.write_text(json.dumps(child));children.append(child);slots.append(dict(spec=str(sp),spec_sha256=digest(sp)))
            proof=root/'dual.json';v=dict(status='TWO_ENVIRONMENT_NATIVE_PILOT_PASSED',children=[dict(ticks=4103,shutdown_clean=True,exit_code=0)]*2);proof.write_text(json.dumps(v))
            b=dict(schema='WM02_PILOT2_TWO_SUPPORTED_POLICY_BATCH_V1',purpose='SUPPORTED_POLICY_DUAL_COMMISSION',
                slots=slots,native_runs_root=str(runs),registered_root=str(batchroot),campaign_root=str(root),
                physical_owner_lock='SYNTHETIC_OWNER_PATH',source_sha='SYNTHETIC',cpus=list(range(8)),
                pilot1_binding='SYNTHETIC_BINDING',pilot1_binding_sha256='SYNTHETIC_SHA',pilot2_plan_sha='SYNTHETIC_PLAN',
                parallel_qualification=str(proof),parallel_qualification_sha256=digest(proof))
            with patch('go2arx5_vla.world_model.pilot2_supported.validate',return_value=6703):
                validate_layout(b,children);b['purpose']='SUPPORTED_POLICY_PRODUCTION'
                with self.assertRaises(PermissionError):validate_layout(b,children)
                for c in v['children']:c['ticks']=6703
                proof.write_text(json.dumps(v));b['parallel_qualification_sha256']=digest(proof);validate_layout(b,children)


if __name__=='__main__':unittest.main()
