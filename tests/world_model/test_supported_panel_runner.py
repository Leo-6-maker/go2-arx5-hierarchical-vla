"""Controller seam only: fake native workers never qualify actual physics."""
import json,tempfile,unittest,sys,time
from pathlib import Path
from unittest.mock import patch
from go2arx5_vla.world_model.pilot2_contact_evaluation import panel
from go2arx5_vla.world_model.storage import digest


class SupportedPanelRunner(unittest.TestCase):
    def execute(self,fail_first=False):
        from scripts.wm02.run_pilot2_supported_panel import run
        from go2arx5_vla.world_model.pilot2_supported_policy_runtime import validate_single_qualification
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);campaign=root/'campaign';campaign.mkdir();stage=root/'panel';stage.mkdir();(stage/'control').mkdir()
            runs=root/'runs';runs.mkdir();source=root/'source';(source/'experiments/wm02').mkdir(parents=True)
            recipe=Path(__file__).parents[2]/'experiments/wm02/pilot2_supported_collection_v1.json'
            (source/'experiments/wm02/pilot2_supported_collection_v1.json').write_bytes(recipe.read_bytes())
            registry=root/'registry.json';registry.write_text('{"entries":[]}');binary=root/'native';binary.write_bytes(b'SOFTWARE_TEST_NOT_NATIVE')
            scenes=[]
            for i in range(4):
                model=root/('scene'+str(i));model.write_bytes(b'SOFTWARE_TEST_NOT_NATIVE')
                scenes.append(dict(parent='P'+str(i),scene_id='S'+str(i),split='DEV',model_path=str(model),model_sha256=digest(model)))
            assets=root/'assets.json';assets.write_text(json.dumps(dict(scenes=scenes)))
            variants=[dict(name=x,method=x)for x in ['GOAL_FSM','CONSTANT_KEEP','CONSTANT_RELEASE']]
            for x in ['FSM_STUDENT','WM_STUDENT']:
                variants.append(dict(name=x,method='TRAINED_GOAL_POLICY',supported_policy_numpy=x,
                    supported_policy_numpy_sha256=x,supported_policy_training_result=x,supported_policy_training_result_sha256=x))
            plan=panel(scenes,variants,seed=363701);planpath=stage/'PANEL.json';planpath.write_text(json.dumps(plan))
            qualification=root/'qualified.json';qualification.write_text('{}')
            base=dict(campaign_root=str(campaign),physical_owner_lock=str(root/'physical.lock'),registry=str(registry),
                host='SYNTHETIC',owner='SYNTHETIC',pilot1_binding='SYNTHETIC',native_environment={},python=sys.executable,
                scene_allowlist=str(assets),scene_allowlist_sha256=digest(assets),model_path=scenes[0]['model_path'],
                frozen_files={},native_binary=str(binary),pilot1_binding_sha256='SYNTHETIC')
            basepath=stage/'BASE.json';basepath.write_text(json.dumps(base));inp=dict(panel=str(planpath),panel_sha256=digest(planpath),
                base_spec=str(basepath),source_root=str(source),source_files={},source_sha='SYNTHETIC_SOFTWARE_TEST_NOT_NATIVE',
                output_parent=str(runs),variants=variants,scene_qualifications={s['scene_id']:dict(path=str(qualification),sha256=digest(qualification))for s in scenes},
                program_qualifications={s['scene_id']:[]for s in scenes},pilot2_plan_sha='SYNTHETIC',storage_proof='SYNTHETIC',storage_proof_sha256='SYNTHETIC',
                native_binary_sha256=digest(binary),parallel_qualification='SYNTHETIC',parallel_qualification_sha256='SYNTHETIC',deadline_epoch=time.time()+120)
            inputpath=stage/'INPUT.json';inputpath.write_text(json.dumps(inp));lease=stage/'control/LEASE.json';lease.write_text(json.dumps(dict(status='ACTIVE',run_uuid=stage.name,input_sha256=digest(inputpath))))
            calls=[]
            def snapshot(path):return json.loads(Path(path).read_text()),digest(path)
            def once(path,value,*args,**kwargs):Path(path).write_text(json.dumps(value))
            def produce(specpath,failed=False):
                s=json.loads(Path(specpath).read_text());rr=Path(s['registered_root']);e=json.loads(Path(s['episode_manifest']).read_text());n=rr/'native-01';n.mkdir()
                if 'parallel_batch_spec'in s:validate_single_qualification(s,e)
                goal=int(e['task_goal']=='RELEASE_FROM_SUPPORTED_GRASP');choice=goal
                if e['method']=='CONSTANT_KEEP':choice=0
                if e['method']=='CONSTANT_RELEASE':choice=1
                native=dict(status='NATIVE_STAGE_FAILED'if failed else'NATIVE_STAGE_PASSED',shutdown_clean=True,physics_stops=[],
                    actual_target_integrations=50 if failed else 6703,native_binary_sha256=digest(binary),scene_sha256=digest(s['model_path']))
                decision=dict(native_step=5303,future_response_observed=False,method=e['method'],task_goal=e['task_goal'],
                    policy_sha256=s.get('supported_policy_numpy_sha256'),decision_prefix_eligible=True)
                score=dict(goal_success=not failed and choice==goal)
                for name,value in [('RESULT.json',native),('SUPPORTED_POLICY_DECISION.private.json',decision),('SUPPORTED_CLOSE_RESULT.private.json',score)]:
                    (n/name).write_text(json.dumps(value))
                lp=Path(s['lease']);v=json.loads(lp.read_text());v['status']='CLOSED';lp.write_text(json.dumps(v))
            class FakePopen:
                def __init__(self,argv,**kwargs):
                    specpath=argv[-1];s=json.loads(Path(specpath).read_text());calls.append(specpath);self.pid=12345
                    failed=fail_first and len(calls)==1;self.returncode=int(failed)
                    if 'slots'in s:
                        for child in s['slots']:produce(child['spec'])
                    else:produce(specpath,failed)
                def poll(self):return self.returncode
            with patch('scripts.pi0294.night10.native_relay.snapshot',snapshot),patch('scripts.pi0294.night10.native_relay.once',once),\
                    patch('go2arx5_vla.world_model.pilot2_supported.validate'),patch('go2arx5_vla.world_model.target_parallel.validate_layout'),\
                    patch('subprocess.Popen',FakePopen):
                code=run(inputpath)
            result=json.loads((stage/'SUMMARY.private.json').read_text());closed=json.loads(lease.read_text());return code,result,closed,len(calls)

    def test_full_fixture_keeps_constant_action_failures_and_closes_actual_parent_role(self):
        code,r,lease,calls=self.execute();self.assertEqual(code,0);self.assertEqual(calls,60)
        self.assertEqual(r['attempted'],80);self.assertEqual(r['not_run'],0);self.assertEqual(lease['status'],'CLOSED')
        self.assertEqual(r['variants']['CONSTANT_KEEP']['strict_successes'],8)
        self.assertEqual(r['variants']['CONSTANT_RELEASE']['strict_successes'],8)
        self.assertEqual(r['variants']['GOAL_FSM']['strict_successes'],16)
        self.assertFalse(r['CONTACT_CONTROL_GAIN'])

    def test_native_failure_stops_and_preserves_not_run_denominator(self):
        code,r,lease,calls=self.execute(True);self.assertEqual(code,1);self.assertEqual(calls,1)
        self.assertEqual(r['attempted'],1);self.assertEqual(r['native_failed'],1);self.assertEqual(r['not_run'],79)
        self.assertEqual(lease['status'],'CLOSED');self.assertEqual(lease['worker_exit_code'],1)


if __name__=='__main__':unittest.main()
