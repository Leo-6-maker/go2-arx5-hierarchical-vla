"""V7 finite-law and source gates; execution records here are synthetic."""
import copy,hashlib,json,subprocess,sys,os,shutil
from pathlib import Path
import numpy as np
import pytest

from scripts.pi0292.eval_contract import matrix10
from go2arx5_vla.world_model import fusion_task_phase_pose_continuation_v7 as phase
from go2arx5_vla.world_model import fusion_task_phase_pose_continuation_contract_v7 as contract
from go2arx5_vla.world_model import fusion_task_phase_pose_continuation_native_v7 as native
from go2arx5_vla.world_model import fusion_task_phase_pose_feedback_v6 as v6
from go2arx5_vla.world_model import fusion_task_phase_initialization_native as original_native
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_native_v4 as n4
from go2arx5_vla.world_model import fusion_trajectory_qualification_native as common_native
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_evidence as evidence
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model import auto1_native,target_sequential
from scripts.wm02 import fusion_task_phase_pose_continuation_check_v7 as cli
from test_fusion_task_phase_initialization import setup as pure_setup,samples
from test_fusion_task_phase_pose_feedback_v6 import recipe as v6_recipe,completed_v6
from test_fusion_task_phase_xyz_feedback_continuation_v5 import QualifiedAdapter
from test_fusion_live_proposal_qualification_contract import prior_files,put,close
import test_fusion_live_proposal_qualification_contract as prior_fixture
from test_fusion_reference_qualification_contract import reference
from test_fusion_trajectory_qualification import setup_flow


@pytest.fixture
def setup(monkeypatch):return pure_setup.__wrapped__(monkeypatch)


def make(setup,grade):
 e=phase.make_episode(run='/fixture/V7',geometry_scale=grade)
 return phase.make_program(setup['inputs'],e,setup['initial'],setup['decision'],setup['base']),e


@pytest.mark.parametrize('grade,count,end',[(.5,16,34303),(1.,24,46303)])
def test_finite_law_original_goal_and_strict_endpoints(setup,grade,count,end):
 p,e=make(setup,grade);assert phase.endpoint(e)==end and len(phase.feedback_steps(e))==count
 old=v6.original.make_program(setup['inputs'],phase.parent_recipe_view(e),setup['initial'],setup['decision'],setup['base'])
 np.testing.assert_array_equal(p.desired_path[:e['ramp_atoms']+1],old.desired_path)
 desired=p.desired_path.copy()
 for i,step in enumerate(phase.feedback_steps(e)):
  actual=p.graded_goal_TCP_target.copy();actual[:3,3]-=[.05,0.,0.]
  actual[:3,:3]=v6._exp_so3([0.,.06,0.])@actual[:3,:3]
  u=p.prepare_feedback(step,actual)
  assert np.isclose(u.evidence['applied_delta_norm_m'],.008)
  assert np.isclose(u.evidence['applied_rotation_norm_rad'],.01)
  assert u.evidence['peak_translation_speed_m_s']<=.05
  assert u.evidence['peak_translation_acceleration_m_s2']<=.25
  p.commit_feedback(u,saved_decision_sha256=hashlib.sha256(str(i).encode()).hexdigest())
  phase.validate_install_preview(p.preview((step-3703)//25),initial_tcp=setup['initial'],program=p,
      step=step,base=setup['base'],frozen_base=setup['base'],end_step=end)
 assert np.isclose(p.evidence()['cumulative_feedback_reference_path_length_m'],count*.008)
 np.testing.assert_array_equal(p.desired_path,desired)
 with pytest.raises(ValueError):p.prepare_feedback(end,actual)
 rows=samples(setup,grade)
 for row,step in zip(rows,range(end-200,end+1,25)):row.update(native_step=step,time_s=step*.002)
 assert phase.query_readiness(rows,e,setup['inputs'])['actual_sample_steps']==list(range(end-200,end+1,25))


@pytest.mark.parametrize('grade',[.25,.03,True,.75])
def test_no_repeat_completed_grade_or_unregistered_amplitude(grade):
 with pytest.raises(ValueError):phase.make_episode(run='/fixture/no',geometry_scale=grade)


def test_v6_default_math_and_all_revisions_bitwise_equal_pinned_original(setup):
 root=Path(__file__).resolve().parents[2]
 raw=subprocess.check_output(['git','show','da177fe856fb184c50c9d7763132ec626ecfc172:src/go2arx5_vla/world_model/fusion_task_phase_pose_feedback_v6.py'],cwd=root)
 scope=dict(__name__='go2arx5_vla.world_model._pinned_v6',__package__='go2arx5_vla.world_model')
 exec(compile(raw,'pinned-v6','exec'),scope)
 e=v6.make_episode(run='/fixture/parity',geometry_scale=.25)
 a=v6.make_program(setup['inputs'],e,setup['initial'],setup['decision'],setup['base'])
 b=scope['make_program'](setup['inputs'],e,setup['initial'],setup['decision'],setup['base'])
 for i,step in enumerate(v6.feedback_steps(e)):
  actual=a.graded_goal_TCP_target.copy();actual[:3,3]-=[.025,.01,.01]
  actual[:3,:3]=v6._exp_so3([.02,.03,-.01])@actual[:3,:3]
  x,y=a.prepare_feedback(step,actual),b.prepare_feedback(step,actual)
  assert x.evidence==y.evidence;np.testing.assert_array_equal(x.reference_path,y.reference_path)
  for p,u in ((a,x),(b,y)):p.commit_feedback(u,saved_decision_sha256=hashlib.sha256(str(i).encode()).hexdigest())
  assert a.evidence()==b.evidence()


@pytest.fixture
def recipe(tmp_path,monkeypatch):
 spec,_,_,setup,_=v6_recipe.__wrapped__(tmp_path,monkeypatch)
 root=Path(__file__).resolve().parents[2];capsule=Path(spec['source_root'])
 for name in contract.SOURCE_MEMBERS:
  dest=capsule/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((root/name).read_bytes());spec['source_files'][name]=digest(dest)
 prior,result,path,pin,specpin,_=completed_v6(tmp_path/'SYNTHETIC_QUALIFIED_V6',spec,setup,.25,monkeypatch)
 monkeypatch.setattr(contract,'FIRST_PREDECESSOR_SOURCE',prior['source_sha'])
 monkeypatch.setattr(contract,'FIRST_PREDECESSOR_SPEC',specpin);monkeypatch.setattr(contract,'FIRST_PREDECESSOR',pin)
 spec.update(phase=phase.PHASE,kind=contract.KIND,previous_stage=str(path),previous_stage_sha256=pin,max_ticks=34303)
 e=phase.make_episode(run=spec['registered_root'],geometry_scale=.5)
 spec['episode_manifest_sha256']=put(spec['episode_manifest'],e)
 return spec,result,e,setup


def test_qualified_predecessor_gate_does_not_grant_execution(recipe,tmp_path):
 spec,previous,_,_=recipe;assert contract.episode_limit(spec,previous)==34303
 path=tmp_path/'SPEC7.json';put(path,spec);assert cli.check(path)['native_execution_authorized']is False
 assert auto1_native.phase_limit(phase.PHASE,previous,spec)==34303
 assert target_sequential.native_task_bucket(phase.PHASE,True)=='96'


@pytest.mark.parametrize('fault',['false-position','wrong-source','skip-grade','missing-cache-source'])
def test_failed_or_mismatched_baseline_cannot_advance(recipe,monkeypatch,fault):
 spec,previous,e,_=recipe
 if fault=='false-position':previous['reference_qualification']['task_phase_initialization']['readiness']['predicates']['position']=False
 elif fault=='wrong-source':monkeypatch.setattr(contract,'FIRST_PREDECESSOR_SOURCE','0'*40)
 elif fault=='missing-cache-source':spec['source_files'].pop('src/go2arx5_vla/world_model/planner_event_log.py')
 else:
  e=phase.make_episode(run=spec['registered_root'],geometry_scale=1.)
  spec['episode_manifest_sha256']=put(spec['episode_manifest'],e);spec['max_ticks']=46303
 if fault=='false-position':
  spec['previous_stage_sha256']=put(spec['previous_stage'],previous);monkeypatch.setattr(contract,'FIRST_PREDECESSOR',spec['previous_stage_sha256'])
 with pytest.raises((ValueError,PermissionError,KeyError)):contract.episode_limit(spec,previous)


class Adapter(QualifiedAdapter):
 def install_reference(self,action):
  native.validate_adapter_install(self,action);value=reference(round(self.step*.002*1e9)/1e9)
  wire=json.dumps(dict(synthetic_not_native_command=True,native_step=self.step,H40=action.tolist()))
  pin=hashlib.sha256(wire.encode()).hexdigest();value.update(wire=wire,wire_sha256=pin)
  for row in value['receiver_events']:
   row['ref_id']=pin
   if row['kind']=='accept':row['raw']=wire
  self.active_reference=dict(value,action_EEF10=action.tolist());self.installs.append((self.step,action.copy()));return self.active_reference


def completed(root,spec,setup,grade):
 spec=copy.deepcopy(spec);spec.update(registered_root=str(root),phase=phase.PHASE,kind=contract.KIND)
 e=phase.make_episode(run=str(root),geometry_scale=grade)
 # This fixture's legacy constructor otherwise applies V3's33000 ceiling.
 # Use the actual V7 endpoint only while building explicitly synthetic records.
 with pytest.MonkeyPatch.context()as m:
  m.setattr(prior_fixture.c,'endpoint',phase.endpoint)
  outer,path,_,_=prior_files(root,e,spec,source=spec['source_sha'])
 spec.update(episode_manifest=str(root/'EPISODE_MANIFEST.private.json'),episode_manifest_sha256=digest(root/'EPISODE_MANIFEST.private.json'),max_ticks=phase.endpoint(e))
 adapter=Adapter(spec,root/'native-01',setup)
 with pytest.MonkeyPatch.context()as m:
  setup_flow(m,adapter);consume=common_native.consumed_window
  m.setattr(common_native,'consumed_window',lambda a,*args:dict(consume(a,*args),reference_sha256=a.active_reference['wire_sha256']))
  def fk(a,p,state,**kwargs):return dict(status='SYNTHETIC_NOT_FK',native_step=p['step'],native_time_s=p['time_s'],model_sha256=v6.original.SCENE,reader_sha256=a.spec['source_files']['scripts/pi0294/task_state.py'],packet_qpos_qvel_ctrl_exact=True,original_reader_instance_retained=True)
  m.setattr(original_native,'_verify_actual_fk',fk);m.setattr(n4,'_verify_actual_fk',fk)
  cfg=dict(source_observation_bytes=setup['args']['source_observation_bytes'],T_source_from_target=setup['args']['T_source_from_target'],scene_sha256=v6.original.SCENE)
  m.setattr(contract,'episode_config',lambda s:(e,cfg));out=native.run(adapter)
 outer['reference_qualification']=out;pin=put(path,outer);close(root,spec)
 return spec,outer,path,pin


def test_all_sixteen_decisions_native_wrapper_rejoin_and_only_then_full_grade_gate(recipe,tmp_path,monkeypatch):
 spec,_,_,setup=recipe;prior,result,path,pin=completed(tmp_path/'SYNTHETIC_V7_HALF',spec,setup,.5)
 checked=evidence.audit_completed_feedback(prior,result,recipe=phase)
 assert checked['actual_feedback_decisions']==16 and checked['complete_task_atoms']==1272
 assert checked['all_nine_non_full_grade_predicates_passed']
 spec.update(previous_stage=str(path),previous_stage_sha256=pin)
 e=phase.make_episode(run=spec['registered_root'],geometry_scale=1.)
 spec['episode_manifest_sha256']=put(spec['episode_manifest'],e);spec['max_ticks']=46303
 assert contract.episode_limit(spec,result)==46303
 result['reference_qualification']['finite_xyz_feedback']['committed_feedback_decisions']=15
 with pytest.raises(ValueError):evidence.audit_completed_feedback(prior,result,recipe=phase)


@pytest.mark.skipif(not shutil.which('g++'),reason='Original C++ quota ABI compiler required')
def test_original_cpp_bucket96_accepts_both_finite_bounds_but_never_extra_ticks(tmp_path):
 root=Path(__file__).resolve().parents[2];header=root/'scripts/pi0294/task_quota_r1.hpp'
 assert header.read_bytes()==subprocess.check_output(['git','show','da177fe856fb184c50c9d7763132ec626ecfc172:scripts/pi0294/task_quota_r1.hpp'],cwd=root)
 source=tmp_path/'q.cpp';source.write_text('#include "task_quota_r1.hpp"\nint main(int argc,char**argv){uint64_t n=0;return gr00t_quota_r1::read_quota(argv[1],n)?0:1;}\n')
 exe=tmp_path/'quota';subprocess.run(['g++','-std=c++17','-I'+str(header.parent),str(source),'-o',str(exe)],check=True)
 binding=tmp_path/'binding';quota=tmp_path/'quota.txt'
 env=dict(os.environ,GR00T_TASK_QUOTA_BINDING=str(binding),GR00T_TASK_RUN_UUID='SYNTHETIC_V7',GR00T_TASK_CARD_SHA256='CARD',GR00T_TASK_RESERVATION_ID='OWNER')
 for end in (34303,46303):
  binding.write_text(f'GRTQUOTA2\nSYNTHETIC_V7\nCARD\nOWNER\n{end}\n5\n96\n')
  for requested,expected in ((end,0),(end+1,1)):
   quota.write_text(str(requested)+'\n');assert subprocess.run([str(exe),str(quota)],env=env).returncode==expected
