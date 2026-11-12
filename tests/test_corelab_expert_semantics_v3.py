import copy
import pytest
from go2arx5_vla.world_model import corelab_expert_semantics_v3 as expert
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model import task_learning_collection_v1 as collection
from go2arx5_vla.world_model import task_learning_teacher_panel_v1 as teacher
from go2arx5_vla.world_model.task_learning_scene_v1 import scene_id
from go2arx5_vla.world_model.native_body_camera_capture_v1 import session_configuration
from go2arx5_vla.world_model.native_collection_performance_v1 import configuration, VERSION

def panel():
 return dict(schema=expert.SCHEMA,protocol=expert.protocol(),task_learning_role=expert.ROLE,
  Teacher_implementation_source_sections_sha256=teacher.FROZEN_TEACHER,base_Teacher_qualification={'fixture':'existing-qualification'},
  assignments=[dict(layout=layout,native_run='new-'+layout,parent=scene_id(layout),split='TRAIN',command=task.COMMAND,
   purpose=expert.PURPOSE,restart=1,Teacher_mode='NORMAL_TEACHER_V7',model_sha256='a'*64)for layout in expert.LAYOUTS])

def test_exact_teacher_and_four_train_requests_reuse_completed_qualification(monkeypatch):
 calls=[];monkeypatch.setattr(collection,'qualified_base',lambda p:calls.append(p))
 p=panel();assert expert.validate_panel(p)==p['assignments'];assert calls==[p['base_Teacher_qualification']]
 assert not task.student_role(expert.ROLE)
 assert expert.protocol()['expert_training_eligible'] is False

@pytest.mark.parametrize('mutation',['order','duplicate','DEV','mode','restart'])
def test_changed_population_or_expert_is_rejected_before_qualification(monkeypatch,mutation):
 monkeypatch.setattr(collection,'qualified_base',lambda p:pytest.fail('invalid panel reached teacher qualification'))
 p=panel();r=p['assignments']
 if mutation=='order':r.reverse()
 elif mutation=='duplicate':r[1]['native_run']=r[0]['native_run']
 elif mutation=='DEV':r[0]['split']='DEV'
 elif mutation=='mode':r[0]['Teacher_mode']='RECOVERY'
 else:r[0]['restart']=2
 with pytest.raises(ValueError):expert.validate_panel(p)

def test_new_role_requires_explicit_body_session_and_never_adopts_old_collection():
 s=dict(phase=task.PHASE,task_learning_role=expert.ROLE,observation_camera_profile='BODY_BASE_FIXED_V1_20261111',
  body_camera_session_v3=dict(version='BODY_BASE_EPISODE_SESSION_V3',mode='PERSISTENT_COMPATIBILITY'),
  collection_performance=dict(version=VERSION,mode='events_camera_cache'))
 assert session_configuration(s) and configuration(s)
 missing=copy.deepcopy(s);missing.pop('body_camera_session_v3')
 with pytest.raises(ValueError):configuration(missing)
 old=copy.deepcopy(s);old['task_learning_role']=task.COLLECTION_ROLE
 with pytest.raises(ValueError):session_configuration(old)
 with pytest.raises(ValueError):configuration(old)
