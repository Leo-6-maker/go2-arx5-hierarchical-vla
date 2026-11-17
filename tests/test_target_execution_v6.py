import copy,hashlib,json
import pytest
from go2arx5_vla.world_model import target_scope_v6 as scope
from scripts.wm02.target_offline_report_v6 import decision


@pytest.fixture
def interface(tmp_path,monkeypatch):
    files={};digests={};frozen={}
    for key in scope.BINARIES:
        path=tmp_path/key;path.write_bytes(('CPU_FIXTURE_'+key).encode());files[key]=str(path);digests[key]=hashlib.sha256(path.read_bytes()).hexdigest();frozen[str(path)]=digests[key]
    monkeypatch.setattr(scope,'BINARIES',digests)
    auth=dict(schema='USER_PROJECT_AUTHORIZED_ISOLATED_MUJOCO_V6_T',plan_commit=scope.PLAN,source_sha='a'*40,
        source='CURRENT_USER_EXPLICIT_FIXED_SLA_V6_PLAN',hardware_allowed=False,deadline_epoch=1795027000)
    path=tmp_path/'authority.json';path.write_text(json.dumps(auth))
    return dict(sim_only_project_authorization=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        source_sha='a'*40,scope=scope.SCOPE,plan_commit=scope.PLAN,mobile_experiment_version=scope.VERSION,
        native_backend='MUJOCO_ONLY',physics=True,hardware_connections_allowed=False,execution_clock_profile='SIM_PAUSED_NATIVE',physics_dt_s=.002,
        mobile_condition=dict(mode='M_FREE',family='F1',method='SIM_INTERFACE_HOLD'),optimizer_updates=0,WBC_changes=False,safety_changes=False,
        model_service_timeout_s=30,sim_only_role='INTERFACE',deadline_epoch=1795026900,new_physical_requests_budget=1,max_ticks=scope.INTERFACE_TICKS,
        frozen_files=frozen,native_environment=dict(ROS_LOCALHOST_ONLY='1',ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',CUDA_VISIBLE_DEVICES=''),ROS_DOMAIN_ID=198,**files)


def test_v6_new_identity_preserves_frozen_interface_contract(interface):
    assert scope.validate(interface)['original_full_H40_P10_TTL_STOP']


@pytest.mark.parametrize('fault',['hardware','clock','binary','contact_gate','ticks','source','GPU','mode'])
def test_v6_authority_does_not_bypass_original_limits(interface,fault):
    s=copy.deepcopy(interface)
    if fault=='hardware':s['hardware_connections_allowed']=True
    elif fault=='clock':s['execution_clock_profile']='WALL_CLOCK_NATIVE'
    elif fault=='binary':s['frozen_files'][s['controller_binary']]='b'*64
    elif fault=='contact_gate':s['safety_changes']=True
    elif fault=='ticks':s['max_ticks']+=1
    elif fault=='source':s['source_sha']='b'*40
    elif fault=='GPU':s['native_environment']['CUDA_VISIBLE_DEVICES']='0'
    else:s['mobile_condition']['mode']='M_ASSIST'
    with pytest.raises((PermissionError,ValueError)):scope.validate(s)


def test_target_selection_requires_full_panel_and_all_eight_historical_rejections():
    rows=[];offline={'by_arm':{}}
    keys=('TRAIN96_all','CN38_180_all','saved_reject3_all','zero_CLOSE_omissions','CLOSE_within100ms_no_regression','old3_and_E2reject5_all')
    for arm in('T-B1','T-FM','T-H40'):
        offline['by_arm'][arm]=dict(promotion_offline_gates={k:True for k in keys})
        for f in('F1','F2'):
            for i in range(4):rows.append(dict(role='STUDENT',arm=arm,family=f,condition=dict(layout='T%d'%i),actual_native_requests=1,technical_failure=False,
                hard_STOP=False,completed=True,task_and_mobile_success=True,valid_policy_rejection=False,task_ticks=10000,native_ticks=15703))
    with pytest.raises(ValueError,match='COMPLETE_NEW24'):decision(rows[:23],offline)
    assert decision(rows,offline)['selected_arm']=='T-B1'
    offline['by_arm']['T-B1']['promotion_offline_gates']['old3_and_E2reject5_all']=False
    assert decision(rows,offline)['selected_arm']=='T-FM'
    offline['by_arm']['T-FM']['promotion_offline_gates']['old3_and_E2reject5_all']=False
    offline['by_arm']['T-H40']['promotion_offline_gates']['old3_and_E2reject5_all']=False
    assert decision(rows,offline)['selected_arm']is None
