"""Finite PI0258 kinematic registry and deterministic, bounded arm IK. No physics."""
from pathlib import Path
import argparse
import copy
import itertools
import json
import math
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

import pi0255_motion3 as old
from go2arx5_vla.wbc.native.kinematics import NativeArmKinematics

ROOT = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0258')
PREVIOUS = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0255')
AUTHORITY = '8944d569d9f83baa74f0cb28c1113ba9659e528a'


def instructions(task, command, colors):
    if task != 'M3':
        return old.instructions(task, command, colors)
    start, end, direction = ('black', 'white', 'counterclockwise') if command == 0 else ('white', 'black', 'clockwise')
    return [f'Approach the {start} endpoint, then follow the open contour {direction} to the {end} endpoint as seen from above.',
            f'Trace the open arc from {start} to {end}, moving {direction} when viewed from above.',
            f'Go to the {start} end of the open contour and trace it {direction} to {end}, viewed from above.',
            f'Follow the open contour {direction} from the {start} endpoint to the {end} endpoint as viewed from above.']


def freeze():
    target = ROOT/'design/registry.json'
    assert not target.exists(), 'finite registry already frozen'
    previous = json.loads((PREVIOUS/'design/task_registry_v1.json').read_text())
    config = json.loads((PREVIOUS/'evidence/resolved_native_config_cpu.json').read_text())
    source = json.loads((PREVIOUS/'evidence/platform_source_readback.json').read_text())
    kin = NativeArmKinematics.from_urdf(source['components']['native_urdf']['path'])
    default = np.asarray(config['default_q'])
    assert np.all(default >= kin.joint_limits[:, 0]) and np.all(default <= kin.joint_limits[:, 1]), 'DEFAULT_HARD_LIMIT_INVALID'
    factors = copy.deepcopy(previous['candidate_factor_order'])
    factors.sort(key=lambda f: old.ordering(f, 'PI0258_FAMILY_ORDER:25801'))
    assert len(factors) >= 76 and len({json.dumps(x,sort_keys=True) for x in factors}) == len(factors)
    families = []; rows = []
    for i, f in enumerate(factors[:76]):
        split = 'DEVELOPMENT' if i<6 else 'TRAIN' if i<46 else 'KIN-VAL' if i<56 else 'TEST_COMPOSITION'
        families.append(dict(family_id=f'pi0258-family-{i:03d}', split=split, factors=f,
            center_world=(np.array(previous['geometry']['nominal_center_world']) + [f['x_shift'],f['y_shift'],f['z_shift']]).tolist(),
            initial_root_position=config['nominal_base_world'],initial_root_quaternion=[0.,0.,0.,1.],
            initial_arm_joints=default.tolist(),state_origin='KINEMATIC_FK_NOT_DYNAMIC_MEASUREMENT'))
    for i, f in enumerate(previous['families'][-20:]):
        f=copy.deepcopy(f);f.update(family_id=f'pi0258-geometry-test-{i:03d}',split='TEST_GEOMETRY',manifest_only=True)
        f.pop('reset_seed',None);families.append(f)
    for family in families:
        for task, command in itertools.product(old.TASKS,(0,1)):
            texts=instructions(task,command,family['factors']['colors'])
            language_index=int(old.ordering([family['family_id'],task,command],'PI0258_LANGUAGE:25802'),16)%2
            if family['split'] not in ('DEVELOPMENT','TRAIN'):language_index+=2
            rows.append(dict(sequence_id=family['family_id']+f'-{task}-c{command}',family_id=family['family_id'],
                task=task,command=command,split=family['split'],instruction=texts[language_index],paraphrases=texts,
                execute=not family['split'].startswith('TEST')))
    reg=dict(schema='PI0258_KIN_MT_V1',authority=AUTHORITY,families=families,sequences=rows,
        source_hashes={k:old.sha(PREVIOUS/'evidence'/k) for k in ['resolved_native_config_cpu.json','platform_source_readback.json']},
        urdf=source['components']['native_urdf'],geometry=previous['geometry'],experts=previous['experts'],
        scoring=previous['scoring'],candidate_factor_order=factors,
        family_order='SHA256 canonical JSON namespace PI0258_FAMILY_ORDER:25801',language_order='PI0258_LANGUAGE:25802',
        duplicate_log=[],duplicate_definition='same scene factors AND actual root/DOF initial state; old stochastic physical resets differ from declared nominal kinematic state',
        old_scene_factor_reuse=True,old_failed_physical_conditions_used_as_validation=False,
        label='KINEMATIC_RENDERED_REFERENCE',physics_steps_after_placement=0,
        root_pose=config['nominal_base_world']+[0,0,0,1],arm_default=default.tolist(),
        action_period_s=.05,observation_period_s=.1,horizon=40,neutral_gripper=.5,
        solver=dict(method='deterministic damped least squares',max_iterations=200,damping=.001,
            orientation_weight_m_per_rad=.2,jacobian='central finite difference',jacobian_step=1e-6,
            max_joint_step_rad=.1,line_search=[1.,.5,.25,.125,.0625,.03125,.015625,.0078125],
            joint_candidates='project into original hard bounds; no random restarts',position_tolerance_m=.002,rotation_tolerance_rad=.01,
            warm_start='previous timestamp including failed solution; no reference changes'),
        collision=dict(method='independent original URDF collision geometry; source adjacent/fixed-link exemptions only',
            status='IMPLEMENTATION_AND_VERIFICATION_REQUIRED_BEFORE_DATA_ADMISSION',world_obstacle='original grey box and ground',
            self_collision_disabled_in_native_does_not_waive_check=True),
        renderer=dict(adapter='isolated IsaacGym direct state placement',resolution=[96,96],
            body_attachment=[1.15,-1.15,.95],wrist_attachment=[.28,-.10,.12],local_y_rotation_deg=135,
            status='TRANSFORM_PROJECTION_READBACK_REQUIRED_BEFORE_DATA_ADMISSION'),
        budgets=dict(planned_sequences=336,attempt_cap=360,technical_reserve=24,active_windows=60000,image_pairs=120000),
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    reg['geometry']=copy.deepcopy(reg['geometry']);reg['geometry']['scene_classes_all_tasks'][-2:]=['open_270_degree_contour','black_A_white_B_endpoints']
    reg['experts']=copy.deepcopy(reg['experts']);reg['experts']['M3'].update(start_angle_rad=-3*math.pi/4,end_angle_rad=3*math.pi/4,command0_direction='CCW',command1_direction='CW')
    assert sum(r['execute'] for r in rows)==336
    old.write(target,reg)
    print(json.dumps(dict(registry_sha256=old.sha(target),planned=336)))


def reference(reg,family,task,command,times):
    p0=np.array(reg['geometry']['nominal_tcp_world']);q0=np.array([0.,0.,0.,1.])
    if task!='M3':
        f=dict(family,split='PILOT') # old pure geometry helper's split guard; no TEST caller allowed here.
        return old.expert_at(reg,f,task,command,p0,q0,times)
    data=old.scene(reg,family);a=(-3 if command==0 else 3)*math.pi/4
    entry=data['center']+[data['radius']*math.cos(a),data['radius']*math.sin(a),0]
    u=np.clip(times/4,0,1);u=u*u*(3-2*u)
    p=p0+(entry-p0)*u[:,None];active=times>=4
    angle=a+(1 if command==0 else -1)*1.5*math.pi*np.clip((times[active]-4)/12,0,1)
    p[active]=data['center']+np.column_stack((data['radius']*np.cos(angle),data['radius']*np.sin(angle),np.zeros(len(angle))))
    return p,np.tile(q0,(len(times),1))


def solve(kin,p,r,q,settings):
    q=q.copy();weight=np.array([1.,1.,1.]+[settings['orientation_weight_m_per_rad']]*3)
    def error(candidate):
        pose=kin.tcp_pose(candidate)
        return np.r_[p-pose[:3,3],Rotation.from_matrix(r@pose[:3,:3].T).as_rotvec()]
    for iteration in range(settings['max_iterations']):
        e=error(q)
        if np.linalg.norm(e[:3])<=settings['position_tolerance_m'] and np.linalg.norm(e[3:])<=settings['rotation_tolerance_rad']:
            return q,e,iteration,True
        j=weight[:,None]*kin.finite_difference_jacobian(q,settings['jacobian_step'])
        delta=j.T@np.linalg.solve(j@j.T+settings['damping']**2*np.eye(6),weight*e)
        delta*=min(1.,settings['max_joint_step_rad']/max(np.max(abs(delta)),1e-12))
        best=q;cost=np.linalg.norm(weight*e)
        for scale in settings['line_search']:
            candidate=np.clip(q+scale*delta,kin.joint_limits[:,0],kin.joint_limits[:,1]);candidate_cost=np.linalg.norm(weight*error(candidate))
            if candidate_cost < cost:best=candidate;break
        if np.array_equal(best,q):return q,e,iteration+1,False
        q=best
    e=error(q)
    return q,e,settings['max_iterations'],bool(np.linalg.norm(e[:3])<=settings['position_tolerance_m'] and np.linalg.norm(e[3:])<=settings['rotation_tolerance_rad'])


def check():
    reg=json.loads((ROOT/'design/registry.json').read_text());kin=NativeArmKinematics.from_urdf(reg['urdf']['path'])
    q=np.array([.1,.2,.3,.1,-.1,.1]);target=kin.tcp_pose(q)
    recovered,e,n,ok=solve(kin,target[:3,3],target[:3,:3],q+.001,reg['solver'])
    assert ok and np.linalg.norm(e[:3])<=.002 and np.linalg.norm(e[3:])<=.01
    for f in reg['families'][:6]:
        t=np.arange(361)*.05
        for c in (0,1):
            p,r=reference(reg,f,'M3',c,t)
            assert np.allclose(p[0],reg['geometry']['nominal_tcp_world'])
            relative=p[80:321,:2]-np.asarray(f['center_world'])[:2]
            phase=np.unwrap(np.arctan2(relative[:,1],relative[:,0]))
            assert np.isclose(phase[-1]-phase[0],(1 if c==0 else -1)*1.5*math.pi)
    old.write(ROOT/'evidence/kinematic_cpu_check.json',dict(status='PASS_BOUNDED_IK_AND_OPEN_CONTOUR_FIXTURES',physics=0,queries=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.freeze:freeze()
    elif args.check:check()
