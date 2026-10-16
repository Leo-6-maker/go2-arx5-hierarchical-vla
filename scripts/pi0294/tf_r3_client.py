"""Bounded search RPC and original Cartesian certifier; no actual state edits."""
import hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
from .mp_client import Client as OriginalClient,PlannerFailure,serial,contact_permitted
from .tf_r3_plan_runtime import FOLDER,reserve_bundle,reserve_internal,finish_internal
from .task_capture_runtime import read,sha
from go2arx5_vla.trajectory.so3 import matrix_to_quat,quat_to_matrix,quat_slerp,geodesic_angle

class Client(OriginalClient):
    def __init__(self,root,engine):
        super().__init__(root,engine)
        self.old_series=self.series;self.series=Path(root)/FOLDER
        self.context=None
        self.timing_changed=False
        self.executed_reference_history=[]

    def bundle(self,state,packages,stage,condition,index,*,attached=False,hold=False,frozen_poses=None,frozen_stages=None):
        from scripts.pi0292.durable import write
        folder=self.series/'requests'/condition/f'{index:04d}';folder.mkdir(parents=True,exist_ok=True)
        config=__import__('scripts.pi0294.tf_r3_plan_runtime',fromlist=['condition']).condition(self.root,condition)
        batch=config['kind']=='SAVED_INPUT_REGISTERED_SCENE_PREVIEW'
        goals=[]
        for package in packages:
            if frozen_poses is not None or stage!='approach':entries=[('target',package['target'],stage,False)]
            elif package.get('direct_join'):entries=[('join',package['contact'],'align' if package['task']=='T1' else 'contact',False)]
            else:entries=[('approach',package['approach'],'approach',False),('contact',package['contact'],package.get('entry_contact_stage','align' if package['task']=='T1' else 'contact'),False)]
            source=config['sources'][package['index']] if batch else None
            streams=source['original_streams'] if batch else config['random_streams'][index]
            paired=len(entries)==2
            seedtable=[[streams['RRT'][branch*4+stream*2+link] for link in range(2 if paired else 1)] for branch in range(1 if batch else 2) for stream in range(2)]
            for local,(endpoint,pose,goalstage,closed) in enumerate(entries):
                goal=dict(package=package['index'],endpoint=endpoint,pose=np.asarray(pose).tolist(),stage=goalstage,closed=closed,
                    IK_seed=streams['IK'][1 if batch else local],RRT_seed_table=seedtable)
                if batch and package['task']=='T2':goal.update(reuse_v2_slot=source['reusable_response']['goal_slot'],old_response_sha256=source['reusable_response']['sha256'])
                goals.append(goal)
        if len(goals)>12:raise ValueError('SDR3_GOAL_CAP')
        payload=self.payload(state,np.asarray(goals[0]['pose']),stage,0,attached=attached,hold=hold)
        payload.update(schema='TF_NIGHT_REQUEST_V1',run_root=str(self.root),bundle_key=condition+'/'+str(index),
            goals=goals,condition_kind=config['kind'],implementation_sha=read(self.root/'RUN_MANIFEST.json')['code']['vla_commit'])
        payload['registered_source_programs']=[{k:p[k] for k in ('index','source_id','source_program_sha256','active_duration_s') if k in p} for p in packages]
        if frozen_poses is not None:payload['frozen_window']=dict(poses=np.asarray(frozen_poses).tolist(),stages=list(frozen_stages))
        input_identity=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        reserve_bundle(self.root,condition,index,input_identity)
        request=folder/'INPUT.private.json'
        if request.exists():
            if read(request)!=payload:raise ValueError('SDR3_BUNDLE_INPUT_CHANGED')
        else:write(request,payload)
        final=folder/'FINAL.private.json';npz=folder/'H40.private.npz'
        if final.exists():
            result=read(final)
            if result['input_sha256']!=sha(request):raise ValueError('SDR3_FIRST_RESULT_INPUT_CHANGED')
            if result['status']!='Success':raise PlannerFailure(result['status'],result)
            with np.load(npz,allow_pickle=False) as arrays:poses=arrays['poses'].copy()
            self.last_route=read(folder/f"ROUTE-{result['route']:02d}.private.json")
            return poses,result['certificate'],result['package']
        self.context=dict(folder=folder,key=payload['bundle_key'],route=None,request=request)
        started=time.monotonic();deadline=started+60
        env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[2])+':'+str(Path(__file__).resolve().parents[2]/'src'),
            LD_LIBRARY_PATH='',LD_PRELOAD='',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
            MPLCONFIGDIR=str(self.old_series/'tmp/mpl-config'),XDG_CACHE_HOME=str(self.old_series/'cache'))
        def rpc(action,route=0):
            output=folder/('IK_RESULT.private.json' if action=='IK' else f'ROUTE-{route:02d}.private.json')
            if not output.exists():
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('SDR3_BUNDLE_WALL_CAP')
                log=folder/(action+'-'+str(route)+'-backend.log')
                with log.open('a') as handle:
                    subprocess.run([str(self.old_series/'planner-env/bin/python'),'-m','scripts.pi0294.tf_r3_backend',
                        '--request',str(request),'--action',action,'--route',str(route)],env=env,stdout=handle,stderr=subprocess.STDOUT,
                        timeout=remaining,check=True)
            value=read(output)
            if value['input_sha256']!=sha(request):raise ValueError('SDR3_RPC_INPUT_SHA')
            return value
        failures=[]
        try:
            if hold:
                q=np.asarray(state['qpos'])[self.binding['joint_qpos_addresses']]
                indices=[i for i,n in enumerate(self.binding['joint_names']) if n.startswith('x5_joint')]
                response=dict(status='Success',time=[0.,2.],position=[q[indices].tolist()]*2,package=packages[0]['index'],route=0)
                responses=[response]
            else:
                ik=rpc('IK');fk=np.asarray(ik['initial_fk_world'])
                error=[float(np.linalg.norm(fk[:3,3]-state['tcp'][:3,3])),float(geodesic_angle(fk[:3,:3],state['tcp'][:3,:3]))]
                if error[0]>1e-5 or error[1]>1e-4:raise ValueError('SDR3_NATIVE_FK_PARITY')
                if not ik['pairs']:raise PlannerFailure('NO_LEGAL_REGISTERED_PAIR',dict(goals=ik['goals']))
                responses=None
            for route in range(1 if hold else 2*len(ik['pairs'])):
                response=responses[0] if hold else rpc('ROUTE',route)
                if response['status']!='Success':failures.append(dict(route=route,status=response['status']));continue
                self.context['route']=route
                response.update(request_sha256=sha(request),response_sha256=sha(request) if hold else sha(folder/f'ROUTE-{route:02d}.private.json'),
                    FK_error=[0.,0.] if hold else error)
                slots=read(self.series/'SEARCH_LEDGER.json')['bundles'][payload['bundle_key']]['slots']
                response.update(IK_calls=sum(k.startswith('IK/') and v['state']=='COMPLETE' for k,v in slots.items()),
                    RRT_calls=sum(k.startswith('RRT/') and v['state']=='COMPLETE' for k,v in slots.items()))
                try:
                    if frozen_poses is None:
                        poses,certificate=self.cartesian(state,response,speed=.06,attached=attached,stage='contact' if packages[0]['task']=='T2' else stage)
                    else:poses,certificate=self.certify_frozen(state,response,np.asarray(frozen_poses),list(frozen_stages),attached=attached)
                except PlannerFailure as e:
                    failures.append(dict(route=route,status=str(e)))
                    used=sum(k.startswith('DENSE/') for k in read(self.series/'SEARCH_LEDGER.json')['bundles'][payload['bundle_key']]['slots'])
                    if used>=config_dense_cap(self.root,payload['bundle_key']):raise PlannerFailure(str(e),dict(routes=failures))
                    continue
                self.last_route=response
                package=next(p for p in packages if p['index']==response['package'])
                if frozen_poses is None and stage=='approach' and certificate['duration_s']+package.get('active_duration_s',0)>config['active_budget_s']:
                    failures.append(dict(route=route,status='SOURCE_ENTRY_AND_ACTIVITY_TIME_INFEASIBLE',entry_duration_s=certificate['duration_s'],active_duration_s=package.get('active_duration_s')))
                    continue
                if npz.exists():
                    with np.load(npz,allow_pickle=False) as arrays:
                        if not np.array_equal(arrays['poses'],poses):raise ValueError('SDR3_FIRST_H40_ARRAY_CHANGED')
                else:np.savez_compressed(npz,poses=poses)
                result=dict(status='Success',package=response['package'],route=route,certificate=certificate,
                    input_sha256=sha(request),service_s=time.monotonic()-started,earlier_failures=failures)
                write(final,result);return poses,certificate,response['package']
            raise PlannerFailure('REGISTERED_ROUTES_EXHAUSTED',dict(routes=failures))
        except (TimeoutError,subprocess.TimeoutExpired):
            result=dict(status='TIME_BUDGET_INCOMPLETE',input_sha256=sha(request),service_s=time.monotonic()-started,earlier_failures=failures)
            write(final,result);raise PlannerFailure(result['status'],result)
        except PlannerFailure as e:
            result=dict(status=str(e),input_sha256=sha(request),service_s=time.monotonic()-started,evidence=e.evidence,earlier_failures=failures)
            write(final,result);raise

    def dense_check(self,state,poses,arm_samples,*,stage,attached):
        from scripts.pi0292.durable import write
        c=self.context;cache=c['folder']/f"DENSE-ROUTE-{c['route']:02d}.private.json"
        identity=hashlib.sha256(np.asarray(poses,dtype='<f8').tobytes()+np.asarray(arm_samples,dtype='<f8').tobytes()).hexdigest()
        if cache.exists():
            r=read(cache)
            if r['window_sha256']!=identity:raise ValueError('SDR3_DENSE_WINDOW_CHANGED')
            if r['status']!='PASS':raise PlannerFailure(r['reason'],r)
            return r
        ledger=read(self.series/'SEARCH_LEDGER.json');slots=ledger['bundles'][c['key']]['slots']
        slot=sum(x.startswith('DENSE/') for x in slots)
        if slot>=config_dense_cap(self.root,c['key']):raise PlannerFailure('SDR3_DENSE_BUDGET_CAP',{})
        reserve_internal(self.root,c['key'],'DENSE',slot)
        import mujoco
        m,d=self.model,self.native_data;d.qpos[:]=state['qpos'];d.qvel[:]=state['qvel'];d.ctrl[:]=state['ctrl']
        poses=np.concatenate([poses,poses[-1:]]);path=np.vstack([arm_samples,arm_samples[-1]])
        jacp=np.zeros((3,m.nv));jacr=np.zeros_like(jacp);relative=np.linalg.inv(state['tcp'])@state['object_pose']
        cubej=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'task_cube_free');cqa=int(m.jnt_qposadr[cubej])
        points=iterations=0;max_pos=max_rot=0.;status='PASS';reason=None
        try:
            for time_s in np.arange(1001)*.002:
                points+=1;i=min(int(time_s/.05),39);fraction=(time_s-i*.05)/.05
                target=np.eye(4);target[:3,3]=(1-fraction)*poses[i,:3,3]+fraction*poses[i+1,:3,3]
                target[:3,:3]=quat_to_matrix(quat_slerp(matrix_to_quat(poses[i,:3,:3]),matrix_to_quat(poses[i+1,:3,:3]),fraction))
                d.qpos[self.arm_q]=(1-fraction)*path[i]+fraction*path[i+1]
                for _ in range(5):
                    iterations+=1;mujoco.mj_forward(m,d);current=d.site_xmat[self.site].reshape(3,3)
                    ep=target[:3,3]-d.site_xpos[self.site];delta=matrix_to_quat(target[:3,:3]@current.T)
                    if delta[3]<0:delta=-delta
                    length=np.linalg.norm(delta[:3]);er=delta[:3]*(2*np.arctan2(length,delta[3])/max(length,1e-15))
                    if np.linalg.norm(ep)<=1e-5 and np.linalg.norm(er)<=1e-4:break
                    mujoco.mj_jacSite(m,d,jacp,jacr,self.site);j=np.vstack([jacp[:,self.arm_v],jacr[:,self.arm_v]])
                    step=np.linalg.lstsq(j,np.r_[ep,er],rcond=1e-10)[0]
                    if not np.isfinite(step).all():raise PlannerFailure('DENSE_NONFINITE_IK',{})
                    d.qpos[self.arm_q]+=step
                else:raise PlannerFailure('CARTESIAN_INTERPOLATION_NOT_NATIVE_IK_FEASIBLE',{})
                if np.any(d.qpos[self.arm_q]<m.jnt_range[self.arm_j,0]) or np.any(d.qpos[self.arm_q]>m.jnt_range[self.arm_j,1]):raise PlannerFailure('DENSE_NATIVE_JOINT_LIMIT',{})
                max_pos=max(max_pos,float(np.linalg.norm(ep)));max_rot=max(max_rot,float(np.linalg.norm(er)))
                if attached:
                    obj=target@relative;d.qpos[cqa:cqa+3]=obj[:3,3];quat=matrix_to_quat(obj[:3,:3]);d.qpos[cqa+3:cqa+7]=np.r_[quat[3],quat[:3]]
                mujoco.mj_forward(m,d)
                for contact in d.contact:
                    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in (contact.geom1,contact.geom2)]
                    if not contact_permitted(names,contact.dist,stage):raise PlannerFailure('DENSE_NATIVE_COLLISION:'+str(names),{})
        except PlannerFailure as e:status='FAIL';reason=str(e)
        except Exception as e:
            status='ENGINEERING_INCOMPLETE';reason=type(e).__name__+': '+str(e);raise
        finally:
            proof=dict(status=status,reason=reason,points=points,local_IK_iterations=iterations,window_sha256=identity,
                max_native_IK_position_residual_m=max_pos,max_native_IK_rotation_residual_rad=max_rot,physics_steps=0,actual_state_modified=False)
            write(cache,proof);finish_internal(self.root,c['key'],'DENSE',slot,dict(points=points,local_IK_iterations=iterations,status=status))
        if status!='PASS':raise PlannerFailure(reason,proof)
        return proof


    def certify_frozen(self,state,response,poses,stages,*,attached):
        """Validate the source window on a clone; never replace its active values."""
        import mujoco
        from scripts.pi0292.durable import write
        from .source_retime_v2_analysis import so3_angles
        if poses.shape!=(40,4,4) or len(stages)!=40:raise ValueError('SDR3_FROZEN_H40_SHAPE')
        active_poses=poses.copy()
        history=self.executed_reference_history[-2:]
        if history and len(history)!=2:raise ValueError('TWO_ACTUALLY_EXECUTED_REFERENCES_REQUIRED')
        prefix=len(history)
        if prefix:
            if abs(history[-1]['native_end_time']-(5.+state['time_s']))>1e-8:raise ValueError('EXECUTED_REFERENCE_HISTORY_NOT_CURRENT_NATIVE_TICK')
            poses=np.concatenate([np.stack([r['pose'] for r in history]),poses]);stages=[stages[0]]*prefix+list(stages)
        speed=np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1)/.05;angular=so3_angles(poses[:-1,:3,:3],poses[1:,:3,:3])/.05
        if max(speed.max()/.06,angular.max()/.3)>1+1e-10:raise PlannerFailure('FROZEN_SOURCE_RATE_REJECT',{})
        c=self.context;ledger=read(self.series/'SEARCH_LEDGER.json');slots=ledger['bundles'][c['key']]['slots'];slot=sum(k.startswith('DENSE/') for k in slots)
        reserve_internal(self.root,c['key'],'DENSE',slot)
        path=c['folder']/f"FROZEN-DENSE-{slot:02d}.private.json";trace=c['folder']/f"FROZEN-DENSE-{slot:02d}.private.npz"
        m,d=self.model,self.native_data;d.qpos[:]=state['qpos'];d.qvel[:]=state['qvel'];d.ctrl[:]=state['ctrl'];d.qpos[self.arm_q]=np.asarray(response['position'])[-1]
        jacp=np.zeros((3,m.nv));jacr=np.zeros_like(jacp);relative=np.linalg.inv(state['tcp'])@state['object_pose']
        cube=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'task_cube_free');cqa=int(m.jnt_qposadr[cube]);points=iterations=0;qs=[];status='PASS';reason=None
        extended=np.concatenate([poses,poses[-1:]])
        try:
            for tick in range((40+prefix)*25+1):
                at=tick*.002;i=min(int(at/.05),39+prefix);fraction=(at-i*.05)/.05;points+=1
                target=np.eye(4);target[:3,3]=(1-fraction)*extended[i,:3,3]+fraction*extended[i+1,:3,3]
                target[:3,:3]=quat_to_matrix(quat_slerp(matrix_to_quat(extended[i,:3,:3]),matrix_to_quat(extended[i+1,:3,:3]),fraction))
                for iteration in range(5):
                    iterations+=1;mujoco.mj_forward(m,d);ep=target[:3,3]-d.site_xpos[self.site];delta=matrix_to_quat(target[:3,:3]@d.site_xmat[self.site].reshape(3,3).T)
                    if delta[3]<0:delta=-delta
                    length=np.linalg.norm(delta[:3]);er=delta[:3]*(2*np.arctan2(length,delta[3])/max(length,1e-15))
                    if iteration and np.linalg.norm(ep)<=1e-5 and np.linalg.norm(er)<=1e-4:break
                    mujoco.mj_jacSite(m,d,jacp,jacr,self.site);step=np.linalg.lstsq(np.vstack([jacp[:,self.arm_v],jacr[:,self.arm_v]]),np.r_[ep,er],rcond=1e-10)[0]
                    if not np.isfinite(step).all():raise PlannerFailure('FROZEN_SOURCE_NONFINITE_LOCAL_IK',{})
                    d.qpos[self.arm_q]+=step
                else:raise PlannerFailure('FROZEN_SOURCE_NATIVE_IK_REJECT',{})
                if np.any(d.qpos[self.arm_q]<m.jnt_range[self.arm_j,0]) or np.any(d.qpos[self.arm_q]>m.jnt_range[self.arm_j,1]):raise PlannerFailure('FROZEN_SOURCE_NATIVE_LIMIT',{})
                if attached:
                    obj=target@relative;d.qpos[cqa:cqa+3]=obj[:3,3];quat=matrix_to_quat(obj[:3,:3]);d.qpos[cqa+3:cqa+7]=np.r_[quat[3],quat[:3]]
                mujoco.mj_forward(m,d)
                for contact in d.contact if tick>=prefix*25 else ():
                    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in (contact.geom1,contact.geom2)]
                    if not contact_permitted(names,contact.dist,stages[i]):raise PlannerFailure('FROZEN_SOURCE_NATIVE_COLLISION:'+str(names),{})
                qs.append(d.qpos[self.arm_q].copy())
            command_q=np.asarray(qs)[np.arange(40+prefix)*25];vel=np.diff(command_q,axis=0)/.05;acc=np.diff(vel,axis=0)/.05
            if np.max(np.abs(vel))>.7*(1+1e-10):raise PlannerFailure('FROZEN_SOURCE_JOINT_SPEED',{})
            if np.max(np.abs(acc))>1.*(1+1e-10):raise PlannerFailure('FROZEN_SOURCE_JOINT_ACCELERATION',{})
        except PlannerFailure as e:status='FAIL';reason=str(e)
        except Exception as e:
            status='ENGINEERING_INCOMPLETE';reason=type(e).__name__+': '+str(e);raise
        finally:
            np.savez_compressed(trace,joint_q=np.asarray(qs)[prefix*25:],poses=active_poses,history_and_preview_joint_q=np.asarray(qs))
            proof=dict(status=status,reason=reason,points=points,local_IK_iterations=iterations,
                installed_pose_values_changed=False,actual_state_modified=False,physics_steps=0,
                frame_rate_Hz=500,joint_rate_certificate_grid_Hz=20,joint_velocity_limit=.7,joint_acceleration_limit=1.,
                trace_sha256=sha(trace),input_sha256=sha(c['request']),actually_executed_history_points=prefix,
                history_recomputed_in_current_root_and_branch=True,actual_qpos_is_not_command_history=True,
                previous_reference_identities=[{k:r[k] for k in ('wire_sha256','prefix','native_step','native_end_time')} for r in history])
            write(path,proof);finish_internal(self.root,c['key'],'DENSE',slot,dict(status=status,points=points,local_IK_iterations=iterations))
        if status!='PASS':raise PlannerFailure(reason,proof)
        return active_poses,dict(duration_s=2.,dense=proof,max_linear_speed=float(speed.max()),max_angular_speed=float(angular.max()),
            frozen_source_values_preserved=True,input_sha256=sha(c['request']),response_sha256=response['response_sha256'])


def config_dense_cap(root,key):
    from .tf_r3_plan_runtime import condition
    return condition(root,key.split('/')[0]).get('max_dense_windows',1)
