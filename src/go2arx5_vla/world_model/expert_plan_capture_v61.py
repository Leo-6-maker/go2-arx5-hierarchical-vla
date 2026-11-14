"""Prospective final plans and actual origin-command consumption evidence.

Explicit new CPU-expert runs only. The optional transport diagnostic uses the
frozen N20 normalizer;future PD training-view statistics remain unfitted here.
"""
import json,os,selectors,subprocess,time
from pathlib import Path

VERSION='SLA_PROSPECTIVE_FINAL_PLAN_CAPTURE_V6_1'
KEY='expert_plan_capture_v61'
PANEL=VERSION+'_FIRST4_PANEL'
LAYOUTS=('TRAIN01','TRAIN04','TRAIN07','TRAIN10')


def validate_panel(panel):
    from .task_learning_teacher_panel_v1 import FROZEN_TEACHER,teacher_fingerprint
    rows=panel['assignments']
    if(panel['schema']!=PANEL or len(rows)!=4 or tuple(r['layout']for r in rows)!=LAYOUTS
            or len({r['native_run']for r in rows})!=4
            or panel['Teacher_implementation_source_sections_sha256']!=FROZEN_TEACHER
            or teacher_fingerprint()!=FROZEN_TEACHER
            or panel['old_RP8_passed']is not False or panel['formal_D100_new']is not False
            or panel['first_wave_only']is not True or panel['user_authorized_earlier_first4']is not True):
        raise ValueError('FOUR_FIXED_NEW_EXPERT_EPISODES_OLD_RP8_UNCHANGED')
    return rows


def native_scene_admission(spec):
    from .stable_task_scene_v1 import pinned
    from .task_learning_scene_v1 import validate_static_provenance
    from .sla_label_replay_v61 import _archived_scene
    configuration(spec)
    panel=pinned(dict(path=spec['stable_task_panel'],sha256=spec['stable_task_panel_sha256']))
    rows=validate_panel(panel);matches=[r for r in rows if r['native_run']==Path(spec['registered_root']).name]
    if len(matches)!=1 or panel['source_sha']!=spec['source_sha']:raise ValueError('UNIQUE_NEW_FIRST4_NATIVE_REQUEST')
    row=matches[0];old=pinned(row['original_spec']);proof,builder=_archived_scene(row['original_spec']['path'],row['original_spec']['sha256'])
    validate_static_provenance(spec,proof,expected_role=old['task_learning_role'],verified_builder_sha256=builder)
    if(row['layout']!=proof['layout'] or old['model_path']!=spec['model_path']
            or old['task_phase_source_from_target']!=spec['task_phase_source_from_target']
            or old['native_binary']!=spec['native_binary'] or old['solver_binary']!=spec['solver_binary']
            or old['controller_binary']!=spec['controller_binary']):
        raise ValueError('ORIGINAL_SCENE_FRAME_AND_FROZEN_NATIVE_STACK')
    return proof


def configuration(spec):
    value=spec.get(KEY)
    if value is None:return None
    from ..sla_adapter.action_contract_v61 import STATS_SHA
    from .storage import digest
    if(value['version']!=VERSION or spec['task_learning_role']!='CORELAB_EXPERT_TRAIN4_SEMANTICS_NOT_BC_V3'
            or spec.get('task_learning_teacher_mode')!='NORMAL_TEACHER_V7'
            or spec.get('target_gr00t_policy') or spec.get('task_learning_control')
            or spec.get('sla_label_replay_v61') or spec.get('native_callback_replay_v61')
            or spec.get('sla_expert_epsilon_v3') or spec.get('sla_pick_place_v5') or spec.get('sla_five_object_hold_v5')
            or spec.get('expert_eef_stream_v2')!=dict(version='EEF_STREAM_V2',prefix_atoms=10)
            or spec.get('expert_dense_observation_v3')is None
            or value['transport_statistics']['sha256']!=STATS_SHA
            or digest(value['transport_statistics']['path'])!=STATS_SHA):
        raise ValueError('EXPLICIT_NEW_NORMAL_EXPERT_FINAL_PLAN_CAPTURE_NOT_OLD_DATA_OR_POLICY')
    return value


class Recorder:
    def __init__(self,adapter):
        from .native_async_journal_v3 import BoundedJournal
        self.adapter=adapter;self.config=configuration(adapter.spec)
        self.writer=BoundedJournal(adapter.output/'FINAL_PLAN_EVENTS.private.jsonl')
        self.plans=[];self.active=None;self.success=False;self.closed=False;self.worker=None

    def emit(self,kind,**fields):
        self.writer.append((json.dumps(dict(kind=kind,**fields),allow_nan=False)+'\n').encode())

    def begin(self,program,*,index,proposal,reference):
        from ..sla_adapter.action_contract_v61 import final_plan
        from .storage import digest
        step=self.adapter.observe()['step'];source=Path(self.adapter.spec['source_root'])
        p=final_plan(program,query_id=Path(self.adapter.spec['registered_root']).name+':'+str(index),
            native_step=step,planner_sha256=digest(source/'src/go2arx5_vla/world_model/eef_stream_v2.py'),
            adapter_sha256=digest(source/'src/go2arx5_vla/world_model/task_learning_grasp_v1.py'),
            normalizer_sha256=None,full_H40_valid=False)
        p.update(reference_before_query=reference.tolist(),base_pose_hold=program.base_pose_hold.tolist(),preplanner_proposal=proposal,
            training_statistics_status='UNFIT_RAW_PHYSICAL_PLAN_NOT_TRAINING_ADMITTED',
            transport_statistics_sha256=self.config['transport_statistics']['sha256'],
            transport_is_oracle_not_student=True,origin_query_index=index,events=[])
        self.plans.append(p);self.active=p;self.emit('FINAL_PLAN_BEFORE_INSTALL',plan=p.copy())

    def transport(self,program,sample):
        import numpy as np
        import mujoco
        from scripts.pi0291_arx5_data.native_dataset import pose10
        from scripts.pi0292.eval_contract import matrix10
        from .sla_dense_policy_v5 import compile_proposal
        from ..trajectory.so3 import rot6d_rows_to_matrix
        from .task_learning_grasp_v1 import MECHANICAL_TRAVEL_M,WORKING_OPEN_M,WORKING_CLOSED_M
        from ..sla_adapter.gripper_latent_v2 import EPSILON
        from ..sla_adapter.action_contract_v61 import STATS_SHA
        a=self.adapter;p=self.active;frame=np.asarray(a.spec['task_phase_source_from_target'])
        state=pose10(frame@np.asarray(sample['tcp']),sample['gripper_fraction'])
        if self.worker is None:
            body=mujoco.mj_name2id(a.actual_state.model,mujoco.mjtObj.mjOBJ_BODY,'x5_base_link')
            if body<0:raise ValueError('ACTUAL_CALIBRATED_MOUNT_BODY_REQUIRED')
            mount=np.eye(4);mount[:3,:3]=a.actual_state.data.xmat[body].reshape(3,3);mount[:3,3]=a.actual_state.data.xpos[body]
            self.mount=frame@mount
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                PYTHONPATH=str(Path(a.spec['source_root'])/'src')+':'+a.spec['source_root'])
            env.pop('LD_PRELOAD',None)
            self.stderr=(a.output/'FINAL_PLAN_CODEC_WORKER.stderr.log').open('x')
            self.worker=subprocess.Popen([self.config['python'],'-B','-m','scripts.sla.final_plan_transport_v61',
                self.config['transport_statistics']['path']],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,
                text=True,bufsize=1,env=env,start_new_session=False)
        folder=a.output/'FINAL_PLAN_TRANSPORT'/str(p['origin_query_index']).zfill(3)
        p.update(transport_query_state_eef10=state.tolist(),transport_policy_from_mount=self.mount.tolist(),
                 policy_from_native_target=frame.tolist())
        request=dict(output=str(folder),state10=state.tolist(),policy_H40=pose10(frame@matrix10(program.decoded_H40),program.decoded_H40[:,9]).tolist(),mount=self.mount.tolist())
        self.worker.stdin.write(json.dumps(request)+'\n');self.worker.stdin.flush()
        with selectors.DefaultSelector()as selector:
            selector.register(self.worker.stdout,selectors.EVENT_READ)
            if not selector.select(90):raise TimeoutError('FINITE_CPU_CODEC_RESPONSE')
            response=json.loads(self.worker.stdout.readline())
        if not response.get('passes'):raise ValueError('ORIGINAL_CODEC_WORKER_FAILED:'+repr(response))
        compiled,proof=compile_proposal(response['action']['path'],response['action']['sha256'],source_from_target=frame,
            query_state10=state,actual_TCP=sample['tcp'],consumed_reference_TCP=program.initial_reference_TCP_target,
            base_hold=program.base_pose_hold)
        old,new=program.reference_path,compiled.reference_path
        xyz=float(np.max(np.linalg.norm(old[:,:3]-new[:,:3],axis=1)))
        r=rot6d_rows_to_matrix(old[:,3:9])@np.swapaxes(rot6d_rows_to_matrix(new[:,3:9]),1,2)
        angle=float(np.max(np.arccos(np.clip((np.trace(r,axis1=1,axis2=2)-1)/2,-1,1))))
        width=float(np.max(np.abs(old[:,9]-new[:,9]))*MECHANICAL_TRAVEL_M)
        if xyz>1e-6 or angle>1e-5 or width>EPSILON*(WORKING_OPEN_M-WORKING_CLOSED_M)+1e-12:
            raise ValueError('EXISTING_FP32_CODEC_REFERENCE_TOLERANCE')
        # The expert plan itself also passes the unchanged full native certifier.
        from .target_sequential import supported_object_hypothesis
        from .native_replay import forbid_integration
        from scripts.pi0294.task_path_model import certify_chunk
        from scripts.pi0292.eval_contract import matrix10
        stage={'CLOSE':'close','LIFT':'lift','HOLD':'hold'}.get(a.task_learning_stage,'approach')
        with forbid_integration(mujoco):
            certificate=certify_chunk(a.actual_state.model,sample,matrix10(program.preview(0)),stage=stage,
                history=a.reference_history[-2:]if len(a.reference_history)>=2 else(),
                planning_object_motion=supported_object_hypothesis(a.spec['phase'],stage,sample))
        p.update(transported_final_H40=compiled.decoded_H40.tolist(),transported_reference_H40=compiled.reference_path.tolist(),
            original_plan_native_certificate=certificate,transport_proof=proof,transport_packet=response['action'],
            transport_reference_error=dict(xyz_m=xyz,so3_rad=angle,mechanical_grip_m=width),
            transport_statistics_sha256=STATS_SHA)
        self.emit('FP32_RAW_STUDENT_TRANSPORT',query_id=p['query_id'],packet=response['action'],proof=proof,
                  reference_error=p['transport_reference_error'])
        return compiled

    def certify(self,reference):
        from ..sla_adapter.action_contract_v61 import identity
        p=self.active;c=reference.get('native_reference_certificate')or{}
        if(reference['native_step']!=p['native_step']or reference['action_EEF10']!=p['transported_reference_H40']
                or c.get('status')!='PASS_CLONED_NATIVE_REFERENCE_CERTIFICATE_NOT_DYNAMIC_WBC_PROOF'
                or c.get('physics_integrations')!=0):raise ValueError('ORIGINAL_FULL_NATIVE_CERTIFICATE_AND_EXACT_ACKED_PACKET')
        p.update(full_H40_valid=True,supervision_mask=[True]*40,native_certificate_sha256=identity(c))
        self.emit('ORIGINAL_NATIVE_CERTIFICATE_AND_INSTALL_ACK',query_id=p['query_id'],
            plan_sha256=p['plan_sha256'],certificate_sha256=p['native_certificate_sha256'],supervision_mask=p['supervision_mask'])

    def confirm(self,k,row):
        from ..sla_adapter.action_contract_v61 import consumed_mask
        p=self.active
        if not p['full_H40_valid']or row['actual_installed_consumed_EEF10_BASE7'][:10]!=p['transported_reference_H40'][k]:
            raise ValueError('EXACT_ORIGIN_TRANSPORTED_COMMAND_CONSUMPTION')
        event=dict(kind='NATIVE_ATOM',k=k,origin_query_id=p['query_id'],command_id=p['command_ids'][k],
            native_start_step=row['native_start_step'],native_end_step=row['native_end_step'],actual_ticks=row['actual_ticks'],
            solver_policies_with_exact_installed_target=bool(row['solver_policies_with_exact_installed_target']),
            gripper_command_consumed=row['gripper']['command_consumed'])
        mask=consumed_mask(p,p['events']+[event])
        p['events'].append(event);p['consumed_mask']=mask;self.emit('CONSUMPTION',event=event)

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            if self.worker is not None:
                self.worker.stdin.close()
                try:self.worker.wait(timeout=10)
                except subprocess.TimeoutExpired:self.worker.terminate();self.worker.wait(timeout=5)
                self.stderr.close()
        finally:
            report=self.writer.close()
        from .storage import write_json
        write_json(self.adapter.output/'FINAL_PLAN_CAPTURE_ACTUAL.private.json',dict(schema=VERSION,
            plans=self.plans,queries=len(self.plans),supervised_points=sum(sum(p['supervision_mask'])for p in self.plans),
            consumed_points=sum(sum(p['consumed_mask'])for p in self.plans),expert_task_success_observed=self.success,
            training_dataset_admitted=False,outer_native_shutdown_qualified=False,oracle_is_student=False,
            SLA_model_queries=0,optimizer_updates=0,writer=report))
