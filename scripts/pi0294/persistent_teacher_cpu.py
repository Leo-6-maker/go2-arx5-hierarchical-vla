"""Actual saved input and strict prefix contracts; no physics/renderer/model."""
import ast,json,unittest
from pathlib import Path
import numpy as np
from .persistent_teacher import Teacher,cards,VERSION,pad_corners
from .task_state import ActualState
from .task_capture_runtime import read,sha


def qualify(root,manifest):
    import mujoco
    from scripts.pi0292.durable import write
    modules=('scripts.pi0294.test_persistent_teacher','scripts.pi0294.test_persistent_approach_draft',
        'scripts.pi0294.test_task_clock','scripts.pi0294.test_task_contacts','scripts.pi0294.test_native_receiver')
    suite=unittest.TestLoader().loadTestsFromNames(modules)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise ValueError('PERSISTENT_TEACHER_CPU_CONTRACT_FAILURE')
    folder=Path(root)/'teacher-qualification-v1';rows=[]
    for name in ('TASK-PILOT-W0-T2-engineering04','TASK-PILOT-W1-T1','TASK-PILOT-W1-T2'):
        old=Path(root)/'episodes'/name
        samples=[json.loads(s) for s in (old/'samples.jsonl').read_text().splitlines()]
        intents={r['index']:r for r in (json.loads(s) for s in (old/'committed_intents.jsonl').read_text().splitlines())
            if r.get('phase')=='SUBMITTED_REQUIRES_INSTALL_ACK'}
        m=mujoco.MjModel.from_binary_path(str(old/'task-scene.private.mjb'));engine=ActualState(m)
        oldcard=read(old/'CARD.json');card=next(c for c in cards() if c['task']==oldcard['task'] and c['scene_class']==oldcard['scene_class'] and c['duration_s']==12.)
        teacher=None;queries=[]
        for i in range(57):
            raw=samples[i];packet={k:raw[k] for k in ('qpos','qvel','ctrl')}
            packet.update(time_s=5.+i*.05,step=2500+25*i,contacts=[],contact_time_s=5.+i*.05-.002)
            state=engine.update(packet);state['time_s']=i*.05
            for key in ('native_grasp','safety_valid','robot_object_contact','left_object_contact','right_object_contact'):
                state[key]=raw[key]
            # Use the saved measured mount, never combine unrelated RGB/state.
            intent=intents[i]
            base=np.asarray(intent['actual_BASE_mount'])
            if teacher is None:teacher=Teacher(card,state,engine,base)
            teacher.observe(state,base)
            if teacher.stage!='approach':raise ValueError('FAR_SAVED_INPUT_FORCED_CONTACT')
            if i%8==0:
                plan=teacher.plan(state,base)
                if plan['poses'].shape!=(40,4,4) or plan['BASE_soft_velocity'].shape!=(40,3):raise ValueError('ACTUAL_INPUT_H40_SHAPE')
                lead=np.linalg.norm(plan['poses'][:,:3,3]-state['tcp'][:3,3],axis=1)
                # Old actual motion may drift away from an already-issued
                # cursor; record this explicitly rather than clipping state.
                queries.append(dict(index=i,time_s=state['time_s'],max_lead_m=float(lead.max()),
                    cursor=plan['persistent']['cursor'],segment=plan['persistent']['segment'],
                    far_precontact_ready=plan['geometry']['precontact_geometry_ready']))
        rows.append(dict(id=name,actual_input_queries=queries,source_samples_sha256=sha(old/'samples.jsonl'),
            compiled_model_sha256=sha(old/'task-scene.private.mjb'),events=teacher.events))
    code=Path(__file__).parent
    for name in ('persistent_teacher.py','persistent_teacher_worker.py','persistent_teacher_receiver.py','teacher_qualification.py'):
        ast.parse((code/name).read_text())
    receipt=dict(status='PASS_ZERO_PHYSICS_RUNTIME_CONTRACT',version=VERSION,source_sha=manifest['code']['vla_commit'],
        tests=result.testsRun,saved_actual_input_cases=rows,new_integrations=0,new_rendered_pairs=0,new_GR00T_queries=0,
        full_robot_dynamic_reachability='UNPROVEN_CONTROLLED_DEVELOPMENT_USES_ORIGINAL_NATIVE_SAFETY',
        static_asset_and_initial_contacts='RECHECKED_BEFORE_EACH_NATIVE_START',
        selected_MPC_policy_consumption='UNKNOWN_REQUIRES_LIVE_OBSERVATION_NOT_INSTALL_ACK',
        does_not_assert_physical_milestones=True)
    write(folder/'CPU_ADMISSION.json',receipt)
    return receipt
