"""Source-bound V2 adapter qualification, actual saved inputs, no physics."""
import ast,inspect,json,unittest
from pathlib import Path
import numpy as np
from .component_progress import ComponentProgress,VERSION
from .component_progress_draft import ComponentProgressDraft
from .component_teacher import Teacher,cards,geometry
from .persistent_teacher import Teacher as PreviousTeacher,geometry as previous_geometry
from .component_qualification import require_adoption
from .task_capture_runtime import read,sha
from .task_state import ActualState


def qualify(root,manifest):
    import mujoco
    from scripts.pi0292.durable import write
    require_adoption(root,manifest)
    modules=('scripts.pi0294.test_persistent_teacher','scripts.pi0294.test_task_clock','scripts.pi0294.test_task_contacts')
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromNames(modules))
    if not tests.wasSuccessful():raise ValueError('V2_CPU_TIME_NATIVE_CONTRACT')
    if ComponentProgress._advance is not ComponentProgressDraft._advance:raise ValueError('ADOPTED_DRAFT_ALGORITHM_CHANGED')
    for method in ('__init__','observe','plan'):
        if inspect.getsource(getattr(Teacher,method))!=inspect.getsource(getattr(PreviousTeacher,method)):
            raise ValueError('UNAUTHORIZED_TASK_TEACHER_CHANGE:'+method)
    if inspect.getsource(geometry)!=inspect.getsource(previous_geometry):raise ValueError('ACTUAL_GEOMETRY_CHANGED')
    if len(cards())!=6 or sum(round((c['duration_s']+5)/.002)+3 for c in cards())!=71018:
        raise ValueError('FIXED_V2_CARDS_BUDGET')
    initial=np.eye(4);initial[2,3]=.72;target=initial.copy();target[0,3]=.5
    def fixture(restore=False):
        core=ComponentProgress(task='T1',actual_tcp=initial,stage_target=target,actual_BASE=np.eye(4),
            station_world=np.eye(4),pad_radius_m=.05);values=[]
        for i in range(8):
            actual=core.cursor.copy();t=i*.4
            core.observe(time_s=t,actual_tcp=actual,actual_BASE=np.eye(4))
            p=core.plan(time_s=t,actual_tcp=actual,actual_BASE=np.eye(4))
            values.append(np.r_[p['H40_world_poses'].ravel(),p['BASE_soft_velocity'].ravel()])
            if restore and i==3:core=ComponentProgress.restore(json.loads(json.dumps(core.snapshot())))
        return np.array(values)
    if not np.array_equal(fixture(),fixture(True)):raise ValueError('EXACT_V2_CORE_RESTORE_8_VS_4_PLUS_4')
    saved=[]
    for old in sorted((Path(root)/'teacher-qualification-v1/episodes').iterdir()):
        samples=[json.loads(s) for s in (old/'samples.jsonl').read_text().splitlines()]
        card0=read(old/'CARD.json');card=next(c for c in cards() if c['task']==card0['task'] and c['scene_class']==card0['scene_class'] and c['duration_s']==12.)
        model=mujoco.MjModel.from_binary_path(str(old/'task-scene.private.mjb'));engine=ActualState(model)
        teacher=None;count=0
        for i in range(57):
            raw=samples[i];packet={k:raw[k] for k in ('qpos','qvel','ctrl')}
            packet.update(step=2500+25*i,time_s=5.+i*.05,contacts=[],contact_time_s=5.+i*.05-.002)
            state=engine.update(packet);state['time_s']=i*.05
            for key in ('native_grasp','safety_valid','robot_object_contact','left_object_contact','right_object_contact'):state[key]=raw[key]
            base=np.asarray(raw['actual_BASE_mount'])
            if teacher is None:teacher=Teacher(card,state,engine,base)
            teacher.observe(state,base)
            if teacher.stage!='approach':raise ValueError('SAVED_FAR_INPUT_FORCED_CONTACT')
            if i%8==0:
                plan=teacher.plan(state,base);count+=1
                if plan['poses'].shape!=(40,4,4) or plan['BASE_soft_velocity'].shape!=(40,3):raise ValueError('V2_REAL_INPUT_SHAPE')
                if plan['persistent']['version']!=VERSION:raise ValueError('ACTUAL_CORE_VERSION_BINDING')
        saved.append(dict(id=old.name,matched_card=card['id'],queries=count,events=teacher.events,
            raw_samples_sha256=sha(old/'samples.jsonl'),model_sha256=sha(old/'task-scene.private.mjb')))
    code=Path(__file__).parent
    for name in ('component_progress.py','component_teacher.py','component_teacher_worker.py','component_qualification.py'):
        ast.parse((code/name).read_text())
    expected=(code/'persistent_teacher_worker.py').read_text()
    expected=expected.replace('from .persistent_teacher import VERSION,Teacher,cards','from .component_teacher import VERSION,Teacher,cards')
    expected=expected.replace('from .teacher_qualification import require_card,require_adoption','from .component_qualification import require_card,require_adoption')
    expected=expected.replace("series=root/'teacher-qualification-v1'","series=root/'teacher-qualification-v2'")
    if expected!=(code/'component_teacher_worker.py').read_text():raise ValueError('UNEXPECTED_NATIVE_WORKER_CHANGE')
    proof=dict(status='PASS_ZERO_PHYSICS_RUNTIME_CONTRACT',version=VERSION,source_sha=manifest['code']['vla_commit'],
        narrow_tests=tests.testsRun,continuous8_equals4_JSON_restore4=True,
        actual_saved_input_cases=saved,teacher_geometry_and_observe_identical=True,
        native_worker_only_namespace_and_import_changes=True,clock_repair_carried_unchanged=True,
        algorithm_is_exact_adopted_component_draft=True,new_physics=0,new_render=0,new_GR00T_queries=0,
        physical_milestones='NOT_YET_OBSERVED',selected='NONE',deployment='NONE')
    folder=Path(root)/'teacher-qualification-v2'
    if (folder/'CPU_ADMISSION.json').exists():raise FileExistsError('FIRST_CPU_ADMISSION_PRESERVED')
    write(folder/'CPU_ADMISSION.json',proof)
    return proof
