"""Seal the complete new TRAIN population; this grants no Processor admission."""
from pathlib import Path
import shutil
import numpy as np
from . import task_learning_grasp_v1 as task
from . import task_learning_collection_v1 as collection
from .stable_task_dataset_v1 import load
from .task_learning_dataset_v1 import audited_episode,fit_statistics
from .storage import digest,write_json

SCHEMA='TASK_LEARNING_CONTINUOUS_GR00T_DATASET_V1'
DOMAIN='GO2_ARX5_GRASP_RED_LIFT20MM_HOLD1S_SIM_V1'
PROFILE='TASK_LEARNING_GR00T_V1'


def terminal_group(group,panel,panel_pin):
    """A finite group may stop at a closed prefix; missing requests stay missing."""
    group=Path(group)
    inp,ip=load(group/'INPUT.private.json',maximum=16*1024**2)
    lease,lp=load(group/'control/LEASE.json',maximum=1024**2)
    result,rp=load(group/'RESULT.private.json',maximum=16*1024**2)
    outcomes,op=load(group/'OUTCOMES.private.json',maximum=16*1024**2)
    selection=inp.get('assignment_slice',[0,len(panel['assignments'])])
    task.need(inp.get('task_learning_role')==task.COLLECTION_ROLE
        and inp.get('source_sha')==panel['source_sha']
        and inp.get('panel_sha256')==panel_pin
        and type(selection)is list and len(selection)==2
        and all(type(v)is int for v in selection)
        and 0<=selection[0]<selection[1]<=100,'ACTUAL_DECLARED_CONTIGUOUS_COLLECTION_GROUP')
    expected=panel['assignments'][selection[0]:selection[1]]
    complete=result.get('exit_code')==0
    finite_prefix=(result.get('exit_code')==1
        and str(result.get('failure','')).startswith('RuntimeError: FINITE_PANEL_NO_LATE_NEW_EPISODE;'))
    task.need(lease.get('status')=='CLOSED'and lease.get('input_sha256')==ip['sha256']
        and lease.get('worker_exit_code')==result.get('exit_code')
        and result.get('source_sha')==panel['source_sha']and result.get('panel_sha256')==panel_pin
        and result.get('assignment_slice')==selection and result.get('selected_requests')==len(expected)
        and result.get('completed_in_group')==len(outcomes)
        and 0<len(outcomes)<=len(expected)
        and [v['native_run']for v in outcomes]==[v['native_run']for v in expected[:len(outcomes)]]
        and ((complete and len(outcomes)==len(expected)
              and result.get('status')=='TASK_LEARNING_NATIVE_GROUP_COMPLETE'and result.get('failure')is None)
             or(finite_prefix and result.get('status')=='TASK_LEARNING_NATIVE_GROUP_INCOMPLETE_PRESERVED')),
        'CLOSED_GROUP_COMPLETE_OR_EXPLICIT_FINITE_PREFIX_NEVER_UNKNOWN_RETRY')
    task.need(result.get('summary')==collection.summarize(panel,outcomes),'GROUP_COUNTS_RECOMPUTE_FROM_ACTUAL_OUTCOMES')
    return outcomes,dict(input=ip,lease=lp,result=rp,outcomes=op)


def audit_population(panel_path,groups):
    panel,pp=load(panel_path,maximum=16*1024**2)
    rows=collection.validate(panel)
    task.need(panel.get('task_learning_role')==task.COLLECTION_ROLE and len(rows)==100,
        'COMPLETE_NEW_TRAIN100_NOT_CONSTRUCTION_OR_QUALIFICATION')
    outcomes=[];roots={};group_proofs=[]
    for group in groups:
        values,proof=terminal_group(group,panel,pp['sha256']);group_proofs.append(proof)
        for value in values:
            name=value['native_run']
            task.need(name not in roots,'NO_REPEATED_CASE_ACROSS_FINITE_GROUPS')
            roots[name]=Path(group)/'runs'/name
        outcomes.extend(values)
    summary=collection.summarize(panel,outcomes)
    task.need(summary['complete']is True and summary['passes']is True,
        'ALL100_ATTEMPTS_MIN80_FULL_MIN20_LAYOUTS_MIN20_RECOVERIES_MIN10_LAYOUTS_HARD0_TECHNICAL0')
    by_name={v['native_run']:v for v in outcomes};kept=[];attempts=[]
    for row in rows:
        name=row['native_run'];outcome=by_name[name];root=roots[name]
        worker,wp=load(root/'WORKER_EXIT.private.json',maximum=1024**2)
        lease,lp=load(root/'control/LEASE.json',maximum=1024**2)
        spec,sp=load(root/'EXECUTION_SPEC.json',maximum=16*1024**2)
        task.need(lease.get('status')=='CLOSED'and lease.get('spec_sha256')==sp['sha256']
            and lease.get('worker_exit_code')==worker.get('exit_code')
            and spec.get('registered_root')==str(root)and spec.get('source_sha')==panel['source_sha']
            and spec.get('phase')==task.PHASE and spec.get('kind')==task.KIND
            and spec.get('task_learning_role')==task.COLLECTION_ROLE
            and spec.get('stable_task_panel_sha256')==pp['sha256']
            and spec.get('data_role')==row['split']=='TRAIN'and spec.get('data_parent')==row['parent']
            and spec.get('scene_id')==row['parent']and spec.get('task_command')==row['command']==task.COMMAND
            and digest(root/'native-01/RESULT.json')==outcome.get('result_sha256'),
            'EVERY_SUCCESS_AND_FAILURE_RETAINS_ITS_ACTUAL_CLOSED_CASE_IDENTITY')
        if outcome['task_success']and outcome['reset_ready']and outcome['technical_valid']:
            episode=audited_episode(root,row,pp['sha256'],panel['source_sha'])
            task.need(episode['genuine_recovery_success']==outcome.get('genuine_recovery_success'),
                'ACTUAL_RECOVERY_WITNESS_MATCHES_POPULATION_COUNTS')
            kept.append(episode)
            attempts.append(dict(row,status='FULL_CONTINUOUS_NATIVE_EPISODE_PENDING_PROCESSOR',proof=episode['proof']))
        else:
            attempts.append(dict(row,status='RETAINED_FAILED_OR_PARTIAL_EPISODE_NO_FULL_H40_SUPERVISION',
                outcome=outcome,proof=dict(worker=wp,lease=lp,spec=sp)))
    recovered=[e for e in kept if e['genuine_recovery_success']]
    task.need(len(kept)==summary['task_successes']and len(recovered)==summary['genuine_recovery_successes']
        and len({e['native_binary_sha256']for e in kept})==1,'SAME_QUALIFIED_NATIVE_RUNTIME_AND_AUDITED_COUNTS')
    frame=kept[0]['captures'][0]['frame']
    task.need(all(np.array_equal(e['captures'][0]['frame'],frame)for e in kept),'ONE_DECLARED_TARGET_TO_POLICY_FRAME_ACROSS_POPULATION')
    return dict(panel=pp,groups=group_proofs,summary=summary,episodes=kept,attempts=attempts)


def write_episode(e,output):
    """Serialize an already audited episode without changing its original split."""
    output=Path(output);row=e['row'];split=row['split']
    task.need((split,row['purpose'])in(('TRAIN','CONTINUOUS_TEACHER_DEMONSTRATION'),
        ('DEV','DEV_REFERENCE_NOT_BC')),'EXPLICIT_TRAIN_OR_DEV_REFERENCE_SERIALIZATION')
    name=row['native_run'];folder=output/'media'/name;folder.mkdir();paths=[];pins=[]
    for i,capture in enumerate(e['captures']):
        frame_paths=[];frame_pins=[]
        for camera,original in zip(('body_rgb','wrist_rgb'),capture['images']):
            target=folder/f'{i:06d}-{camera}.png';shutil.copyfile(original,target)
            task.need(digest(target)==digest(original),'UNCHANGED_REAL_DUAL_RGB_MEDIA')
            frame_paths.append(target.relative_to(output).as_posix());frame_pins.append(digest(target))
        paths.append(frame_paths);pins.append(frame_pins)
    packet=dict(e['packet'],rgb_paths=np.asarray(paths),rgb_sha256=np.asarray(pins))
    path=output/'episodes'/(name+'.npz');np.savez_compressed(path,**packet)
    return dict(id=name,split=split,parent=row['parent'],command=row['command'],purpose=row['purpose'],
        file=path.relative_to(output).as_posix(),sha256=digest(path),
        supervised_anchors=np.flatnonzero(packet['valid_window']).tolist(),
        optimization_anchors=e['optimization_anchor_schedule']if split=='TRAIN'else[],real_policy_decisions=15,
        supervised_policy_decisions=int(packet['valid_window'].sum()),
        expert_stage_audit_only=packet['expert_stage_audit_only'].tolist(),
        genuine_recovery_success=e['genuine_recovery_success'],proof=e['proof'])


def build_train_base(panel_path,groups,output):
    """TRAIN-only base; DEV references and the actual Processor still gate SFT."""
    output=Path(output)
    task.need(output.is_absolute()and output.parent.resolve()==output.parent and not output.exists(),'FRESH_CANONICAL_TRAIN_BASE_EXPORT')
    checked=audit_population(panel_path,groups)
    output.mkdir();(output/'episodes').mkdir();(output/'media').mkdir()
    rows=[write_episode(e,output)for e in checked['episodes']]
    write_json(output/'statistics.json',fit_statistics(checked['episodes']))
    manifest=dict(schema=SCHEMA,profile=PROFILE,domain=DOMAIN,bundle_kind='CONTINUOUS_TRAIN_BASE',
        training_admitted=False,requires='FIXED_DEV_REFERENCES_AND_ACTUAL_ORIGINAL_PROCESSOR_ADMISSION',
        panel=checked['panel'],groups=checked['groups'],population=checked['summary'],
        episodes=rows,attempts=checked['attempts'],statistics_sha256=digest(output/'statistics.json'),
        TRAIN_only_statistics=True,statistics_fit_before_oversampling=True,
        native_binary_sha256=checked['episodes'][0]['native_binary_sha256'],
        T_policy_from_target=checked['episodes'][0]['captures'][0]['frame'].tolist(),
        equal_episode_optimization_visits=32,boundary_visits_per_episode=8,
        policy_H40_period_s=.05,compiler_time_stretch_factor=1,physics_dt_s=.002,
        labels='ACTUAL_CONTINUOUS_NOMINAL_TEACHER_H40_WITH_EACH_KNOT_SOLVER_GRIPPER_CONFIRMED',
        policy_action_boundary='BEFORE_ORIGINAL_REFERENCE_TRANSPORT_AND_WORKING_GRIPPER_AFFINE',
        student_state='ORIGINAL_CURRENT_SHADOW_ABI_EEF10_UNMODIFIED',
        original_Processor_route='new_embodiment',target_category=10,
        embodiment_provenance='EXPLICIT_TARGET_INTERFACE_INITIALIZED_FROM_COMPATIBLE_SOURCE_CATEGORY_NOT_PRETRAINED_GO2',
        privileged_Teacher_metadata_is_student_input=False,interventions_masked=True,
        construction_or_qualification_relabelled=False,TEST_results_read=False)
    write_json(output/'DATASET_MANIFEST.json',manifest)
    return dict(manifest_sha256=digest(output/'DATASET_MANIFEST.json'),statistics_sha256=manifest['statistics_sha256'],
        TRAIN_episodes=len(rows),unique_supervised_decisions=sum(v['supervised_policy_decisions']for v in rows),
        optimization_visits_per_population_pass=32*len(rows),training_admitted=False,
        new_GR00T_updates=0,new_native_integrations=0)
