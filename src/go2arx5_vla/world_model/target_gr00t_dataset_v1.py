"""Real target demonstrations -> original Processor/NativeDataset seam.

This dataset has its own target domain and admission. Old HOLD, diagnostics,
failed tasks, WM records and unexecuted previews cannot enter optimization.
"""
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image

from scripts.pi0292.relay_protocol import unseal
from scripts.pi0291_arx5_data.native_dataset import matrix_from10, pose10
from scripts.pi0292.train_adapter import NativeDataset as OriginalDataset, TAG
from scripts.pi0294.multitask.source_export import statistics
from .stable_task_dataset_v1 import load, NATIVE
from . import stable_task_data_v1 as data
from . import target_gr00t_collection_v1 as collection
from .storage import digest, write_json

DOMAIN = 'GO2_ARX5_FLOATING_BASE_TARGET_SHORT_TASK_SIM_V1'
SCHEMA = 'WM02_TARGET_GR00T_DEMONSTRATIONS_V1'


def need(ok, reason):
    if not ok:
        raise ValueError('TARGET_GR00T_DATASET_' + reason)


def actual_commands(q):
    """Use each solver/gripper-confirmed 50ms command exactly once."""
    from .stable_task_dataset_v1 import consumed_label
    consumed_label(q)  # Existing full40, 176 interval and exact sequence joins.
    rows = q['generated_segment']['windows']
    return np.asarray([r['actual_installed_consumed_EEF10_BASE7'][:10] for r in rows],float)


def policy_commands(root,q):
    """One actual Teacher proposal before deterministic compilation, not renewals."""
    proposal,pp=load(root/'native-01/STABLE_TASK_EXPERT_PROPOSAL.private.json',
                     q['executed_reference_proposal_sha256'],maximum=16*1024**2)
    decision,dp=load(root/'native-01/LIVE_SKILL_SEGMENT_DECISION.private.json',maximum=16*1024**2)
    raw=np.asarray(proposal['original_physical_H40'],float)
    need(raw.shape==(40,10)and np.isfinite(raw).all()
         and np.array_equal(raw,np.asarray(decision['raw_H40']))
         and np.array_equal(raw,np.asarray(q['generated_segment']['original_raw_H40'])),
         'SAME_PRECOMMITTED_TEACHER_POLICY_PROPOSAL')
    desired=np.asarray(decision['desired_path']);reference=np.asarray(decision['nominal_reference_path'])
    knots=q['adaptation']['clock']['knot_indices'];rows=q['generated_segment']['windows']
    from .fusion_trajectory_adapter import WORKING_CLOSED_M,WORKING_OPEN_M,MECHANICAL_TRAVEL_M
    expected=raw.copy()
    expected[:,9]=(WORKING_CLOSED_M+(WORKING_OPEN_M-WORKING_CLOSED_M)*raw[:,9])/MECHANICAL_TRAVEL_M
    need(len(knots)==40 and np.allclose(desired[knots],expected,rtol=0,atol=1e-12),
         'ALL_TEACHER_KNOTS_PRESERVED_BY_COMPILER')
    consumed=actual_commands(q)
    need(all(np.array_equal(consumed[k],reference[k])for k in knots),
         'EVERY_RAW_TEACHER_KNOT_ACTUALLY_COMPILED_INSTALLED_AND_CONSUMED')
    return raw,dict(proposal=pp,compiler_decision=dp,raw_knot_to_consumed_atom=knots,
        policy_decisions=1,renewed_previews_as_additional_expert_decisions=0,
        working_gripper_affine_applied_once_by_compiler=True,
        raw_policy_grip_preserved_in_label=True)


def capture(root, spec, spec_pin, identity):
    from .fusion_observation import shadow_inference_payload
    p = Path(identity['path'])
    need(p.is_relative_to(root/'native-01/PAUSED_OBSERVATIONS'),'OWN_CAPTURE')
    record, descriptor = load(p,identity['sha256'],maximum=16*1024**2)
    b = unseal(record)
    need(b['native_step'] == identity['native_step'] == collection.B+25*identity['atom']
         and b['world']['execution_spec_sha256'] == spec_pin
         and b['world']['scene_sha256'] == digest(spec['model_path'])
         and b['task']['language'] == collection.LANGUAGES[spec['task_command']], 'SAME_PACKET_TASK_SCENE_SPEC')
    for name,pin in b['files'].items():
        member = p.parent/name
        need(member.resolve().is_relative_to(p.parent) and digest(member) == pin,'MEDIA_AND_STATE_HASH')
    shadow = unseal(b['original_shadow_ABI'])
    payload = shadow_inference_payload(b['original_shadow_ABI'])
    need(shadow['native_step'] == b['native_step']
         and shadow['target_observation']['native_state_sha256'] == b['files']['STATE.private.bin'],
         'ACTUAL_OBSERVATION_NOT_ACHIEVED_FUTURE')
    images = []
    for camera in ('body_rgb','wrist_rgb'):
        path = p.parent/'images'/('000000-'+camera+'.png')
        with Image.open(path) as im:
            need(im.mode == 'RGB' and im.size == (256,256),'REAL_RGB256')
        images.append(path)
    return dict(state=np.asarray(payload['state_eef10'],float),images=images,
        frame=np.asarray(shadow['source_policy_view']['T_source_from_target'],float),descriptor=descriptor)


def episode(root, row, panel_pin, *, qualification_only=False):
    root = Path(root)
    spec, sp = load(root/'EXECUTION_SPEC.json',maximum=16*1024**2)
    need(spec.get('phase') == data.PHASE and spec.get('kind') == data.KIND
         and spec.get('stable_task_panel_sha256') == panel_pin
         and spec.get('data_role') == row['split'] and spec.get('data_parent') == row['parent']
         and spec.get('task_command') == row['command']
         and spec.get('target_gr00t_attempt_role') == row['purpose'], 'NEW_ROLE_AND_PROFILE_NO_RELABEL')
    data.validate(spec)
    need(row['purpose'] == ('QUALIFICATION' if qualification_only else 'DEMONSTRATION'), 'QUALIFICATION_NOT_BC')
    lease, lp = load(root/'control/LEASE.json',maximum=1024**2)
    worker, wp = load(root/'WORKER_EXIT.private.json',maximum=1024**2)
    need(lease.get('status') == 'CLOSED' and lease.get('spec_sha256') == sp['sha256']
         and type(worker.get('exit_code')) is int and worker['exit_code'] == lease.get('worker_exit_code') == 0
         and worker.get('failure') is None,'ACTUAL_CLOSED0')
    result, rp = load(root/'native-01/RESULT.json')
    trace = result.get('trace_audit') or {}; writer = trace.get('writer') or {}
    children = result.get('owned_process_exits') or []
    need(result.get('status') == 'NATIVE_STAGE_PASSED' and result.get('phase') == data.PHASE
         and result.get('failure') is None and result.get('actual_target_integrations') == collection.E
         and result.get('native_binary_sha256') == NATIVE and result.get('shutdown_clean') is True
         and len(children) == 4 and {r['name'] for r in children} == {'sim','solver','controller','planner'}
         and all(type(r.get('exit_code')) is int and r['exit_code'] == 0 for r in children)
         and result.get('original_joint_or_warning_samples') == 0
         and result.get('native_outer_guard_rejection_count') == 0 and not result.get('physics_stops')
         and result.get('trace_audit_error') is None
         and trace.get('status') == 'RECORDED_TRACE_CHECKS_PASS' and trace.get('native_step_records') == collection.E
         and writer.get('dropped') == 0 and writer.get('events') == writer.get('written'), 'SAFE_COMPLETE_NATIVE_PROOF')
    q = result['reference_qualification']
    need(q.get('schema') == 'WM02_TARGET_GR00T_EXPERT_NATIVE_ACTUAL_V1'
         and q.get('target_gr00t_collection') == collection.contract()
         and q.get('target_gr00t_attempt_role') == row['purpose']
         and q.get('task_condition_success') is True
         and (q.get('target_gr00t_task_score') or {}).get('task_success') is True
         and q.get('raw_reference_is_GR00T') is False
         and q.get('GR00T_updates') == q.get('GR00T_queries',0) == 0, 'SUCCESSFUL_REAL_EXPERT_NOT_WM_OR_FAILURE')
    commands = actual_commands(q)
    teacher,policy_proof=policy_commands(root,q)
    descriptor = q['demonstration_capture_index']
    index, ip = load(descriptor['path'],descriptor['sha256'],maximum=16*1024**2)
    identities = index.get('captures') or []
    need(index.get('complete') is True and index.get('task_success') is True
         and [r['atom'] for r in identities] == list(collection.CAPTURE_ATOMS), 'WHOLE_EPISODE_IMAGES')
    captures = [capture(root,spec,sp['sha256'],r) for r in identities]
    frame = captures[0]['frame']
    need(all(np.array_equal(r['frame'],frame) for r in captures),'SAME_FIXED_POLICY_FRAME')
    mapped_commands = pose10(frame @ matrix_from10(teacher),teacher[:,9])
    mapped_consumed = pose10(frame @ matrix_from10(commands),commands[:,9])
    states = np.asarray([r['state'] for r in captures])
    need(states.shape == (27,10) and mapped_commands.shape == (40,10)
         and np.isfinite(states).all() and np.isfinite(mapped_commands).all(),'ACTUAL_NUMERIC_DIMENSIONS')
    anchors = np.asarray(collection.CAPTURE_ATOMS,dtype=np.int64)
    # The Teacher made ONE policy proposal at B. Renewing its controller
    # preview at later captures does not create another expert policy query.
    valid = anchors==0
    return dict(packet=dict(actual_state10=states,committed_absolute10=mapped_commands,
        native_command_anchor=anchors,valid_window=valid,
        installed_consumed_50ms_audit_only=mapped_consumed,
        instruction=np.asarray(collection.LANGUAGES[spec['task_command']]),
        label_representation=np.asarray('ABSOLUTE_COMMITTED_EEF10'),
        command_period_s=np.asarray(.05), native_dt_s=np.asarray(.002)),
        captures=captures,proof=dict(spec=sp,lease=lp,worker=wp,result=rp,capture_index=ip,policy=policy_proof),
        row=row, qualification_only=qualification_only)


def fit_statistics(episodes):
    """TRAIN-only actual inputs and the corresponding executed H40 windows."""
    states, actions, relative = [], [], []
    for e in episodes:
        need(e['row']['split'] == 'TRAIN' and e['row']['purpose'] == 'DEMONSTRATION'
             and e['qualification_only'] is False,'TRAIN_ONLY_STATISTICS')
        p = e['packet']; valid = p['valid_window']
        s = p['actual_state10'][valid]
        need(int(valid.sum())==1 and p['native_command_anchor'][valid].tolist()==[0],
             'ONE_REAL_TEACHER_POLICY_DECISION_NOT_RENEWED_PREVIEWS')
        a = p['committed_absolute10'][None]
        states.append(s);actions.append(a)
        relative.append(pose10(np.linalg.inv(matrix_from10(s))[:,None]@matrix_from10(a),a[...,9])[...,:9])
    need(bool(states),'NONEMPTY_REAL_TRAIN')
    s,a,r = np.concatenate(states),np.concatenate(actions),np.concatenate(relative)
    return {TAG:dict(state=dict(eef_9d=statistics(s[:,:9],0),gripper_position=statistics(s[:,9:],0)),
        action=dict(eef_9d=statistics(a[...,:9],(0,1)),gripper_position=statistics(a[...,9:],(0,1))),
        relative_action=dict(eef_9d=statistics(r,(0,1))))}


def build(panel_path, runs_root, output, *, bundle_kind='FORMAL'):
    panel, pp = load(panel_path,maximum=16*1024**2)
    need(panel.get('schema') == collection.PANEL_SCHEMA and panel.get('collection') == collection.contract(), 'FROZEN_PANEL')
    need(bundle_kind in ('FORMAL','FIT4_DIAGNOSTIC'),'EXPLICIT_DATA_BUNDLE_PURPOSE')
    output = Path(output)
    need(output.is_absolute() and output.parent.resolve() == output.parent and not output.exists(),'FRESH_EXPORT')
    kept, attempts = [], []
    for row in panel['assignments']:
        root = Path(runs_root)/row['native_run']
        if row['purpose'] == 'QUALIFICATION':
            attempts.append(dict(row,status='QUALIFICATION_EXCLUDED_FROM_BC'));continue
        if bundle_kind == 'FIT4_DIAGNOSTIC' and not (
                row['split']=='TRAIN' and row['parent'] in ('STABLE_TASK_LAYOUT_TRAIN01','STABLE_TASK_LAYOUT_TRAIN02')
                and row['restart']==1):
            attempts.append(dict(row,status='PREREGISTERED_NOT_INCLUDED_IN_FIT4_DIAGNOSTIC'));continue
        need(row['split'] in ('TRAIN','DEV'),'NO_TEST_READ_DURING_DEVELOPMENT')
        # A missing/running attempt prevents finalization; it is not called a failure.
        need((root/'WORKER_EXIT.private.json').is_file(),'ATTEMPT_NOT_TERMINAL:'+root.name)
        lease,_ = load(root/'control/LEASE.json',maximum=1024**2)
        need(lease.get('status') == 'CLOSED','ATTEMPT_STILL_OPEN:'+root.name)
        try:
            e = episode(root,row,pp['sha256'])
        except (ValueError,KeyError,FileNotFoundError) as error:
            attempts.append(dict(row,status='EXCLUDED_RETAINED_FAILURE',reason=type(error).__name__+': '+str(error)))
        else:
            kept.append(e);attempts.append(dict(row,status='ELIGIBLE_SUCCESSFUL_SHORT_DEMONSTRATION',proof=e['proof']))
    train = [e for e in kept if e['row']['split'] == 'TRAIN']
    need(len(train) >= (4 if bundle_kind=='FIT4_DIAGNOSTIC'else panel['minimum_TRAIN_episode_admission'])
         and len({e['row']['parent'] for e in train}) >= (2 if bundle_kind=='FIT4_DIAGNOSTIC'else panel['minimum_TRAIN_layouts'])
         and {e['row']['command'] for e in train} == set(collection.COMMANDS),'MINIMUM_DIVERSE_TRUE_EXPERT_TRAIN')
    if bundle_kind=='FORMAL':
        need({e['row']['command'] for e in kept if e['row']['split']=='DEV'} == set(collection.COMMANDS),'INDEPENDENT_DEV_REQUIRED')
    output.mkdir();(output/'episodes').mkdir();(output/'media').mkdir()
    rows = []
    for e in kept:
        row = e['row']; name = row['native_run'];media=output/'media'/name;media.mkdir()
        paths,pins = [],[]
        for i,c in enumerate(e['captures']):
            dest=[];hashes=[]
            for camera,p in zip(('body_rgb','wrist_rgb'),c['images']):
                target = media/f'{i:06d}-{camera}.png';shutil.copyfile(p,target)
                need(digest(target)==digest(p),'PRIVATE_MEDIA_TRANSFER_BYTES')
                dest.append(target.relative_to(output).as_posix());hashes.append(digest(target))
            paths.append(dest);pins.append(hashes)
        packet=dict(e['packet'],rgb_paths=np.asarray(paths),rgb_sha256=np.asarray(pins))
        numeric=output/'episodes'/(name+'.npz');np.savez_compressed(numeric,**packet)
        rows.append(dict(id=name,split=row['split'],parent=row['parent'],command=row['command'],
            file=numeric.relative_to(output).as_posix(),sha256=digest(numeric),windows=1,
            raw_root=str(output),task='TARGET_DIRECTIONAL_SHORT_V1',family='GO2_ARX5_REAL_TARGET',
            origin='NEW_EXECUTED_EXPERT_SHORT_EPISODE',proof=e['proof']))
    write_json(output/'statistics.json',fit_statistics(train))
    manifest=dict(schema=SCHEMA,profile='TARGET_GR00T_V1',domain=DOMAIN,bundle_kind=bundle_kind,
        training_admitted='REQUIRES_PINNED_TARGET_PROCESSOR_ADMISSION',
        statistics_sha256=digest(output/'statistics.json'),episodes=rows,attempts=attempts,
        panel=pp,collection=collection.contract(),TRAIN_only_statistics=True,
        original_Processor_route=TAG,target_category=10,
        embodiment_provenance='EXPLICIT_TARGET_INTERFACE_INITIALIZED_FROM_COMPATIBLE_SOURCE_CATEGORY_NOT_PRETRAINED_GO2',
        labels='ONE_REAL_TEACHER_POLICY_H40_VERIFIED_THROUGH_ALL_ORIGINAL_COMPILER_KNOTS',
        policy_action_boundary='BEFORE_ORIGINAL_TRAJECTORY_ADAPTATION_NOT_ALREADY_ADAPTED_REFERENCE',
        policy_H40_period_s=.05,compiler_time_stretch_factor=2,
        controller_interval_s=.05,RGB_and_state='SAME_ACTUAL_PAUSED_CAPTURE_AT_VALID_ANCHORS',
        all_process_RGB_pairs_per_episode=27,supervised_policy_decisions_per_episode=1,
        remaining_process_images_are_new_expert_queries=False,
        TEST_results_read=False,WM_or_HOLD_or_diagnostic_relabelled=False)
    write_json(output/'DATASET_MANIFEST.json',manifest)
    return dict(manifest_sha256=digest(output/'DATASET_MANIFEST.json'),statistics_sha256=manifest['statistics_sha256'],
        TRAIN_episodes=len(train),DEV_episodes=len(kept)-len(train),TRAIN_windows=len(train),
        target_GR00T_updates=0,training_admitted=False)


class TargetNativeDataset(OriginalDataset):
    """Use the original Processor/collator; select only real H40 anchor rows."""
    def __init__(self,root,source,cosmos,split='TRAIN',*,for_optimization=False,trace_root=None):
        need(split in ('TRAIN','DEV') and (not for_optimization or split=='TRAIN'),'NO_DEV_OPTIMIZATION')
        self.root=Path(root);self.source=str(source);self.cosmos=str(cosmos);self.split=split
        self.for_optimization=for_optimization
        self.manifest=json.loads((self.root/'DATASET_MANIFEST.json').read_bytes())
        need(self.manifest.get('schema')==SCHEMA and self.manifest.get('domain')==DOMAIN
             and self.manifest.get('profile')=='TARGET_GR00T_V1','TARGET_PROFILE_NOT_OLD_SOURCE')
        self.statistics_sha=self.manifest['statistics_sha256'];self.statistics=self.get_dataset_statistics()
        self.manifest_sha=digest(self.root/'DATASET_MANIFEST.json')
        if for_optimization:
            admission=json.loads((self.root/'PROCESSOR_TRAINING_ADMISSION.json').read_bytes())
            need(admission.get('schema')=='WM02_TARGET_GR00T_PROCESSOR_ADMISSION_V1'
                 and admission.get('status')=='PASS_ACTUAL_OFFICIAL_PROCESSOR_ROUNDTRIP'
                 and admission.get('manifest_sha256')==self.manifest_sha
                 and admission.get('statistics_sha256')==self.statistics_sha
                 and admission.get('TRAIN_only_statistics')is True,'ACTUAL_TARGET_PROCESSOR_QUALIFICATION_REQUIRED')
        self.rows=[(r,a)for r in self.manifest['episodes']if r['split']==split for a in range(r['windows'])]
        need(bool(self.rows),'NONEMPTY_SPLIT')
        self.processor=None;self.cached_id=None;self.cached=None;self.raw_bindings={};self.seed=20261108
        self.trace_root=Path(trace_root)if trace_root is not None else None
        if self.trace_root is not None:self.trace_root.mkdir(parents=True,exist_ok=True)

    def raw(self,index):
        row,anchor=self.rows[index];path=self.root/row['file']
        need(path.resolve().is_relative_to(self.root.resolve())and digest(path)==row['sha256'],'NUMERIC_HASH')
        with np.load(path,allow_pickle=False)as z:p={k:z[k].copy()for k in z.files}
        need(p['valid_window'][anchor]and p['label_representation'].item()=='ABSOLUTE_COMMITTED_EEF10','FULL_EXECUTED_LABEL')
        atom=int(p['native_command_anchor'][anchor]);actions=p['committed_absolute10'][atom:atom+40]
        need(actions.shape==(40,10),'NO_PARTIAL_OR_CROSS_EPISODE_H40')
        images=[]
        for relative,pin in zip(p['rgb_paths'][anchor],p['rgb_sha256'][anchor]):
            frame=self.root/str(relative)
            need(frame.resolve().is_relative_to(self.root.resolve())and digest(frame)==str(pin),'OWN_REAL_MEDIA_HASH')
            with Image.open(frame)as im:
                need(im.mode=='RGB'and im.size==(256,256),'RGB_FORMAT')
                images.append(np.asarray(im).copy())
        return p['actual_state10'][anchor],actions,images,p['instruction'].item()
