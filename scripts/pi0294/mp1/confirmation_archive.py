"""Inventory existing audited confirmation clips without rendering or uploading."""
from pathlib import Path

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.public_checkpoint import reject_private
from .audit import require_frozen_attempt
from .confirmation_config import validate_plan
from .confirmation_evidence import AUDIT_PASS,canonical_row,coverage


def owned_file(root,relative):
    root=Path(root);relative=Path(relative);path=root/relative
    if (relative.is_absolute() or '..' in relative.parts or not path.is_file()
            or any(p.is_symlink() for p in [path,*path.parents] if p.is_relative_to(root))
            or not path.resolve().is_relative_to(root.resolve())):
        raise PermissionError('ORIGINAL_OWNED_FILE_NO_SYMLINK_OR_ESCAPE')
    return path


def inventory(root,plan,rows,*,unresolved=()):
    """Join only terminal raw attempts; preserve unaudited and engineering gaps."""
    root=Path(root);rows=[canonical_row(r) for r in rows]
    if validate_plan(root)!=plan:raise PermissionError('ONLY_ACTUAL_SEALED_CONFIRMATION_PLAN')
    coverage(plan,rows,{},unresolved=unresolved)
    audits={};index=[];files={};sources={};gaps=[]
    for row in rows:
        identifier=row['attempt_id'];actual=require_frozen_attempt(root,identifier)
        if actual!=row:raise PermissionError('EXACT_TERMINAL_RAW_ADJUDICATION_REQUIRED')
        result=owned_file(root,Path('episodes')/identifier/'RESULT.json')
        if sha(result)!=row['result_sha256']:raise ValueError('ORIGINAL_RESULT_BYTES_CHANGED')
        zero=row['main_integrations']==0 and row['status']=='ENGINEERING_INCOMPLETE'
        folder=Path('evidence')/('zero-step-audits' if zero else 'audits')/identifier
        audit_file=root/folder/'AUDIT.private.json'
        if not audit_file.exists():
            gaps.append(dict(attempt_id=identifier,status='MISSING_INDEPENDENT_AUDIT'));continue
        audit=read(owned_file(root,folder/'AUDIT.private.json'))
        if (audit['status'] not in AUDIT_PASS or audit.get('adjudication')!=row
                or audit.get('attempt_id')!=identifier or not audit.get('checks')
                or not all(audit['checks'].values()) or audit.get('errors')!=[]):
            gaps.append(dict(attempt_id=identifier,status='PRESERVED_UNQUALIFIED_AUDIT',audit_status=audit['status']));continue
        movie=audit['video'];relative=folder/('actual-initial-body-wrist.mp4' if zero else 'actual-body-wrist.mp4')
        path=owned_file(root,relative)
        if (Path(movie['relative_path'])!=relative or sha(path)!=movie['sha256']
                or path.stat().st_size!=movie['bytes'] or movie['fps']!=20
                or movie['actual_paired_frames']!=movie['decoded_frames'] or movie['decoded_frames']!=row['RGB_pairs']
                or movie['native_duration_s']!=row['duration_s'] or movie['padding_or_interpolation'] is not False
                or abs(movie['playback_duration_s']-movie['decoded_frames']/20)>1e-6):
            raise ValueError('ORIGINAL_FULL_DECODE_VIDEO_HASH_COUNT_CLOCK_AND_DURATION')
        if zero:
            if movie.get('initial_frame_only') is not True or movie.get('task_trajectory_video') is not False:
                raise ValueError('ZERO_STEP_INITIAL_FRAME_IS_NOT_A_TASK_TRAJECTORY')
        elif movie.get('initial_frame_included') is not True:
            raise ValueError('ORIGINAL_INITIAL_FRAME_MUST_BE_RETAINED')
        name='movies/'+identifier+'.mp4';sources[name]=path
        files[name]=dict(bytes=movie['bytes'],sha256=movie['sha256'])
        audits[identifier]=audit
        index.append(dict(attempt_id=identifier,status=row['status'],task=row['task'],scene_group=row['scene_group'],
            model=row['model'],split=row['split'],failure_stage=row['failure_stage'],
            engineering_incomplete=row['status']=='ENGINEERING_INCOMPLETE',
            pipeline_trustworthy=row['pipeline_trustworthy'],task_success=row['status']=='SUCCESS',
            S1_rejections=row['S1_rejections'],live_native_failures=row['live_native_failures'],
            teacher_takeover=False,native_sim_s=row['duration_s'],actual_frames=movie['decoded_frames'],
            initial_frame_only=zero,artifact_path=name,actual_execution_source_sha=audit['execution_source_sha'],
            audit_sha256=sha(audit_file),result_sha256=row['result_sha256'],**files[name]))
    value=dict(schema='MP1_SOURCE_CONFIRM_ACTUAL_AUDITED_VIDEO_INVENTORY_V1',
        coverage=coverage(plan,rows,audits,unresolved=unresolved),rows=index,audit_gaps=gaps,
        actual_videos=len(index),new_model_queries=0,new_physics=0,new_training_updates=0,
        rerendered_videos=0,public_upload_performed=False,private_source_bytes_unchanged=True,
        same_condition_replacement_is_independent_scene=False,selected='NONE',deployment='NONE')
    reject_private(value)
    return value,files,sources
