"""Freeze a new shared TRAIN reader version from the qualified source59 bytes.

Caller supplies the existing owner/Journal and output capacity before invoking.
The old candidate and experiments remain immutable. No RGB re-render, statistics
fit, Processor work, model or training launch occurs here.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil

from .shared_task_sampling import validate_fixed59

DATA_SCHEMA = 'SHARED4_RAW_M32_FROZEN_TRAIN59_V1'
POPULATION_SCHEMA = 'SHARED4_RAW_M32_FROZEN_EXPERT_POPULATION_V1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(source,destination,*,source_manifest_sha256,qualification_path,
        qualification_sha256,task_DEV_path,task_DEV_sha256):
    source,destination=Path(source),Path(destination)
    if destination.exists()or destination.is_symlink():raise PermissionError('NEW_SHARED_DATA_VERSION_NO_OVERWRITE')
    if sha(source/'DATASET_MANIFEST.json')!=source_manifest_sha256 or sha(qualification_path)!=qualification_sha256:
        raise PermissionError('EXACT_ACTUAL_SOURCE59_AND_QUALIFICATION_BYTES')
    m=json.loads((source/'DATASET_MANIFEST.json').read_bytes());q=json.loads(Path(qualification_path).read_bytes())
    if (m.get('schema')!='SOURCE_COLLECTION_EXPANDED_FOUR_TASK_TRAIN_CANDIDATE_V1'
            or m.get('training_admitted')is not False or m.get('DEV_windows')!=0
            or q.get('status')!='PASS_OFFICIAL_TRAIN_CODEC_ALL_WINDOWS_AND_REAL_RGB'
            or q.get('manifest_sha256')!=source_manifest_sha256
            or q.get('statistics_sha256')!=m.get('statistics_sha256')
            or q.get('windows')!=34563 or q.get('torch_CUDA_initialized')is not False
            or m.get('TRAIN_windows')!=34563):
        raise PermissionError('ACTUAL_OFFICIAL_ALL_WINDOW_AND_REAL_RGB_TRAIN_QUALIFICATION')
    rows=[(r,a)for r in m['episodes']for a in range(r['windows'])];validate_fixed59(rows)
    qualified={r['id']:(r['task'],r['windows'])for r in q['episodes']}
    if (len(q['episodes'])!=59 or qualified!={r['id']:(r['task'],r['windows'])for r in m['episodes']}
            or len(qualified)!=59 or sha(source/'statistics.json')!=m['statistics_sha256']
            or sha(task_DEV_path)!=task_DEV_sha256):
        raise PermissionError('UNMODIFIED_TRAIN_STATS_AND_COMPLETE_EPISODE_QUALIFICATION_JOIN')
    # Validate/copy one bounded numerical packet at a time; actual raw RGB remains
    # at its already-qualified node-local immutable roots and is checked by reader.
    destination.mkdir(mode=0o700);(destination/'episodes').mkdir()
    def put(name,value):
        data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
        with(destination/name).open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
    population=[];copied=0
    for row in m['episodes']:
        if not row['id']or any(not(c.isalnum()or c in'_-')for c in row['id']):
            raise PermissionError('SAFE_IMMUTABLE_SHARED_EPISODE_NAME')
        src=source/row['file'];dst=destination/'episodes'/(row['id']+'.npz')
        if not src.resolve().is_relative_to(source.resolve())or src.is_symlink()or sha(src)!=row['sha256']:
            raise PermissionError('EXACT_ACTUAL_SOURCE_EXPERT_NUMERICAL_BYTES')
        shutil.copyfile(src,dst)
        if sha(dst)!=row['sha256']:raise PermissionError('EXACT_SHARED_EXPERT_COPY')
        copied+=dst.stat().st_size;population.append(dict(row,file=str(dst.relative_to(destination))))
    shutil.copyfile(source/'statistics.json',destination/'statistics.json')
    shutil.copyfile(task_DEV_path,destination/'TASK_DEV_POPULATION.json')
    if sha(destination/'statistics.json')!=m['statistics_sha256']or sha(destination/'TASK_DEV_POPULATION.json')!=task_DEV_sha256:
        raise PermissionError('SHARED_STATS_AND_DEV_CHANGED_DURING_COPY')
    pop=dict(schema=POPULATION_SCHEMA,episodes=population,full_expert_episodes=46,expert_suffix_entries=13,
        source_manifest_sha256=source_manifest_sha256,source_qualification_sha256=qualification_sha256,
        statistics_sha256=m['statistics_sha256'],DEV_TEST_confirm_admitted=False)
    put('TRAIN_POPULATION.json',pop)
    put('TRAIN_DEV_SEAL.json',dict(schema='SHARED4_FIXED_TRAIN_AND_EXISTING_DEV_REFERENCE_V1',
        task_DEV_population_sha256=task_DEV_sha256,TRAIN_entries=[r['id']for r in population],
        DEV_in_statistics_or_optimizer=False,DEV_execution_qualification_claim=False))
    manifest=dict(m,schema=DATA_SCHEMA,training_admitted=True,episodes=population,
        source_manifest_sha256=source_manifest_sha256,source_qualification_sha256=qualification_sha256,
        shared_training_model_GPU_zero_step_DEV_qualification=False)
    put('DATASET_MANIFEST.json',manifest)
    return dict(status='FROZEN_EXACT_SHARED_TRAIN_READER_INPUTS_NOT_MODEL_ADMISSION',root=str(destination),
        manifest_sha256=sha(destination/'DATASET_MANIFEST.json'),population_sha256=sha(destination/'TRAIN_POPULATION.json'),
        statistics_sha256=sha(destination/'statistics.json'),task_DEV_sha256=sha(destination/'TASK_DEV_POPULATION.json'),
        entries=59,full_expert_episodes=46,expert_suffix_entries=13,windows=34563,
        numeric_bytes_copied=copied,statistics_refit=False,new_models_updates_physics=0,
        model_and_GPU_training_admitted=False)
