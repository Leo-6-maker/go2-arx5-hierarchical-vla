"""Saved-byte identity/clock witness, never a replacement score or demo audit."""
from pathlib import Path

import numpy as np

from scripts.pi0292.durable import read
from scripts.pi0294.fd1 import sha
from .plan import validate_group


def score_input_failure_witness(folder,card,guard=lambda:None):
    folder=Path(folder);validate_group(card)
    result=read(folder/'RESULT.json');s1=read(folder/'S1_RESULT.json')
    worker=read(folder/'WORKER_RECEIPT.json')
    if (read(folder/'CARD.json')!=card or result['id']!=card['id']
            or result['status']!='FAILED_PRESERVED' or result.get('error') is not None
            or result.get('score_error')!='ValueError: source-goal separation must be at least 0.15 m'
            or result['physics_steps']!=16000 or result['rgb_pairs']!=641
            or 'score' in result or worker['source_sha']!=result['source_sha']
            or worker['physics_steps']!=16000 or worker['samples_sha256']!=sha(folder/'samples.json')):
        raise PermissionError('DATA_COV1_ONLY_EXACT_FULL_UNSCORABLE_FIRST_ATTEMPT')
    if (s1['live_native_failures'] or s1['pre_execution_rejections'] or s1['actions_modified']
            or s1['shadow_integrations']!=16000 or s1['maximum_live_shadow_difference']>1e-10):
        raise PermissionError('DATA_COV1_ORIGINAL_S1_PARITY_AND_NO_NATIVE_FAULT')
    entries=read(folder/'FILE_MANIFEST.json');names=[]
    for row in entries:
        guard();p=folder/row['path'];names.append(row['path'])
        if (p.is_symlink() or not p.resolve().is_relative_to(folder.resolve())
                or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']):
            raise ValueError('DATA_COV1_FAILED_RAW_MEMBER_IDENTITY')
    if len(set(names))!=len(names):raise ValueError('DATA_COV1_FAILED_RAW_DUPLICATE_MEMBER')
    samples=read(folder/'samples.json');references={r['ref_id']:r for r in read(folder/'references.json')}
    if len(samples)!=641:raise ValueError('DATA_COV1_FAILED_RAW_COMPLETE_SAMPLE_CLOCK')
    for index,sample in enumerate(samples):
        guard();obs=sample['policy_obs']
        if abs(obs['timestamp_s']-index*.05)>1e-8 or obs['instruction']!=card['instruction']:
            raise ValueError('DATA_COV1_FAILED_RAW_OBSERVATION_CLOCK_OR_TASK')
        for camera in ('body_rgb','wrist_rgb'):
            image=folder/obs[camera]['path']
            if image.is_symlink() or not image.resolve().is_relative_to(folder.resolve()) or sha(image)!=obs[camera]['sha256']:
                raise ValueError('DATA_COV1_FAILED_RAW_ACTUAL_IMAGE_IDENTITY')
        label=sample['supervision']
        if index==640:
            if label is not None:raise ValueError('DATA_COV1_FINAL_OBSERVATION_HAS_NO_NEXT_COMMAND')
            continue
        if not label or not label['actually_committed'] or not label['valid']:
            raise ValueError('DATA_COV1_FAILED_RAW_ACTUAL_SUBMITTED_LABEL')
        ref=references[label['ref_id']];i=label['point_index']
        if (any(not np.array_equal(label[k],ref['planned_H40'][i]) for k in ('desired','accepted','installed','committed'))
                or label['committed_gripper']!=ref['gripper_H40'][i]):
            raise ValueError('DATA_COV1_FAILED_RAW_LABEL_REFERENCE_IDENTITY')
    for camera in ('body_rgb','wrist_rgb'):
        if len(list((folder/camera).glob('*.png')))!=641:
            raise ValueError('DATA_COV1_FAILED_RAW_ALL_FRAMES_RETAINED')
    with np.load(folder/'highrate.npz',allow_pickle=False) as z:
        rows=z['rows'];external=z['external_force_max']
        if (len(rows)!=16000 or not np.isfinite(rows).all() or external.shape!=(16000,2)
                or np.any(external) or not np.allclose(rows[:,0],np.arange(1,16001)*.002,atol=1e-8,rtol=0)):
            raise ValueError('DATA_COV1_FAILED_RAW_HIGH_RATE_CLOCK_OR_EXTERNAL_FORCE')
    with np.load(folder/'physics.npz',allow_pickle=False) as z:
        if len(z['time_s'])!=641 or not np.allclose(z['time_s'],np.arange(641)*.05,atol=1e-8,rtol=0):
            raise ValueError('DATA_COV1_FAILED_RAW_PHYSICAL_SAMPLE_CLOCK')
    return dict(status='ENGINEERING_INCOMPLETE',id=card['id'],scene_group=card['group_id'],task=card['task'],
        auditable_identity_and_clock=True,task_success=False,task_success_known=False,
        admitted=False,raw_admitted=False,codec_admitted=False,native_faults=0,unknown_effects=[],
        valid_task_failure=False,reason='FROZEN_SCORER_INPUT_GEOMETRY_CONFLICT',
        raw_manifest_sha256=sha(folder/'FILE_MANIFEST.json'),result_sha256=sha(folder/'RESULT.json'),
        actual_frame_pairs_checked=641,actual_high_rate_rows_checked=16000,
        physical_replacements=0,new_physics=0,new_constructors=0,new_GR00T_queries=0,
        original_result_score_and_manifest_unchanged=True,not_a_full_FK_or_success_demo_audit=True)
