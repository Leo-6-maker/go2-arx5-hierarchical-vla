"""Saved-byte negative witness only, never admits incomplete demonstrations."""
from pathlib import Path

import numpy as np

from scripts.pi0292.durable import read
from scripts.pi0294.fd1 import put, sha


def audit(root,card,result,s1):
    root=Path(root)
    folder=root/'episodes'/card['id']
    if result['status']=='VALID_COMPLETE' or result['physics_steps']>=16000:
        raise PermissionError('PV1_PARTIAL_WITNESS_IS_NOT_FULL_TASK_AUDIT')
    error=result.get('error') or ''
    static='STATIC_QUALIFICATION_FAILED_NO_INTEGRATION' in error
    reject='PRE_EXECUTION_SAFETY_REJECT' in error
    if not (static or reject):
        raise PermissionError('PV1_PARTIAL_WITNESS_ONLY_KNOWN_LEGAL_NEGATIVES')
    for entry in read(folder/'FILE_MANIFEST.json'):
        p=folder/entry['path']
        if (p.is_symlink() or not p.resolve().is_relative_to(folder.resolve())
                or p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']):
            raise ValueError('PV1_PARTIAL_RAW_IDENTITY')
    worker=read(folder/'WORKER_RECEIPT.json')
    if (worker['source_sha']!=result['source_sha'] or worker['physics_steps']!=result['physics_steps']
            or worker['samples_sha256']!=sha(folder/'samples.json')):
        raise ValueError('PV1_PARTIAL_NATIVE_RECEIPT')
    with np.load(folder/'highrate.npz',allow_pickle=False) as z:
        high=z['rows'].copy()
        external=z['external_force_max'].copy()
    steps=result['physics_steps']
    if (len(high)!=steps or not np.isfinite(high).all() or external.shape!=(steps,2)
            or np.any(external) or (steps and not np.allclose(high[:,0],np.arange(1,steps+1)*.002,atol=1e-8,rtol=0))):
        raise ValueError('PV1_PARTIAL_ACTUAL_CLOCK_OR_FORCE')
    forecasts=read(folder/'S1_PREFLIGHTS.json') if (folder/'S1_PREFLIGHTS.json').exists() else []
    rejected=[f for f in forecasts if f['decision']=='REJECT']
    if reject and (not s1 or s1['pre_execution_rejections']!=1 or len(rejected)!=1
            or steps!=rejected[0]['sequence']*25):
        raise ValueError('PV1_PARTIAL_REJECTED_PREFIX_MUST_EXECUTE_ZERO_TICKS')
    if static and steps!=0:
        raise ValueError('PV1_STATIC_REJECT_MUST_EXECUTE_ZERO_TICKS')
    samples=read(folder/'samples.json')
    references={r['ref_id']:r for r in read(folder/'references.json')}
    for index,sample in enumerate(samples):
        if abs(sample['policy_obs']['timestamp_s']-index*.05)>1e-8:
            raise ValueError('PV1_PARTIAL_OBSERVATION_CLOCK')
        for camera in ['body_rgb','wrist_rgb']:
            obs=sample['policy_obs'][camera]
            if sha(folder/obs['path'])!=obs['sha256']:
                raise ValueError('PV1_PARTIAL_CAMERA_IDENTITY')
        label=sample['supervision']
        if label is not None:
            ref=references[label['ref_id']]
            i=label['point_index']
            if (not label['actually_committed'] or not label['valid']
                    or any(not np.array_equal(label[key],ref['planned_H40'][i])
                           for key in ['desired','accepted','installed','committed'])
                    or label['committed_gripper']!=ref['gripper_H40'][i]):
                raise ValueError('PV1_PARTIAL_SUBMITTED_ACTION_LABEL')
    for camera in ['body_rgb','wrist_rgb']:
        if len(list((folder/camera).glob('*.png')))!=result['rgb_pairs']:
            raise ValueError('PV1_PARTIAL_ALL_CAPTURED_FRAMES_RETAINED')
    value=dict(status='PASS_INDEPENDENT_PARTIAL_NEGATIVE_WITNESS',admitted=False,
        id=card['id'],manifest_sha256=sha(folder/'FILE_MANIFEST.json'),
        complete=False,physics_steps=steps,actual_paired_frames=result['rgb_pairs'],
        task_success=False,rejected_prefix_live_ticks=0,partial_not_full_FK_or_score_audit=True,
        integrations=0,engine_starts=0,valid_scientific_negative_not_engineering_replacement=True)
    put(folder/'PARTIAL_NEGATIVE_AUDIT.json',value)
    return value
