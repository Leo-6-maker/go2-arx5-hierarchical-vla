"""Convert the pinned21 full source BOOK demonstrations, without new physics."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from scripts.pi0292.durable import write
from scripts.pi0294.e1.artifacts import atomic_npz
from scripts.pi0294.multitask.source_export import KINDS, packet
from go2arx5_vla.sla_adapter.codec_v1 import Quantiles, pad32, roundtrip_error, rigid
from go2arx5_vla.sla_adapter.dense_dataset_v4 import member, sha
from go2arx5_vla.sla_adapter.gripper_v2 import EPSILON, VERSION, GripLogitCodec
from go2arx5_vla.sla_adapter.source_book_v5 import (
    ANCHORS, DATASET_ID, IDS, SCHEMA, SOURCE_MANIFEST_SHA, SourceBookDataset,
    full_source_rows, manifest_contract, nominal_commands)


def source_inputs(source_root, frame_proof):
    source_root = Path(source_root)
    if sha(source_root/'DATASET_MANIFEST.json') != SOURCE_MANIFEST_SHA:
        raise ValueError('EXACT_EXISTING_D1_SOURCE_MANIFEST_REQUIRED')
    rows = full_source_rows(json.loads((source_root/'DATASET_MANIFEST.json').read_bytes()))
    if (frame_proof.get('schema') != 'SLA_EXISTING_BOOK21_SOURCE_FRAME_FK_V5'
            or frame_proof.get('passes') is not True or frame_proof['actual_saved_state_FK_checks'] != 13461
            or frame_proof['object_truth_used_for_frame'] is not False
            or frame_proof['new_integrations'] != 0
            or tuple(r['id'] for r in frame_proof['rows']) != IDS):
        raise ValueError('ACTUAL_ALL21_SAVED_SOURCE_MODEL_FK_AND_FIXED_MOUNT_PROOF_REQUIRED')
    for row, frame in zip(rows, frame_proof['rows']):
        raw = Path(row['raw_root']);audit_path = raw/'INDEPENDENT_RAW_AUDIT.json'
        if (sha(member(source_root,row['file'])) != row['sha256']
                or sha(raw/'ACTUAL_RECORDS.jsonl') != row['raw_sha256']
                or sha(audit_path) != row['audit_sha256'] or sha(raw/'TERMINAL.json') != row['terminal_sha256']
                or frame['source_numeric_sha256'] != row['sha256'] or frame['raw_sha256'] != row['raw_sha256']
                or sha(raw/'NATIVE_SCENE.private.xml') != frame['scene_sha256']
                or frame['checks'] != 641 or frame['max_TCP_matrix_error'] > 1e-10
                or frame['max_fixed_mount_error'] != 0 or frame['mount_body'] != 'x5_base_link'
                or frame['policy_frame'] != 'ACTUAL_SOURCE_WORLD'):
            raise ValueError('PINNED_BOOK_SOURCE_COMMAND_RAW_SCENE_AND_FRAME_IDENTITY_CHANGED')
        rigid(frame['T_policy_from_mount'])
        audit = json.loads(audit_path.read_bytes())
        if (audit['integrity_problems'] or audit['raw_sha256'] != row['raw_sha256']
                or any(audit[k] is not True for k in ('task_success','qualified_source_pilot',
                    'complete_episode','original_native_safety_pass','additional_support_release_withdraw_hold_pass'))
                or audit['actual_integrations'] != 18500):
            raise ValueError('ORIGINAL_FULL_SAFE_RELEASE_WITHDRAW_BOOK_AUDIT_REQUIRED')
    return rows


def derive_episode(source_root, output, row, frame):
    raw = Path(row['raw_root']);folder = output/row['id'];folder.mkdir()
    with np.load(member(source_root,row['file']),allow_pickle=False) as f:
        original = {k:f[k].copy() for k in f.files}
    kept = []
    with (raw/'ACTUAL_RECORDS.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            if r['kind'] in KINDS:kept.append(r)
    card = json.loads((raw/'CARD.json').read_bytes());recovered = packet(kept,card)
    if any(not np.array_equal(recovered[k],original[k]) for k in recovered):
        raise ValueError('ACTUAL_SOURCE_COMMAND_OBSERVATION_CLOCK_MUST_MATCH_SAVED_PACKET')
    observations = [r['value'] for r in kept if r['kind'] == 'observation']
    nominal = nominal_commands(original['committed_absolute10'])
    states = original['actual_state10'];codec = GripLogitCodec(frame['T_policy_from_mount'])
    state7,latent7 = [],[];phases = [];maximum = dict(xyz_m=0.,so3_rad=0.,gripper=0.)
    for i in ANCHORS:
        s,a = codec.encode(states[i],nominal[i:i+40]);state7.append(s);latent7.append(a)
        error = roundtrip_error(nominal[i:i+40],codec.decode(states[i],pad32(a)))
        if error['xyz_m']>1e-6 or error['so3_rad']>1e-5 or error['gripper']>EPSILON+1e-12:
            raise ValueError('BOOK_SOURCE_COMMAND_CODEC_ROUNDTRIP')
        for k in maximum:maximum[k] = max(maximum[k],error[k])
        phase = observations[i]['privileged_teacher_stage']
        if not isinstance(phase,str) or not phase:raise ValueError('ORIGINAL_AUDIT_STAGE_REQUIRED')
        phases.append(phase)
    # Check all original image byte identities; decode/export only the actual
    # 2Hz training anchors. Original full-rate captures remain immutable.
    images = [];image_bytes = 0
    for i,(paths,pins) in enumerate(zip(original['rgb_paths'],original['rgb_sha256'])):
        pair = []
        for camera,(name,pin) in enumerate(zip(paths,pins)):
            p = member(raw,str(name));receipt = observations[i]['policy_obs'][('body_rgb','wrist_rgb')[camera]]
            if p.is_symlink() or sha(p)!=str(pin) or receipt['sha256']!=str(pin) or p.stat().st_size!=receipt['bytes']:
                raise ValueError('IMMUTABLE_ORIGINAL_BOOK_RGB_BYTES_REQUIRED')
            image_bytes += p.stat().st_size
            if i in ANCHORS:
                with Image.open(p) as im:
                    if im.mode!='RGB' or im.size!=(256,256):raise ValueError('ORIGINAL_BOOK_RGB256')
                    pair.append(np.asarray(im).copy())
        if i in ANCHORS:images.append(pair)
    state7,latent7 = np.asarray(state7),np.asarray(latent7)
    native_step = 2500+ANCHORS*25
    label = folder/'TRAIN_WINDOWS.npz'
    atomic_npz(label,state7=state7,latent7=latent7,state_eef10=states[ANCHORS],nominal_policy_H40=np.stack([nominal[i:i+40]for i in ANCHORS]),
        source_committed_absolute10=original['committed_absolute10'],native_step=native_step,frame_index=np.arange(61),phase=np.asarray(phases))
    descriptors = [];images = np.asarray(images,dtype=np.uint8)
    for first in range(0,61,32):
        last = min(61,first+32);p = folder/f'RGB_{first:03}.npz'
        atomic_npz(p,rgb=images[first:last],native_step=native_step[first:last],state_eef10=states[ANCHORS[first:last]])
        descriptors.append(dict(file=p.relative_to(output).as_posix(),sha256=sha(p),first_frame=first,frames=last-first))
    mount = {k:frame[k]for k in ('mount_body','T_policy_from_mount','policy_frame','scene_sha256','checks','max_TCP_matrix_error','max_fixed_mount_error')}
    ep = dict(id=row['id'],task='BOOK_FROM_TABLE_TO_SLOT',split='TRAIN',windows=61,frames=61,source_frames=641,
        commands=640,source_native_integrations=18500,expert_suffix=False,language=str(original['instruction']),
        mount=mount,labels=dict(file=label.relative_to(output).as_posix(),sha256=sha(label)),frame_shards=descriptors,
        expert_stage_audit_only=phases,source_numeric_sha256=row['sha256'],source_raw_sha256=row['raw_sha256'],
        source_audit_sha256=row['audit_sha256'],source_terminal_sha256=row['terminal_sha256'])
    return ep,state7,latent7,dict(all_source_image_files_verified=1282,source_image_bytes_verified=image_bytes,codec_roundtrip=maximum)


def audit_dataset(root, manifest_sha):
    from openpi_client.image_tools import resize_with_pad
    start = time.time();dataset = SourceBookDataset(root,manifest_sha,resize_with_pad=resize_with_pad)
    phases = {};maximum = dict(xyz_m=0.,so3_rad=0.,gripper=0.)
    for i in range(len(dataset)):
        ep,anchor = dataset.rows[i];state,nominal,images,text = dataset.raw(i);item = dataset[i]
        if (item['state'].shape!=(32,) or item['actions'].shape!=(40,32) or not np.isfinite(item['actions']).all()
                or set(item)!= {'image','image_mask','image_padding_mask','state','actions','prompt'}
                or np.any(item['state'][7:]!=0) or np.any(item['actions'][:,7:]!=0) or item['prompt']!=text):
            raise ValueError('ORDINARY_RGB_STATE32_H40_32_WITHOUT_PRIVILEGED_INPUTS')
        decoded = dataset.codecs[ep['id']].decode(state,pad32(dataset.statistics['actions'].unnormalize(item['actions'][:,:7])))
        err = roundtrip_error(nominal,decoded)
        if err['xyz_m']>1e-6 or err['so3_rad']>1e-5 or err['gripper']>EPSILON+1e-6:
            raise ValueError('FP32_BOOK_TRAIN_CODEC_ROUNDTRIP')
        for k in maximum:maximum[k]=max(maximum[k],err[k])
        phase = ep['expert_stage_audit_only'][anchor];phases[phase]=phases.get(phase,0)+1
    return dict(schema='SLA_BOOK21_ALL_WINDOW_RGB_READBACK_V5',passes=True,manifest_sha256=manifest_sha,
        dataset_id=DATASET_ID,episodes=21,read_windows=len(dataset),phase_counts_audit_only=phases,
        FP32_roundtrip_max_error=maximum,model_loads=0,optimizer_updates=0,new_physical_integrations=0,
        target_WBC_qualified=False,DEV_TEST_reads=0,wall_s=time.time()-start)


def prepare(source_root, frame_proof_path, output):
    start = time.time();cpu = time.process_time();source_root,output = Path(source_root),Path(output)
    proof = json.loads(Path(frame_proof_path).read_bytes());rows = source_inputs(source_root,proof)
    output.mkdir();write(output/'STATE.private.json',dict(state='preparing',target_WBC_qualified=False))
    try:
        episodes,states,actions,costs = [],[],[],[]
        for row,frame in zip(rows,proof['rows']):
            ep,s,a,c = derive_episode(source_root,output,row,frame)
            episodes.append(ep);states.extend(s);actions.extend(a);costs.append(c)
        states,actions = np.asarray(states),np.asarray(actions).reshape(-1,7)
        stat = output/'STATISTICS.private.json'
        write(stat,dict(state=Quantiles.fit(states).summary(states),actions=Quantiles.fit(actions).summary(actions),
            split='TRAIN',domain='SOURCE_FIXED_ARM',unique_windows=1281,action_knots=51240,
            fit_before_oversampling=True,state_gripper_transformed=False,codec=VERSION,codec_version=VERSION,epsilon=EPSILON))
        manifest = dict(schema=SCHEMA,dataset_id=DATASET_ID,state='committed',domain='SOURCE_FIXED_ARM',
            source_manifest_sha256=SOURCE_MANIFEST_SHA,source_frame_proof_sha256=sha(frame_proof_path),
            codec=VERSION,split='TRAIN',tasks=['BOOK_FROM_TABLE_TO_SLOT'],target_WBC_qualified=False,
            public_RoboTwin_S_cmd_added=0,supervised_windows=1281,phase_or_object_truth_in_student=False,
            state_gripper_transformed=False,command_period_ticks=25,physics_dt_s=.002,anchor_period_ticks=250,
            source_frame_period_ticks=25,source_frame_pairs=13461,training_frame_pairs=1281,
            statistics=dict(file=stat.name,sha256=sha(stat)),episodes=episodes)
        manifest_contract(manifest);write(output/'MANIFEST.private.json',manifest)
        audit = audit_dataset(output,sha(output/'MANIFEST.private.json'));assert audit['read_windows']==1281
        write(output/'READBACK.private.json',audit)
        result = dict(schema='SLA_BOOK21_SOURCE_PREPARATION_ACTUAL_V5',passes=True,episodes=21,source_commands=13440,
            source_observations=13461,all_source_image_files_verified=sum(c['all_source_image_files_verified']for c in costs),
            source_image_bytes_verified=sum(c['source_image_bytes_verified']for c in costs),training_frame_pairs=1281,
            read_windows=1281,manifest_sha256=sha(output/'MANIFEST.private.json'),readback_sha256=sha(output/'READBACK.private.json'),
            target_WBC_qualified=False,model_loads=0,optimizer_updates=0,new_physical_integrations=0,
            dataset_bytes=sum(p.stat().st_size for p in output.rglob('*')if p.is_file()),
            wall_s=time.time()-start,CPU_s=time.process_time()-cpu,readback=audit)
        write(output/'STATE.private.json',dict(state='committed',target_WBC_qualified=False));return result
    except BaseException as error:
        write(output/'STATE.private.json',dict(state='incomplete',error=repr(error),target_WBC_qualified=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--frame-proof',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.source,a.frame_proof,a.output)))
