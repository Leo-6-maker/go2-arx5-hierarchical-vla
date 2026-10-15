"""Official CPU adapter for a sealed actual full-body bank, no neural model."""
import copy
from pathlib import Path
import numpy as np
from scripts.pi0292.train_adapter import official_import,sha,TAG
from scripts.pi0292.durable import read
from .night_r1_data import DOMAIN,counts,admitted


def modality():
    from gr00t.data.types import ModalityConfig,ActionConfig,ActionFormat,ActionType,ActionRepresentation
    return {TAG:{
        'video':ModalityConfig(delta_indices=[0],modality_keys=['external_rgb','wrist_rgb']),
        'state':ModalityConfig(delta_indices=[0],modality_keys=['eef_9d','gripper_position','base_observed9']),
        'action':ModalityConfig(delta_indices=list(range(40)),modality_keys=['eef_9d','gripper_position','base_soft_velocity3'],action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE,type=ActionType.EEF,format=ActionFormat.XYZ_ROT6D,state_key='eef_9d'),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE,type=ActionType.NON_EEF,format=ActionFormat.DEFAULT,state_key='gripper_position'),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE,type=ActionType.NON_EEF,format=ActionFormat.DEFAULT)]),
        'language':ModalityConfig(delta_indices=[0],modality_keys=['annotation.language.task_description'])}}


def build_processor(source,cosmos,statistics):
    official_import(source)
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    p=Gr00tN1d7Processor(modality_configs=modality(),statistics=copy.deepcopy(statistics),
        use_percentiles=False,clip_outliers=False,use_relative_action=True,image_crop_size=[256,256],
        image_target_size=[256,256],use_albumentations=False,random_rotation_angle=0,color_jitter_params=None,
        crop_fraction=1.,model_name=str(cosmos),model_type='qwen',max_state_dim=132,max_action_dim=132,
        max_action_horizon=40,state_dropout_prob=0.,exclude_state=False,
        transformers_loading_kwargs={'local_files_only':True,'trust_remote_code':True})
    p.train_image_transform=p.eval_image_transform;p.eval()
    if p.embodiment_id_mapping[TAG]!=10:raise ValueError('OFFICIAL_EMBODIMENT_ROUTE')
    return p


def states(s):return dict(eef_9d=s[None,:9],gripper_position=s[None,9:10],base_observed9=s[None,10:19])
def actions(a):return dict(eef_9d=a[:,:9],gripper_position=a[:,9:10],base_soft_velocity3=a[:,10:13])


def consume(bank,source,cosmos):
    """Independent process reads every member/window, images stratified by stage.

    No empty-set success; bank creation/statistics must happen first and exactly
    once on admitted TRAIN. This entry does not manufacture those receipts.
    """
    import torch
    from PIL import Image
    bank=Path(bank).resolve();manifest=read(bank/'DATASET_MANIFEST.json');seal=read(bank/'BANK_SEAL.json')
    if (manifest.get('domain')!=DOMAIN or manifest.get('fixture') is not False or manifest.get('statistics_fit_count')!=1
        or not manifest.get('episodes') or not any(r['split']=='TRAIN' for r in manifest['episodes'])
        or seal.get('manifest_sha256')!=sha(bank/'DATASET_MANIFEST.json')
        or manifest.get('statistics_sha256')!=sha(bank/'statistics.json')):
        raise ValueError('NONEMPTY_ACTUAL_TRAIN_FROZEN_BANK_REQUIRED')
    if {p.relative_to(bank).as_posix() for p in bank.rglob('*') if p.is_file()}!=set(seal['file_sha256'])|{'BANK_SEAL.json'}:raise ValueError('EXACT_BANK_MEMBER_SET')
    for name,digest in seal['file_sha256'].items():
        p=(bank/name).resolve()
        if not p.is_relative_to(bank) or sha(p)!=digest:raise ValueError('BANK_MEMBER_READBACK')
    torch.set_num_threads(1);processor=build_processor(source,cosmos,read(bank/'statistics.json'))
    from gr00t.data.types import VLAStepData,MessageType,ActionType,ActionFormat
    from gr00t.data.embodiment_tags import EmbodimentTag
    from scripts.pi0292.eval_contract import matrix10
    from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows,geodesic_angle
    windows=images=0;max_xyz=max_so3=max_other=0.;strata=[]
    for row in manifest['episodes']:
        eroot=(bank/row['export']).resolve();receipt=read(eroot/'EXPORT_RECEIPT.json');raw=Path(receipt['raw_root']).resolve()
        if not eroot.is_relative_to(bank) or receipt['fixture'] is not False or receipt['domain']!=DOMAIN or sha(eroot/'episode.npz')!=receipt['numeric_sha256']:raise ValueError('ACTUAL_EXPORT_REQUIRED')
        admitted(raw)
        for name,digest in receipt['raw_members'].items():
            p=(raw/name).resolve()
            if not p.is_relative_to(raw) or sha(p)!=digest:raise ValueError('ALL_RAW_MEMBERS_READBACK')
        with np.load(eroot/'episode.npz',allow_pickle=False) as f:packet={k:f[k].copy() for k in f.files}
        nc,no,nw=counts(receipt['duration_s']);s=packet['actual_state19'];a=packet['committed_absolute13']
        if s.shape!=(no,19) or a.shape!=(nc,13) or packet['valid_window'].shape!=(nw,) or not packet['valid_window'].all():raise ValueError('VARIABLE_DURATION_SHAPE')
        samples=[__import__('json').loads(l) for l in (raw/'samples.jsonl').read_text().splitlines()]
        stages={}
        for i,sample in enumerate(samples[:nw]):stages.setdefault(sample['teacher_stage'],i)
        for i in range(nw):
            state=states(s[i]);target=a[i:i+40];target_dict=actions(target)
            normalized=processor.state_action_processor.apply_action(target_dict,TAG,state=state)
            packed=np.concatenate([normalized[k] for k in ('eef_9d','gripper_position','base_soft_velocity3')],axis=-1).astype(np.float32)
            decoded=processor.decode_action(np.pad(packed,((0,0),(0,119)))[None],EmbodimentTag.NEW_EMBODIMENT,{k:v[None] for k,v in state.items()})
            recovered=np.concatenate([decoded[k] for k in ('eef_9d','gripper_position','base_soft_velocity3')],axis=-1)[0]
            expected=np.linalg.inv(matrix10(s[i,:10]))@matrix10(target[:,:10])
            relative=processor.state_action_processor._convert_to_relative_action(target[:,:9],s[i,:9],ActionType.EEF,ActionFormat.XYZ_ROT6D)
            expected9=np.concatenate([expected[:,:3,3],rotation_matrix_to_rot6d_rows(expected[:,:3,:3])],axis=-1)
            if np.max(np.abs(relative-expected9))>1e-8:raise ValueError('SINGLE_RELATIVE_EEF_CONVERSION')
            xyz=float(np.max(np.linalg.norm(recovered[:,:3]-target[:,:3],axis=-1)));so3=float(np.max(geodesic_angle(matrix10(recovered[:,:10])[:,:3,:3],matrix10(target[:,:10])[:,:3,:3])))
            other=float(np.max(np.abs(recovered[:,9:]-target[:,9:])))
            max_xyz,max_so3,max_other=max(max_xyz,xyz),max(max_so3,so3),max(max_other,other)
            if not np.isfinite(recovered).all() or xyz>1e-5 or so3>1e-4 or other>1e-5:raise ValueError('OFFICIAL_FULLBODY_CODEC_GATE')
            if i in stages.values():
                pictures=[]
                for camera in range(2):
                    path=raw/str(packet['rgb_paths'][i,camera])
                    with Image.open(path) as opened:pictures.append(np.asarray(opened).copy())
                data=VLAStepData(images=dict(external_rgb=[pictures[0]],wrist_rgb=[pictures[1]]),states=state,actions=target_dict,masks=None,text=packet['instruction'].item(),embodiment=EmbodimentTag.NEW_EMBODIMENT,metadata={})
                value=processor([dict(type=MessageType.EPISODE_STEP.value,content=data)])
                batch=processor.collator([value])['inputs'];mask=batch['action_mask']
                if tuple(batch['action'].shape)!=(1,40,132) or not bool((mask[...,:13]==1).all()) or not bool((mask[...,13:]==0).all()) or not bool((batch['state'][...,19:]==0).all()):raise ValueError('FULLBODY_13_19_MASK_AND_PADDING')
                images+=2;strata.append(dict(id=row['id'],anchor=i,stage=samples[i]['teacher_stage']))
            windows+=1
    if not windows or torch.cuda.is_initialized():raise ValueError('NONEMPTY_CPU_ONLY_CONSUMPTION_REQUIRED')
    return dict(status='PASS_ACTUAL_INDEPENDENT_FULLBODY_CPU_CONSUMER',windows=windows,RGB_images=images,strata=strata,
        max_XYZ_m=max_xyz,max_SO3_rad=max_so3,max_gripper_BASE_error=max_other,models_loaded=0,physics_steps=0,relative_EEF_count=1,BASE_relative_conversion=False)
