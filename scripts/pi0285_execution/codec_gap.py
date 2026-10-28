"""Read-only CPU decomposition of the original failed frozen codec admission."""
import json,os
from collections import Counter
from pathlib import Path
import numpy as np
from contract import ROOT,sha,write


def run():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    from codec import native_imports,TAG,fields
    native_imports()
    from gr00t.data.state_action.state_action_processor import StateActionProcessor
    from gr00t.data.types import ActionType,ActionFormat
    from gr00t.data.utils import normalize_values_minmax
    from cpu_inputs import physical_errors
    from scipy.spatial.transform import Rotation
    root=ROOT/'data';config=json.loads((root/'processor_config.json').read_text())['processor_kwargs']
    names=('SOURCE_BANK.json','STATISTICS_RELEASE.json','statistics.json','processor_config.json','embodiment_id.json','WINDOW_INDEX.json')
    before={n:sha(root/n) for n in names}
    processor=StateActionProcessor(modality_configs=config['modality_configs'],statistics=json.loads((root/'statistics.json').read_text()),use_percentiles=config['use_percentiles'],clip_outliers=config['clip_outliers'],use_relative_action=config['use_relative_action'])
    assert processor.clip_outliers and not processor.use_percentiles and processor.use_relative_action
    params=processor.norm_params[TAG]['action']['eef_9d'];bank=json.loads((root/'SOURCE_BANK.json').read_text());rows=[];counts=Counter();worst=None
    for ep in bank['admitted_episodes']:
        if ep['split']!='DEV':continue
        name=ep['episode'];state=np.load(root/'policy_observation'/name/'state_v1.npy',mmap_mode='r');commands=np.load(root/'action_target'/name/'action.npy',mmap_mode='r');phase=np.load(root/'action_target'/name/'phase.npy',mmap_mode='r')
        anchors=sorted(set([int(x) for x in np.unique(phase[20:621],return_index=True)[1]+20]+[620]))
        for a in anchors:
            target=commands[a:a+40].copy();actual=state[a].copy();normalized=processor.apply_action(fields(target),TAG,fields(actual[None]))
            decoded_fields=processor.unapply_action(normalized,TAG,fields(actual[None]));decoded=np.concatenate([decoded_fields['eef_9d'],decoded_fields['gripper_position']],axis=-1).astype(np.float32)
            relative=processor._convert_to_relative_action(target[:,:9],actual[:9],ActionType.EEF,ActionFormat.XYZ_ROT6D)
            raw=normalize_values_minmax(relative,params);assert np.array_equal(np.clip(raw,-1,1),normalized['eef_9d'])
            # Counterfactual algebra only: no live processor, statistics, or decoder modification.
            unbounded=(raw+1)/2*(params['max']-params['min'])+params['min']
            absolute=processor._convert_to_absolute_action(unbounded,actual[:9],ActionType.EEF,ActionFormat.XYZ_ROT6D)
            no_clip=np.concatenate([absolute,target[:,9:10]],axis=1).astype(np.float32)
            native_error=physical_errors(target,decoded);counterfactual=physical_errors(target,no_clip)
            def matrix(r):
                x=r[:,:3].astype(float);x/=np.linalg.norm(x,axis=1)[:,None];y=r[:,3:].astype(float);y-=x*(x*y).sum(1)[:,None];y/=np.linalg.norm(y,axis=1)[:,None];return np.stack((x,y,np.cross(x,y)),axis=1)
            independent_angle=Rotation.from_matrix(matrix(decoded[:,3:9])@matrix(target[:,3:9]).transpose(0,2,1)).magnitude()
            assert np.isclose(float(independent_angle.max()),native_error['SO3_max_rad'],rtol=1e-8,atol=1e-10)
            passed=all(native_error[k]<=v for k,v in [('XYZ_max_m',1e-5),('SO3_max_rad',1e-4),('gripper_max_fraction',1e-5)])
            row=dict(episode=name,task=ep['task'],anchor=a,phase=int(phase[a]),native_error=native_error,counterfactual_unbounded_math_error=counterfactual,gate_pass=passed,preclip_OOB_by_EEF_channel=(abs(raw)>1).sum(0).tolist(),preclip_max_abs=float(abs(raw).max()))
            rows.append(row);counts[ep['task']+'_'+('PASS' if passed else 'FAIL')]+=1
            if worst is None or native_error['SO3_max_rad']>worst[0]['native_error']['SO3_max_rad']:worst=(row,dict(target=target,actual_state=actual,native_normalized=normalized['eef_9d'],native_decoded=decoded,relative_native=relative,preclip_normalized=raw,counterfactual_unbounded_decoded=no_clip))
    assert rows and before=={n:sha(root/n) for n in names},'Frozen bank/statistics changed'
    out=ROOT/'audit/codec-gap';out.mkdir(exist_ok=True)
    if worst is not None:np.savez_compressed(out/'WORST_RECORDED_DEV_CODEC.npz',**worst[1])
    report=dict(status='CPU_READ_ONLY_FROZEN_CODEC_GAP_DECOMPOSITION_NOT_ADMISSION',original_failed_receipt_sha256=sha(ROOT/'runtime/codec_check-BANK.CPU_COMPLETE.json'),bindings=before,rows=rows,counts=dict(counts),worst=worst[0],worst_arrays_sha256=sha(out/'WORST_RECORDED_DEV_CODEC.npz'),native_normalization= dict(min=params['min'].tolist(),max=params['max'].tolist()),statistics_changed=False,processor_changed=False,decoder_changed=False,counterfactual_used_for_training_or_inference=False,new_model_queries=0,new_gradients=0,new_optimizer_updates=0,new_physics=0,launch_allowed=False,selected='NONE',deployment='NONE')
    write(out/'REPORT.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('rows','native_normalization')}));return report


if __name__=='__main__':run()
