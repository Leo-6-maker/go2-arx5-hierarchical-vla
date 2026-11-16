"""One CPU input/decode seam check for the new representation, no model calls."""
import json
from pathlib import Path
import time
import numpy as np
from go2arx5_vla.sla_adapter.time_training_v1 import TimeDataset,physical
from go2arx5_vla.sla_adapter.mfree_training_v1 import MobileDataset
from go2arx5_vla.sla_adapter.codec_v1 import pad32
from go2arx5_vla.sla_adapter.runtime_v1 import probe_indices


def run(spec):
    from openpi_client.image_tools import resize_with_pad
    start=time.time();d=TimeDataset(spec['dataset'],spec['dense_manifest_sha256'],time_audit=spec['time_codec_audit'],
                                  resize_with_pad=resize_with_pad,max_cache_bytes=128*1024**2)
    records=[]
    for index in probe_indices(d):
        actual=d[index];before=MobileDataset.__getitem__(d,index)
        if(not np.array_equal(actual['state'],before['state'])or actual['prompt']!=before['prompt']
                or not np.array_equal(actual['actions'][:,:6],before['actions'][:,:6])
                or not np.array_equal(actual['actions'][:,7:],before['actions'][:,7:])):
            raise ValueError('REPRESENTATION_CHANGE_MUST_ONLY_TOUCH_ACTION6')
        for group in ('image','image_mask','image_padding_mask'):
            for name in actual[group]:
                if not np.array_equal(actual[group][name],before[group][name]):raise ValueError('EXISTING_RGB_INPUT_CHANGED')
        state,_,_,_=d.raw(index);ep,_=d.rows[index];frame=d.frames[ep['id']]['T_policy_from_mount']
        original=d.codecs[ep['id']].decode(state,pad32(d.statistics['actions'].unnormalize(before['actions'][:,:7])))
        decoded,detail=physical(actual['actions'][None],state,frame,d.statistics,d.reference(index))
        if not np.array_equal(decoded[:,:9],original[:,:9]):raise ValueError('XYZ_OR_SO3_DECODE_CHANGED')
        width=.002+.040*decoded[:,9];previous=d.reference(index)['width_m']
        peak=float(abs(np.diff(np.r_[previous,width])/.05).max())
        if peak>.04*(1+1e-10):raise ValueError('SAVED_LABEL_EVENT_AND_SEAM_BOUND')
        records.append(dict(index=index,unchanged_XYZ_SO3_bit_exact=True,unchanged_original_RGB_state_prompt=True,
                            peak_width_speed_with_seam_m_s=peak,time_parameters=detail))
    return dict(schema='SLA_TIME_REPRESENTATION_EXISTING_INPUT_CPU_SEAM_V1',exit_code=0,records=records,
        original_data_and_statistics_unchanged=True,only_action_dimension6_representation_changed=True,
        new_model_generations=0,new_optimizer_updates=0,new_physical_requests=0,new_integrations=0,
        new_camera_render_calls=0,new_camera_qualification=False,physical_admission=False,
        wall_s=time.time()-start,CPU_s=time.process_time())


if __name__=='__main__':
    import sys
    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_bytes()))))
