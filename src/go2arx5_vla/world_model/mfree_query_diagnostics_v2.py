"""Read-only exact H40 envelope attribution; no alternative safety gate."""
import numpy as np
from scripts.pi0292.eval_contract import matrix10
from scripts.pi0291_arx5_data.native_dataset import pose10
from . import fusion_trajectory_adapter as limits
from .task_learning_executability_v3 import path_series
from .task_learning_grasp_v1 import FiftyMsProgram
CHANNELS=(('linear_velocity','maximum_translation_speed_m_s'),('angular_velocity','maximum_angular_speed_rad_s'),('finger_velocity','maximum_finger_speed_m_s'),('linear_acceleration','maximum_discrete_translation_acceleration_m_s2'),('angular_acceleration','maximum_discrete_angular_acceleration_rad_s2'))

def analyze(raw,state,frame,reference):
    raw=np.asarray(raw);actual=np.linalg.inv(frame)@matrix10(state);target=pose10(np.linalg.inv(frame)@matrix10(raw),raw[:,9]);desired=target.copy()
    desired[:,9]=(limits.WORKING_CLOSED_M+(limits.WORKING_OPEN_M-limits.WORKING_CLOSED_M)*raw[:,9])/limits.MECHANICAL_TRAVEL_M
    installed=desired.copy();installed[:,:3]=reference[:3,3]+target[:,:3]-actual[:3,3]
    rotation=reference[:3,:3]@actual[:3,:3].T@matrix10(target)[:,:3,:3]
    from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows
    installed[:,3:9]=rotation_matrix_to_rot6d_rows(rotation)
    paths={};events=[]
    for name,path,anchor in (('nominal',desired,actual),('reference',installed,reference)):
        series=path_series(path,grip_is_mechanical_fraction=True);original=limits._geometry(path,anchor);channels={}
        for order,(channel,key)in enumerate(CHANNELS):
            info=series[channel];values=np.asarray(info['magnitudes']);threshold=info['limit'];peak=int(values.argmax());bad=np.flatnonzero(values>threshold*(1+1e-10))
            if not np.isclose(values[peak],original[key],atol=1e-9,rtol=1e-8):raise ValueError('EXACT_ORIGINAL_ENVELOPE_ARITHMETIC:'+channel)
            summary=dict(peak_value=float(values[peak]),threshold=threshold,peak_threshold_ratio=float(values[peak]/threshold),margin=threshold-float(values[peak]),peak_k=peak,first_exceedance_k=int(bad[0])if len(bad)else None,exceedance_indices=bad.tolist(),prefix_peak=float(values[:10].max()),suffix_peak=float(values[10:].max()),zero_velocity_endpoint_padding=channel.endswith('acceleration'))
            channels[channel]=summary
            if len(bad):
                k=int(bad[0]);events.append(dict(path=name,channel=channel,k=k,time_s=.05*k,region='START_REST_BOUNDARY'if channel.endswith('acceleration')and k==0 else'END_REST_BOUNDARY'if channel.endswith('acceleration')and k==39 else'P10'if k<10 else'P10_SUFFIX_JOIN'if k==10 else'SUFFIX',first_ratio=float(values[k]/threshold),peak_ratio=summary['peak_threshold_ratio'],order=order))
        paths[name]=dict(channels=channels,full_H40_pass=not any(v['exceedance_indices']for v in channels.values()),first_point_anchor_gap_m=float(np.linalg.norm(path[0,:3]-anchor[:3,3])),path_length_m=original['path_length_m'],endpoint_progress_m=float(np.linalg.norm(path[-1,:3]-path[0,:3])))
    passed=not events
    try:FiftyMsProgram(target,actual,reference,np.r_[np.zeros(3),0.,0.,0.,1.]);gate=True
    except ValueError as e:
        if 'ORIGINAL_FULL_PREVIEW_RATE_AND_ACCELERATION_LIMITS'not in str(e):raise
        gate=False
    if gate!=passed:raise ValueError('ORIGINAL_COMPILER_GATE_DISAGREES_WITH_ATTRIBUTION')
    first=min(events,key=lambda x:(x['k'],x['path']!='nominal',x['order']))if events else None
    short=next((e for path in ('nominal','reference')for channel,_ in CHANNELS for e in events if e['path']==path and e['channel']==channel),None)
    return dict(original_full_H40_pass=passed,paths=paths,first_temporal_exceedance=first,original_short_circuit_component=short,rotation_derivative='RELATIVE_SO3_LOG_NOT_EULER',dt_s=.05,prefix_suffix_index_definition='k<10 prefix; k=10 replan join; acceleration39 is last-segment-to-rest; velocity k is segment k to k+1',full_suffix_gate_retained=True)
