"""One-time sidecar derivation; original C20 files/statistics are never written."""
import json
from pathlib import Path
import numpy as np
from .checkpoint_v1 import sha
from .gripper_time_v1 import label_parameters,decode,reference_width


def read(path):return json.loads(Path(path).read_bytes())


def build(dataset,manifest_sha256):
    root=Path(dataset);manifest=read(root/'MANIFEST.private.json')
    if sha(root/'MANIFEST.private.json')!=manifest_sha256 or manifest['windows']!=1376 or len(manifest['episodes'])!=20:
        raise ValueError('ORIGINAL_FIXED_C20_1376_ONLY')
    stats=read(root/'STATISTICS.json');stats_pin=sha(root/'STATISTICS.json')
    if stats_pin!=manifest['statistics']['sha256']:raise ValueError('ORIGINAL_FROZEN_C20_STATISTICS')
    low,high=stats['actions']['q01'][6],stats['actions']['q99'][6]
    contexts=[];targets=[];records=[];pins={}
    for ep in manifest['episodes']:
        result=Path(ep['native_result']['path']);capture=read(ep['original_live_capture']['path'])
        if sha(result)!=ep['native_result']['sha256']or sha(capture['labels']['path'])!=capture['labels']['sha256']:
            raise ValueError('ORIGINAL_NATIVE_OR_SUBMITTED_LABEL_IDENTITY_CHANGED')
        live=read(capture['labels']['path']);sample_file=result.parent/'native-01/MOBILE_SAMPLES.private.jsonl'
        samples=[json.loads(l)for l in sample_file.read_text().splitlines()]
        by_step={r['native_end_step']:r for r in samples}
        if len(by_step)!=len(samples):raise ValueError('AMBIGUOUS_ORIGINAL_COMMAND_CLOCK')
        path=root/ep['labels']['file']
        if sha(path)!=ep['labels']['sha256']:raise ValueError('C20_LABEL_BYTES_CHANGED')
        with np.load(path,allow_pickle=False)as z:
            steps=z['native_step'];nominal=z['nominal_policy_H40'];phases=z['phase']
        for anchor,(tick,trajectory)in enumerate(zip(steps,nominal)):
            sample=by_step[int(tick)];label=live[anchor]
            if label['native_step']!=int(tick)or abs(label['native_time_s']-.002*int(tick))>1e-8:
                raise ValueError('EXACT_C20_OBSERVATION_SOURCE_CLOCK')
            context=dict(width_m=.044*float(sample['installed_EEF10'][9]),
                source_native_s=sample['gripper_command']['source_native_s'],query_native_s=label['native_time_s'])
            reference_width(context)
            expected=.002+.040*trajectory[:,9];target,detail=label_parameters(expected,context,low,high)
            actual,_=decode(target,context,low,high)
            speed=float(np.max(abs(np.diff(np.r_[context['width_m'],actual])/.05)))
            crossing=lambda w:next((int(k)for k,x in enumerate(w)if x<=.022),None)
            truth=crossing(expected)if expected[0]>.022 and expected[-1]<=.022 else None
            prediction=crossing(actual);event_error=None if truth is None or prediction is None else .05*abs(truth-prediction)
            if(detail['maximum_width_reconstruction_error_m']>5e-5 or speed>.04*(1+1e-10)
                    or actual.min()<.002-1e-12 or actual.max()>.042+1e-12
                    or truth is not None and(event_error is None or event_error>.100000001)):
                raise ValueError('TIME_CODEC_CANNOT_FAITHFULLY_REPRESENT_C20:'+str((ep['id'],anchor,detail,speed,event_error)))
            contexts.append(dict(index=len(contexts),episode=ep['id'],anchor=anchor,**context));targets.append(target)
            records.append(dict(index=len(records),phase=str(phases[anchor]),width_error_m=detail['maximum_width_reconstruction_error_m'],
                peak_speed_with_seam_m_s=speed,annotated_crossing=truth is not None,crossing_error_s=event_error,**detail))
        pins[ep['id']]=dict(native_result_sha256=sha(result),live_labels_sha256=capture['labels']['sha256'],
                            actual_command_samples_sha256=sha(sample_file),sealed_labels_sha256=ep['labels']['sha256'])
    if len(targets)!=1376:raise ValueError('ALL_C20_WINDOWS_REQUIRED')
    return np.asarray(targets,np.float32),contexts,dict(records=records,source_pins=pins,
        manifest_sha256=manifest_sha256,statistics_sha256=stats_pin,low_grip_quantile=low,high_grip_quantile=high,
        maximum_width_error_m=max(r['width_error_m']for r in records),
        maximum_speed_with_seam_m_s=max(r['peak_speed_with_seam_m_s']for r in records),
        annotated_crossings=sum(r['annotated_crossing']for r in records),
        annotated_crossings_within100ms=sum(r['annotated_crossing']and r['crossing_error_s']is not None and r['crossing_error_s']<=.100000001 for r in records))
