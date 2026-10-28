"""Read-only raster replay; preserve original rendering and correction evidence."""
import json
import copy
import numpy as np
from pi0259_kinematic import ROOT
from pi0259_observation import visibility
from pi0255_motion3 import sha,write


def replay():
    original=ROOT/'evidence/camera_calibration.json';camera=copy.deepcopy(json.loads(original.read_text()));rows=[]
    fixture_frames={c:np.zeros((96,96,3),np.uint8) for c in ('body','wrist')}
    for c in ('body','wrist'):fixture_frames[c+'_segmentation']=np.zeros((96,96),int)
    ids=dict(black_A=100,white_B=101,**{'contour'+str(i):102+i for i in range(24)})
    fixture_frames['body_segmentation'][0,:4]=100;fixture_frames['body_segmentation'][1,:4]=101
    assert not visibility(fixture_frames,ids,'M3',0,True)['pass_necessary']
    assert visibility(fixture_frames,ids,'M3',0,False)['pass_necessary']
    for outcome in camera['rigs']:
        rig=outcome['rig_index'];starts=[];later=[]
        for path in sorted((ROOT/'calibration_rigs'/('rig'+str(rig))).glob('*/*.json')):
            record=json.loads(path.read_text())
            if 'array_file' not in record:continue
            packet=path.parent/record['array_file'];assert sha(packet)==record['array_sha256']
            with np.load(packet) as a:data={k:a[k].copy() for k in a.files}
            frames={('body' if k=='body_rgb' else 'wrist' if k=='wrist_rgb' else k):v for k,v in data.items()}
            checks={t:{str(c):visibility(frames,record['scene_ids'],t,c,record['synthetic_reference_time']==0) for c in (0,1)} for t in ('M1','M2','M3')}
            row=dict(record_path=str(path.relative_to(ROOT)),source_sha256=sha(path),array_sha256=sha(packet),visibility=checks)
            rows.append(row)
            if path.stem=='common_start':starts.append(row)
            elif record.get('sequence'):
                later.append(dict(sequence=record['sequence'],task=record['task'],time=record['synthetic_reference_time'],
                    pass_necessary=checks[record['task']][str(record['command'])]['pass_necessary']))
        outcome['start_supported_tasks']=[t for t in ('M1','M2','M3') if all(x['visibility'][t][str(c)]['pass_necessary'] for x in starts for c in (0,1))]
        outcome['calibration_frames']=later
        outcome['supported_tasks']=[t for t in outcome['start_supported_tasks'] if not any(x['task']==t for x in outcome['missing_geometry'])
            and len([x for x in later if x['task']==t])==48 and all(x['pass_necessary'] for x in later if x['task']==t)]
    supported=[r['rig_index'] for r in camera['rigs'] if len(r['supported_tasks'])>=2]
    camera.update(selected_rig=min(supported) if supported else None,status='H0_REVIEW_PENDING' if supported else 'OBSERVATION_DESIGN_BLOCKED',
        original_camera_report_sha256=sha(original),raster_replay_source=sha(__file__),rules_source=sha(ROOT/'runtime/pi0259_observation.py'),
        correction='Eight contour sectors apply at common start only. Original per-frame reports retained. Endpoint thresholds unchanged.',
        renderer_exit='Native destructor SIGSEGV after final JSON was durably written. No rerender; all NPZ hashes/readback checked in CPU replay.')
    write(ROOT/'evidence/raster_cpu_replay.json',dict(schema='PI0259_RASTER_CPU_REPLAY_V1',records=rows,camera_outcome=camera,
        fixture='PASS_START_ONLY_SECTOR_RULE',physics_steps=0,render_calls=0))
    write(ROOT/'evidence/camera_calibration_cpu_corrected.json',camera)
    print(json.dumps(dict(records=len(rows),rigs=[dict(rig=x['rig_index'],start=x['start_supported_tasks'],full=x['supported_tasks']) for x in camera['rigs']],status=camera['status'])),flush=True)


if __name__=='__main__':replay()
