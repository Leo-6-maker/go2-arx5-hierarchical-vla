"""Seal actual H0 rasters and inherited geometry without modifying old evidence."""
import copy
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from pi0260_design import ROOT,OLD,AUTHORITY
from pi0255_motion3 import sha,write,scene


def seal(commit):
    path=ROOT/'handoff/H0_v1/manifest.json';assert not path.exists()
    reg=json.loads((ROOT/'design/registry.json').read_text())
    camera=json.loads((ROOT/'evidence/camera_calibration.json').read_text())
    cues=json.loads((OLD/'design/cue_geometry.json').read_text())
    for row in cues:
        f=next(f for f in reg['calibration_families'] if f['family_id']==row['family_id'])
        d=scene(reg,f);g=reg['geometry']
        for i,p in enumerate(d['targets']):
            rotation=np.array(reg['initial_TCP_world'])[:3,:3]@Rotation.from_euler('z',g['target_yaw_rad'][i]).as_matrix()
            for a in range(3):row['cues']['axis%d_%d'%(i,a)].update(position=(p+rotation[:,a]*g['axes_length_m']).tolist(),radius=g['axes_marker_radius_m'])
        for name in ('black_A','white_B'):row['cues'][name]['radius']=g['start_marker_radius_m']
        for i in range(24):row['cues']['contour'+str(i)]['radius']=g['contour_marker_radius_m']
    write(ROOT/'design/cue_geometry.json',cues)
    binding=json.loads((OLD/'design/FK_source_binding.json').read_text())
    assert sha(binding['urdf']['path'])==binding['urdf']['sha256']
    binding.update(parent_binding_sha256=sha(OLD/'design/FK_source_binding.json'),
        full_fk_source=dict(path=str(ROOT/'runtime/pi0258_geometry.py'),sha256=sha(ROOT/'runtime/pi0258_geometry.py')))
    write(ROOT/'design/FK_source_binding.json',binding)
    geometry=[json.loads((ROOT/'calibration'/s['sequence_id']/'result.json').read_text()) for s in reg['calibration_sequences']]
    oldcoverage={f['family_id']:f['factors'] for f in reg['calibration_families'] if f['family_id'] in camera['complete_paired_family_ids']}
    write(ROOT/'evidence/complete_geometry_coverage.json',dict(families=oldcoverage,
        note='Four original IDs; cal-001/cal-004 are color aliases of one numerical geometry. IDs and numeric geometry both disclosed.'))
    entries=[]
    for name in ('design','calibration','calibration_rigs','evidence','runtime'):
        for p in sorted((ROOT/name).rglob('*')):
            if not p.is_file() or (name=='runtime' and (p.parent!=ROOT/'runtime' or p.suffix!='.py')):continue
            assert not p.is_symlink()
            entries.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    entries.sort(key=lambda x:x['path'])
    write(path,dict(schema='PI0260_H0_V1',stage='H0',root=str(ROOT),protocol_id=AUTHORITY,code_commit=commit,
        producer_role='M',entries=entries,identities={name:sha(ROOT/'design'/name) for name in ('registry.json','cue_geometry.json','FK_source_binding.json')},
        camera_outcome=camera,outcome_index=geometry,review_state='PENDING_AGENT_V',
        state_origin='KINEMATIC_FK_NOT_DYNAMIC_MEASUREMENT',evidence_class='KINEMATIC_RENDERED_REFERENCE',
        cohort='OLD_EXPOSED_QUARANTINED_CALIBRATION_ONLY',
        raster_conventions=dict(rgb='uint8 RGB actual96x96',depth='camera-axis negative metres, -inf background',
            projection='row world homogeneous @ view @ projection; x=(ndcx+1)*48, y=(1-ndcy)*48; pixel centers +.5'),
        counters=dict(new_IK_calls=0,calibration_sequences_reused=24,calibration_image_pairs=camera['image_pairs'],
            physics_steps=0,WBC_actor_calls=0,model_queries=0,optimizer_updates=0),
        missing_fields=['No fresh proposal outcomes, TRAIN data, processor, canary or model queries before H0 admission.',
            'Missing old geometric suffixes excluded from camera measurements, not promoted to usable geometry.'],
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print(json.dumps(dict(manifest=str(path),sha256=sha(path),entries=len(entries),status=camera['status'])))


if __name__=='__main__':seal(sys.argv[1])
