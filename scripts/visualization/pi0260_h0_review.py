"""R1 read-only source adapter around pinned V FK/raster/depth checks."""
from pathlib import Path
import json
import sys
import cv2
import numpy as np
import pi0259_review as v
import pi0259_h0 as old

SCIENCE='90cbb926a842f08c3f9a78756d7a24e0d7a63f78'
WORKFLOW='1f27d9c14ce0a91518930a8d8a2f6f0ffc2dd468'


class Packet(v.Packet):
    def __init__(self,manifest,digest):
        self.path=Path(manifest).resolve();v.require(v.sha(self.path)==digest,'MANIFEST_HASH_MISMATCH')
        self.m=v.read(self.path);self.root=Path(self.m['root']).resolve();self.files={}
        v.require(self.m['schema']=='PI0260_H0_V1' and self.m['protocol_id']==SCIENCE,'WRONG_SCIENCE')
        v.require(self.path.is_relative_to(self.root),'manifest outside source root')
        for row in self.m['entries']:
            p=v.local(self.root,row['path'])
            v.require(row['path'] not in self.files and p.stat().st_size==row['bytes'] and v.sha(p)==row['sha256'],'FILE_IDENTITY_MISMATCH: '+row['path'])
            self.files[row['path']]=row


def coverage(available,observed,missing):
    v.require(set(observed)==set(available),'AVAILABLE_POSE_COVERAGE_MISMATCH')
    v.require(not set(missing)&set(observed),'MISSING_GEOMETRY_MISLABELLED_AS_IMAGE')


def main(manifest,digest,out,urdf):
    out=Path(out);packet=Packet(manifest,digest);old.Packet=Packet
    # Reuse unchanged native FK/state/camera/depth/raster replay. Its old cohort verdict is NOT used.
    legacy=old.review_h0(manifest,digest,out,urdf)
    reg=v.read(packet.file('design/registry.json'));records=v.read(out/'raster_records.json')
    available={};missing={};complete=[]
    for family in reg['calibration_families']:
        family_complete=True
        for seq in [s for s in reg['calibration_sequences'] if s['family_id']==family['family_id']]:
            with np.load(packet.file('calibration/'+seq['sequence_id']+'/geometry.npz'),allow_pickle=False) as z:
                finite=np.isfinite(z['joints']).all(axis=1)&np.isfinite(z['FK_world']).all(axis=(1,2))
                family_complete=family_complete and bool(finite.all())
                for t in reg['renderer']['calibration_times'][seq['task']][1:]:
                    i=int(round(t/.05));key=(seq['sequence_id'],t)
                    (available if finite[i] else missing)[key]=dict(task=seq['task'],command=seq['command'])
        if family_complete:complete.append(family['family_id'])
    v.require(len(complete)>=4,'LESS_THAN_FOUR_PAIRED_CALIBRATION_IDS')
    rigs=[]
    for rig in sorted({r['rig'] for r in records}):
        starts=[r for r in records if r['rig']==rig and r['source'].endswith('/common_start.json')]
        later=[r for r in records if r['rig']==rig and r['sequence']]
        coverage(available,{(r['sequence'],r['time']):r for r in later},missing)
        start_ok=len(starts)==6 and all(r['checks'][t][str(c)]['necessary_pass'] for r in starts for t in ('M1','M3') for c in (0,1))
        later_ok=all(r['checks'][r['task']][str(r['command'])]['necessary_pass'] for r in later)
        rigs.append(dict(rig=rig,start_pass=start_ok,later_pass=later_ok,qualified=start_ok and later_ok,
            starts=len(starts),later=len(later),missing=len(missing)))
    selected=next((r['rig'] for r in rigs if r['qualified']),None)
    v.require(selected==packet.m['camera_outcome']['selected_rig'],'PRODUCER_SELECTION_CONTRADICTS_ARRAYS')
    v.require([r['rig'] for r in rigs]==list(range(len(rigs))),'RIG_ORDER_CHANGED')
    tests=[]
    for args in [({('missing',1):{}},{},{ }),({}, {('missing',1):{}},{('missing',1):{}})]:
        try:coverage(*args)
        except ValueError:tests.append('missing_geometry_rejected')
        else:raise AssertionError('missing geometry negative accepted')
    canvas=np.full((((len(records)+5)//6)*118,6*204,3),25,np.uint8)
    for i,r in enumerate(sorted(records,key=lambda r:r['source'])):
        y=(i//6)*118;x=(i%6)*204
        with np.load(packet.file(r['array']),allow_pickle=False) as a:
            canvas[y+20:y+116,x:x+192]=np.concatenate([a['body_rgb'],a['wrist_rgb']],axis=1)[...,::-1]
        cv2.putText(canvas,('%02d '%i)+r['source'].split('/')[-1].replace('pi0259-cal-','')[:28],(x,y+12),cv2.FONT_HERSHEY_SIMPLEX,.27,(230,230,230),1)
    cv2.imwrite(str(out/'all_native_rgb.png'),canvas)
    receipt=dict(status='H0_NUMERIC_PASS_PENDING_NATURAL_RGB' if selected is not None else 'PI0260_CAMERA_DESIGN_BLOCKED',
        stage='H0',scientific_authority=SCIENCE,workflow_authority=WORKFLOW,execution_owner='PI0260_SINGLE_OWNER_M',
        review_mode='SAME_OWNER_READONLY_REPLAY',producer_manifest_sha256=digest,
        verifier_source_sha256=v.sha(__file__),shared_dependencies='Pinned V full_fk/kinematics, independent V raster/depth logic; no producer PASS trusted',
        verifier_dependencies={p.name:v.sha(p) for p in Path(__file__).parent.glob('*.py')},
        source_files_checked=len(packet.files),image_pairs=len(records),complete_paired_family_ids=complete,
        alias_note='cal001 and cal004 are color aliases; four original IDs, three unique numeric geometries',rigs=rigs,selected_rig=selected,
        missing_geometry=[dict(sequence=s,time=t,**row) for (s,t),row in missing.items()],
        maximum_saved_fk_delta=legacy['maximum_saved_fk_delta'],maximum_state_delta=legacy['maximum_state_delta'],
        maximum_camera_rotation_matrix_delta=legacy['maximum_camera_rotation_matrix_delta'],
        depth_surface_reconstruction=legacy['depth_surface_reconstruction'],repeat_pairs=legacy['exact_common_start_repeat_pairs'],
        negative_checks=tests,new_calls=v.ZERO,next_stage='NATURAL_RGB_REVIEW',
        old_verdict_scope='receipt.json is legacy PI0259 cohort criterion, inapplicable to PI0260; numeric outputs reused only',
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    v.write(out/'PI0260_H0_NUMERIC_RECEIPT.json',receipt)
    (out/'index.html').write_text('<h1>PI0260 H0 — SAME_OWNER_READONLY_REPLAY</h1><p>84 native96x96 pairs; kinematic reference, no physics/model.</p><img src="all_native_rgb.png"><p>See PI0260_H0_NUMERIC_RECEIPT.json. Legacy PI0259 cohort verdict is not applicable.</p>')
    print(json.dumps({k:receipt[k] for k in ['status','image_pairs','selected_rig','maximum_saved_fk_delta','maximum_state_delta']}))


if __name__=='__main__':main(*sys.argv[1:])
