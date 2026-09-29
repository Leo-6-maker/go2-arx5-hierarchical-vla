"""CPU-only fault injection. All images/poses here are test fixtures, not experiment evidence."""
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import pi0259_review as v
from pi0259_h0 import raster


def fixture(root):
    root.mkdir()
    (root/'test.urdf').write_text('<robot name="CPU_FIXTURE"><link name="base"/><link name="tcp"/><joint name="j" type="revolute"><parent link="base"/><child link="tcp"/><origin xyz="0 0 0"/><axis xyz="0 0 1"/><limit lower="-1" upper="1"/></joint></robot>')
    n=4;t=np.array([0.,2.,4.,5.9]);fk=np.tile(np.eye(4),(n,1,1));fk[:,:3,3]=[.5,.2,0]
    angle=.3;r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    fk[:,:3,:3]=r
    anchor=np.eye(4);anchor[:3,3]=[.5,.2,0]
    state=np.tile(np.r_[0.,0.,0.,r[:2].reshape(-1),.5],(n,1))
    a=dict(time=t,fk_world=fk,reference_world=fk.copy(),T_world_native=anchor,StateV1=state,joints=np.zeros((n,1)),root_world=fk.copy(),render_tcp_world=fk.copy(),active=np.ones(n,bool),window_start_index=np.arange(n),window_geometry_valid=np.ones(n,bool),raw_gate_reject=np.zeros(n,bool),codec_gate_reject=np.zeros(n,bool))
    attachment=np.eye(4);attachment[2,3]=1
    for stream in ('body','wrist'):
        rgb=np.zeros((n,96,96,3),np.uint8);rgb[:,46:48,46:48]=[255,0,0];rgb[:,50:52,50:52]=[0,0,255]
        seg=np.zeros((n,96,96),np.int32);seg[:,46:48,46:48]=101;seg[:,50:52,50:52]=102
        camera=fk@attachment
        projection=np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,-1],[0,0,-.2,0]])
        a.update({stream+'_rgb':rgb,stream+'_segmentation':seg,stream+'_depth':np.full((n,96,96),-1.),stream+'_time':t.copy(),stream+'_world':camera,stream+'_view':np.linalg.inv(camera).swapaxes(-1,-2),stream+'_projection':np.tile(projection,(n,1,1))})
    np.savez_compressed(root/'arrays.npz',**a)
    row=dict(source_id='CPU_FIXTURE_NOT_SCIENCE',task='M1',command=0,instruction='Approach the red target and align the tool with its axes.',colors=['red','blue'],target_id='target0',command_targets={'0':'target0','1':'target1'},frame='world',state_frame='native',units=dict(position='m',rotation='rad',time='s'),state_origin='KINEMATIC_FK_NOT_DYNAMIC_MEASUREMENT',time_origin='synthetic_reference_time',rgb_order='RGB',arrays={k:dict(path='arrays.npz',key=k) for k in a},urdf='test.urdf',joint_names=['j'],tcp_link='tcp',tcp_offset_m=[0,0,0],camera_links=dict(body='base',wrist='tcp'),camera_attachments=dict(body=attachment.tolist(),wrist=attachment.tolist()),camera_convention='row_vector_opengl',depth_convention='negative_camera_z_m',cues=[dict(id=101,name='target0',role='target',world_m=[.5,.2,0]),dict(id=102,name='target1',role='target',world_m=[.5,.2,0])],required_cues=['target0'],complete_sequence=True,geometry_tuple=[0,0,0,.035],fixed_root_dofs_sha256='test-root',camera_sha256='test-camera',asset_sha256='test-asset',split='TRAIN',collator_fields=sorted(v.ALLOW),collator_arrays={k:dict(path='arrays.npz',key=k) for k in v.ALLOW-{'language'}},collator_language='Approach the red target and align the tool with its axes.',paired_prefix_indices=[0],outcome='TEST_FIXTURE')
    v.write(root/'records.json',[row])
    m=dict(sealed=True,producer_role='Agent M',protocol_commit=v.AUTH,code_commit='a'*40,stage='H0',review_records='records.json',source_ids=[row['source_id']],files=[dict(path=p.name,bytes=p.stat().st_size,sha256=v.sha(p)) for p in sorted(root.iterdir())])
    v.write(root/'manifest.json',m)
    return row,a


def main():
    checks=[]
    def rejected(name,fn,contains=None):
        try:fn()
        except (ValueError,RuntimeError,KeyError) as e:
            if contains:assert contains in str(e),(name,str(e))
            checks.append(name)
        else:raise AssertionError('negative was accepted: '+name)
    with tempfile.TemporaryDirectory() as d:
        base=Path(d);root=base/'sealed';row,a=fixture(root)
        m=root/'manifest.json';digest=v.sha(m);packet=v.Packet(m,digest)
        r=v.check_record(packet,row);assert r['visibility_fraction']==1 and r['position_max_m']==0
        checks.append('source_96px_raw_masks_nonzero_root_fk')
        for name,key,value in [('changed_target','target_id','target1'),('swapped_command','command',1),('mixed_frame','state_frame','world')]:
            bad=copy.deepcopy(row);bad[key]=value;rejected(name,lambda:v.check_record(packet,bad))
        class Changed:
            def __init__(self,field,mutate):self.field=field;self.mutate=mutate
            def file(self,name):return packet.file(name)
            def array(self,ref):
                x=packet.array(ref)
                return self.mutate(x) if ref['key']==self.field else x
        rejected('dropped_initial_quarter',lambda:v.check_record(Changed('active',lambda x:np.array([False,True,True,True])),row),'DROPPED_MOTION_QUARTER')
        rejected('camera_fetch_pose_mismatch',lambda:v.check_record(Changed('render_tcp_world',lambda x:x+np.eye(4)*.1),row))
        rejected('image_state_timestamp_mismatch',lambda:v.check_record(Changed('body_time',lambda x:x+.05),row),'TIME_MISMATCH')
        invisible=v.check_record(Changed('body_segmentation',lambda x:np.zeros_like(x)),row)
        assert invisible['cue_counts']['body']['target0']==[0]*4 and invisible['cue_counts']['wrist']['target0']==[4]*4
        checks.append('visibility_recomputed_from_masks_not_pass_or_boxes')
        bad=copy.deepcopy(row);bad['collator_fields'].append('target_id')
        rejected('target_metadata_in_collator',lambda:v.check_cohort([bad],{row['source_id']:r},packet),'COLLATOR')
        pair=copy.deepcopy(row);pair.update(source_id='CPU_PAIR',command=1,target_id='target1',instruction='Reach the blue marker and match the marked tool frame.',collator_language='Reach the blue marker and match the marked tool frame.')
        cohort=v.check_cohort([row,pair],{row['source_id']:r,pair['source_id']:r},packet)
        assert cohort['complete_input_pairs']=={'M1/TRAIN':1}
        checks.append('paired_inputs_byte_compared')
        other=copy.deepcopy(pair);other['split']='KIN-VAL'
        assert any('GEOMETRY_SPLIT_LEAK' in f for f in v.check_cohort([row,other],{row['source_id']:r,other['source_id']:r},packet)['findings'])
        checks.append('color_alias_cannot_cross_geometry_split')
        r2=copy.deepcopy(r);r2['arrays']['StateV1'][0,0]+=.1
        rejected('paired_input_changed',lambda:v.check_cohort([row,pair],{row['source_id']:r,pair['source_id']:r2},packet))
        rejected('wrong_manifest_hash',lambda:v.Packet(m,'0'*64),'HASH_MISMATCH')
        rejected('path_traversal',lambda:v.local(root,'../outside.npz'),'escapes')
        rejected('source_overwrite',lambda:v.review(m,digest,root/'review','H0'),'OVERLAPS')
        before={p.name:v.sha(p) for p in root.iterdir()}
        receipt=v.review(m,digest,base/'review','H0')
        assert receipt['status']=='H0_NUMERIC_REVIEW_PARTIAL',receipt
        assert len(receipt['cards'])==4
        assert before=={p.name:v.sha(p) for p in root.iterdir()}
        assert all(__import__('cv2').imread(str(base/'review'/c['path'])).shape==(720,1280,3) for c in receipt['cards'])
        checks.append('readonly_companion_cards_receipt_no_false_design_pass')
        # H2 renderer takes sealed arrays only, including a known nonzero SE3 error.
        pred=np.tile(np.eye(4),(40,1,1));ref=pred.copy();pred[:,0,3]=.01;noise=np.arange(8,dtype=np.float32)
        np.savez_compressed(root/'pred.npz',prediction=pred,reference=ref,time=np.arange(40)*.05,noise=noise)
        v.write(root/'cpu.json',{'status':'FIXTURE_ONLY_NOT_EXPERIMENT_QA'})
        hr=dict(source_id='H2_CPU_FIXTURE',task='M1',instruction=row['instruction'],model='MT-G',step=0,evidence_class='OFFLINE_MODEL_PREDICTION',frame='native',checkpoint_sha256='a'*64,processor_sha256='b'*64,initializer_revision='c'*40,noise_sha256=v.hashlib.sha256(noise.tobytes()).hexdigest(),cpu_receipt='cpu.json',first_valid_or_reuse='FIRST_VALID',inference_latency_s=1.,arrays={k:dict(path='pred.npz',key=k) for k in ['prediction','reference','time','noise']})
        v.write(root/'h2-records.json',[hr]);hm=v.read(m)
        hm.update(stage='H2',review_records='h2-records.json',source_ids=[hr['source_id']])
        hm['files']=[dict(path=p.name,bytes=p.stat().st_size,sha256=v.sha(p)) for p in sorted(root.iterdir()) if p.name not in ('manifest.json','h2-manifest.json')]
        v.write(root/'h2-manifest.json',hm)
        hp=v.Packet(root/'h2-manifest.json',v.sha(root/'h2-manifest.json'))
        pr=v.prediction_record(hp,hr);assert abs(pr['xyz_rmse_m']-.01)<1e-12
        h2=v.review(root/'h2-manifest.json',v.sha(root/'h2-manifest.json'),base/'h2-review','H2')
        assert h2['status']=='H2_NUMERIC_REVIEW_PARTIAL' and len(h2['cards'])==1
        bad=copy.deepcopy(hr);bad['noise_sha256']='0'*64
        rejected('h2_noise_identity',lambda:v.prediction_record(hp,bad),'noise mismatch')
        bad=copy.deepcopy(hr);bad['step']=1000
        rejected('h2_intermediate_endpoint_denied',lambda:v.prediction_record(hp,bad),'endpoint')
        checks.append('h2_sealed_numeric_metrics_and_card_no_model')
        ids={'black_A':100,'white_B':101,**{f'contour{i}':102+i for i in range(24)}}
        pixels={s+'_rgb':np.zeros((96,96,3),np.uint8) for s in ('body','wrist')}
        pixels.update({s+'_segmentation':np.zeros((96,96),np.uint32) for s in ('body','wrist')})
        pixels['body_segmentation'][0,:4]=100;pixels['body_segmentation'][1,:4]=101
        assert raster(pixels,ids,'M3',0,False)['necessary_pass']
        assert not raster(pixels,ids,'M3',0,True)['necessary_pass']
        pixels['body_segmentation'][0,3]=0
        assert not raster(pixels,ids,'M3',0,False)['necessary_pass']
        checks.append('native_h0_start_only_sectors_and_later_endpoint_threshold')
        for module in ('torch','transformers','isaacgym','isaacsim'):
            rejected('blocked_import_'+module,lambda module=module:__import__(module),'FORBIDDEN_RENDER_IMPORT')
        run=subprocess.run([sys.executable,str(Path(v.__file__)),'run-episode'],capture_output=True,text=True)
        assert run.returncode!=0 and 'GUARD_STUB_ONLY' in run.stderr
        checks.append('run_episode_guard_stub_only')
    assert not {'torch','transformers','isaacgym','isaacsim'} & set(sys.modules)
    print(json.dumps(dict(status='PASS_CPU_NEGATIVE_TESTS',checks=checks,new_calls=v.ZERO),indent=2))


if __name__=='__main__':main()
