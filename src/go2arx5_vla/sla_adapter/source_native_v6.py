"""Typed native dual-arm H50 recorded-reference reader, separate from C20.

No source pose is a Go2 command. TEST numerical labels are not prepared here.
"""
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA='SLA_SOURCE_FUTURE_LOGGED_REFERENCE_V1'
COORDINATES='SOURCE_ARM_LOCAL_V1'
CAMERAS=('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')
SOURCE_CAMERAS=('cam_head','cam_left_wrist','cam_right_wrist')
HORIZON=50


def pad32(value):
    array=np.asarray(value)
    if array.shape[-1]!=14 or not np.isfinite(array).all():raise ValueError('TWO_FINITE_NATIVE_ARM7_REQUIRED')
    return np.pad(array,[(0,0)]*(array.ndim-1)+[(0,18)])


def relative_arrays(state_poses,action_poses,state_grips,action_grips):
    """Spatial relative SO3 R_future @ R_current.T in each arm-local frame."""
    state_parts=[];action_parts=[];errors=[];lengths={len(x)for x in[*state_poses,*action_poses,*state_grips,*action_grips]}
    if len(lengths)!=1:raise ValueError('SAME_INDEXED_RECORDED_ARRAY_LENGTHS_REQUIRED')
    count=lengths.pop();valid=count-HORIZON
    if valid<=0:raise ValueError('REAL_FUTURE_H50_WINDOW_REQUIRED')
    index=np.arange(valid)[:,None]+np.arange(HORIZON)[None,:]
    for s,a,gs,ga in zip(state_poses,action_poses,state_grips,action_grips,strict=True):
        s,a=np.asarray(s),np.asarray(a);gs,ga=np.asarray(gs).reshape(-1),np.asarray(ga).reshape(-1)
        if s.shape!=(count,7)or a.shape!=(count,7)or not all(np.isfinite(x).all()for x in(s,a,gs,ga)):
            raise ValueError('FINITE_NATIVE_POSE_WXYZ_AND_GRIPPER_SCALAR_REQUIRED')
        if not np.array_equal(a[:-1],s[1:])or not np.array_equal(ga[:-1],gs[1:]):
            raise ValueError('DECLARED_ONE_FRAME_LOGGED_REFERENCE_RELATION_NOT_VERIFIED')
        if np.max(abs(np.linalg.norm(s[:,3:],axis=1)-1))>1e-6 or np.max(abs(np.linalg.norm(a[:,3:],axis=1)-1))>1e-6:
            raise ValueError('UNIT_SOURCE_QUATERNION_REQUIRED_NO_SILENT_REPAIR')
        rs=Rotation.from_quat(s[:,[4,5,6,3]]).as_matrix();ra=Rotation.from_quat(a[:,[4,5,6,3]]).as_matrix()
        relative=ra[index]@np.swapaxes(rs[:valid,None],-1,-2)
        delta_xyz=a[index,:3]-s[:valid,None,:3]
        rotvec=Rotation.from_matrix(relative.reshape(-1,3,3)).as_rotvec().reshape(valid,HORIZON,3)
        action=np.concatenate([delta_xyz,rotvec,ga[index,None]],axis=-1).astype(np.float32)
        state=np.c_[s[:valid,:3],Rotation.from_matrix(rs[:valid]).as_rotvec(),gs[:valid]].astype(np.float32)
        rebuilt=Rotation.from_rotvec(action[:,:,3:6].reshape(-1,3)).as_matrix().reshape(valid,HORIZON,3,3)@rs[:valid,None]
        discrepancy=Rotation.from_matrix((rebuilt@np.swapaxes(ra[index],-1,-2)).reshape(-1,3,3)).magnitude()
        errors.append(dict(XYZ_native_max_abs=float(np.max(abs(action[:,:,:3]+s[:valid,None,:3]-a[index,:3]))),
            SO3_max_rad=float(discrepancy.max()),native_g_max_abs=float(np.max(abs(action[:,:,6]-ga[index])))))
        state_parts.append(state);action_parts.append(action)
    return np.concatenate(state_parts,axis=-1),np.concatenate(action_parts,axis=-1),dict(original_frames=count,
        valid_H50_windows=valid,first_future_state_offset_frames=1,last_copied_action_excluded=True,
        supervised_action_indices='i..i+49',verified_state_indices='i+1..i+50',roundtrip=errors,
        labels='FUTURE_LOGGED_STATE_REFERENCE',coordinates=COORDINATES,physical_calibration='UNKNOWN')


def load_numeric(path):
    import h5py
    with h5py.File(path,'r')as f:
        if float(f['additional_info/frequency'][()])!=30.:raise ValueError('FIXED_NOMINAL30HZ_REQUIRED')
        poses=[[np.asarray(f[group+'/'+arm+'_ee_poses'][()])for arm in('left','right')]for group in('state','action')]
        grips=[[np.asarray(f[group+'/'+arm+'_ee_joint_states'][()])for arm in('left','right')]for group in('state','action')]
        text=f['instruction'][()];text=text.decode('utf-8')if isinstance(text,bytes)else str(text)
        n=len(poses[0][0])
        if any(len(f['vision/'+cam+'/colors'])!=n for cam in SOURCE_CAMERAS):raise ValueError('CURRENT_SOURCE_RGB_LENGTHS_MATCH_REQUIRED')
    state,action,audit=relative_arrays(*poses,*grips)
    if not text.strip():raise ValueError('ORIGINAL_FULL_TASK_INSTRUCTION_REQUIRED')
    return state,action,text,audit


class Quantiles14:
    def __init__(self,summary):
        self.q01=np.asarray(summary['q01'],dtype=np.float64);self.q99=np.asarray(summary['q99'],dtype=np.float64)
        if self.q01.shape!=(14,)or self.q99.shape!=(14,)or not np.isfinite([self.q01,self.q99]).all()or(self.q99<self.q01).any():
            raise ValueError('TRAIN_ONLY_TYPED_SOURCE14_QUANTILES_REQUIRED')
    def normalize(self,value):return (np.asarray(value)-self.q01)/(self.q99-self.q01+1e-6)*2-1
    def unnormalize(self,value):return (np.asarray(value)+1)*.5*(self.q99-self.q01+1e-6)+self.q01


def native_rgb(encoded):
    from PIL import Image
    stream=Image.open(io.BytesIO(np.asarray(encoded,dtype=np.uint8).tobytes().rstrip(b'\0')))
    standard=stream.info.get('comment')==b'XPL-RGB1'
    image=np.asarray(stream.convert('RGB'))
    # Pinned upstream's JPEG marker is the source of this branch, not colours
    # guessed from a demonstration. Stored encoded bytes remain unchanged.
    return image if standard else image[:,:,::-1].copy()


class SourceDataset:
    def __init__(self,root,manifest_sha256,*,split,resize_with_pad,max_cache_bytes=128*1024**2):
        self.root=Path(root);path=self.root/'MANIFEST.private.json'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest_sha256:raise ValueError('FIXED_PREPARED_SOURCE_MANIFEST_REQUIRED')
        self.manifest=json.loads(path.read_bytes())
        if self.manifest['schema']!=SCHEMA or self.manifest['coordinates']!=COORDINATES or split not in('TRAIN','SOURCE_DEV'):
            raise ValueError('EXPLICIT_TRAIN_OR_DEV_SOURCE_ONLY_READER')
        stat_path=self.root/self.manifest['statistics']['file']
        if hashlib.sha256(stat_path.read_bytes()).hexdigest()!=self.manifest['statistics']['sha256']:raise ValueError('SOURCE_TRAIN_STATISTICS_CHANGED')
        stats=json.loads(stat_path.read_bytes());self.statistics={k:Quantiles14(stats[k])for k in('state','actions')}
        self.episodes={e['id']:e for e in self.manifest['episodes']if e['split']==split and e['valid']}
        self.rows=[(e['id'],i)for e in self.episodes.values()for i in range(e['windows'])]
        self.resize=resize_with_pad;self.cache=OrderedDict();self.cache_bytes=0;self.cache_limit=max_cache_bytes;self.numeric=OrderedDict()

    def __len__(self):return len(self.rows)

    def arrays(self,episode):
        if episode not in self.numeric:
            ep=self.episodes[episode];p=self.root/ep['arrays']['file']
            if hashlib.sha256(p.read_bytes()).hexdigest()!=ep['arrays']['sha256']:raise ValueError('SOURCE_LABEL_ARRAY_BYTES_CHANGED')
            with np.load(p,allow_pickle=False)as z:self.numeric[episode]={k:z[k].copy()for k in('state','actions','dynamic')}
            while len(self.numeric)>2:self.numeric.popitem(last=False)
        self.numeric.move_to_end(episode);return self.numeric[episode]

    def __getitem__(self,index):
        import h5py
        episode,anchor=self.rows[index];ep=self.episodes[episode];arrays=self.arrays(episode);key=(episode,anchor)
        if key not in self.cache:
            images={};padding={}
            with h5py.File(ep['raw_path'],'r')as f:
                for name,cam in zip(CAMERAS,SOURCE_CAMERAS,strict=True):
                    image=native_rgb(f['vision/'+cam+'/colors'][anchor])
                    if image.shape!=(480,640,3):raise ValueError('DECLARED_SOURCE_RGB480_640_REQUIRED')
                    images[name],padding[name]=self.resize(image,224,224,return_mask=True)
            size=sum(x.nbytes for x in[*images.values(),*padding.values()]);self.cache[key]=(images,padding,size);self.cache_bytes+=size
            while self.cache_bytes>self.cache_limit and len(self.cache)>1:
                _,(_,_,amount)=self.cache.popitem(last=False);self.cache_bytes-=amount
        self.cache.move_to_end(key);images,padding,_=self.cache[key]
        return dict(image=images,image_padding_mask=padding,image_mask={k:True for k in CAMERAS},
            state=pad32(self.statistics['state'].normalize(arrays['state'][anchor])).astype(np.float32),
            actions=pad32(self.statistics['actions'].normalize(arrays['actions'][anchor])).astype(np.float32),prompt=ep['instruction'])

    def identity(self,index):
        episode,anchor=self.rows[index]
        return dict(episode=episode,anchor=anchor,input_camera_index=anchor,input_record_state_index=anchor,
            future_record_state_range=[anchor+1,anchor+50],native_frequency_Hz=30,command_dispatch='UNKNOWN')
