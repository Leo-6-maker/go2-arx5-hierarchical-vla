"""Lazy content-addressed native CPU tensors and ordered two-worker parity proof."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from codec import ROOT, OFFICIAL, load_codec, sha
from data import write


def tensor_hash(inputs):
    h=hashlib.sha256()
    for k,v in sorted(inputs.items()):
        array=v.detach().cpu().contiguous().numpy()
        h.update(k.encode());h.update(str(array.dtype).encode());h.update(str(array.shape).encode());h.update(array.tobytes())
    return h.hexdigest()


def collate(items):
    """Batch fixed256/current-frame native tensors; parity checked against native collator."""
    import torch
    keys=set(items[0]);assert all(set(x)==keys for x in items)
    # Every image has identical dimensions and every instruction/template is fixed.
    # Therefore native token lengths agree and no substitute tokenizer/padding is needed.
    assert len({x['input_ids'].shape[-1] for x in items})==1
    return {'inputs':{k:torch.cat([x[k] for x in items],dim=0) for k in keys}}


class FrozenDataset:
    def __init__(self, root=ROOT, plan_path=None, cache=True):
        self.root=Path(root)
        self.rows=[json.loads(line) for line in Path(plan_path or self.root/'PLAN16384.jsonl').read_text().splitlines()]
        self.codec=None
        self.cache=cache
        self.maps={}
        official=Path(os.environ.get('PI0282_OFFICIAL_DIR',str(OFFICIAL)))
        self.binding={'contract':sha(self.root/'DATA_CONTRACT.json'),
                      'codec':sha(Path(__file__).with_name('codec.py')),'cpu_inputs':sha(__file__),
                      'official':{str(p.relative_to(official)):sha(p) for p in [
                          official/'gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py',
                          official/'gr00t/model/gr00t_n1d7/image_augmentations.py',
                          official/'gr00t/data/state_action/state_action_processor.py',
                          official/'gr00t/data/state_action/pose.py',official/'gr00t/data/utils.py']}}
        self.namespace=hashlib.sha256(json.dumps(self.binding,sort_keys=True).encode()).hexdigest()

    def __len__(self):return len(self.rows)

    def raw(self,row):
        import numpy as np
        name=f"{row['split']}-{row['seed']}"
        if name not in self.maps:
            ep=self.root/'episodes'/name
            self.maps[name]={k:np.load(ep/(k+'.npy'),mmap_mode='r') for k in ('state_v1','action','base_camera_raw','hand_camera_raw')}
        a=row['anchor'];m=self.maps[name]
        return m['base_camera_raw'][a].copy(),m['hand_camera_raw'][a].copy(),m['state_v1'][a].copy(),m['action'][a:a+40].copy()

    def __getitem__(self,index):
        import torch
        row=self.rows[index]
        key=hashlib.sha256(json.dumps({'namespace':self.namespace,'uid':row['uid'],
              'command_sha256':row['command_sha256'],'observation_sha256':row['observation_sha256']},sort_keys=True).encode()).hexdigest()
        path=self.root/'cache'/self.namespace/(key+'.pt')
        if self.cache and path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=True)
            assert saved['key']==key and saved['uid']==row['uid']
            assert tensor_hash(saved['inputs'])==saved['tensor_sha256']
            return saved['inputs']
        if self.codec is None:self.codec=load_codec(self.root)
        inputs=self.codec.encode_item(*self.raw(row))
        assert inputs['state'].shape==(1,1,132) and inputs['action'].shape==(1,40,132)
        mask=inputs['action_mask'];assert bool((mask[:,:,:10]==1).all()) and bool((mask[:,:,10:]==0).all())
        assert int(inputs['embodiment_id'][0])==10
        if self.cache:
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_suffix(f'.{os.getpid()}.tmp')
            torch.save({'key':key,'uid':row['uid'],'tensor_sha256':tensor_hash(inputs),'inputs':inputs},temp)
            temp.replace(path)
        return inputs


def materialize(root):
    import numpy as np
    contract=json.loads((root/'DATA_CONTRACT.json').read_text())
    receipts=[]
    for episode in contract['admitted_episodes']:
        name=f"{episode['split']}-{episode['seed']}";target=root/'episodes'/name;target.mkdir(parents=True,exist_ok=True)
        original=Path(episode['path']);files={}
        for archive,keys in [('observations.npz',('state_v1','base_camera_raw','hand_camera_raw')),('commands.npz',('action',))]:
            with np.load(original/archive) as source:
                for key in keys:
                    path=target/(key+'.npy');array=source[key]
                    if not path.exists():np.save(path,array,allow_pickle=False)
                    material=np.load(path,mmap_mode='r');assert np.array_equal(material,array)
                    files[key]={'sha256':sha(path),'shape':list(array.shape),'dtype':str(array.dtype),'bytes':path.stat().st_size,
                                'array_sha256':hashlib.sha256(array.tobytes()).hexdigest()}
        receipts.append({'episode':name,'source_command_sha256':episode['command_sha256'],
                         'source_observation_sha256':episode['observation_sha256'],'files':files})
        print('MATERIALIZED',name,flush=True)
    write(root/'RAW_MATERIALIZATION.json',{'episodes':receipts,'new_RGB':0,'original_member_byte_parity':True,
                                         'total_bytes':sum(f['bytes'] for e in receipts for f in e['files'].values())})


def check(root):
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    torch.set_num_threads(4)
    codec=load_codec(root)
    dataset=FrozenDataset(root,root/'DIAG48.jsonl')
    plain=[];features=[];checks=[]
    for i,row in enumerate(dataset.rows):
        raw=dataset.raw(row);base,wrist,state,commands=raw
        inputs=codec.encode_item(*raw)
        inference=codec.observation(base,wrist,state)
        shared=set(inputs)-{'action','action_mask'}
        assert shared==set(inference)-{'action_mask'}
        for k in shared:
            # Official inference embodiment dtype int32 vs native training int64.
            assert torch.equal(inputs[k],inference[k]),k
        cached=dataset[i];assert tensor_hash(cached)==tensor_hash(inputs)
        reconstructed=codec.decode_actions(inputs['action'][0].numpy(),state)
        errors=physical_errors(commands,reconstructed)
        checks.append({'uid':row['uid'],'split':row['split'],'tensor_sha256':tensor_hash(inputs),**errors,
                       'command_achieved_max_abs_difference':float(np.abs(commands[0]-state).max())})
        plain.append(inputs);features.append(codec.feature(*raw))
    direct=codec.proc.collator(features[:4])['inputs'];assembled=collate(plain[:4])['inputs']
    assert tensor_hash(direct)==tensor_hash(assembled)
    loader=torch.utils.data.DataLoader(dataset,batch_size=4,num_workers=2,collate_fn=collate,shuffle=False,pin_memory=True)
    worker_hashes=[]
    for offset,batch in enumerate(loader):
        value=batch['inputs'];expected=collate(plain[offset*4:offset*4+4])['inputs']
        assert tensor_hash(value)==tensor_hash(expected)
        worker_hashes.append(tensor_hash(value))
    # Moving anchor/nonidentity rotation exercises actual native SE(3), without statistics clipping.
    state=dataset.raw(dataset.rows[0])[2];commands=dataset.raw(dataset.rows[0])[3]
    from gr00t.data.types import ActionType,ActionFormat
    p=codec.proc.state_action_processor
    relative=p._convert_to_relative_action(commands[:,:9],state[:9],ActionType.EEF,ActionFormat.XYZ_ROT6D)
    restored=p._convert_to_absolute_action(relative,state[:9],ActionType.EEF,ActionFormat.XYZ_ROT6D)
    moved=state.copy();moved[:3]+=[.01,-.02,.03]
    rotation=Rotation.from_euler('xyz',[.2,-.1,.3]).as_matrix()
    moved[3:9]=rotation[:2].reshape(6)
    relative2=p._convert_to_relative_action(commands[:,:9],moved[:9],ActionType.EEF,ActionFormat.XYZ_ROT6D)
    restored2=p._convert_to_absolute_action(relative2,moved[:9],ActionType.EEF,ActionFormat.XYZ_ROT6D)
    assert not np.array_equal(relative,relative2)
    assert np.max(np.abs(restored-commands[:,:9]))<1e-5 and np.max(np.abs(restored2-commands[:,:9]))<1e-5
    write(root/'CPU_INPUT_AUDIT.json',{'status':'PASS_NATIVE_INPUT_MASK_LEAK_AND_TWO_WORKER_BYTE_PARITY',
          'allowlist':['actual_external_RGB','actual_wrist_RGB','actual_TCP_XYZ_ROT6D','measured_gripper_open_fraction','fixed_language'],
          'state_shape':[1,132],'action_mask_shape':[40,132],'active_dimensions':10,
          'diagnostic_rows':checks,'native_collator_vs_cached_batch_sha256':tensor_hash(direct),
          'worker_batch_sha256':worker_hashes,'ordered_workers':2,'pin_memory_requested':True,
          'pin_memory_effective_cpu_only':False,'inference_input_labels_removed_parity':True,
          'moving_anchor_nonidentity_native_conversion':True,'cache_namespace':dataset.namespace,'cache_binding':dataset.binding,
          'model_forward':0,'physics':0,'training':0})


def physical_errors(command,reconstructed):
    import numpy as np
    from scipy.spatial.transform import Rotation
    def matrix(r):
        a=r[:,:3];a=a/np.linalg.norm(a,axis=-1,keepdims=True)
        b=r[:,3:];b=b-(a*b).sum(-1,keepdims=True)*a;b=b/np.linalg.norm(b,axis=-1,keepdims=True)
        return np.stack([a,b,np.cross(a,b)],axis=-2)
    first=matrix(command[:,3:9].astype(np.float64));second=matrix(reconstructed[:,3:9].astype(np.float64))
    diff=second@np.transpose(first,(0,2,1))
    def validity(r):
        a=r[:,:3];b=r[:,3:]
        return {'minimum_first_row_norm':float(np.linalg.norm(a,axis=-1).min()),
                'minimum_cross_norm':float(np.linalg.norm(np.cross(a,b),axis=-1).min()),
                'degenerate_count':int(((np.linalg.norm(a,axis=-1)<1e-8)|(np.linalg.norm(np.cross(a,b),axis=-1)<1e-8)).sum())}
    return {'XYZ_max_m':float(np.linalg.norm(command[:,:3]-reconstructed[:,:3],axis=-1).max()),
            'SO3_max_rad':float(Rotation.from_matrix(diff).magnitude().max()),
            'gripper_max_fraction':float(np.abs(command[:,9]-reconstructed[:,9]).max()),
            'nonfinite':bool(not np.isfinite(reconstructed).all()),
            'command_rot6d_validity':validity(command[:,3:9]),'decoded_rot6d_validity':validity(reconstructed[:,3:9])}


def full_roundtrip(root):
    import numpy as np
    from codec import fields,TAG
    codec=load_codec(root);p=codec.proc.state_action_processor
    from gr00t.data.utils import normalize_values_minmax
    from gr00t.data.types import ActionType,ActionFormat
    contract=json.loads((root/'DATA_CONTRACT.json').read_text());maxima={};rows=0;clamps=0;values=0;invalid=0
    for episode in contract['admitted_episodes']:
        if episode['split']!='TRAIN':continue
        d=FrozenDataset(root);row={'split':'TRAIN','seed':episode['seed'],'anchor':0}
        d.raw(row);arrays=d.maps[f"TRAIN-{episode['seed']}"];states=arrays['state_v1'];commands=arrays['action']
        episode_max={'XYZ_max_m':0.,'SO3_max_rad':0.,'gripper_max_fraction':0.}
        for anchor in range(601):
            command=commands[anchor:anchor+40].copy();state=states[anchor].copy()
            normalized=codec.encode_actions(command,state);reconstructed=codec.decode_actions(normalized,state)
            error=physical_errors(command,reconstructed)
            for k in episode_max:episode_max[k]=max(episode_max[k],error[k])
            invalid+=error['nonfinite'];rows+=1
            rel=p._convert_to_relative_action(command[:,:9],state[:9],ActionType.EEF,ActionFormat.XYZ_ROT6D)
            norm=normalize_values_minmax(rel,p.norm_params[TAG]['action']['eef_9d'])
            clamps+=int((np.abs(norm)>1).sum());values+=norm.size
        maxima[str(episode['seed'])]=episode_max
        print('ROUNDTRIP',episode['seed'],episode_max,flush=True)
    overall={k:max(e[k] for e in maxima.values()) for k in ('XYZ_max_m','SO3_max_rad','gripper_max_fraction')}
    passed=overall['XYZ_max_m']<=1e-5 and overall['SO3_max_rad']<=1e-4 and overall['gripper_max_fraction']<=1e-6 and invalid==0
    write(root/'FULL_TRAIN_ROUNDTRIP.json',{'status':'PASS' if passed else 'FAIL_REQUIRES_CONCRETE_CODEC_DECISION',
          'TRAIN_windows':rows,'maxima':overall,'per_episode':maxima,'preclip_out_of_range_values':clamps,
          'relative_values':values,'preclip_fraction':clamps/values,'nonfinite_windows':invalid,
          'native_inverse_clipping_retained':True,'no_constant_or_rotation_repair':True})
    assert passed,overall


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['materialize','check','roundtrip'])
    parser.add_argument('--root',default=str(ROOT));args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    begin=time.monotonic();root=Path(args.root)
    {'materialize':materialize,'check':check,'roundtrip':full_roundtrip}[args.operation](root)
    write(root/('USAGE_'+args.operation+'.json'),{'elapsed_seconds':time.monotonic()-begin,
          'rusage_self':list(resource.getrusage(resource.RUSAGE_SELF)),'rusage_children':list(resource.getrusage(resource.RUSAGE_CHILDREN)),
          'CUDA_VISIBLE_DEVICES':'','model_forward':0,'physics':0,'training':0})


if __name__=='__main__':main()
