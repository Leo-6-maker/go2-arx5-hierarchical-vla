"""New common-scene bank and native hierarchical plans; no old demonstrations."""
import argparse,hashlib,json,os
from collections import Counter,defaultdict
from functools import lru_cache
from pathlib import Path
from contract import ROOT,INSTRUCTIONS,sha,write

WEIGHTS=(25,25,25,20,5)


def immutable(path,value):
    path=Path(path)
    if path.exists():assert json.loads(path.read_text())==value,str(path)
    else:write(path,value)


def quotas(total,weights):
    denominator=sum(weights);out=[total*w//denominator for w in weights]
    remainder=sorted(range(len(weights)),key=lambda i:(-(total*weights[i]%denominator),i))
    for i in remainder[:total-sum(out)]:out[i]+=1
    assert sum(out)==total
    return out


def ordered(values,label):
    return sorted(values,key=lambda v:hashlib.sha256(f'PI0285:{label}:{v}'.encode()).digest())


def window_rows(row):
    import numpy as np
    folder=Path(row['folder'])
    with np.load(folder/'commands.npz') as f:commands=f['action'];phase=f['phase']
    with np.load(folder/'physics.npz') as f:grasp=f['is_cubeA_grasped'][::5]
    assert len(commands)==len(phase)==660 and (phase[:20]==-1).all()
    assert all((phase[20:]==p).any() for p in range(5)),'Incomplete behavior coverage'
    changes=np.zeros(660,dtype=bool)
    changes[1:]|=phase[1:]!=phase[:-1]
    changes[1:]|=grasp[1:660]!=grasp[:659]
    changes[1:]|=abs(np.diff(commands[:,-1]))>.01
    identity={name:sha(folder/name) for name in ('observations.npz','commands.npz','EPISODE_CARD.json','TEACHER_EVENTS.json')}
    return [dict(uid=f"{row['task']}-{row['split']}-{row['layout']}-{a}",task=row['task'],split=row['split'],layout=row['layout'],episode=f"{row['task']}-{row['split']}-{row['layout']}",anchor=a,phase=int(phase[a]),source='expert',instruction=INSTRUCTIONS[row['task']],event_first8=bool(changes[a:a+8].any()),event_tail_only=bool(not changes[a:a+8].any() and changes[a+8:a+40].any()),source_identity=identity) for a in range(20,621)]


def plan(rows,tasks,updates):
    groups=defaultdict(list)
    for r in rows:
        if r['split']=='TRAIN':groups[r['task'],r['layout'],r['phase']].append(r)
    layouts={task:sorted({key[1] for key in groups if key[0]==task}) for task in tasks}
    assert all(layouts.values())
    counts=quotas(updates*8,[1]*len(tasks));streams={}
    for task,count in zip(tasks,counts):
        bucket_counts=quotas(count,WEIGHTS)
        streams[task]=iter(ordered([(p,j) for p,n in enumerate(bucket_counts) for j in range(n)],task))
    visits=Counter();bucket_visits=Counter();pool_visits=Counter();out=[]
    for i in range(updates*8):
        task=tasks[i%len(tasks)];visit=i//len(tasks);seed=layouts[task][visit%len(layouts[task])]
        phase,_=next(streams[task]);key=(task,seed,phase);group=groups[key];assert group,key
        event=[r for r in group if r['event_first8']];prioritize=phase!=4 and bucket_visits[task,phase]%2==0
        pool=event if prioritize and event else group
        pool=sorted(pool,key=lambda r:r['anchor'])
        # ponytail: deterministic eight strata, sufficient for the fixed finite stream.
        poolkey=key+(bool(prioritize and event),);ordinal=pool_visits[poolkey]
        strata=min(8,len(pool));stratum=ordinal%strata
        eligible=pool[stratum::strata];chosen=eligible[(ordinal//strata)%len(eligible)]
        out.append(dict(chosen,index=i,sampling_event_priority=prioritize,event_fallback=bool(prioritize and not event),stratum=stratum))
        visits[key]+=1;bucket_visits[task,phase]+=1;pool_visits[poolkey]+=1
    assert Counter(r['task'] for r in out)==dict(zip(tasks,counts))
    for task,count in zip(tasks,counts):assert [sum(r['task']==task and r['phase']==p for r in out) for p in range(5)]==quotas(count,WEIGHTS)
    return out


def collect(root):
    admitted=[];attempts=[];gates={};panels={}
    for split in ('TRAIN','DEV'):
        panel=ROOT/'source/evidence'/f'{split}.json';value=json.loads(panel.read_text());panels[split]=sha(panel)
        audit=ROOT/'audit'/f'{split}_CPU.json';receipt=json.loads(audit.read_text())
        assert receipt['status'].startswith('PASS_INDEPENDENT_CPU_') and receipt['source_sha256']==sha(panel)
        gates[split]=value['tasks']
        for r in value['rows']:
            attempts.append({k:r[k] for k in ('task','split','layout','status','persistent_success','folder')})
            if r['status']=='VALID_SOURCE_COMPLETE' and r['persistent_success']:admitted.append(r)
    tasks=[task for task in INSTRUCTIONS if all(gates[s].get(task,{}).get('admitted',False) for s in ('TRAIN','DEV'))]
    assert 'T1' in tasks,'T1 teacher/data gate is not met'
    admitted=[r for r in admitted if r['task'] in tasks]
    rows=[];episodes=[]
    import numpy as np
    for r in admitted:
        folder=Path(r['folder']);result=json.loads((folder/'RESULT.json').read_text())
        for name,h in result['files'].items():assert sha(folder/name)==h
        card=json.loads((folder/'EPISODE_CARD.json').read_text());assert card['instruction']==INSTRUCTIONS[r['task']]
        name=f"{r['task']}-{r['split']}-{r['layout']}";members={}
        for archive,namespace,keys in [('observations.npz','policy_observation',('state_v1','base_camera_raw','hand_camera_raw','time_s')),('commands.npz','action_target',('action','phase'))]:
            with np.load(folder/archive) as f:
                for key in keys:
                    path=root/namespace/name/(key+'.npy');path.parent.mkdir(parents=True,exist_ok=True)
                    array=f[key]
                    if not path.exists():np.save(path,array,allow_pickle=False)
                    material=np.load(path,mmap_mode='r');assert np.array_equal(material,array)
                    members[f'{namespace}/{name}/{key}.npy']=dict(sha256=sha(path),shape=list(array.shape),dtype=str(array.dtype),bytes=path.stat().st_size)
        rr=window_rows(r);rows.extend(rr)
        episodes.append(dict(task=r['task'],split=r['split'],layout=r['layout'],episode=name,folder=str(folder),source_result_sha256=sha(folder/'RESULT.json'),members=members,eligible_H40=len(rr)))
    immutable(root/'SOURCE_BANK.json',dict(status='PASS_NEW_TEACHER_BANKS_CPU_READBACK',tasks=tasks,teacher_gates=gates,panels=panels,admitted_episodes=episodes,all_attempts=attempts,feedback_evidence='ORIGINAL_SOURCE_RAW_CONTACTS_PHYSICS_INSTALLATIONS',provenance='ORIGINAL_EPISODE_CARDS_AND_HASHES',new_teacher_calls=0,new_RGB=0))
    return tasks,episodes,rows


def statistics(root,episodes,rows):
    import numpy as np
    from codec import PRISTINE,COSMOS,TAG,modality,native_imports
    native_imports()
    from gr00t.data.state_action.state_action_processor import StateActionProcessor
    from gr00t.data.types import ActionType,ActionFormat
    # The established reducer and native converter are reused unchanged.
    import importlib.util
    path=ROOT.parent/'PI0282/code/data.py'
    spec=importlib.util.spec_from_file_location('pi0282_stats_reducer',path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    converter=StateActionProcessor({TAG:modality()},use_relative_action=True)
    states=[];absolute=[];relative=[]
    for ep in episodes:
        if ep['split']!='TRAIN':continue
        name=ep['episode'];state=np.load(root/'policy_observation'/name/'state_v1.npy',mmap_mode='r');commands=np.load(root/'action_target'/name/'action.npy',mmap_mode='r')
        states.append(state[20:621].copy())
        for a in range(20,621):
            chunk=commands[a:a+40].copy();assert chunk.shape==(40,10)
            absolute.append(chunk);relative.append(converter._convert_to_relative_action(chunk[:,:9],state[a,:9],ActionType.EEF,ActionFormat.XYZ_ROT6D))
    state=np.concatenate(states);action=np.concatenate(absolute);rel=np.concatenate(relative)
    summary=helper.summary
    new={'state':{'eef_9d':summary(state[:,:9]),'gripper_position':summary(state[:,9:])},'action':{'eef_9d':summary(action[:,:9]),'gripper_position':summary(action[:,9:])},'relative_action':{'eef_9d':summary(rel)}}
    stats=json.loads((PRISTINE/'statistics.json').read_text());assert TAG not in stats;stats[TAG]=new
    config=json.loads((PRISTINE/'processor_config.json').read_text());kwargs=config['processor_kwargs'];kwargs['modality_configs'][TAG]=modality()
    kwargs.update(use_percentiles=False,clip_outliers=True,shortest_image_edge=256,crop_fraction=1.,image_crop_size=[256,256],image_target_size=[256,256],random_rotation_angle=0,color_jitter_params={k:0. for k in ('brightness','contrast','saturation','hue')},extra_augmentation_config=None,state_dropout_prob=0.,exclude_state=False,max_state_dim=132,max_action_dim=132,max_action_horizon=40,use_relative_action=True,model_name=str(COSMOS),model_type='qwen')
    for name,value in [('processor_config.json',config),('statistics.json',stats),('embodiment_id.json',json.loads((PRISTINE/'embodiment_id.json').read_text()))]:immutable(root/name,value)
    immutable(root/'STATISTICS_RELEASE.json',dict(status='FROZEN_ONCE_FULL_ADMITTED_TRAIN_NATIVE_MINMAX',tasks=sorted({e['task'] for e in episodes if e['split']=='TRAIN'}),TRAIN_episodes=sum(e['split']=='TRAIN' for e in episodes),state_population=len(state),action_population=len(action),known_task_set_preprocessing=True,DEV_or_confirmation_in_fit=0,old_demonstrations_in_fit=0,files={n:sha(root/n) for n in ('processor_config.json','statistics.json','embodiment_id.json')}))


@lru_cache(maxsize=1)
def codec(root):
    from language import LanguageCodec
    return LanguageCodec(root)


class Dataset:
    def __init__(self,root,plan_path):
        self.root=Path(root);self.rows=[json.loads(x) for x in Path(plan_path).read_text().splitlines()];self.maps={}
    def __len__(self):return len(self.rows)
    def raw(self,row):
        import numpy as np
        name=row['episode']
        if name not in self.maps:
            self.maps[name]={key:np.load(self.root/namespace/name/(key+'.npy'),mmap_mode='r') for namespace,keys in [('policy_observation',('base_camera_raw','hand_camera_raw','state_v1')),('action_target',('action',))] for key in keys}
        m=self.maps[name];a=row['anchor']
        return m['base_camera_raw'][a].copy(),m['hand_camera_raw'][a].copy(),m['state_v1'][a].copy(),m['action'][a:a+40].copy()
    def __getitem__(self,i):
        row=self.rows[i];result=codec(str(self.root)).feature(*self.raw(row),row['instruction'])
        result['_pi0282_visit']=row
        return result


def collate(items,root):
    visits=[v.pop('_pi0282_visit') for v in items]
    result=codec(str(root)).batch(items);result['_pi0282_visits']=visits
    return result


def build(root):
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    root=Path(root);tasks,episodes,rows=collect(root)
    statistics(root,episodes,rows)
    for name,selected in [('PLACE4096',['T1']),('SHARED4096',tasks)]:
        values=plan(rows,selected,4096);path=root/(name+'.jsonl');payload=''.join(json.dumps(r,sort_keys=True)+'\n' for r in values)
        if path.exists():assert path.read_text()==payload
        else:path.write_text(payload)
        dataset=Dataset(root,path);images=set();states=set();pairs=set();seen=set()
        for r in values:
            if r['uid'] in seen:continue
            seen.add(r['uid']);base,wrist,state,_=dataset.raw(r)
            rgb=hashlib.sha256(base.tobytes()+wrist.tobytes()).hexdigest();actual=hashlib.sha256(state.tobytes()).hexdigest()
            images.add(rgb);states.add(actual);pairs.add((rgb,actual))
        immutable(root/(name+'_PLAN_AUDIT.json'),dict(status='FROZEN_PLAN_NOT_TRAINING_ADMISSION',plan_sha256=sha(path),rows=len(values),task_counts=dict(Counter(r['task'] for r in values)),phase_counts=dict(Counter(r['phase'] for r in values)),source_counts=dict(Counter(r['source'] for r in values)),unique_layouts=len({r['layout'] for r in values}),unique_UIDs=len(seen),unique_actual_two_view_RGB=len(images),unique_actual_states=len(states),unique_actual_RGB_state_pairs=len(pairs),event_first8=sum(r['event_first8'] for r in values),tail_only=sum(r['event_tail_only'] for r in values),event_fallback=sum(r['event_fallback'] for r in values),A_T2_gradient_allowed=False,B_launch_requires_CONFIRMED_A=True))
    immutable(root/'WINDOW_INDEX.json',dict(rows=rows,full_H40_real_future=True,neutral_excluded=True,failed_teacher_supervision=0,phase_or_object_truth_in_policy_input=False))


def self_check():
    assert quotas(32768,WEIGHTS)==[8192,8192,8192,6554,1638]
    rows=[dict(task=task,layout=seed,phase=p,split='TRAIN',uid=f'{task}:{seed}:{p}:{a}',anchor=a,event_first8=a==0,event_tail_only=a==1,source='expert') for task in ('T1','T2') for seed in range(2) for p in range(5) for a in range(4)]
    a=plan(rows,['T1'],4096);b=plan(rows,['T1','T2'],4096)
    assert all(r['task']=='T1' for r in a) and Counter(r['task'] for r in b)=={'T1':16384,'T2':16384}
    for values in (a,b):
        for task in {r['task'] for r in values}:
            for p in range(4):
                group=[r for r in values if r['task']==task and r['phase']==p]
                assert sum(r['sampling_event_priority'] for r in group)==quotas(len(group),[1,1])[0]
    assert a==plan(rows,['T1'],4096)
    assert max(Counter(r['layout'] for r in a).values())-min(Counter(r['layout'] for r in a).values())<=1
    print('PASS_LARGEST_REMAINDER_TASK_EPISODE_EVENT_PLAN; model/optimizer/physics=0')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT/'data'));p.add_argument('--self-check',action='store_true');a=p.parse_args()
    self_check() if a.self_check else build(a.root)
