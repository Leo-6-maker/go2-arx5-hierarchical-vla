"""One task bank on the sealed S4 Trainer; only new membership, bounds and budget."""
import argparse,ast,hashlib,importlib.util,json,sys
from functools import lru_cache
from pathlib import Path
ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0276');CONTROL=ROOT.with_name('PI0273')
spec=importlib.util.spec_from_file_location('pi0276_qualified_train',CONTROL/'code/train.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
old,paired,np,torch,read,read_lines=previous.old,previous.paired,previous.np,previous.torch,previous.read,previous.read_lines
from torch.utils.data import Dataset,SequentialSampler
from run import sha
RELEASE='57fd8a9c53ce241c05a8d5b351034bed7ba56b14'
STATS_SHA=sha(ROOT/'contracts/TASK/source/dataset/statistics.json')

def seed_for(unit):return int.from_bytes(hashlib.sha256(f'PI0276:TRAIN:27603:{unit}'.encode()).digest()[:8],'big')%2**31

class Pairs(Dataset):
    def __init__(self,processor,output,arm,resume):
        assert arm=='TASK'
        self.processor,self.output=processor,output
        self.plan=read(ROOT/'design/TRAIN_PLAN.json')['stream'];assert len(self.plan)==16384
        card=read(ROOT/'design/DATASET_CARD.json');allowed={(e['split'],e['seed']) for e in card['episodes'] if e['admitted']}
        self.rows={r['uid']:r for r in card['rows'] if r['eligible'] and (r['split'],r['seed']) in allowed}
        self.proof={r['uid']:r for r in read(ROOT/'design/PROCESSED_INPUTS.json')['rows']}
    def __len__(self):return len(self.plan)
    def get_dataset_statistics(self):return read(ROOT/'contracts/TASK/source/dataset/statistics.json')
    @lru_cache(maxsize=8)
    def episode(self,seed,command_hash,observation_hash):
        folder=ROOT/'source/data'/f'TRAIN-{seed}'
        assert sha(folder/'commands.npz')==command_hash and sha(folder/'observations.npz')==observation_hash
        with np.load(folder/'commands.npz') as z:commands=z['action'].copy()
        with np.load(folder/'observations.npz') as z:obs={k:z[k].copy() for k in ('state_v1','base_camera_96','hand_camera_96')}
        return commands,obs
    def __getitem__(self,index):
        visit=self.plan[index];r=self.rows[visit['uid']];j=r['anchor']
        assert r['split']=='TRAIN' and visit['group']==r['group']
        commands,obs=self.episode(r['seed'],r['command_sha256'],r['observation_sha256'])
        packet=dict(body_rgb=obs['base_camera_96'][j],wrist_rgb=obs['hand_camera_96'][j],state_v1=obs['state_v1'][j],
            language='Stack the red cube on the green cube and release it.')
        value=old.batch(self.processor,packet,commands[j:j+40].astype(np.float32));condition={k:v for k,v in value['inputs'].items() if k not in ('action','action_mask')}
        proof=self.proof[r['uid']];assert old.identity(condition)==proof['condition_sha256']
        assert old.identity({k:value['inputs'][k] for k in ('action','action_mask')})==proof['labels_masks_sha256']
        value['_pi0264_meta']=dict(unit=visit['unit'],micro=visit['micro'],command=r['group'],condition_sha256=proof['condition_sha256'])
        value['_pi0276_visit']=dict(visit,seed=r['seed'],anchor=j,condition_sha256=proof['condition_sha256'],labels_masks_sha256=proof['labels_masks_sha256'])
        return value

def adapted():
    tree=ast.parse((CONTROL/'code/train.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('hooks','run','equal','compare')]
    edits=[]
    class Binding(ast.NodeTransformer):
        def visit_Constant(self,n):
            mapping={196608:155648,10528:8464,5120:8192,2560:4096,27303:27603,'ROT':'TASK',
                'PASS_PI0273_FULL_BASE_ENCODING_AND_BOTH_10240_SCHEDULES':'PASS_PI0276_ACTUAL_TASK_INPUTS_AND16384_SCHEDULE',
                'PASS_PI0273_ACTUAL_FROZEN_HELPERS_AND_HOOKS':'PASS_PI0276_ACTUAL_FROZEN_HELPERS_AND_HOOKS',
                'PASS_PI0273_REAL_EXACT_RESUME_CANARY':'PASS_PI0276_REAL_EXACT_RESUME_CANARY',
                'PI0273_READOUT_CONTRACT.json':'PI0276_READOUT_CONTRACT.json','pi0273_provenance':'pi0276_provenance',
                'pi0273_provenance/CPU_ENCODER_KERNEL.json':'pi0276_provenance/CPU_ENCODER_KERNEL.json',
                'PI0273_SEGMENT_COMPLETE':'PI0276_SEGMENT_COMPLETE'}
            if n.value in mapping:edits.append(str(n.value));return ast.copy_location(ast.Constant(mapping[n.value]),n)
            return n
        def visit_For(self,n):
            if isinstance(n.iter,ast.Tuple) and all(isinstance(x,ast.Constant) for x in n.iter.elts) and ast.literal_eval(n.iter)==('BROAD','TARGET'):
                n.iter=ast.Tuple(elts=[ast.Constant('TASK')],ctx=ast.Load());edits.append('single_task_canary')
            return self.generic_visit(n)
    nodes=[Binding().visit(n) for n in nodes];assert len(nodes)==4 and edits.count('single_task_canary')==1
    scope=vars(previous).copy();scope.update(ROOT=ROOT,CONTROL=CONTROL,RELEASE=RELEASE,STATS_SHA=STATS_SHA,Pairs=Pairs,seed_for=seed_for,sha=sha)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(CONTROL/'code/train.py'),'exec'),scope)
    return scope,edits

runtime,ADAPTATIONS=adapted();base_hooks=runtime['hooks'];compare=runtime['compare']

def hooks(args,output):
    audit=base_hooks(args,output)
    from gr00t.experiment import experiment
    base=experiment.Gr00tTrainer
    class ConsumedTrainer(base):
        def _get_train_sampler(self,*unused):return SequentialSampler(self.train_dataset)
        def training_step(self,model,inputs,*a,**kw):
            visit=inputs.pop('_pi0276_visit');assert visit['micro_index']==2*self.state.global_step+visit['micro']
            with (output/'sample_audit.jsonl').open('a') as f:f.write(json.dumps(visit,sort_keys=True)+'\n')
            if not getattr(self,'gripper_hook_installed',False):
                def observe_gradient(module,args,result):
                    if torch.is_grad_enabled() and isinstance(result,torch.Tensor) and result.requires_grad:
                        def record(gradient):
                            value=float(gradient[...,9].detach().float().norm())
                            assert np.isfinite(value)
                            with (output/'gripper_output_gradients.jsonl').open('a') as f:f.write(json.dumps(dict(unit=self.state.global_step+1,norm=value))+'\n')
                        result.register_hook(record)
                model.action_head.model.register_forward_hook(observe_gradient);self.gripper_hook_installed=True
                name,param=next((n,p) for n,p in model.named_parameters() if p.requires_grad)
                self.changed_weight_name=name;self.start_weight=param.detach().clone()
            loss=super().training_step(model,inputs,*a,**kw)
            if args.phase=='main' and self.state.global_step==127 and visit['micro']==1:
                param=dict(model.named_parameters())[self.changed_weight_name]
                assert not torch.equal(param.detach(),self.start_weight)
                rows=read_lines(output/'gripper_output_gradients.jsonl');assert any(r['norm']>0 for r in rows)
                (output/'HEALTH128.json').write_text(json.dumps(dict(status='PASS_CONTINUING_TASK_LEARNING_HEALTH',
                    permitted_tensor_changed=self.changed_weight_name,nonzero_native_gripper_gradient=True,reset=False))+'\n')
            return loss
    experiment.Gr00tTrainer=ConsumedTrainer
    return audit

runtime['hooks']=hooks;run=runtime['run']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('TASK',),required=True);p.add_argument('--phase',choices=('continuous','split','main'))
    p.add_argument('--limit',type=int);p.add_argument('--resume',type=int,default=0);p.add_argument('--compare',action='store_true');a=p.parse_args()
    if a.compare:compare(a.arm)
    else:assert a.phase and 0<=a.resume<a.limit<=8192;run(a)
