"""One declared objective phase on the existing FOCUS recovery, only after its trigger."""
import argparse,fcntl,inspect,json,sys
from datetime import datetime,timezone
from pathlib import Path

PHASE='FOCUS_GRIP_BOUND_R3'

def rewrite(source):
    edits={
        "result['loss']=result['loss']+.1*penalty":
        "result['loss']=result['loss']+.1*penalty\n        add_box(result,g,m,captured['states'],output,audit['forwards'],b.resume)",
        "data.write(cp/'PI0277_READOUT_CONTRACT.json',data.read(root/'design'/f'{b.arm}_READOUT.json'))":
        "data.write(cp/'PI0277_READOUT_CONTRACT.json',dict(data.read(root/'design'/f'{b.arm}_READOUT.json'),objective_phase=PHASE,objective_phase_seal=phase_seal))\n            data.write(cp/'FOCUS_GRIP_BOUND_R3.json',phase_seal)",
    }
    for old,new in edits.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    compile(source,__file__,'exec')
    return source

def add_box(result,g,mask,states,output,forward,resume):
    import torch
    from bound_loss import box_loss
    from train import append
    value,fraction=box_loss(g,mask)
    result['loss']=result['loss']+.1*value
    row=dict(objective_phase=PHASE,forward=forward,box=float(value.detach()),weighted_box=float((.1*value).detach()),
        active_violations=int((((fraction<0)|(fraction>1)) & mask[...,9].bool()).sum()),total=float(result['loss'].detach()),additional_generation=False)
    append(output/'box_objective.jsonl',row)
    if forward in (2*resume+1,2*resume+2):
        def observe(gradient,kind,index=None):
            assert torch.isfinite(gradient).all()
            append(output/'BOX_FIRST_UPDATE_GRADIENT.jsonl',dict(**row,kind=kind,state_index=index,norm=float(gradient.float().norm()),finite=True,
                gradient_scope='isolated_box_branch' if kind=='fraction' else 'total_objective',zero_box_gradient_is_valid=True))
        fraction.register_hook(lambda gradient:observe(gradient,'fraction'))
        states=[state for state in states if state.requires_grad]
        assert len(states)==4 and states[-1] is g
        for index,state in enumerate(states,1):
            state.register_hook(lambda gradient,index=index:observe(gradient,'Euler',index))

def main(root,resume,seal):
    assert datetime.now(timezone.utc).isoformat()<'2026-10-06T23:00:00+00:00','0700_NO_NEW_TRAINING'
    sys.path.insert(0,str(root/'code'))
    import cpu_inputs as data,train,focus_train
    phase_seal=data.read(seal)
    assert phase_seal['phase']==PHASE and phase_seal['launch_allowed'] and phase_seal['resume_update']==resume<4096
    assert len(phase_seal['trigger']['suffix_only_terminal_failures'])>=2 and not phase_seal['qualified_candidate_exists']
    for row in phase_seal['files']:assert data.sha(row['path'])==row['sha256']
    checkpoint=root/'FOCUS/main/training'/f'checkpoint-{resume}'
    assert not (root/'runtime/STOP_TRAINING.json').exists()
    assert data.read(checkpoint/'trainer_state.json')['global_step']==resume
    for row in phase_seal['resume_state']:assert data.sha(checkpoint/row['path'])==row['sha256']
    original=train.run
    def continued(binding):
        assert binding.arm=='FOCUS' and binding.phase=='main' and binding.resume==resume and binding.limit==4096
        scope=dict(vars(train),add_box=add_box,PHASE=PHASE,phase_seal=dict(path=str(seal),sha256=data.sha(seal),phase=PHASE,
            governing_release=phase_seal['release'],resume_update=resume,old_source_sha256=phase_seal['old_train_sha256']))
        exec(compile(rewrite(inspect.getsource(original)),__file__,'exec'),scope)
        return scope['run'](binding)
    train.run=continued
    focus_train.main(root,'main','FAST_CACHE2',4096,resume)

def check(source):
    import ast
    tree=ast.parse(rewrite(source))
    assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='add_box' for n in ast.walk(tree))==1
    assert 'config.training.resume_from_checkpoint=bool(b.resume)' in source
    assert 'config.training.max_steps=4096' in source
    return dict(status='PASS_SINGLE_BOX_INSERTION_AND_UNCHANGED_RESUME_SCHEDULE',launch_allowed=False,objective_applied=False)

def check_graph(root):
    import hashlib,tempfile,types,torch
    torch.set_num_threads(4)
    assert not torch.cuda.is_available()
    # CPU-only logger stub; production imports the unchanged Hopper append helper.
    logger=types.ModuleType('train')
    def append(path,row):
        with Path(path).open('a') as f:f.write(json.dumps(row)+'\n')
    logger.append=append;previous=sys.modules.get('train');sys.modules['train']=logger
    try:
        a=torch.zeros(1,40,132,requires_grad=True);b=a+0.;c=b+0.;g=(c+0.).clone();g[0,0,9]=g[0,0,9]+1.4
        mask=torch.zeros_like(g);mask[...,:10]=1
        with tempfile.TemporaryDirectory(dir=root/'evidence') as tmp:
            result={'loss':g.sum()*0+9}
            add_box(result,g,mask,[torch.zeros_like(a),a,b,c,g],Path(tmp),1,0)
            assert abs(result['loss'].item()-9.04)<1e-5
            result['loss'].backward()
            rows=[json.loads(s) for s in (Path(tmp)/'BOX_FIRST_UPDATE_GRADIENT.jsonl').read_text().splitlines()]
            assert len(rows)==5 and all(x['finite'] for x in rows)
            assert [x['state_index'] for x in rows if x['kind']=='Euler']==[4,3,2,1]
            assert abs(a.grad[0,0,9].item()-.2)<1e-5
    finally:
        if previous is None:del sys.modules['train']
        else:sys.modules['train']=previous
    return dict(status='PASS_CPU_SINGLE_ADDITION_AND_FOUR_GRAD_STATES_WITH_NONGRAD_INITIAL_NOISE',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),logger='CPU_STUB_ONLY',formal_updates=0,new_model_queries=0,launch_allowed=False,objective_applied=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--resume',type=int);p.add_argument('--seal',type=Path);p.add_argument('--check-source',type=Path);p.add_argument('--check-graph',action='store_true');a=p.parse_args()
    if a.check_source:print(json.dumps(check(a.check_source.read_text())))
    elif a.check_graph:print(json.dumps(check_graph(a.root)))
    else:
        assert a.root and a.resume and a.seal
        with (a.root.parent/'runtime/owner.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            main(a.root,a.resume,a.seal)
