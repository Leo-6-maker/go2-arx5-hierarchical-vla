"""Version the proven runtime with narrow PI0283 carry/cap/route changes."""
from pathlib import Path


def build():
    here=Path(__file__).resolve().parent;old=here.parent/'pi0282_execution';out=here/'runtime';out.mkdir(exist_ok=True)
    names=('codec','cpu_inputs','data','scope','native_model','learning','continuation','train','training_launch','canary','model_view','source','source_launch','inference','evaluate')
    for name in names:(out/(name+'.py')).write_bytes((old/(name+'.py')).read_bytes())
    for name in ('authority','phase','plan','audit','transport','launch','cpu_repair_check'):
        source=here/(name+'.py')
        if source.exists():(out/source.name).write_bytes(source.read_bytes())
    def replace(name,pairs):
        path=out/(name+'.py');s=path.read_text()
        for before,after in pairs:
            assert before in s,(name,before);s=s.replace(before,after)
        path.write_text(s,newline='\n')
    replace('learning',[("memory=32,technical=128,canary=64","memory=16,technical=32,canary=16")])
    replace('canary',[("choices=('ACTION2048','INTERFACE2048')","choices=('MORE','EVENT')")])
    replace('source',[("ROOT=BASE/'PI0282'","ROOT=BASE/'PI0283'"),("starts=100,control=72000,physics=360000,rgb=72000,teacher=8192","starts=40,control=30000,physics=160000,rgb=30000,teacher=0"),("<=8","<=4")])
    replace('inference',[("import torch\n","os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')\nassert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'\nimport torch\n"),("model,codec,binding=load(root,args.point)","from authority import model_process\n    process=model_process(root,'start')\n    model,codec,binding=load(root,args.point)"),("for name,p in model.named_parameters():assert array_hash(p.detach().cpu().contiguous().view(torch.uint8).numpy())==expected[name],name\n", "for name,p in model.named_parameters():assert array_hash(p.detach().cpu().contiguous().view(torch.uint8).numpy())==expected[name],name\n    model_process(root,'end',process)\n")])
    replace('inference',[("def query(root,model,codec,request):","def inference_paths(root,source,target):\n    allowed=[root.resolve()]\n    tier=root/'evidence/OUTPUT_TIERING.json'\n    if tier.exists():\n        assert json.loads(tier.read_text())['status']=='PASS_EXACT_BYTES_DURABLE_NVME_TIERING_AT_DEAD_PROCESS_BOUNDARY'\n        assert root==Path('/home/liuyu-nfs/Projects/go2-arx5/PI0283')\n        allowed.append(Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0283/source').resolve())\n    for path in (source,target):\n        assert path.is_relative_to(root) and any(path.resolve().is_relative_to(p) for p in allowed),'UNAUTHORIZED_INFERENCE_PATH'\n\n\ndef query(root,model,codec,request):"),("assert source.resolve().is_relative_to(root.resolve()) and target.resolve().is_relative_to(root.resolve())","inference_paths(root,source,target)")])
    replace('train',[
        ("get_cosine_schedule_with_warmup(optimizer or self.optimizer,self.warmup,2048)","get_cosine_schedule_with_warmup(optimizer or self.optimizer,self.warmup,self.phase_updates)"),
        ("with native_rng(step,micro) as observed:","with native_rng(self.lineage_offset+step,micro) as observed:"),
        ("torch.cuda.synchronize()\n        assert observed", "torch.cuda.synchronize()\n        assert math.isclose(float(loss.detach()),self.actual_native_loss/self.accumulation,rel_tol=1e-6,abs_tol=1e-8),'NATIVE_TRAINER_ACCUMULATION_REDUCTION_CONTRADICTION'\n        assert observed"),
        ("(step+1)%16==0","(step+1)%128==0"),
        ("if self.phase=='main' and state.global_step in (128,512):control.should_save=True", "if self.phase=='main' and state.global_step in (1024,self.trainer.phase_updates):control.should_save=True"),
        ("if time.time()+30>=load(self.root)['training_cutoff_epoch']:","if time.time()+60>=self.trainer.lease_deadline:"),
        ("if self.arm.startswith('ACTION'):","if self.arm in ('MORE','EVENT'):"),
        ("state.global_step in (512,1024,2048)","state.global_step in (1024,self.trainer.phase_updates)"),
        ("snapshot=Path(args.output_dir).parent/'snapshots'/f'step-{state.global_step}'", "snapshot=self.root/'delivery'/(self.arm+str(state.global_step))"),
        ("self.arm.replace('2048','')+str(state.global_step)","self.arm+str(state.global_step)"),
        ("assert len(self.base)==len(self.plan)==16384", "assert len(self.base)==len(self.plan) and len(self.plan) in (8192,16384)"),
        ("result=dict(self.base[i]);result['_pi0282_visit']=self.plan[i];return result", "result=dict(self.cached_item(i));result['_pi0282_visit']=self.plan[i];return result"),
        ("def run(a):",'''    def cached_item(self,i):
            from cpu_inputs import tensor_hash
            row=self.plan[i]
            key=hashlib.sha256(json.dumps({'namespace':self.base.namespace,'uid':row['uid'],'command_sha256':row['command_sha256'],'observation_sha256':row['observation_sha256']},sort_keys=True).encode()).hexdigest()
            old=Path(self.base.root).parent.parent/'PI0282/data/cache'/self.base.namespace/(key+'.pt')
            if old.exists():
                saved=torch.load(old,map_location='cpu',weights_only=True)
                assert saved['key']==key and saved['uid']==row['uid'] and tensor_hash(saved['inputs'])==saved['tensor_sha256']
                return saved['inputs']
            # ponytail: reuse the deterministic encoder; bound only the new private cache.
            if i%64==0:
                import fcntl
                folder=self.base.root/'cache'/self.base.namespace;folder.mkdir(parents=True,exist_ok=True)
                with (folder/'eviction.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    files=sorted(folder.glob('*.pt'),key=lambda p:p.stat().st_mtime);size=sum(p.stat().st_size for p in files)
                    for p in files:
                        if size<=3*1024**3:break
                        if time.time()-p.stat().st_mtime>60:size-=p.stat().st_size;p.unlink()
            try:return self.base[i]
            except FileNotFoundError:return self.base[i]


def run(a):'''),
        ("continuation=None\n    if a.continuation_card:","continuation=None\n    assert a.continuation_card\n    if a.continuation_card:"),
        ("from continuation import validate","from phase import validate"),
        ("assert a.arm=='ACTION_F2048' and a.phase=='main' and a.resume==0", "assert a.arm==continuation['stage']"),
        ("('ACTION2048' if continuation else a.arm)","a.arm"),
        ("configure(model,'ACTION2048' if continuation else a.arm)","configure(model,'ACTION2048')"),
        ("from continuation import restore_model","from phase import restore_model"),
        ("for group in groups:group['lr']*=.5", "for i,group in enumerate(groups):group['lr']=continuation['peak_LRs'][i]"),
        ("phase_scheduler_warmup=32","phase_scheduler_warmup=continuation['warmup']"),
        ("max_steps=2048,per_device", "max_steps=continuation['updates'],per_device"),
        ("learning_rate=5e-5 if continuation else 1e-4", "learning_rate=continuation['peak_LRs'][0]"),
        ("warmup_steps=32 if continuation else 103", "warmup_steps=continuation['warmup']"),
        ("trainer.warmup=32 if continuation else 103", "trainer.warmup=continuation['warmup'];trainer.phase_updates=continuation['updates'];trainer.lineage_offset=continuation['parent_global_update'];trainer.lease_deadline=json.loads((root/'runtime/TRAINING_ALLOCATION.json').read_text())['started_epoch']+allocation['training_GPU_reservation_s']-120"),
        ("from continuation import restore_optimizer","from phase import restore_optimizer"),
        ("trainer.add_callback(Provenance", "from phase import install_audit\n    audit_hooks=install_audit(trainer,model)\n    trainer.add_callback(Provenance"),
        ("receipt=dict(consumption=", "receipt=dict(native_loss_modes_LR=trainer.mode_journal.close(),consumption="),
        ("choices=('ACTION2048','INTERFACE2048','ACTION_F2048')", "choices=('MORE','EVENT')"),
    ])
    # Keep source/codec/prediction semantic release binding; the new authority is
    # independently enforced by authority.py and the sealed phase card.
    return out


if __name__=='__main__':print(build())
