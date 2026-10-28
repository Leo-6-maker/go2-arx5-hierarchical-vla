"""Small stage adapter around the proven native Trainer, not a new training loop."""
import argparse,hashlib,json,sys
from pathlib import Path

def completed_kind(rows,arm,phase,step,limits):
    assert len(rows)<limits['all_updates_cap']
    duplicate=any(r['arm']==arm and r['phase']==phase and r['step']==step for r in rows)
    kind='technical' if duplicate else ('memory' if phase=='memory' else ('formal' if phase=='main' else 'canary'))
    caps=dict(formal=limits['formal_cap'],memory=limits['memory_cap'],technical=limits['technical_cap'],canary=limits['canary_cap'])
    assert sum(r['work_kind']==kind for r in rows)<caps[kind]
    return kind

def run(a):
    root=Path(a.root)
    from authority import allowed
    limits=allowed(root,'update')
    card=json.loads(Path(a.continuation_card).read_text());assert a.arm==card['stage'] and a.arm in ('CTRL2048','LR5X2048')
    prior_code=root.parent/'PI0283/code'
    # These imports intentionally retain their observed __file__; their native loop is reused.
    sys.path.append(str(prior_code))
    import learning
    learning.completed_kind=completed_kind
    import train as native
    assert Path(native.__file__).resolve()==(prior_code/'train.py').resolve()
    from scope import configure as original_configure
    def configure(model,scope):
        groups,proof=original_configure(model,scope)
        model.action_head.state_dropout_prob=0.
        model.config.state_dropout_prob=0.
        model.action_head.config.state_dropout_prob=0.
        proof['PI0284_state_dropout']=0.;proof['PI0284_stage']=a.arm
        return groups,proof
    native.configure=configure
    prior_callback=native.Provenance
    class Provenance(prior_callback):
        def on_save(self,args,state,control,**kwargs):
            result=super().on_save(args,state,control,**kwargs)
            from scope import digest
            assert any(digest(p)!=self.scope['initial_DiT_hashes'][n] for n,p in kwargs['model'].named_parameters() if n.startswith('action_head.model.')),'DIT_DID_NOT_CHANGE'
            return result
    native.Provenance=Provenance
    manifest=json.loads((root/'design/IMPLEMENTATION.json').read_text())
    for name,digest in manifest['native_dependencies'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    from budget import process
    started=process(root,'start')
    try:native.run(a)
    finally:process(root,'end',started)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--base',required=True);p.add_argument('--data-root',required=True);p.add_argument('--arm',choices=('CTRL2048','LR5X2048'),required=True);p.add_argument('--phase',choices=('continuous','split','main','memory'),required=True);p.add_argument('--continuation-card',required=True);p.add_argument('--microbatch',type=int,choices=(4,2,1),default=4);p.add_argument('--resume',type=int,default=0);p.add_argument('--stop',type=int,required=True)
    a=p.parse_args();assert 0<=a.resume<a.stop<=2048;run(a)
