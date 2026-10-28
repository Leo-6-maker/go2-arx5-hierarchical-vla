"""Explicit GEN hooks for the existing numerical Trainer and terminal audit."""
from pathlib import Path
from . import amd_bank as bank,amd_dataset,amd_learning as learning,amd_pilot_scope as scope
from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1.mix1024 import step_boundary

VISITS=32768
STATS=scope.a.STATS
validate=learning.validate
trainer_arguments=learning.trainer_arguments
make_trainer_class=learning.make_trainer_class

def dataset(root,arm,evidence):
    scope.need(arm==bank.IDENTITY,'TRAIN_READER_EXPLICIT_GEN_NOT_OLD_ARM')
    root=Path(root);b=read(root/'NEURAL_BINDINGS.private.json');i=read(root/'plans/PV1_TRAIN_INPUTS.private.json')
    gen=read(root/'plans/GEN1_TRAIN_INPUTS.private.json')
    value=amd_dataset.Dataset(b['dataset'],b['source'],b['cosmos'],for_optimization=True,
        raw_bindings=read(b['raw_bindings']),trace_root=Path(evidence)/'worker_trace',
        correction_root=i['C1_dataset'],correction_bindings=read(i['C1_raw_bindings']),
        new_root=gen['dataset'],new_raw_bindings=read(gen['raw_bindings']),
        packet_bytes=(root/'plans/GEN1_V5_PREVIEW_V2.private.json').read_bytes())
    scope.need(value.new_reader.manifest_sha==gen['bank_manifest_sha256'],'TRAIN_SAME_REGISTERED_GEN_BANK')
    value.initialize();return value

def checked_resume(root,arm,checkpoint,recipe,data):
    root=Path(root);cp=Path(checkpoint)
    scope.need(arm==bank.IDENTITY and not cp.is_symlink()
        and cp.parent.resolve()==(root/'training'/arm).resolve(),'TRAIN_OWN_EXACT_COMPLETE_RESUME')
    complete=read(cp/'CHECKPOINT_COMPLETE.json');hashes={}
    for n,h in complete['files'].items():
        p=cp/n
        scope.need(not p.is_symlink()and p.resolve().is_relative_to(cp.resolve())and sha(p)==h,
            'TRAIN_COMPLETE_RESUME_MEMBER:'+n);hashes[n]=h
    return learning.resume_state(read(cp/'SAMPLER_STATE.json'),complete,read(cp/'trainer_state.json'),
        recipe,data.make_sampler(data,stop=VISITS),hashes)

def begin_update(v,*,arm,local_step,request_id,recipe):
    scope.need(arm==bank.IDENTITY,'TRAIN_BEGIN_ONLY_GEN')
    return learning.begin_update(v,local_step=local_step,request_id=request_id,recipe_value=recipe)

def commit_update(v,*,arm,local_step_after,request_id):
    scope.need(arm==bank.IDENTITY,'TRAIN_COMMIT_ONLY_GEN')
    return learning.commit_update(v,local_step_after=local_step_after,request_id=request_id)

def run(*args,**kwargs):
    import sys
    from scripts.pi0294.pv1.mix_training import run as original
    return original(*args,**kwargs,_backend=sys.modules[__name__])

def audit(root,request):
    import sys
    from scripts.pi0294.pv1.mix_terminal import audit as original
    return original(root,request,_backend=sys.modules[__name__])
