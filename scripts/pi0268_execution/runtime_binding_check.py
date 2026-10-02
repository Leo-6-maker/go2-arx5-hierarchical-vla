"""CPU-only import/signature and hook-install check before PBS allocation."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import argparse
import inspect
import json
from types import SimpleNamespace
from pathlib import Path
import torch
from train import ROOT,paired,seed_for,Pairs,read,sha

def main(arm):
    torch.set_num_threads(4)
    import pi0264_compare_canary as compare
    import pi0264_training_audit as audit_module
    import pi0264_archive_trainer as archive
    import pi0260_r3_inputs as inputs
    import pi0260_r2_codec as codec
    import pi0260_r3_physical as physical
    expected={r['path']:r for r in read(ROOT.with_name('PI0267')/'EXECUTION_FILESET.json')['files']}
    # These read-only closeout helpers were added after the original training seal.
    # Their bytes match scripts/ at the authorizing e5ec6d5 release.
    for name,digest in [('pi0264_training_audit.py','ff0da2dbf4281df2ab4efca1c74f62575bf000f427b497c049b2648d1e733457'),
                        ('pi0264_archive_trainer.py','5743b8951e284d3f4acf8d0cb6c1e1ce6b9d61c0f21c269b6e264599cbb03e57')]:
        expected[str(ROOT.with_name('PI0267')/'code'/name)]=dict(sha256=digest)
    modules=[paired,paired.r3,compare,audit_module,archive,inputs,codec,physical]
    provenance=[]
    for module in modules:
        p=Path(inspect.getfile(module));assert str(p) in expected and sha(p)==expected[str(p)]['sha256'],str(p)
        provenance.append(dict(module=module.__name__,path=str(p),sha256=sha(p)))
    assert Path(paired.__file__).parent==ROOT.with_name('PI0267')/'code'
    assert {'raw_head_cap','raw_update_cap'}<=set(inspect.signature(paired.install).parameters)
    assert {'max_steps','split_phase'}<=set(inspect.signature(compare.compare).parameters)
    assert {'max_steps','campaign'}<=set(inspect.signature(audit_module.graph).parameters)
    paired.ROOT=ROOT;paired.r3.ROOT=ROOT/'contracts'/arm;paired.seed_for=seed_for;paired.Pairs=Pairs
    paired.r3.Visits=lambda processor,*unused:SimpleNamespace(processor=processor)
    def forbidden(*a,**kw):raise RuntimeError('CPU_BINDING_CHECK_NO_MODEL_OR_OPTIMIZER')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden
    output=ROOT/'runtime'/('cpu-hook-install-'+arm);output.mkdir(exist_ok=True)
    audit=paired.install(SimpleNamespace(resume=0,limit=8,arm=arm,phase='continuous'),output,raw_head_cap=90000,raw_update_cap=4384)
    assert audit['forwards']==0 and not torch.cuda.is_available()
    p=ROOT/'evidence'/('RUNTIME_BINDING_'+arm+'.json');assert not p.exists()
    p.write_text(json.dumps(dict(status='PASS_EXACT_FROZEN_HELPER_IMPORTS_AND_CPU_HOOK_INSTALL',arm=arm,modules=provenance,
        model_queries=0,gradient_queries=0,optimizer_updates=0,script_sha256=sha(__file__)),indent=2)+'\n')
    print('PASS_CPU_RUNTIME_BINDING',arm,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=('OLD','ROT'));main(p.parse_args().arm)
