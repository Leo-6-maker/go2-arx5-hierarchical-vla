"""Actual frozen helper imports and hook install; CPU with zero model calls."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from train import ROOT, CONTROL, paired, read, sha, seed_for, Pairs


def main():
    assert read(ROOT/'evidence/HOPPER_CPU_ADMISSION.json')['status'] == 'PASS_PI0270_ACTUAL_FROZEN_MAPPING_AND_CONTROL_INPUTS'
    torch.set_num_threads(4)
    import pi0264_compare_canary as compare
    import pi0264_training_audit as audit_module
    import pi0264_archive_trainer as archive
    import pi0260_r3_inputs as inputs
    import pi0260_r2_codec as codec
    import pi0260_r3_physical as physical
    import temporal
    previous = read(CONTROL/'evidence/RUNTIME_BINDING_PREFIX.json')
    assert previous['status'] == 'PASS_EXACT_FROZEN_HELPER_IMPORTS_AND_CPU_HOOK_INSTALL'
    expected = {r['module']: r for r in previous['modules']}
    modules = [paired, paired.r3, compare, audit_module, archive, inputs, codec, physical]
    provenance = []
    for module in modules:
        p = Path(inspect.getfile(module)); old = expected[module.__name__]
        assert str(p) == old['path'] and sha(p) == old['sha256']
        provenance.append(dict(module=module.__name__,path=str(p),sha256=sha(p)))
    p = Path(inspect.getfile(temporal)); assert p == CONTROL/'code/temporal.py'
    assert sha(p) == '011ea6bae8ab0d18638684886b9782b366f68e9c4d8a64464ccf3d89a661ad2d'
    provenance.append(dict(module='temporal',path=str(p),sha256=sha(p)))
    assert {'raw_head_cap','raw_update_cap'} <= set(inspect.signature(paired.install).parameters)
    assert {'max_steps','split_phase'} <= set(inspect.signature(compare.compare).parameters)
    paired.ROOT = ROOT; paired.r3.ROOT = ROOT/'contracts/ROT'; paired.seed_for = seed_for; paired.Pairs = Pairs
    paired.r3.Visits = lambda processor, *unused: SimpleNamespace(processor=processor)
    def forbidden(*a, **kw): raise RuntimeError('PI0270_CPU_BINDING_NO_MODEL_OR_OPTIMIZER')
    torch.nn.Module.__call__ = forbidden; torch.optim.Optimizer.__init__ = forbidden
    output = ROOT/'runtime/cpu-hook-install-AGG'; output.mkdir(parents=True,exist_ok=True)
    audit = paired.install(SimpleNamespace(resume=0,limit=8,arm='AGG',phase='continuous'), output,
                           raw_head_cap=80000,raw_update_cap=4368)
    temporal.install('PREFIX', output)
    assert audit['forwards'] == 0 and not torch.cuda.is_available()
    value = dict(status='PASS_EXACT_FROZEN_HELPER_IMPORTS_AND_CPU_HOOK_INSTALL',arm='AGG',modules=provenance,
                 control_binding_sha256=sha(CONTROL/'evidence/RUNTIME_BINDING_PREFIX.json'),
                 inherited_temporal_recipe='PREFIX',model_queries=0,gradient_queries=0,optimizer_updates=0,
                 script_sha256=sha(__file__))
    p = ROOT/'evidence/RUNTIME_BINDING_AGG.json'; text = json.dumps(value,indent=2,sort_keys=True)+'\n'
    if p.exists(): assert p.read_text() == text
    else: p.write_text(text)
    print(value['status'],'AGG',flush=True)


if __name__ == '__main__': main()
