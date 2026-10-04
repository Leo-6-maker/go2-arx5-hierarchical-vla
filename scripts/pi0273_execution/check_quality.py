"""Constructed admission/invalid-output checks; no neural model or renderer."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import ast
from pathlib import Path
import numpy as np
import quality_gate as gate
import quality_worker as worker
from quality_offline import physical_score
from quality_evaluate import complete

def main():
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def forbidden(*a,**kw):raise RuntimeError('PI0273_CPU_NO_LEARNED_QUERY_OR_OPTIMIZER')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden
    assert not torch.cuda.is_available()
    root=gate.ROOT;fixture=root/'runtime/quality-fixtures';fixture.mkdir(parents=True,exist_ok=True)
    gate.ROOT=fixture
    try:
        try:gate.admitted()
        except FileNotFoundError as error:assert str(error.filename).endswith('QUALITY_EXECUTION_FILESET.json')
        else:raise AssertionError('UNSEALED_EVALUATION_ADMITTED')
    finally:gate.ROOT=root
    assert not complete([fixture/'missing.json'],'BROAD5120',root/'design/SUPPORT_CARD.json')
    sv=np.zeros(10);sv[3]=sv[7]=1.;sv[9]=.5
    expert=np.tile(sv,(40,1));decoded=expert[None].copy()
    metrics,result=physical_score(decoded,expert,sv,'PI0273-CONSTRUCTED')
    assert metrics['xyz_rmse_m']==0 and metrics['so3_rmse_rad']<1e-6 and result['decision']!='REJECT'
    decoded[:,:,3:9]=0
    metrics,result=physical_score(decoded,expert,sv,'PI0273-CONSTRUCTED-INVALID')
    assert metrics is None and result['reason']=='INVALID_ROT6D_POLICY_OUTPUT'
    decoded[:]=np.nan
    metrics,result=physical_score(decoded,expert,sv,'PI0273-CONSTRUCTED-NONFINITE')
    assert metrics is None and result['reason']=='NONFINITE_POLICY_OUTPUT'
    files=[root/'code'/n for n in ('quality_worker.py','quality_gate.py','quality_virtual.py','quality_components.py',
        'quality_offline.py','newfit.py','late.py','quality_evaluate.py','materialize.py','check_quality.py')]
    for p in files:ast.parse(p.read_text(),filename=str(p))
    virtual=(root/'code/quality_virtual.py').read_text();assert 'ordinal<=208' in virtual and 'frame_ordinal<=15360' in virtual
    source=(root/'code/quality_worker.py').read_text();assert 'ordinal<=18432' in source and 'heads<=73728' in source
    worker.write(root/'evidence/QUALITY_CPU_FIXTURES.json',dict(status='PASS_CONSTRUCTED_ADMISSION_INVALID_RETENTION_AND_PHYSICAL_SCORE',
        source_files=[dict(path=str(p),sha256=worker.sha(p)) for p in files],model_queries=0,optimizer_updates=0,renders=0,physics=0))
    print('PASS_CONSTRUCTED_ADMISSION_INVALID_RETENTION_AND_PHYSICAL_SCORE',flush=True)

if __name__=='__main__':main()
