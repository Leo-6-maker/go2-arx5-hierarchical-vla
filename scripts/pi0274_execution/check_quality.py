"""Bounded CPU checks of the actual adapted entry and original physical scoring."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import argparse,ast
from pathlib import Path
import numpy as np
import quality_gate as gate
from quality_offline import physical_score,write
from compat import load,NAMES,OLD
from local_map import self_check
def main():
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);assert not torch.cuda.is_available()
    def forbidden(*a,**kw):raise RuntimeError('PI0274_CPU_NO_MODEL_OR_OPTIMIZER')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden
    sv=np.zeros(10);sv[3]=sv[7]=1.;sv[9]=.5;expert=np.tile(sv,(40,1));decoded=expert[None].copy()
    metrics,verdict=physical_score(decoded,expert,sv,'PI0274-CONSTRUCTED');assert metrics['xyz_rmse_m']==0 and metrics['so3_rmse_rad']<1e-6 and verdict['decision']!='REJECT'
    decoded[:,:,3:9]=0;metrics,verdict=physical_score(decoded,expert,sv,'PI0274-INVALID');assert metrics is None and verdict['reason']=='INVALID_ROT6D_POLICY_OUTPUT'
    decoded[:]=np.nan;metrics,verdict=physical_score(decoded,expert,sv,'PI0274-NONFINITE');assert metrics is None and verdict['reason']=='NONFINITE_POLICY_OUTPUT'
    self_check();original=gate.ROOT;gate.ROOT=original/'runtime/missing-seal-fixture'
    try:
        try:gate.admitted('BROAD5120')
        except FileNotFoundError as e:assert str(e.filename).endswith('QUALITY_EXECUTION_FILESET.json')
        else:raise AssertionError('UNSEALED_QUERY_ADMITTED')
    finally:gate.ROOT=original
    for p in (original/'code').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    worker=load('quality_worker');assert len(worker._source_binding['registered_adaptations'])>=6
    write(original/'evidence/QUALITY_CPU_FIXTURES.json',dict(status='PASS_PI0274_ADMISSION_INVALID_RETENTION_FIELD_AND_S4_WRAPPER_CPU',source_files=[dict(path=str(p),sha256=gate.sha(p)) for p in (original/'code').glob('*.py')],worker_adaptation=worker._source_binding,model_queries=0,optimizer_updates=0,renders=0,physics=0))
    print('PASS_PI0274_ACTUAL_QUALITY_ENTRY_CPU_FIXTURES',flush=True)
def seal():
    assert gate.read(gate.ROOT/'evidence/QUALITY_CPU_FIXTURES.json')['status']=='PASS_PI0274_ADMISSION_INVALID_RETENTION_FIELD_AND_S4_WRAPPER_CPU'
    for row in gate.read(gate.ROOT/'evidence/QUALITY_CPU_FIXTURES.json')['source_files']:assert gate.sha(row['path'])==row['sha256']
    paths=list((gate.ROOT/'code').glob('*.py'))+list((gate.ROOT/'design').glob('*.json'))
    paths.extend(OLD/'code'/(n+'.py') for n in NAMES)
    for row in gate.read(OLD/'design/QUALITY_EXECUTION_FILESET.json')['files']:
        p=Path(row['path'])
        if p.suffix=='.py':assert gate.sha(p)==row['sha256'];paths.append(p)
    paths.extend(gate.ROOT/p for p in ('contracts/ROT/source/dataset/statistics.json','evidence/RESULT_ARCHIVE_data.json','evidence/HOPPER_CPU_ADMISSION.json','evidence/PRECISION_CONTRACT_RESOLUTION.json','evidence/EXPOSURE_AND_ACTUAL_SEEN.json','evidence/QUALITY_CPU_FIXTURES.json','model_bindings/BROAD5120.json'))
    paths=sorted({p for p in paths if p.name!='QUALITY_EXECUTION_FILESET.json'})
    write(gate.ROOT/'design/QUALITY_EXECUTION_FILESET.json',dict(status='FROZEN_PI0274_BEFORE_NEW_QUALITY',files=[dict(path=str(p),sha256=gate.sha(p)) for p in paths],old_BROAD_final_integrity_reused=True,EQUIL_final_requires_own_actual_training_audit=True,model_queries=0))
    gate.admitted('BROAD5120');print('PASS_SEALED_BROAD_FIELD_PRETRAIN_ADMISSION',len(paths),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('check','seal'));a=p.parse_args();main() if a.action=='check' else seal()
