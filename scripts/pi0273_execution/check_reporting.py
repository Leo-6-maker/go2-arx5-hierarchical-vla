"""Bounded CPU checks of the actual report/retention/archive entry points."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import ast
from pathlib import Path
import torch
import report
import transitions
import archive
from quality_worker import ROOT,read,sha,write

def main():
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30))
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def forbidden(*a,**kw):raise RuntimeError('PI0273_REPORT_CHECK_NO_LEARNED_MODEL_OR_OPTIMIZER')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden
    assert not torch.cuda.is_available()
    report.self_check();transitions.self_check();archive.prepared()
    assert [name for _,name in archive.roots('predictions')]==['offline','newfit','late','virtual','component']
    files=[ROOT/'code'/name for name in ('report.py','transitions.py','archive.py','query_inventory.py','final_audit.py','reuse.py','trajectory_replay.py','video.py','check_reporting.py')]
    for path in files:ast.parse(path.read_text(),filename=str(path))
    write(ROOT/'evidence/REPORT_CPU_FIXTURES.json',dict(status='PASS_REPORT_AND_ARCHIVE_CPU_FIXTURES_WITH_NO_LEARNED_CALLS',
        source_files=[dict(path=str(p),sha256=sha(p)) for p in files],saved_predecessor_numeric_fixture_reused=True,
        constructed_qualification_pairing_and_exit_boundaries=True,model_queries=0,optimizer_updates=0,renders=0,physics=0))
    print('PASS_REPORT_AND_ARCHIVE_CPU_FIXTURES_WITH_NO_LEARNED_CALLS',flush=True)

if __name__=='__main__':main()
