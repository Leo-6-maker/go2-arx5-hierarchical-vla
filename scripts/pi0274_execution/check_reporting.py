"""CPU-only original-boundary fixtures and one real saved field tensor."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import ast,importlib.util,resource
from quality_gate import ROOT,read,sha
from quality_worker import write

def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30))
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def forbidden(*a,**kw):raise RuntimeError('PI0274_CPU_FIXTURE_NO_LEARNED_CALL_OR_OPTIMIZER')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden;assert not torch.cuda.is_available()
    modules={}
    for name in ('report','transitions','video','query_inventory','local_map','compare','archive_results','final_audit','late_final'):
        path=ROOT/'code'/(name+'.py');spec=importlib.util.spec_from_file_location('pi0274_fixture_'+name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules[name]=module
    modules['report'].self_check();modules['transitions'].self_check();modules['video'].prepared();modules['query_inventory'].prepared();modules['local_map'].self_check()
    modules['compare'].self_check();modules['archive_results'].prepared();modules['final_audit'].prepared()
    late_main=modules['late_final'].prepared();actual=late_main.__globals__['evaluate']
    assert actual.__globals__['ROOT']==ROOT and 296 in actual.__code__.co_consts
    assert len(read(ROOT/'design/SUPPORT_CARD.json')['late'])==296
    assert len(read(ROOT/'design/NEW_FIT192.json')['rows'])==192
    files=[ROOT/'code'/n for n in ('report.py','reuse.py','trajectory_replay.py','transitions.py','onset_final.py','video.py','query_inventory.py','field_report.py','field_cpu.py','compare.py','archive_results.py','final_audit.py','research_return.py','postprocess.py','quality_evaluate.py','late_final.py','seal_quality_final.py','check_reporting.py')]
    for path in files:ast.parse(path.read_text(),filename=str(path))
    write(ROOT/'evidence/REPORT_CPU_FIXTURES.json',dict(status='PASS_PI0274_ORIGINAL_BOUNDARIES_AND_NEW_FIELD_REPORT_WRAPPERS',
        files=[dict(path=str(p),sha256=sha(p)) for p in files],model_queries=0,optimizer_updates=0,teacher_calls=0,renders=0,physics=0,
        full_final_quality_replay='STILL_REQUIRED'))
    print('PASS_PI0274_REPORT_CPU_BOUNDARIES_AND_REGISTERED_WRAPPERS',flush=True)
if __name__=='__main__':main()
