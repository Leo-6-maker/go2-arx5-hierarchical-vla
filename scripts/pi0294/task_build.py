"""Compile only our task contact/finite-gripper adapters, never native WBC."""
import argparse
from pathlib import Path
import subprocess
from .task_capture_runtime import cpu_stage,sha
from .task_contacts import make_observer


def run(root,native,startup,stage):
    from scripts.pi0292.durable import write
    root,native,startup=map(Path,(root,native,startup))
    with cpu_stage(root,stage,cores=1,cpu_s_cap=600):
        out=root/'runtime'/stage;out.mkdir(exist_ok=False)
        observer=startup/'startup-v2/observer.cpp'
        make_observer(observer,out/'task_observer.cpp')
        mujoco=native/'src/ctrl/rl_sar/library/mujoco'
        common=['g++','-std=c++17','-O2','-shared','-fPIC','-I'+str(mujoco/'include'),
                '-L'+str(mujoco/'lib'),'-Wl,-rpath,'+str(mujoco/'lib')]
        sources={'task_observer':out/'task_observer.cpp','task_actuation':Path(__file__).with_name('task_actuation.cpp')}
        for name,path in sources.items():
            result=subprocess.run(common+[str(path),'-lmujoco','-ldl','-o',str(out/(name+'.so'))],text=True,capture_output=True)
            write(out/(name+'-BUILD.json'),dict(exit=result.returncode,stdout=result.stdout,stderr=result.stderr,source_sha256=sha(path)))
            if result.returncode:raise RuntimeError(result.stderr)
        result=dict(status='TASK_ADAPTER_BUILD_PASS_NOT_PHYSICS_QUALIFICATION',
            original_observer_sha256=sha(observer),sources={k:sha(v) for k,v in sources.items()},
            binaries={name:sha(out/(name+'.so')) for name in sources},
            native_WBC_rebuild=False,native_threshold_changes=False,new_physics=0,new_neural_queries=0)
        write(out/'BUILD_RECEIPT.json',result);print(__import__('json').dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','native','startup','stage'):p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.root,a.native,a.startup,a.stage)
