"""One post-freeze source TEST with the existing serial GPU owner."""
import copy
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback

from scripts.pi0292.durable import write
from scripts.pi0294.e1.journal import Journal
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.source_training_v6 import PLAN
from go2arx5_vla.world_model.mfree_recovery_v2 import register
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close, dirs, unit_state
from go2arx5_vla.world_model.storage import digest


def pin(path):
    return dict(path=str(path),sha256=digest(path))


def read(path):
    return json.loads(Path(path).read_bytes())


def wait_terminal(spec, predecessor, phase):
    root=Path(spec['registered_root'])
    prior=read_pin(predecessor)
    folder=Path(prior['registered_root'])
    while time.time()<spec['deadline_epoch']-600 and not (root/'STOP.json').exists():
        lease=read(folder/'control/LEASE.json')
        if lease['spec_sha256']!=predecessor['sha256']:
            raise ValueError('WAITED_SOURCE_TEST_PREDECESSOR_SPEC_CHANGED')
        unit=unit_state('wm02-'+folder.name)
        write(root/'PROGRESS.private.json',dict(phase=phase,epoch=time.time(),
              waited_lease=lease['status'],waited_MainPID=unit['pid'],new_updates=0,
              new_native_requests=0))
        if lease['status']=='CLOSED' and not unit['pid'] and not unit['cgroup_pids']:
            return pin(folder/'RESULT.private.json')
        time.sleep(30)
    raise TimeoutError('FINITE_SOURCE_TEST_DEPENDENCY_WINDOW')


def child_spec(spec, frozen, stage, prepared=None):
    root=Path(spec['registered_root'])/stage
    dirs(root)
    child=copy.deepcopy(spec)
    child.update(registered_root=str(root),role='SLA_V6_SOURCE_TEST_'+stage,
                 new_model_generations_budget=0 if prepared is None else 96,**frozen)
    child['native_environment']=dict(spec['native_environment'],TMPDIR=str(root/'tmp'),
                                     XDG_CACHE_HOME=str(root/'cache'))
    if prepared is None:
        child['native_environment'].update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    else:
        child['compilation_cache_dir']=str(root/'cache')
        child['test_preparation_result']=prepared
        site=read_pin(spec['site_binding'])
        site['allocation']['campaign_root']=spec['registered_root']
        site['paths']['compilation_cache_dir']=child['compilation_cache_dir']
        write(root/'SITE_BINDING.private.json',site)
        child['site_binding']=pin(root/'SITE_BINDING.private.json')
    register(spec,[(root,child)])
    return pin(root/'EXECUTION_SPEC.json')


def run_child(spec, child_pin, journal, stage, route):
    child=read_pin(child_pin);folder=Path(child['registered_root'])
    root=Path(spec['registered_root'])

    def execute():
        def set_cpu_memory():
            limit=4*1024**3
            resource.setrlimit(resource.RLIMIT_AS,(limit,limit))

        entry=Path(spec['source_root'])/'scripts/wm02/v6_bootstrap.py'
        with (folder/'logs/supervisor.log').open('xb') as log:
            process=subprocess.Popen([child['python'],'-B',str(entry),route,child_pin['path']],
                stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,**child['native_environment']),
                preexec_fn=set_cpu_memory if stage=='PREPARE' else None)
            write(root/'CURRENT_CHILD.private.json',dict(pid=process.pid,stage=stage,spec=child_pin))
            while process.poll() is None:
                if (root/'STOP.json').exists() or time.time()>=spec['deadline_epoch']-60:
                    write(folder/'STOP.json',dict(reason='FINITE_SOURCE_TEST_PARENT_LIMIT_OR_STOP'))
                time.sleep(3)
        result=pin(folder/'RESULT.private.json')
        return dict(exit_code=process.returncode,result=result)

    actual=journal.once('RUN-'+stage,dict(spec=child_pin),execute)
    if actual['exit_code'] or read_pin(actual['result'])['exit_code']:
        raise RuntimeError('PRESERVED_SOURCE_TEST_CHILD_FAILURE:'+stage)
    return actual['result']


def run(spec):
    root=Path(spec['registered_root'])
    journal=Journal(root/'control/journal',root.name)
    value=dict(source_sha=spec['source_sha'],new_updates=0,new_visits=0,new_native_requests=0)
    try:
        if (spec['plan_commit']!=PLAN or spec['new_optimizer_updates_budget'] or
                spec['new_training_visits_budget'] or spec['new_model_generations_budget']!=96 or
                spec['new_physical_requests_budget'] or spec['physics']):
            raise PermissionError('FIXED_SOURCE_TEST96_NO_TRAINING_OR_NATIVE_REQUESTS')
        source_result=wait_terminal(spec,spec['wait_source'],'WAIT_FIXED_S1000_AND_INDEPENDENT_RESTORES')
        source=read_pin(source_result)
        if source['exit_code'] or source.get('new_updates')!=1000 or len(source['records'])!=3:
            raise ValueError('FIXED_SOURCE_ENDPOINT_AND_RESTORES_FAILED_TEST_UNOPENED')
        train=source['records'][0]
        frozen=dict(source_campaign_result=source_result,
                    frozen_training_result=dict(path=train['result_path'],sha256=train['result_sha256']),
                    training_spec=spec['source_training_spec'])
        prepare=journal.once('REGISTER-PREPARE',frozen,lambda:child_spec(spec,frozen,'PREPARE'))
        prepared=run_child(spec,prepare,journal,'PREPARE','--source-test-prepare')
        value['preparation_result']=prepared
        wait_terminal(spec,spec['wait_target_followup'],'WAIT_EXISTING_T_FOLLOWUP_GPU_COLD_BOUNDARY')
        evaluate=journal.once('REGISTER-EVALUATE',prepared,lambda:child_spec(spec,frozen,'EVALUATE',prepared))
        result=run_child(spec,evaluate,journal,'EVALUATE','--source-test')
        value.update(exit_code=0,status='SOURCE_TEST96_ONCE_COMPLETE',test_result=result,new_generations=96)
    except BaseException:
        value.update(exit_code=1,status='SOURCE_TEST_FINITE_TERMINAL_NO_REPLAY',failure=traceback.format_exc())
    write(root/'RESULT.private.json',value)
    close(spec,value['exit_code'])
    return value['exit_code']
