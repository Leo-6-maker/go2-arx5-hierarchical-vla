"""Measure the actual Core collector without loading or instrumenting its runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.pi0292.durable import read, write


def fields(path):
    return {line.split(maxsplit=1)[0].rstrip(':'): line.split(maxsplit=1)[1]
            for line in path.read_text().splitlines() if len(line.split(maxsplit=1))==2}


def measure_process(pid):
    root = Path('/proc')/str(pid)
    try:
        state = fields(root/'status')
        memory = fields(root/'smaps_rollup')
        usage = fields(root/'io')
        counters = (root/'stat').read_text().rsplit(') ', 1)[1].split()
        return dict(pid=int(pid), command=state['Name'], cpus=state['Cpus_allowed_list'],
                    threads=int(state['Threads']), rss_kib=int(memory['Rss'].split()[0]),
                    pss_kib=int(memory['Pss'].split()[0]), minor_faults=int(counters[7]),
                    major_faults=int(counters[9]), read_bytes=int(usage['read_bytes']),
                    write_bytes=int(usage['write_bytes']))
    except (OSError, KeyError):
        return dict(pid=int(pid), exited_or_unreadable=True)


def sample(group, producer):
    processes = [measure_process(p) for p in (group/'cgroup.procs').read_text().split()]
    group_io = group/'io.stat'
    return dict(epoch=time.time(), progress=read(producer/'PROGRESS.private.json'),
                cpu_usec=int(fields(group/'cpu.stat')['usage_usec']),
                memory_current=int((group/'memory.current').read_text()),
                memory_peak=int((group/'memory.peak').read_text()),
                cgroup_io=group_io.read_text() if group_io.is_file() else None, processes=processes,
                process_pss_kib=sum(p.get('pss_kib',0) for p in processes),
                memory_available_kib=int(fields(Path('/proc/meminfo'))['MemAvailable'].split()[0]),
                host_pressure={key:Path(f'/proc/pressure/{key}').read_text()
                               for key in ('cpu','memory','io')},
                shared_memory_free_bytes=shutil.disk_usage('/dev/shm').free,
                data_filesystem_free_bytes=shutil.disk_usage(producer).free)


def observe(root, config):
    producer = Path(config['producer'])
    unit = 'wm02-'+producer.name+'.service'
    path = producer/'EXECUTION_SPEC.json'
    spec = read(path)
    status = dict(line.split('=',1) for line in subprocess.check_output(
        ['systemctl','--user','show',unit,'-p','ControlGroup','-p','InvocationID','-p','ActiveState'],
        text=True).splitlines())
    if (hashlib.sha256(path.read_bytes()).hexdigest()!=config['producer_spec']
            or spec['registered_root']!=str(producer) or spec['uid']!=os.getuid()
            or spec['scope']!='SLA_CORE_MULTIENV_V1_EXPERT'
            or status['InvocationID']!=config['invocation'] or status['ActiveState']!='active'
            or not os.environ.get('INVOCATION_ID') or config['deadline']>spec['deadline_epoch']):
        raise PermissionError('The bound Core collector is not the live site owner')
    group = Path('/sys/fs/cgroup')/status['ControlGroup'].lstrip('/')
    start = sample(group,producer)
    write(root/'OWNER.json',dict(pid=os.getpid(),observed_unit=unit,cgroup=status['ControlGroup'],
                                  producer_spec=config['producer_spec'],deadline=config['deadline']))
    latest = start
    peak_pss = 0
    ticks = 0
    with (root/'SAMPLES.jsonl').open('x') as stream:
        while (time.time()<config['deadline'] and not (root/'STOP').exists()
                and not (producer/'RESULT.private.json').exists() and group.is_dir()):
            latest = sample(group,producer)
            stream.write(json.dumps(latest)+'\n')
            stream.flush()
            ticks += 1
            peak_pss = max(peak_pss,latest['process_pss_kib'])
            if latest['progress']['completed']>=4:
                break
            time.sleep(10)
    write(root/'RESULT.json',dict(scope='N1_PRODUCTION_OBSERVATION',samples=ticks,
        cpu_core_seconds=(latest['cpu_usec']-start['cpu_usec'])/1_000_000,
        peak_pss_kib=peak_pss,cgroup_peak_bytes=latest['memory_peak'],
        observed_seconds=latest['epoch']-start['epoch'],progress=latest['progress'],
        new_optimizer_updates=0,new_physical_requests=0,new_model_generations=0))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('config_sha')
    args = parser.parse_args()
    os.umask(0o077)
    file = args.root/'CONFIG.json'
    if hashlib.sha256(file.read_bytes()).hexdigest()!=args.config_sha:
        raise PermissionError('Profile binding changed')
    observe(args.root, read(file))
