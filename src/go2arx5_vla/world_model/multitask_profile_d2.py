"""Measure the unmodified paused mobile timing and all owned native children."""

from collections import defaultdict
import os
from pathlib import Path
from threading import Event, Thread
import time

from .native_collection_performance_v1 import PerformanceTrace


def fields(path, *, delimiter=None):
    try:
        return dict(line.split(delimiter, 1) for line in path.read_text().splitlines()
                    if line.strip())
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return {}


def process_row(pid):
    root = Path('/proc')/str(pid)
    try:
        stat = (root/'stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    status = fields(root/'status', delimiter=':')
    io = fields(root/'io', delimiter=':')
    memory = fields(root/'smaps_rollup', delimiter=':')
    return dict(pid=pid, ppid=int(stat[1]), started_ticks=int(stat[19]),
                cpu_ticks=int(stat[11])+int(stat[12]), threads=int(stat[17]),
                rss_kib=int(status.get('VmRSS', '0').split()[0]),
                pss_kib=int(memory['Pss'].split()[0]) if 'Pss' in memory else None,
                read_bytes=int(io['read_bytes']) if 'read_bytes' in io else None,
                write_bytes=int(io['write_bytes']) if 'write_bytes' in io else None,
                name=status.get('Name', '').strip())


class MobileProfile:
    def __init__(self, adapter):
        self.adapter = adapter
        adapter.performance = PerformanceTrace()
        self.started = time.monotonic()
        self.pid = os.getpid()
        group = next(line.split('::', 1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines()
                     if line.startswith('0::'))
        self.cgroup = Path('/sys/fs/cgroup') / group.lstrip('/')
        self.start_cpu = self.cgroup_cpu()
        self.samples = []
        self.finish = Event()
        self.worker = Thread(target=self.watch, name='mobile-process-tree-profile', daemon=True)
        self.worker.start()

    def cgroup_cpu(self):
        return {key:int(value) for key,value in fields(self.cgroup/'cpu.stat').items()}

    def sample(self):
        pids = set()
        for path in (self.cgroup/'cgroup.procs', *self.cgroup.glob('**/cgroup.procs')):
            try:
                pids.update(map(int, path.read_text().split()))
            except FileNotFoundError:
                continue
        # Find every descendant first, then collect the comparatively expensive
        # PSS/I/O metrics for the selected world only.
        by_parent = defaultdict(list)
        for pid in pids:
            try:
                stat = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            by_parent[int(stat[1])].append(pid)
        descendants = [self.pid]
        visited = {self.pid}
        for pid in descendants:
            for child in by_parent[pid]:
                if child not in visited:
                    visited.add(child)
                    descendants.append(child)
        rows = [row for pid in descendants if (row := process_row(pid)) is not None]
        temp = getattr(self.adapter,'live_control_output', None)
        live_bytes = 0
        if temp is not None:
            for p in Path(temp).iterdir():
                if not p.is_symlink():
                    try:
                        live_bytes += p.stat().st_size
                    except FileNotFoundError:
                        pass
        self.samples.append(dict(wall_s=time.monotonic()-self.started, rows=rows,
                                 tmpfs_bytes=live_bytes, cpu_affinity=sorted(os.sched_getaffinity(0))))

    def watch(self):
        while not self.finish.wait(2.):
            self.sample()

    def close(self):
        self.finish.set()
        self.worker.join(timeout=5.)
        final_cpu = self.cgroup_cpu()
        peaks = defaultdict(int)
        for s in self.samples:
            for row in s['rows']:
                for key in ('cpu_ticks','read_bytes','write_bytes'):
                    if row[key] is not None:
                        ident=(row['pid'],row['started_ticks'],key)
                        peaks[ident]=max(peaks[ident], row[key])
        stats = self.adapter.performance.report()
        totals = {key:sum(v for (_,_,k),v in peaks.items() if k==key)
                  for key in ('cpu_ticks','read_bytes','write_bytes')}
        totals['observed_process_tree_cpu_core_s_lower_bound'] = totals['cpu_ticks']/os.sysconf('SC_CLK_TCK')
        return dict(schema='SLA_MOBILE_D2_LEGACY_TIMING_PROFILE', self_wall_samples=self.samples,
                    world_tree_observed_totals_lower_bound=totals,
                    group_cpu_delta_us={k:final_cpu[k]-v for k,v in self.start_cpu.items() if k in final_cpu},
                    group_cpu_scope='SYSTEMD_BATCH_INCLUDING_OWN_WORLDS_AND_SUPERVISOR',
                    controller_selected_reference='UNKNOWN', fixed_post_atom_wait_s=.15, **{
                    'timing':stats, 'task_wall_s':time.monotonic()-self.started})
