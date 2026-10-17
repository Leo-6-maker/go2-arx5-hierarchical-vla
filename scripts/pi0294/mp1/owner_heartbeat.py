"""Revalidate a stale cached owner heartbeat without extending its 300s lease."""
import math
import os
from pathlib import Path
import time

from scripts.pi0294.e1.control_read import read
from .directory_visibility_audit import force_server_attributes


class HeartbeatLost(TimeoutError):
    def __init__(self,proof):
        super().__init__('MP1_SINGLE_OWNER_HEARTBEAT_LOST_AFTER_SERVER_REVALIDATION')
        self.diagnostics=proof


def read_live_owner(root,run_uuid,*,now=None,reader=read,synchronize=force_server_attributes,force_revalidate=False):
    root=Path(root);path=root/'OWNER_HEARTBEAT.json'
    if root.is_symlink() or path.is_symlink() or not root.is_dir():
        raise PermissionError('EXACT_REGISTERED_HEARTBEAT_PATH_NO_SYMLINK')
    if ((now is not None and (type(now) not in [int,float] or not math.isfinite(now)))
            or type(force_revalidate) is not bool):raise ValueError('FINITE_NATIVE_CLOCK_AND_EXPLICIT_REVALIDATION')
    def age(value):
        # The owner may replace the file during I/O; compare each read to its subsequent clock sample.
        observed_now=time.time() if now is None else now
        if type(observed_now) not in [int,float] or not math.isfinite(observed_now):raise ValueError('FINITE_NATIVE_CLOCK')
        if value['run_uuid']!=run_uuid:raise PermissionError('EXACT_OWNER_HEARTBEAT_RUN')
        epoch=value['observed_epoch']
        if type(epoch) not in [int,float] or not math.isfinite(epoch):raise ValueError('FINITE_REAL_OWNER_HEARTBEAT_EPOCH')
        return observed_now-epoch,observed_now
    value=reader(path);cached_age,cached_clock=age(value)
    if 0<=cached_age<=300 and not force_revalidate:
        return dict(status='PASS_ORIGINAL_300S_OWNER_HEARTBEAT',heartbeat=value,cached_age_s=cached_age,
            server_revalidation=False,revalidated_age_s=None,heartbeat_written=False,watchdog_s=300,
            cached_clock_epoch=cached_clock,clock_sampled_after_read=now is None)
    synchronize(root)
    with os.scandir(root) as entries:
        if sum(1 for _ in entries)>512:raise PermissionError('BOUNDED_HEARTBEAT_PARENT_ENUMERATION')
    if path.is_symlink():raise PermissionError('HEARTBEAT_PATH_REPLACED_WITH_SYMLINK')
    synchronize(path);value=reader(path);fresh_age,fresh_clock=age(value)
    if not 0<=fresh_age<=300:raise HeartbeatLost(dict(cached_age_s=cached_age,revalidated_age_s=fresh_age,
        heartbeat=value,watchdog_s=300,heartbeat_written=False,cached_clock_epoch=cached_clock,
        revalidated_clock_epoch=fresh_clock,clock_sampled_after_read=now is None))
    return dict(status='PASS_ORIGINAL_300S_OWNER_HEARTBEAT',heartbeat=value,cached_age_s=cached_age,
        server_revalidation=True,revalidated_age_s=fresh_age,heartbeat_written=False,watchdog_s=300,
        cached_clock_epoch=cached_clock,revalidated_clock_epoch=fresh_clock,clock_sampled_after_read=now is None)
