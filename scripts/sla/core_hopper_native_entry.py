"""Retain the RGB16 numeric imports while providing the original native IPC modules."""
from importlib import import_module
from pathlib import Path
import os
import runpy
import sys


def bind_imports(source):
    for name in ('go2arx5_vla','go2arx5_vla.sla_adapter','go2arx5_vla.world_model'):
        module=import_module(name)
        fallback=str(source/'src'/Path(*name.split('.')))
        if fallback not in module.__path__:
            module.__path__=[*module.__path__,fallback]


if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1]!='--worker':
        raise ValueError('Expected a registered student IPC worker')
    source=Path(__file__).resolve().parents[2]
    bind_imports(source)
    if os.environ.get('SLA_CORE_CANONICAL_WORKER') == '1':
        # Resolve the new package/proof entry points from the pinned native
        # overlay, while leaving the original numerical module order in place.
        scope = import_module('scripts.sla')
        scope.__path__ = [str(source/'scripts/sla'), *scope.__path__]
    runpy.run_path(str(source/'scripts/sla/hybrid_ipc_worker_v3.py'),run_name='__main__')
