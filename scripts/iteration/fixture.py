"""Spawn isolated CPU fixture workers, without simulator/model imports or SSH."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import time

from scripts.pi0292.durable import write
from .plan import collection_plan
from .queue import Coordinator


def worker(ticket):
    if ticket['fixture'] is not True:raise ValueError('FIXTURE_ONLY')
    before=resource.getrusage(resource.RUSAGE_SELF)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    began=time.monotonic()
    payload=json.dumps({k:ticket[k] for k in ('id','seed','split')},sort_keys=True).encode()
    path=Path(ticket['output'])/'SYNTHETIC.json';path.write_bytes(payload)
    time.sleep(.05)
    after=resource.getrusage(resource.RUSAGE_SELF)
    receipt=dict(id=ticket['id'],fixture=True,status='FIXTURE_COMPLETE',pid=os.getpid(),
        proc_start=Path('/proc/self/stat').read_text().split()[21],
        physics_steps=0,rgb_pairs=0,model_calls=0,
        cpu_s=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        wall_s=time.monotonic()-began,output_bytes=len(payload),
        payload_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write(Path(ticket['output'])/'WORKER_RECEIPT.json',receipt)
    return receipt


def ready(barrier):
    barrier.wait(timeout=20)


def run_fixture(root, card):
    plan=collection_plan(card);coordinator=Coordinator(root,plan,fixture=True)
    order=iter(row['id'] for row in plan['episodes']);pending={};exhausted=False
    try:
        context=multiprocessing.get_context('spawn');barrier=context.Barrier(plan['workers'])
        with ProcessPoolExecutor(max_workers=plan['workers'],mp_context=context,
                                 initializer=ready,initargs=(barrier,)) as pool:
            while pending or not exhausted:
                while not exhausted and len(pending)<plan['workers'] and coordinator.value['status']=='FIXTURE_RUNNING_ZERO_PHYSICS':
                    identifier=next(order,None)
                    if identifier is None:exhausted=True;break
                    ticket=coordinator.reserve(identifier)
                    pending[pool.submit(worker,ticket)]=identifier
                if not pending:break
                completed,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in completed:
                    identifier=pending.pop(future)
                    try:coordinator.complete(future.result())
                    except Exception:
                        coordinator.unknown(identifier);exhausted=True
            return coordinator.finish()
    finally:coordinator.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--card',required=True);p.add_argument('--root',required=True);a=p.parse_args()
    print(json.dumps(run_fixture(a.root,json.loads(Path(a.card).read_text())),indent=2))
