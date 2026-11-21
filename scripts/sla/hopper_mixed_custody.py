"""Qualify and copy genuine mixed-producer TRAIN snapshots to a private Hopper staging directory."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from core_hopper_mixed_readback import inputs, LEGACY
from hopper_reader_compat import READER, READER_MANIFEST
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.sla.hopper_parent_custody import PLAN, digest, write_once
from scripts.sla.hopper_site import DEADLINE, site_epoch, wall_seconds
from scripts.sla.hopper_snapshot_relay import transfer


def actual_input(root, cfg, name):
    receipt = read(Path(cfg['intake_receipts'])/(name+'.json'))
    mixed = receipt['status']=='MIXED_COLD_READER_REQUIRED'
    expected = cfg['mixed_custody'] if mixed else cfg['original_custody']
    folder = Path(expected)/name
    seal_path = folder/'SEALED.private.json'
    seal = read(seal_path)
    if (receipt['key']!=name or receipt['repository']!='leoyuer/sla-core-hopper-train-v1'
            or not re.fullmatch('[0-9a-f]{40}', receipt['revision']) or digest(seal_path)!=receipt['seal']
            or seal['plan']!=PLAN or seal['count']!=receipt['count'] or seal['windows']!=receipt['windows']
            or (not mixed and receipt['status']!='VANDA_INBOX_VERIFIED')):
        raise ValueError('Public revision, VANDA intake and physical snapshot differ')
    return dict(path=str(seal_path), sha256=receipt['seal'], membership=seal['episode_sha256'],
                count=seal['count'], windows=seal['windows']), folder, mixed


def qualify_one(root, cfg, file, old, stop):
    entry, folder, mixed = actual_input(root,cfg,file.stem)
    if not mixed or entry['membership'][:old['episodes']]!=old['member_sha256']:
        raise ValueError('Mixed TRAIN no longer extends the verified reference')
    info = inputs(dict(cfg,mode='candidate'),entry,old)
    if not info['mixed']:
        raise ValueError('The actual receipt lacks per-episode provenance')
    for condition, registered in cfg['repairs'].items():
        case = info['cases'].get(condition)
        if case is not None and any(case[k]!=v for k,v in registered.items()):
            raise ValueError('The repaired expert native or admission identity changed')
    if not any(c in info['cases'] for c in cfg['repairs']):
        raise ValueError('No original repaired admission in mixed custody')
    reply = transfer(folder,Path(cfg['population']['path']),dict(cfg['relay'],deadline_epoch=stop),entry['sha256'])
    if reply['manifest_sha256']!=info['manifest'] or reply['count']!=info['episodes']:
        raise ValueError('Hopper staging and actual VANDA reader disagree')
    return dict(plan=PLAN,status='MIXED_TRAIN_VANDA_READ_AND_HOPPER_STAGED',
        qualified_reader=READER, original_prefix_windows=old['windows'], actual=info,
        destination=reply, revision=read(file)['revision'], job=os.environ['PBS_JOBID'],
        optimizer_updates=0,model_generations=0,native_requests=0,course_switches=0)


def stage(root, cfg, stop):
    if Path(cfg['relay']['hopper_ready']).name != 'mixed-verified':
        raise ValueError('Cold validation must use the separate Hopper staging directory')
    if digest(Path(cfg['capsule'])/'candidate/SOURCE_BINDING.private.json')!=READER_MANIFEST:
        raise ValueError('VANDA reader bundle differs from the registered H200 bundle')
    legacy = root/'LEGACY.json'
    if not legacy.exists():
        rec, _, mixed = actual_input(root,cfg,'TRAIN-016')
        if mixed or rec['sha256']!=LEGACY:
            raise ValueError('The original TRAIN-016 reference changed')
        original = inputs(dict(cfg,mode='candidate'),rec,None)
        if hashlib.sha256(json.dumps(original,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=cfg['core_legacy']:
            raise ValueError('VANDA and both Core readers disagree on actual original inputs')
        write_once(legacy,original)
    old = read(legacy)
    while time.time()<stop-300 and not (root/'STOP').exists():
        for file in sorted(Path(cfg['intake_receipts']).glob('TRAIN-???.json')):
            if read(file)['status']!='MIXED_COLD_READER_REQUIRED':
                continue
            result=qualify_one(root,cfg,file,old,stop)
            write_once(root/'RESULT.json',result)
            return
        write(root/'HEARTBEAT.json',dict(job=os.environ['PBS_JOBID'],pid=os.getpid(),stage='WAIT_MIXED_INTAKE',epoch=time.time()))
        time.sleep(min(45,max(0,stop-300-time.time())))
    write_once(root/'CLOSED.json',dict(status='STOP_OR_WALL_LIMIT',epoch=time.time()))


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('root', type=Path)
    cli.add_argument('config_sha256')
    args=cli.parse_args()
    root=args.root
    if digest(root/'CONFIG.json')!=args.config_sha256:
        raise ValueError('VANDA run configuration changed')
    cfg=read(root/'CONFIG.json')
    job=os.environ['PBS_JOBID']
    site=json.loads(subprocess.check_output(['/opt/pbs/bin/qstat','-f','-F','json',job],text=True,timeout=25))['Jobs'][job]
    res=site['Resource_List']
    if (site['job_state']!='R' or site['Job_Owner'].split('@')[0]!=os.environ['USER'] or site['project']!=cfg['project']
            or int(res['ncpus'])!=2 or int(res.get('ngpus',0))!=0 or wall_seconds(res['walltime'])!=3600
            or cfg['plan']!=PLAN or os.environ['JAX_PLATFORMS']!='cpu' or os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise PermissionError('Expected the bounded project CPU allocation')
    with closing(OwnerLock(root/'owner.lock')):
        stage(root,cfg,min(DEADLINE,cfg['grant_end'],site_epoch(site['stime'])+3600)-60)
