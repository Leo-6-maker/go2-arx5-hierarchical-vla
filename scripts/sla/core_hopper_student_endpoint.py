"""Bind Core's actual RGB16 evidence to the new DEV/CONF student population."""
from contextlib import closing
import argparse
import os
from pathlib import Path
import time

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.sla.hopper_core_receive import SCIENCE
from scripts.sla.hopper_parent_custody import PLAN, digest, write_once
from scripts.sla.hopper_public_checkpoint import REPO, PREFIX, ORDINALS, REF_HASHES, allowed
from scripts.sla.hopper_site import DEADLINE, pinned


def pin(path):
    return dict(path=str(path),sha256=digest(path))


def check_endpoint(slot, site):
    receipt = read(slot/'RECEIVED.json')
    manifest = pinned(dict(path=str(slot/'payload/CHECKPOINT.json'),sha256=receipt['manifest']))
    members = allowed(manifest)
    key = f"{manifest['role']}-phase{manifest['phase']:05d}"
    if (slot.name != key or receipt['repository'] != REPO or not receipt['revision']
            or receipt['prefix'] != PREFIX+'/'+key+'-'+receipt['manifest'][:12] or receipt['arrays'] != len(members)
            or manifest['training_source'] != SCIENCE or manifest['checkpoint_sha256'] != receipt['checkpoint']
            or manifest['foundation_sha256'] != site['model']['parent_manifest']['sha256']
            or manifest['state_sha256'] != site['model']['state_sha256'] or manifest['action_sha256'] != site['action']['sha256']
            or manifest['tokenizer_sha256'] != site['tokenizer']['sha256'] or manifest['panel'] != site['panel']['sha256']):
        raise ValueError('Core checkpoint, base, statistics or panel identity differs')
    replay = read(slot/'RESULT.json')
    gpu = read(slot/'GPU_START.json')
    if (replay['checkpoint'] != receipt['checkpoint'] or gpu['gpu'] != site['gpu'] or not gpu['systemd_invocation']
            or not gpu['launch'] < gpu['end'] <= site['deadline'] or replay['native_requests'] or replay['updates']):
        raise ValueError('Core GPU allocation or replay identity differs')
    if replay['status'] != 'CORE_RGB16_PASS':
        return manifest,dict(status='ENDPOINT_NATIVE_HELD',rgb_status=replay['status'],
                             checkpoint=receipt['checkpoint'],core_result=pin(slot/'RESULT.json'),
                             exact=replay.get('exact'),first_mismatch=replay.get('first_mismatch'),matched=replay.get('matched'))
    if (replay['manifest'] != receipt['manifest'] or replay['first_mismatch'] is not None or replay['native_ticks']
            or (replay['attempts'],replay['completed'],replay['exact']) != (16,16,16)
            or replay['matched'] != dict.fromkeys(REF_HASHES,16)):
        raise ValueError('Expected original exact 16 under all signed measures')
    for ending in ('intent','generated','result'):
        if {p.name for p in slot.glob('*.'+ending+'.json')} != {f'{i:03d}.{ending}.json' for i in ORDINALS}:
            raise ValueError('Missing or additional original Core replay ordinal')
    for row in manifest['reference']:
        ident = {k:row[k] for k in ('ordinal','index','noise_id','input_sha256')}
        stem = f"{row['ordinal']:03d}"
        if (read(slot/(stem+'.intent.json')) != ident or read(slot/(stem+'.generated.json')) != dict(ordinal=row['ordinal'])
                or read(slot/(stem+'.result.json')) != row):
            raise ValueError('Cold Hopper and Core replay rows differ')
    for member,expected in members.items():
        if digest(slot/'payload'/member) != expected:
            raise ValueError('Received weights changed after replay')
    return manifest,dict(status='ENDPOINT_RGB16_VERIFIED',checkpoint=receipt['checkpoint'],manifest=receipt['manifest'],
                         source=SCIENCE,receipt=pin(slot/'RECEIVED.json'),core_result=pin(slot/'RESULT.json'),
                         site_gpu=pin(slot/'GPU_START.json'),exact=16)


def cases(pop, key, split):
    if (pop['plan'] != PLAN or pop['old_TEST_reused'] or pop['assignment_uses_outcomes']
            or split not in ('DEV','CONF') or (split=='CONF' and key!='MAIN-phase20000')):
        raise ValueError('Only fixed DEV and final 20K CONF are eligible')
    clusters = {r['id']:r for r in pop['clusters']}
    rows = pop['conditions'][split]
    if len(rows)!=24 or len({r['condition_id'] for r in rows})!=24:
        raise ValueError('Incomplete registered student population')
    out=[]
    for r in rows:
        c=clusters[r['cluster']]
        budget=(35703,120) if r['instruction'].startswith('HOLD-') else (50703,180)
        if (r['split']!=split or r['id']!=r['condition_id'] or r['role']!=('student_development' if split=='DEV'
                else 'student_confirmation') or r['native_request_limit']['key']!='condition_id_and_endpoint_identity'
                or r['native_request_limit']['max_actual_requests']!=1 or not r['native_request_limit']['technical_failure_after_request_consumes_slot']
                or c['reset']!=r['reset'] or c['family']!=r['family'] or c['scene_seed']!=r['scene_seed']
                or digest(Path(c['scene']['path']))!=c['scene']['sha256']
                or (r['native_max_ticks'],r['max_queries_or_generations'])!=budget):
            raise ValueError('Student condition, frozen scene or individual budget changed')
        out.append(dict(id=r['id']+'-'+key,condition=r['id'],arm=key,geometry=r['cluster'],family=r['family'],
                        instruction=r['instruction'],scene=c['scene'],reset=r['reset'],scene_seed=r['scene_seed'],
                        request_cap=1,tick_cap=budget[0],query_cap=budget[1]))
    if sum(r['query_cap'] for r in out)!=3840 or sum(r['tick_cap'] for r in out)!=1096872:
        raise ValueError('Invalid cumulative student panel budget')
    return out


def watch(root,cfg):
    receiver=Path(cfg['receiver']['path']).parent
    site=pinned(cfg['receiver'])
    owner=pinned(cfg['expert'])
    if (cfg['population']!=owner['compiled_population'] or owner['plan_commit']!=PLAN
            or owner['scope']!='SLA_CORE_HOPPER_V1_EXPERT' or site['plan']!=PLAN or site['deadline']!=DEADLINE):
        raise ValueError('Receiver or geometry changes the goal identity')
    population=pinned(cfg['population'])
    with closing(OwnerLock(root/'owner.lock')):
        while time.time()<DEADLINE and not (root/'STOP').exists():
            results=sorted((receiver/'endpoints').glob('*/RESULT.json'))
            if len(results)>7:
                raise ValueError('Deployment allowance exceeded')
            for file in results:
                dst=root/'endpoints'/(file.parent.name+'.json')
                if dst.exists():
                    if read(dst)['gate']['core_result']!=pin(file):
                        raise ValueError('Existing replay verdict changed')
                    continue
                manifest,gate=check_endpoint(file.parent,site)
                panels={}
                if gate['status']=='ENDPOINT_RGB16_VERIFIED':
                    stages=['DEV','CONF'] if file.parent.name=='MAIN-phase20000' else ['DEV']
                    panels={k:cases(population,file.parent.name,k) for k in stages}
                write_once(dst,dict(gate=gate,panels=panels,compiled_population=cfg['population'],
                           core_profile=site['gpu'],source=cfg['source'],plan=PLAN,check_epoch=time.time(),
                           physical_requests=0,model_generations=0,hardware_allowed=False))
            write(root/'HEARTBEAT.json',dict(status='WAIT_FOR_ENDPOINT' if not results else 'ENDPOINTS_RECONCILED',
                    processed=len(list((root/'endpoints').glob('*.json'))),pid=os.getpid(),epoch=time.time()))
            time.sleep(60)
    write_once(root/'CLOSED.json',dict(status='STOP_OR_DEADLINE',epoch=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('config_sha')
    a=p.parse_args()
    cfg=pinned(dict(path=str(a.root/'CONFIG.json'),sha256=a.config_sha))
    if (cfg['plan']!=PLAN or cfg['deadline']!=DEADLINE or cfg['worker_sha']!=digest(Path(__file__))
            or os.getuid()!=2009 or os.uname().nodename!='corelab-4080s' or not os.environ.get('INVOCATION_ID')):
        raise PermissionError('Unregistered student endpoint receiver')
    watch(a.root,cfg)

