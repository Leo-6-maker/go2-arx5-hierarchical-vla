"""Publish only a sealed SCALE training-state directory to the approved Hub."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


REPO = 'leoyuer/sla-goal-v2-checkpoints'
SCHEMA = 'SLA_SCALE20K_COMPLETE_STATE_V1'
GROUPS = {'parameters':45,'optimizer':94,'rng':1}
DIGEST = re.compile(r'[0-9a-f]{64}')
DOMAINS = {'target','robodojo','libero'}
BINDING_HASHES = {'parent_manifest_sha256','partition_sha256','initial_state_sha256',
                  'optimizer_sha256','learning_rates_sha256','target_admission_sha256','tokenizer_sha256'}


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def checked(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=digest:
        raise ValueError('CHECKPOINT_PUBLICATION_PIN_CHANGED')
    return json.loads(data)


def inventory(checkpoint, complete, *, step):
    path = checkpoint/'COMPLETE.json'
    state = checked(path,complete)
    if (set(state)!={'schema','binding','step','cursors','arrays','training_full_generations','physics'}
            or state['schema']!=SCHEMA or state['step']!=step or state['physics'] is not False
            or state['training_full_generations']!=step*4 or len(state['arrays'])!=sum(GROUPS.values())
            or set(state['cursors'])!=DOMAINS or any(type(v) is not int or v<0 for v in state['cursors'].values())):
        raise ValueError('PUBLIC_RELEASE_REQUIRES_A_SEALED_TRAINING_ONLY_STATE')
    binding = state['binding']
    if (set(binding)!={*BINDING_HASHES,'arm','plan_sha','recipe','source_sha','data_manifest_sha256','visit_bank_sha256'}
            or binding['arm'] not in ('SCALE-MIX20K','SCALE-TGT5K')
            or not re.fullmatch(r'[A-Za-z0-9_-]{1,90}',binding['recipe'])
            or any(not re.fullmatch(r'[0-9a-f]{40}',binding[k]) for k in ('plan_sha','source_sha'))
            or any(not DIGEST.fullmatch(binding[key]) for key in BINDING_HASHES)
            or any(set(binding[k])!=DOMAINS or not all(DIGEST.fullmatch(v) for v in binding[k].values())
                   for k in ('data_manifest_sha256','visit_bank_sha256'))):
        raise ValueError('PUBLIC_RELEASE_BINDING_MAY_CONTAIN_ONLY_ORIGINAL_HASHES')
    inventory = {'COMPLETE.json':dict(sha256=complete,bytes=path.stat().st_size)}
    groups = Counter()
    for i,row in enumerate(state['arrays']):
        filename = f'{i:06d}.npy'
        member = checkpoint/filename
        key = row['key']
        group = re.match(r"^\['(parameters|optimizer|rng)'\]",key)
        if (set(row)!={'key','file','dtype','shape','sha256'} or not group or len(key)>512
                or not re.fullmatch(r"[\w.\[\]'():, -]+",key,flags=re.ASCII)
                or row['file']!=filename or not DIGEST.fullmatch(row['sha256'])
                or not re.fullmatch(r'[a-z0-9]{1,20}',row['dtype'])
                or not all(type(n) is int and n>=0 for n in row['shape'])
                or member.is_symlink() or not member.is_file() or sha(member)!=row['sha256']):
            raise ValueError('PUBLIC_RELEASE_REFUSES_AN_UNEXPECTED_CHECKPOINT_MEMBER:'+filename)
        groups[group[1]] += 1
        inventory[filename] = dict(sha256=row['sha256'],bytes=member.stat().st_size)
    if groups!=GROUPS or set(p.name for p in checkpoint.iterdir())!=inventory.keys() or path.is_symlink():
        raise ValueError('PUBLIC_RELEASE_REFUSES_EXTRA_FILES_OR_UNKNOWN_STATE_GROUPS')
    return state, inventory


def write_once(path,value):
    with path.open('x') as handle:
        json.dump(value,handle,allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def verify_remote(public, revision, prefix, inventory, cache):
    from huggingface_hub import hf_hub_download

    actual = {item.path:item for item in public.list_repo_tree(REPO,repo_type='dataset',revision=revision,
              path_in_repo=prefix,recursive=True) if hasattr(item,'size')}
    if set(actual)!={prefix+'/'+name for name in inventory}:
        raise ValueError('PUBLIC_RELEASE_FILE_SET_DIFFERS')
    for name,row in inventory.items():
        path = prefix+'/'+name
        item = actual[path]
        if item.size!=row['bytes']:
            raise ValueError('PUBLIC_RELEASE_REMOTE_FILE_SIZE_CHANGED')
        found = getattr(item.lfs,'sha256',None)
        if found is None:
            if item.size>5*1024**2:
                raise ValueError('PUBLIC_RELEASE_LARGE_FILE_REMOTE_DIGEST_UNAVAILABLE')
            found = sha(Path(hf_hub_download(REPO,path,repo_type='dataset',revision=revision,token=False,cache_dir=cache)))
        if found!=row['sha256']:
            raise ValueError('PUBLIC_RELEASE_REMOTE_MEMBER_DIGEST_CHANGED')


def require_cold_producer(config, producer, train):
    if config.get('producer_site','Corelab')=='Corelab':
        lease = json.loads((producer/'control/LEASE.json').read_bytes())
        live = subprocess.check_output(['systemctl','--user','show',lease['unit'],'-p','MainPID','--value'],
                                       text=True,timeout=12).strip()
        if lease['status']!='CLOSED' or live!='0':
            raise PermissionError('PUBLICATION_REQUIRES_COLD_CORELAB_TRAINING')
        return
    if config['producer_site']!='VANDA' or os.environ.get('PBS_JOBID')!=config['job_id']:
        raise PermissionError('PUBLICATION_REQUIRES_EXACT_VANDA_PBS_OWNER')
    own_id, prior_id = config['job_id'],config['producer_job']
    rows = json.loads(subprocess.check_output(['/opt/pbs/bin/qstat','-xf','-F','json',own_id,prior_id],
                                               text=True,stderr=subprocess.PIPE,timeout=25))['Jobs']
    own, prior = rows[own_id],rows[prior_id]
    result_path = Path(config['producer_result']['path']).resolve(strict=True)
    receipt = checked(result_path,config['producer_result']['sha256'])
    hosts = {value.split('/')[0].split('.')[0] for value in own['exec_host'].split('+')}
    if (config['uid']!=os.getuid() or producer.stat().st_uid!=os.getuid() or own_id==prior_id
            or any(job['project']!=config['project'] or job['Job_Owner'].split('@')[0]!=config['user'] for job in (own,prior))
            or own['job_state']!='R' or int(own['Resource_List'].get('ngpus',0))
            or os.uname().nodename.split('.')[0] not in hosts or prior['job_state']!='F' or prior['Exit_status']!=0
            or result_path!=producer/'RESULT.private.json' or receipt['PBS_JOBID']!=prior_id
            or receipt['status']!='FIXED_SEGMENT_SAVED' or receipt['stages']['train']!=train):
        raise PermissionError('PUBLICATION_REQUIRES_OWN_CPU_PBS_AND_EXACT_COLD_VANDA_SEAL')


def run(config):
    root = Path(config['root']).resolve(strict=True)
    if root.stat().st_uid!=os.getuid() or config['repo']!=REPO or (root/'RESULT.private.json').exists():
        raise PermissionError('PUBLIC_RELEASE_REQUIRES_FRESH_APPROVED_PUBLICATION_SCOPE')
    status = dict(status='PUBLIC_CHECKPOINT_NOT_VERIFIED',repo=REPO,step=config['step'],
                  source_sha=config['source_sha'],optimizer_updates=0,native_requests=0,native_ticks=0,model_generations=0)
    try:
        pin = config['train']
        train_path = Path(pin['path']).resolve(strict=True)
        result = checked(train_path,pin['sha256'])
        producer = train_path.parent.parent
        require_cold_producer(config,producer,pin)
        checkpoint = Path(result['checkpoint']['path']).resolve(strict=True)
        if (result['exit_code']!=0 or result['status']!='FIXED_SEGMENT_SAVED' or result['global_step']!=config['step']
                or result['arm']!=config['arm']
                or checkpoint!=train_path.parent/f"step{config['step']:05d}"):
            raise PermissionError('PUBLICATION_REQUIRES_THE_EXACT_COLD_FORMAL_TRAINING_SEAL')
        complete = result['checkpoint']['sha256']
        state, members = inventory(checkpoint,complete,step=config['step'])
        if state['binding']['arm']!=config['arm'] or state['cursors']!=result['global_cursors']:
            raise ValueError('PUBLIC_RELEASE_STATE_AND_TERMINAL_EFFECTS_DISAGREE')
        if (root/'STOP.json').exists() or time.time()+120>=config['deadline_epoch']:
            raise TimeoutError('PUBLICATION_STOP_OR_FINITE_WINDOW')
        os.environ.update(HF_HUB_OFFLINE='0',HF_HUB_DISABLE_XET='1',HF_HUB_ENABLE_HF_TRANSFER='0')
        from huggingface_hub import HfApi,CommitOperationAdd,get_token

        public = HfApi(token=False)
        info = public.repo_info(REPO,repo_type='dataset',timeout=15)
        credential = Path(config['credential_path']).read_text().strip() if 'credential_path' in config else get_token()
        if not credential:
            raise PermissionError('PUBLICATION_REQUIRES_CONFIGURED_SITE_HF_TOKEN')
        api = HfApi(token=credential)
        if info.private is not False or api.whoami()['name']!='leoyuer':
            raise PermissionError('PUBLICATION_REQUIRES_EXPECTED_PUBLIC_REPO_AND_OWNER')
        prefix = f"{config['arm']}/step-{config['step']:05d}-{complete[:12]}"
        if any(name.startswith(prefix+'/') for name in public.list_repo_files(REPO,repo_type='dataset',revision=info.sha)):
            raise FileExistsError('PUBLICATION_IMMUTABLE_PREFIX_ALREADY_EXISTS')
        intent = dict(repo=REPO,head=info.sha,prefix=prefix,checkpoint_sha256=complete,
                      members=len(members),bytes=sum(row['bytes'] for row in members.values()))
        write_once(root/'HF_PUBLISH_INTENT.private.json',intent)
        status.update(intent)
        if (root/'STOP.json').exists():
            raise InterruptedError('PUBLICATION_STOP_BEFORE_EXTERNAL_WRITE')
        ops = [CommitOperationAdd(path_in_repo=prefix+'/'+name,path_or_fileobj=str(checkpoint/name)) for name in members]
        commit = api.create_commit(REPO,repo_type='dataset',parent_commit=info.sha,
                                  commit_message=f"Publish {config['arm']} sealed step {config['step']}",
                                  operations=ops,num_threads=1)
        status['revision'] = commit.oid
        write_once(root/'HF_COMMIT.private.json',dict(intent,revision=commit.oid))
        verify_remote(public,commit.oid,prefix,members,root/'cache')
        status['status'] = 'PUBLIC_CHECKPOINT_FIXED_REVISION_AND_ALL_MEMBER_HASHES_VERIFIED'
        return 0
    except Exception as error:
        status['failure_type'] = type(error).__name__
        status['failure'] = str(error) if type(error).__module__=='builtins' else 'EXTERNAL_FAILURE_INSPECT_PRIVATE_LOG'
        raise
    finally:
        write_once(root/'RESULT.private.json',status)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec',type=Path)
    parser.add_argument('sha256')
    args = parser.parse_args()
    os.umask(0o077)
    raise SystemExit(run(checked(args.spec,args.sha256)))
