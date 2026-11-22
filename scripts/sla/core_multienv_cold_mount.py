"""Register the next original Wave A cards with the existing Core owner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from go2arx5_vla.world_model import multisource_d4_native as d4
from go2arx5_vla.world_model.mfree_recovery_v2 import launch_service, register
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin, validate
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import read, write
from scripts.sla.core_multienv_payload import capacity, payload_volume
from scripts.sla.core_multienv_scope import order
from scripts.sla.core_multienv_train import PLAN, PRODUCER_SPEC
from scripts.sla.core_multienv_wave_suffix import closed_handoff
from scripts.wm02.multitask_stage_cards_v6 import case_card, cold_native_owner, pin, require_unstarted_cases
from scripts.wm02.multitask_teacher_campaign_v6 import interface_records


def prepare(config):
    root = Path(config['root'])
    source = Path(__file__).resolve().parents[2]
    original = read_pin(config['original'])
    terminal = read_pin(config['catalog_terminal'])
    expected = terminal['latest']
    sealed = read_pin(dict(path=expected['path'],sha256=expected['sha256']))
    old = Path(original['registered_root'])
    closure = read_pin(config['source_members'])
    end = min(original['deadline_epoch'],config['deadline'])
    chain = config.get('closed_storage_chain')
    maximum_window = 6*3600 if chain else 3*3600
    if (os.getuid()!=2009 or os.uname().nodename!='corelab-4080s' or root.exists()
            or not root.name.startswith('sla-core-multienv-wave-a-cold-') or root.parent.stat().st_uid!=os.getuid()
            or config['original']['sha256']!=PRODUCER_SPEC or terminal['status']!='FINAL_ORIGINAL_TRAIN_SEALED'
            or sealed['count']!=expected['count'] or sealed['count']!=69 or sealed['windows']!=6081
            or end-time.time() < 3600 or end-time.time() > maximum_window
            or not re.fullmatch('[0-9a-f]{40}',config['source_sha'])
            or set(original['source_files'])-closure.keys()
            or any(digest(source/name)!=d for name,d in closure.items())):
        raise PermissionError('Expected the final original TRAIN seal, finite Core authority and a complete cold source')
    if subprocess.check_output(['loginctl','show-user','2009','-p','Linger'],text=True).strip()!='Linger=yes':
        raise PermissionError('Core user service persistence is required')
    prior_timeout = config.get('closed_storage_predecessor')
    maximum = len(config['conditions']) if chain else 1 if prior_timeout else 2
    if len(config['conditions'])!=maximum or not 1 <= maximum <= 12:
        raise ValueError('Only the bounded original conditions can be staged')
    spec = dict(original,source_sha=config['source_sha'],source_root=str(source),source_files=closure,
        source_seal=hashlib.sha256(json.dumps(closure,sort_keys=True).encode()).hexdigest(),registered_root=str(root),
        campaign_root=str(root),mobile_group=str(root),unattended_control=str(root/'control'),
        collection_root=str(root/'admission'),deadline_epoch=end,wall_limit_s=int(end-time.time()),
        new_physical_requests_budget=maximum,multienv_payload_storage=config['payload_storage'],
        multienv_cold_handoff=dict(producer=config['original'],terminal=config['original_terminal'],
                                   conditions=config['conditions']))
    if prior_timeout:
        spec['multienv_cold_handoff']['closed_storage_predecessor'] = prior_timeout
    if chain:
        spec['multienv_cold_handoff']['closed_storage_chain'] = chain
        spec['multienv_local_archive'] = True
    ledger = closed_handoff(spec)
    if terminal['paid_TRAIN']+(3 if chain else int(prior_timeout is not None))!=ledger['paid_train']:
        raise ValueError('Original ledger and final TRAIN catalog disagree')
    if chain:
        from scripts.sla.core_multienv_n1_storage import capacity as archive_capacity
        from scripts.sla.core_multienv_payload import measure

        reserved = archive_capacity(root.parent,config['payload_storage'],original['multienv'])
        # A waiting owner uses the small local parent reserve. Actual acquisition
        # still requires both raw and archive reserves under the world lock.
        can_register = reserved['payload']['passed'] and measure(root.parent,original['multienv']['snapshot_reserve_bytes'])['passed']
    else:
        reserved = capacity(root.parent,config['payload_storage'],original['multienv'])
        can_register = reserved['passed']
    if not can_register:
        raise PermissionError('Paired local and NFS disk reserves are insufficient')
    payload = payload_volume(root.parent,config['payload_storage'])
    admission = payload/'_admission'
    if (old/'STOP.json').is_symlink() or admission.exists() or any((payload/x).exists() for x in config['conditions']):
        raise FileExistsError('A new cold condition already has an output location')
    dims = read_pin(original['compiled_population'])
    shaped = {row['id']:row for row in order(dims)}
    cards = {row['id']:row for row in read_pin(original['registered_panel'])}
    interface = cold_native_owner(original['interface_predecessor'])
    interface_records(interface,spec['mobile_world_lock'],identity=(d4.E0_SCOPE,d4.E0_PLAN))
    templates = {s['mobile_condition']['family']:s for s in map(read,interface['cases'])}
    dirs(root)
    (root/'cases').mkdir()
    admission.mkdir(mode=0o700)
    (root/'admission').symlink_to(admission,target_is_directory=True)
    auth = read_pin(original['sim_only_project_authorization'])
    auth.update(source_sha=config['source_sha'],frozen_producer=config['source_sha'],deadline_epoch=end)
    auth_path = root/'AUTHORITY.private.json'
    write(auth_path,auth)
    spec['sim_only_project_authorization']=pin(auth_path)
    spec.update({name:str(root/name) for name in ('tmp','cache','logs','data')})
    spec['native_environment']=dict(original['native_environment'],PYTHONPATH=os.pathsep.join(
        (str(source),str(source/'src'),original['native_environment']['PYTHONPATH'])),
        TMPDIR=spec['tmp'],XDG_CACHE_HOME=spec['cache'],ROS_LOG_DIR=str(root/'logs/ros'))
    cases=[]
    for cid in config['conditions']:
        row=shaped[cid]
        folder, case = case_card(spec,templates[row['family']],cards[cid],spec['registered_panel'])
        case.update(registry_scope=spec['scope'],collection_split=row['split'],compiled_population=spec['compiled_population'],
            expert_training_eligible=row['split']=='TRAIN',multienv=spec['multienv'],
            allocation_deadline_epoch=min(end,time.time()+2160),deadline_epoch=min(end,time.time()+2160)-60)
        if not chain:
            case['multienv_payload_storage'] = spec['multienv_payload_storage']
        validate(case)
        cases.append(str(folder/'EXECUTION_SPEC.json'))
    spec['cases']=cases
    require_unstarted_cases(spec['registry'],set(config['conditions']))
    register(spec,[(root,spec)])
    write(root/'PREFLIGHT.private.json',dict(plan=PLAN,previous=ledger['original_records_sha256'],first=config['conditions'],
        baseline=expected['sha256'],native_requests=0,maximum_new_requests=maximum,source_sha=spec['source_sha'],
        reserved=reserved))
    return launch_service(spec,'wm02-'+root.name,'--core-hopper-collection',bootstrap='v6_bootstrap.py')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('sha')
    a=parser.parse_args()
    os.umask(0o077)
    print(json.dumps(prepare(read_pin(dict(path=a.config,sha256=a.sha)))))
