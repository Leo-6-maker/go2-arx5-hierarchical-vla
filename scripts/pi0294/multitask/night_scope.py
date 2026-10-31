"""One shared night source allowance in the existing owner's run account.

Pure authority, forecast and reservation arithmetic. Native registry/lease,
node-local input, safety and independent scoring admission remain separate.
This module cannot launch a process, download assets or change any ledger.
"""
from copy import deepcopy
from datetime import datetime
import base64,hashlib,json,math
from pathlib import Path

APPROVAL_SHA='d7ba87c7711b484b71907fb22b7c6b334c5d67dceeef369ecb8ba12cdc8364e4'
SCOPE='PINNED_OPEN_SOURCE_NIGHT_20261030_V1'
TASKS=('ROBOCASA_OPEN_CABINET','LIBERO_BOOK_LEFT','LIBERO_BOOK_RIGHT')
STOP_NEW=datetime.fromisoformat('2026-10-30T08:00:00+08:00').timestamp()
DEADLINE=datetime.fromisoformat('2026-10-30T09:00:00+08:00').timestamp()
GiB=1024**3
CAPS=dict(attempts=12,physics_integrations=600000,RGB_pairs=24000,
    Core_CPU_s=41400,PC_CPU_s=1800,render_device_s=7200,
    Core_peak_bytes=20*GiB,PC_evidence_bytes=2*GiB,transfer_bytes=20*GiB)
SUCCESSOR_AUTH='49506e3a135ad6ad931cd672cb9f0b9f6214e47fb072fd160b3f29b5932f007b'
CLOSED_PARENT='fc92d9548b2e5bb134e4b88880df456f1ebc457d071406a71b5bfb05b3f0f468'
CLOSED_SUPPLEMENT='06e3560d5180ff6ce74a1015f40e8a410fa65c4fbb71ab29b43bfdf9d7f302bd'
SUCCESSOR_STOP=datetime.fromisoformat('2026-10-31T08:00:00+08:00').timestamp()
SUCCESSOR_DEADLINE=datetime.fromisoformat('2026-10-31T09:00:00+08:00').timestamp()
# Four retained <=512MiB child roots plus <=128MiB source/assets and
# <=128MiB owner/account metadata. A separate 768MiB margin makes3GiB.
# This is a phase storage forecast, not a reduction of the shared20GiB cap.
FOUR_TASK_CORE_STORAGE=3*GiB
V7_CARRY='v7_source_development_carry'
V7_POLICY='b57b75cacdc3d8611b9f8d13cd5f4f4e15ed0d4dccd9a69b7469bc62f4c9b99a'
V7_FINITE='v7_finite_source_job'
FINITE_UPPER=dict(attempts=1,physics_integrations=50000,RGB_pairs=2000,
    Core_CPU_s=2840,PC_CPU_s=0,render_device_s=600,
    Core_peak_bytes=512*1024**2,PC_evidence_bytes=0,transfer_bytes=0)


def finite_profile(value):
    """One source-only finite job under V7; no total pool is reissued."""
    need(type(value)is dict and set(value)=={'policy_sha256','created_epoch','stop_new_epoch','deadline_epoch','upper'}
         and value['policy_sha256']==V7_POLICY and value['upper']==FINITE_UPPER,
         'EXACT_V7_FINITE_SOURCE_ONLY_PROFILE')
    need(all(type(x)is int for x in value['upper'].values()),'TYPED_FINITE_SOURCE_UPPERS')
    a,b,c=[value[k]for k in('created_epoch','stop_new_epoch','deadline_epoch')]
    need(all(type(x)in(int,float)and math.isfinite(x)for x in(a,b,c))
         and a<b<=a+300 and b+90<=c<=a+1200,'FINITE_FIVE_MINUTE_START_TWENTY_MINUTE_END')
    return value


def successor_authority(body):
    need(type(body)is bytes and hashlib.sha256(body).hexdigest()==SUCCESSOR_AUTH,
         'EXACT_SECTION5_DIRECT_APPROVAL')
    value=json.loads(body)
    need(value['direct_answer']=='正式批准第5节全文并执行'
         and value['source_pool']=='carry exact prior costs and remaining allowance; no fresh pool'
         and value['S1_closed_T2_quarantined']is True and value['infrastructure_changes']is False,
         'SECTION5_SAME_SOURCE_POOL_AND_SAFETY')
    return value


def need(ok,why):
    if not ok:raise PermissionError('NIGHT_SOURCE_'+why)


def authority(body,*,now,successor_body=None,v7_policy_body=None,job=None):
    need(type(body)is bytes and hashlib.sha256(body).hexdigest()==APPROVAL_SHA,'EXACT_DIRECT_HUMAN_AUTHORITY')
    value=json.loads(body)
    if v7_policy_body is not None:
        need(type(v7_policy_body)is bytes and hashlib.sha256(v7_policy_body).hexdigest()==V7_POLICY,
             'EXACT_V7_POLICY_BYTES_NOT_A_LOOSE_FLAG')
        p=finite_profile(job)
        need(type(now)in(int,float)and math.isfinite(now)and p['created_epoch']<=now<p['deadline_epoch'],
             'CURRENT_V7_FINITE_JOB_WINDOW')
        return value
    need(value['direct_answer']=='正式批准全文并执行'
        and value['aggregate_new_source_caps']==dict(physics_attempts=12,all_physics_integrations=600000,
            RGB_pairs=24000,CPU_core_seconds=43200,active_CPU_cores_per_node=4,
            render_device_occupation_seconds=7200,Core_peak_bytes=20*GiB,
            PC_evidence_bytes=2*GiB,cumulative_transfer_bytes=20*GiB)
        and type(now)in(int,float)and math.isfinite(now)
        and now<(SUCCESSOR_DEADLINE if successor_body is not None else DEADLINE),
        'FINITE_APPROVED_WINDOW_AND_AGGREGATE_CAPS')
    if successor_body is not None:successor_authority(successor_body)
    return value


def add_to_original(original,*,Core_storage_bytes=None):
    """Keep every old obligation inside its old cap before adding this scope."""
    from scripts.pi0294.pv1.future_budget import _vector
    planned_storage=CAPS['Core_peak_bytes']if Core_storage_bytes is None else Core_storage_bytes
    need(planned_storage in(CAPS['Core_peak_bytes'],FOUR_TASK_CORE_STORAGE),'FIXED_DECLARED_SOURCE_STORAGE_PHASE')
    extra=_vector(dict(Core_CPU_s=CAPS['Core_CPU_s'],local_CPU_s=CAPS['PC_CPU_s'],
        episode_starts=12,physics_ticks=600000,engine_starts=12,
        Core_storage_bytes=planned_storage,local_storage_bytes=2*GiB,transfer_bytes=20*GiB))
    need(set(extra)==set(original['dimensions']),'COMPLETE_ORIGINAL_FOURTEEN_DIMENSIONS')
    need(original['cap_changes_required']=={},'NEW_ALLOWANCE_CANNOT_FILL_OLD_SHORTFALL')
    rows={}
    for key,amount in extra.items():
        before=original['dimensions'][key];cap=before['cap']
        need(cap is None and key=='PBS_submissions'and before.get('count_still_accounted')is True
            or type(cap)in(int,float)and before['projected']<=cap,'ALL_OLD_WORK_WITHIN_OLD_CAP')
        rows[key]=dict(original_cap=cap,original_all_work_projected=before['projected'],
            new_named_source_allowance=amount,combined_projected=before['projected']+amount,
            combined_named_ceiling=None if cap is None else cap+amount)
    return dict(schema='NIGHT_SOURCE_ORIGINAL_ALL14_PLUS_DISTINCT_NAMED_ALLOWANCE_V1',
        original_forecast=deepcopy(original),dimensions=rows,source_caps=dict(CAPS),
        render_device_s_separate_from_training_GPU=7200,
        original_shared_caps_changed=False,old_work_may_spend_new_allowance=False,
        new_GR00T_training_PBS_neural_queries=0,native_adopted=False,launch_allowed=False)


def forecast(approval,global_bytes,local_bytes,*,observation,now,successor_body=None):
    authority(approval,now=now,successor_body=successor_body)
    for key,b in [('GLOBAL',global_bytes),('LOCAL',local_bytes)]:
        v=observation[key]
        need(v['sha256']==hashlib.sha256(b).hexdigest()and v['bytes']==len(b)
            and 0<=now-v['observed_epoch']<=120,'ACTUAL_FRESH_COMPLETE_'+key)
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    return add_to_original(first512_forecast(global_bytes,local_bytes),
        Core_storage_bytes=FOUR_TASK_CORE_STORAGE if successor_body is not None else None)


def account_window(v):
    if V7_FINITE in v:
        p=finite_profile(v[V7_FINITE])
        return p['stop_new_epoch'],p['deadline_epoch']
    extension=v.get('section5_successor')
    if extension is None:return STOP_NEW,DEADLINE
    need(extension['authority_sha256']==SUCCESSOR_AUTH,'ACCOUNT_SECTION5_AUTHORITY')
    pb=base64.b64decode(extension['closed_parent_base64'],validate=True)
    sb=base64.b64decode(extension['closed_PC_supplement_base64'],validate=True)
    need(hashlib.sha256(pb).hexdigest()==CLOSED_PARENT
         and hashlib.sha256(sb).hexdigest()==CLOSED_SUPPLEMENT,'EXACT_BOTH_CLOSED_PC_COST_RECORDS')
    parent,supp=map(json.loads,(pb,sb));validate_account(parent);validate_account(supp)
    need(parent['native_adopted']is supp['native_adopted']is False
         and supp['sealed_closed_parent_account_sha256']==CLOSED_PARENT
         and supp['operations'][:len(parent['operations'])]==parent['operations']
         and v['operations'][:len(supp['operations'])]==supp['operations']
         and all(v['charged_upper'][k]>=supp['charged_upper'][k]>=parent['charged_upper'][k]for k in CAPS),
         'CLOSED_AGGREGATE_CARRIED_ONCE_NO_REFUND_OR_DOUBLE_POOL')
    return SUCCESSOR_STOP,SUCCESSOR_DEADLINE


def successor_account(parent_bytes,supplement_bytes,authority_bytes,*,destination,now):
    """Pure new account proposal; the two historical accounts stay closed."""
    successor_authority(authority_bytes)
    need(type(destination)is str and destination.endswith('-night-source-four01')
         and type(now)in(int,float)and math.isfinite(now)and now<SUCCESSOR_STOP,
         'ONE_FOUR_CONDITION_SUCCESSOR_BEFORE_STOP')
    old=json.loads(supplement_bytes)
    out={k:deepcopy(old[k])for k in('scope','caps','approval_sha256','charged_upper','operations')}
    out.update(stop_new_epoch=SUCCESSOR_STOP,deadline_epoch=SUCCESSOR_DEADLINE,
        native_adopted=False,revision=old['revision']+1,
        custody='ROOT_SOURCE_SUCCESSOR_PREPARATION_NO_NATIVE_ADOPTION',native_destination=destination,
        section5_successor=dict(authority_sha256=SUCCESSOR_AUTH,
            closed_parent_base64=base64.b64encode(parent_bytes).decode(),
            closed_PC_supplement_base64=base64.b64encode(supplement_bytes).decode()))
    validate_account(out);return out


def for_root(root):
    """Explicit cold-root authority selection; historical roots keep old clocks."""
    from pathlib import Path
    from types import SimpleNamespace
    import sys
    profile=Path(root)/'SOURCE_V7_FINITE_JOB.json'
    if profile.exists():
        p=finite_profile(json.loads(profile.read_bytes()))
        policy=(Path(root)/'SOURCE_V7_EXECUTION_POLICY.md').read_bytes()
        need(hashlib.sha256(policy).hexdigest()==V7_POLICY,'SEALED_V7_RUNTIME_POLICY')
        return SimpleNamespace(SCOPE=SCOPE,CAPS=CAPS,APPROVAL_SHA=APPROVAL_SHA,
            STOP_NEW=p['stop_new_epoch'],DEADLINE=p['deadline_epoch'],
            authority=lambda original,*,now:authority(original,now=now,v7_policy_body=policy,job=p),
            validate_account=validate_account,reserve=reserve,V7_CARRY=V7_CARRY,
            transfer_development_custody=transfer_development_custody)
    path=Path(root)/'SOURCE_SUCCESSOR_AUTHORITY.json'
    if not path.exists():return sys.modules[__name__]
    body=path.read_bytes();successor_authority(body)
    return SimpleNamespace(SCOPE=SCOPE,CAPS=CAPS,APPROVAL_SHA=APPROVAL_SHA,
        STOP_NEW=SUCCESSOR_STOP,DEADLINE=SUCCESSOR_DEADLINE,
        authority=lambda original,*,now:authority(original,now=now,successor_body=body),
        validate_account=validate_account,reserve=reserve,V7_CARRY=V7_CARRY,
        transfer_development_custody=transfer_development_custody,
        forecast=lambda original,g,l,*,observation,now:forecast(original,g,l,observation=observation,now=now,successor_body=body))


def development_account(account_bytes,lease_bytes,terminal_bytes,policy_bytes,*,destination,now,job=None):
    """Reuse this account after a V7 repair; never issue another source pool.

    Old version-specific carries remain immutable inside the closed account.
    Future source revisions use this same continuation instead of another
    cabinet/book-specific budget class. Native custody still needs onceIO and
    independent old-process/registry/space checks by the existing owner.
    """
    need(type(policy_bytes)is bytes and hashlib.sha256(policy_bytes).hexdigest()==V7_POLICY,
         'EXACT_V7_DEVELOPMENT_POLICY')
    old=validate_account(json.loads(account_bytes));stop,_=account_window(old)
    need(old['native_adopted']is True and old.get('custody')=='ONE_NATIVE_CORE_SOURCE_OWNER'
         and type(destination)is str and destination!=old['native_destination']
         and type(now)in(int,float)and math.isfinite(now)and (job is not None or now<stop),
         'CLOSED_SOURCE_CONTINUATION_WITHIN_EXISTING_WINDOW')
    out=deepcopy(old);out.pop('qualification_sha256',None)
    out.update(native_adopted=False,native_destination=destination,revision=old['revision']+1,
               custody='V7_CLOSED_SOURCE_CUSTODY_PENDING_NATIVE_READBACK')
    if job is not None:
        p=finite_profile(job)
        need(p['created_epoch']<=now<p['stop_new_epoch'],'FRESH_V7_COLD_FINITE_JOB')
        out[V7_FINITE]=deepcopy(p)
        out.update(stop_new_epoch=p['stop_new_epoch'],deadline_epoch=p['deadline_epoch'])
    else:
        need(V7_FINITE not in old,'NEW_FINITE_JOB_PROFILE_REQUIRED_AFTER_CLOSED_V7_JOB')
    out[V7_CARRY]=dict(policy_sha256=V7_POLICY,destination=destination,
        account_base64=base64.b64encode(account_bytes).decode(),
        lease_base64=base64.b64encode(lease_bytes).decode(),
        terminal_base64=base64.b64encode(terminal_bytes).decode())
    validate_account(out);return out


def transfer_development_custody(old_bytes,new_account_bytes,*,destination):
    new=validate_account(json.loads(new_account_bytes));carry=new.get(V7_CARRY,{})
    need(base64.b64decode(carry['account_base64'],validate=True)==old_bytes
         and new['native_destination']==destination and new['native_adopted']is False,
         'EXACT_ONCE_V7_PREDECESSOR_CUSTODY')
    old=validate_account(json.loads(old_bytes))
    need(old.get('custody')=='ONE_NATIVE_CORE_SOURCE_OWNER','OLD_SOURCE_NOT_ALREADY_TRANSFERRED')
    out=deepcopy(old);out.update(custody='CLOSED_CARRIED_TO_ONE_V7_SOURCE_DEVELOPMENT',
        custody_transferred_destination=destination,
        custody_transferred_account_sha256=hashlib.sha256(new_account_bytes).hexdigest(),revision=old['revision']+1)
    validate_account(out);return out


def validate_account(v):
    stop,deadline=account_window(v)
    finite=finite_profile(v[V7_FINITE])if V7_FINITE in v else None
    need(v['scope']==SCOPE and v['caps']==CAPS and v['approval_sha256']==APPROVAL_SHA
        and v['stop_new_epoch']==stop and v['deadline_epoch']==deadline
        and type(v['operations'])is list and type(v['native_adopted'])is bool,
        'ONE_SHARED_ACCOUNT_NOT_PER_TASK_REPLICATED')
    need(set(v['charged_upper'])==set(CAPS)
        and all(type(x)in(int,float)and math.isfinite(x)and 0<=x and (finite is not None or x<=CAPS[k])
            for k,x in v['charged_upper'].items())
        and (finite is not None or v['charged_upper']['Core_CPU_s']+v['charged_upper']['PC_CPU_s']<=43200),
        'AGGREGATE_WHOLE_NIGHT_RESOURCE_BOUNDS')
    need(type(v['revision'])is int and v['revision']>=0,'NONNEGATIVE_ACCOUNT_REVISION')
    if V7_CARRY in v:
        carry=v[V7_CARRY]
        need(set(carry)=={'policy_sha256','destination','account_base64','lease_base64','terminal_base64'}
             and carry['policy_sha256']==V7_POLICY,'EXACT_V7_CLOSED_SOURCE_CARRY')
        ab,lb,tb=[base64.b64decode(carry[k+'_base64'],validate=True)for k in ('account','lease','terminal')]
        old=validate_account(json.loads(ab));lease,terminal=map(json.loads,(lb,tb))
        need(old['native_adopted']is True and lease['status']=='CLOSED'and lease['all_children_terminal']is True
             and lease['run_uuid']==Path(old['native_destination']).name
             and lease['approval_sha256']==old['approval_sha256']
             and terminal['status']=='FINITE_SOURCE_CHAIN_TERMINAL'
             and terminal['closed_lease_sha256']==hashlib.sha256(lb).hexdigest(),
             'ACTUAL_CLOSED_CHAIN_NOT_OLD_FAILURE_PROHIBITION_OR_LIVE_OWNER')
        mutable={'native_destination','native_adopted','revision','custody','qualification_sha256',
                 'charged_upper','operations',V7_CARRY}
        if finite is not None:
            mutable.update({V7_FINITE,'stop_new_epoch','deadline_epoch'})
            need(all(0<=v['charged_upper'][k]-old['charged_upper'][k]<=finite['upper'][k]for k in CAPS),
                 'ONE_FINITE_JOB_DELTA_OLD_COSTS_NEVER_RESET')
        need(all(v.get(k)==value for k,value in old.items()if k not in mutable)
             and v['native_destination']==carry['destination']!=old['native_destination']
             and v['operations'][:len(old['operations'])]==old['operations']
             and all(v['charged_upper'][k]>=old['charged_upper'][k]for k in CAPS),
             'V7_SAME_POOL_ALL_OLD_COSTS_HISTORY_AND_CLOCK_RETAINED')
        return v
    need(finite is None,'FINITE_V7_JOB_REQUIRES_FULL_CLOSED_PREDECESSOR_CARRY')
    if 'closed_cabinet_turn_speed_port05_carry'in v:
        carry=v['closed_cabinet_turn_speed_port05_carry']
        pins=dict(account='2ee24cc0795f6a00067427fc8e28c6a5e347dc38cdfeb7ec3de548cbd6dab885',
            lease='d952cd307e7644cd8c6a66dfaedd2c3dfd947c56e149341c807eea5c0b45c15b',
            terminal='185667a503009dccfa7edc2009c84fc0254287a0e8f2ad3780e4ca8b3ed3229b',
            episode_terminal='bd5d34649b03157da79df9646f0b5a2c277f6fe41a222d65e3d55cbf27c894a9',
            audit='5414c3e395360f0c3aaaefcc39a6737f7bd7527c044875e2de8d761d7a874db1')
        need(set(carry)==set(pins),'EXACT_CLOSED_PORT05_CARRY_FIELDS')
        bodies={k:base64.b64decode(b,validate=True)for k,b in carry.items()}
        need(all(hashlib.sha256(b).hexdigest()==pins[k]for k,b in bodies.items()),'EXACT_CLOSED_PORT04_POOL_AND_SAFE_GRASP_SPEED_FAILURE')
        old=json.loads(bodies['account']);validate_account(old)
        need(not any(k in v for k in('closed_cabinet_hinge_port04_carry','closed_cabinet_camera_port03_carry','closed_cabinet_geometry_repair02_carry','closed_book_unrun_cabinet_carry'))
            and v['operations'][:len(old['operations'])]==old['operations']
            and all(v['charged_upper'][k]>=old['charged_upper'][k]for k in CAPS)
            and v['section5_successor']==old['section5_successor']
            and v['native_destination'].endswith('-cabinet-port05-night-source-four01'),
            'PORT05_RETAINS_EVERY_OLD_COST_AND_CLOCK_ONCE')
        for op in v['operations'][len(old['operations']):]:
            if op['kind']=='ATTEMPT':
                need(op['task']=='ROBOCASA_OPEN_CABINET'and op['operation']in(
                    'ROBOCASA_RIGHT_PORT05_SLOW_TURN','ROBOCASA_LEFT_PORT05_SLOW_TURN'),'ONE_SLOW_REFERENCE_FIX_PER_SIDE_NO_RETRY')
    if 'closed_cabinet_hinge_port04_carry'in v:
        carry=v['closed_cabinet_hinge_port04_carry']
        pins=dict(account='014f387d4392f3bed2a6c94974f939eb7bbaace62c84e44324d0e2904e08ebf4',
            lease='749f8d8f2d6cc17de25f95e521b91fe2fd5014d0e54a8ad4a810b3d10136226a',
            terminal='eac473cb56e05ff1681974a8e3397f6e317af6bd6ef462d89daf24046d8824a6',
            episode_terminal='1d591b8135857eb4f44029d22ad63028d173af279d7872a82ea3d8521eb2aed6')
        need(set(carry)==set(pins),'EXACT_CLOSED_PORT04_CARRY_FIELDS')
        bodies={k:base64.b64decode(b,validate=True)for k,b in carry.items()}
        need(all(hashlib.sha256(b).hexdigest()==pins[k]for k,b in bodies.items()),'EXACT_CLOSED_PORT03_POOL_AND_REFERENCE_FAILURE')
        old=json.loads(bodies['account']);validate_account(old)
        need(not any(k in v for k in('closed_cabinet_camera_port03_carry','closed_cabinet_geometry_repair02_carry','closed_book_unrun_cabinet_carry'))
            and v['operations'][:len(old['operations'])]==old['operations']
            and all(v['charged_upper'][k]>=old['charged_upper'][k]for k in CAPS)
            and v['section5_successor']==old['section5_successor']
            and v['native_destination'].endswith('-cabinet-port04-night-source-four01'),
            'PORT04_RETAINS_EVERY_OLD_COST_AND_CLOCK_ONCE')
        for op in v['operations'][len(old['operations']):]:
            if op['kind']=='ATTEMPT':
                need(op['task']=='ROBOCASA_OPEN_CABINET'and op['operation']in(
                    'ROBOCASA_RIGHT_PORT04_HINGE','ROBOCASA_LEFT_PORT04_HINGE'),'ONE_REFERENCE_FIX_PER_SIDE_NO_RETRY')
    if 'closed_cabinet_camera_port03_carry'in v:
        carry=v['closed_cabinet_camera_port03_carry']
        pins=dict(account='a01769f8f49b73a05166837314d5cb3f45ea76f17f0e04912c7961847542e62f',
            lease='ad536698b7fb3a916fdf226ab38fd8f47dc18001141052b3284791fc9ace2a18',
            terminal='00073f699854b814100c1c881d8c70bc35c8ad8c3e3ae0607afcbdc19caafb68',
            episode_terminal='3d885ed7d0869e9e1b6b5957546a6bc79b55e1e0bbf8de635ae8d984b7aacced')
        need(set(carry)==set(pins),'EXACT_CLOSED_PORT03_CARRY_FIELDS')
        bodies={k:base64.b64decode(b,validate=True)for k,b in carry.items()}
        need(all(hashlib.sha256(b).hexdigest()==pins[k]for k,b in bodies.items()),'EXACT_CLOSED_PORT02_POOL_AND_CAMERA_FAILURE')
        old=json.loads(bodies['account']);validate_account(old)
        need(not any(k in v for k in('closed_cabinet_geometry_repair02_carry','closed_book_unrun_cabinet_carry'))
            and v['operations'][:len(old['operations'])]==old['operations']
            and all(v['charged_upper'][k]>=old['charged_upper'][k]for k in CAPS)
            and v['section5_successor']==old['section5_successor']
            and v['native_destination'].endswith('-cabinet-port03-night-source-four01'),
            'PORT03_RETAINS_ALL_OLD_COSTS_AND_CLOCKS_ONCE')
        for op in v['operations'][len(old['operations']):]:
            if op['kind']=='ATTEMPT':
                need(op['task']=='ROBOCASA_OPEN_CABINET'and op['operation']in(
                    'ROBOCASA_RIGHT_PORT03_CAMERA','ROBOCASA_LEFT_PORT03_CAMERA'),'ONE_CAMERA_FIX_PER_SIDE_NO_RETRY')
    if 'closed_cabinet_geometry_repair02_carry'in v:
        carry=v['closed_cabinet_geometry_repair02_carry']
        pins=dict(account='33b195275c1078221ebc9dee4dccf7dcf063666bb3b234df55b3f3a2661161bc',
            lease='5261f2a5ea196ee2484bd401e1d576df7e8d0d5172f04c3ae38a33727a5e2b70',
            terminal='3f8eb186864be93683b0b9879681e9fc9bb8d0c6a3438af6d59f99f524e6c9e2',
            stop='fb6a5cdeafd653c4cfa9e52dc6f13021c65e0be92feb170d9beb83acc1afa3b3')
        need(set(carry)==set(pins),'EXACT_CLOSED_GEOMETRY_REPAIR_CARRY_FIELDS')
        bodies={k:base64.b64decode(b,validate=True)for k,b in carry.items()}
        need(all(hashlib.sha256(b).hexdigest()==pins[k]for k,b in bodies.items()),'EXACT_CLOSED_CABINET_POOL_AND_STOP')
        old=json.loads(bodies['account']);validate_account(old)
        need('closed_book_unrun_cabinet_carry'not in v
            and v['operations'][:len(old['operations'])]==old['operations']
            and all(v['charged_upper'][k]>=old['charged_upper'][k]for k in CAPS)
            and v['section5_successor']==old['section5_successor']
            and v['native_destination'].endswith('-cabinet-repair02-night-source-four01'),
            'GEOMETRY_REPAIR_CARRIES_EVERY_OLD_UPPER_AND_OPERATION_ONCE')
        for op in v['operations'][len(old['operations']):]:
            if op['kind']=='ATTEMPT':
                need(op['task']=='ROBOCASA_OPEN_CABINET'and op['operation']in(
                    'ROBOCASA_RIGHT_REPAIR02','ROBOCASA_LEFT_REPAIR02'),'ONLY_ONE_EVIDENCED_REPAIR_PER_CABINET_SIDE')
    if 'closed_book_unrun_cabinet_carry'in v:
        # Pure byte validation also runs before a native capsule exists.
        # Keep this gate independent of filesystem imports/caller sys.path.
        carry=v['closed_book_unrun_cabinet_carry']
        pins=dict(account='39153fc64af24103ac9f79ffe0908210c75daf24e959a839aee353d9a9824bcf',
            lease='c2e4eb4413fc23ddef235b8055d5b3b1ea980318ed168527cc57b84e588fe3c1',
            terminal='00f37cb3a692f7cc7e61812e0c4b3226b8320cc85e750357d764333fd298dc46',
            stop='c5cec8f8c78f5707c333b9d58eef4d44bd821ee6b55189aeea09a2edc75cf018')
        need(set(carry)==set(pins),'EXACT_CLOSED_NATIVE_CABINET_CARRY_FIELDS')
        bodies={k:base64.b64decode(b,validate=True)for k,b in carry.items()}
        need(all(hashlib.sha256(b).hexdigest()==pins[k]for k,b in bodies.items()),
             'EXACT_NATIVE_BOOK_POOL_AND_CLOSED_SAFETY_EVIDENCE')
        old=json.loads(bodies['account']);validate_account(old)
        need(v['operations'][:len(old['operations'])]==old['operations']
             and all(v['charged_upper'][k]>=old['charged_upper'][k]for k in CAPS)
             and v['section5_successor']==old['section5_successor']
             and v['native_destination'].endswith('-cabinet-unrun-night-source-four01'),
             'CABINET_CARRY_ALL_OLD_UPPERS_NO_NEW_POOL')
        for op in v['operations'][len(old['operations']):]:
            if op['kind']=='ATTEMPT':
                need(op['task']=='ROBOCASA_OPEN_CABINET'and op['operation']in(
                    'ROBOCASA_RIGHT_FIRST01','ROBOCASA_LEFT_FIRST01'),'ONLY_TWO_UNRUN_CABINET_ATTEMPTS')
    return v


def reserve(v,*,operation,kind,upper,now,task=None,qualification=None):
    """Reserve full upper bounds before effects; UNKNOWN reservations stay paid."""
    validate_account(v)
    stop,deadline=account_window(v)
    need(type(now)in(int,float)and math.isfinite(now)and now<deadline,'HARD_FINISH')
    need(type(operation)is str and operation and all(r['operation']!=operation for r in v['operations']),
        'NO_REPLAY_OR_DUPLICATE_OPERATION')
    need(kind in('PREPARATION','ATTEMPT','CLOSEOUT')and set(upper)<=set(CAPS)
        and all(type(x)is int and 0<=x for x in upper.values()),'FINITE_EXPLICIT_EFFECT_UPPERS')
    if kind=='ATTEMPT':
        need(v['native_adopted']is True and now<stop and task in TASKS
            and qualification==v.get('qualification_sha256')and type(qualification)is str
            and len(qualification)==64,'ACTUAL_ADOPTED_SAFE_FIXED_TASK_BEFORE08')
        need(upper.get('attempts')==1 and 0<upper.get('physics_integrations',0)<=50000
            and 0<upper.get('RGB_pairs',0)<=2000 and 0<upper.get('render_device_s',0)<=600,
            'WARMUP_RESET_SHADOW_FAIL_QUALIFICATION_ALL_IN_ONE_ATTEMPT')
    else:
        need(all(upper.get(k,0)==0 for k in('attempts','physics_integrations','RGB_pairs','render_device_s')),
            'PREPARATION_CANNOT_STEP_OR_RENDER')
    out=deepcopy(v)
    for key,amount in upper.items():out['charged_upper'][key]+=amount
    out['revision']+=1;out['operations'].append(dict(operation=operation,kind=kind,task=task,
        upper=dict(upper),status='RESERVED_UNKNOWN_UNTIL_EXACT_NATIVE_RESULT',started_epoch=now))
    validate_account(out);return out
