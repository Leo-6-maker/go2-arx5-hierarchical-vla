"""Unrun cabinet conditions only; carry the one source pool after book stop.

Pure account and closed-evidence checks. The stopped book role/STOP and all
its costs stay intact. Native custody transfer uses the original onceIO.
"""
import base64,hashlib,json,math
from copy import deepcopy
from . import night_scope as scope

PINS={
 'account':'39153fc64af24103ac9f79ffe0908210c75daf24e959a839aee353d9a9824bcf',
 'lease':'c2e4eb4413fc23ddef235b8055d5b3b1ea980318ed168527cc57b84e588fe3c1',
 'terminal':'00f37cb3a692f7cc7e61812e0c4b3226b8320cc85e750357d764333fd298dc46',
 'stop':'c5cec8f8c78f5707c333b9d58eef4d44bd821ee6b55189aeea09a2edc75cf018',
}
FIELD='closed_book_unrun_cabinet_carry'
need=scope.need


def predecessor(bodies):
    need(set(bodies)==set(PINS)and all(type(b)is bytes and hashlib.sha256(b).hexdigest()==PINS[k]
         for k,b in bodies.items()),'EXACT_CLOSED_NATIVE_SOURCE_POOL_AND_BOOK_STOP')
    a,l,t,s=(json.loads(bodies[k])for k in('account','lease','terminal','stop'))
    scope.validate_account(a)
    attempts=[x for x in a['operations']if x['kind']=='ATTEMPT']
    need(a['native_adopted']is True and a['custody']=='ONE_NATIVE_CORE_SOURCE_OWNER'
         and a['charged_upper']['attempts']==1 and len(attempts)==1
         and attempts[0]['operation']=='LIBERO_LEFT_FIRST01'
         and attempts[0]['task']=='LIBERO_BOOK_LEFT'
         and l['status']=='CLOSED'and l['all_children_terminal']is True
         and t['status']=='FINITE_SOURCE_CHAIN_TERMINAL'and t['qualified_source_pilots']==0
         and len(t['results'])==1 and t['results'][0]['task_kind']=='BOOK'
         and t['results'][0]['side']=='left'and t['results'][0]['OS_exit_status']==1
         and s['reason']=='RuntimeError: ILLEGAL_CONTACT_OR_PENETRATION'
         and s['operation']=='LIBERO_LEFT_FIRST01','ONLY_UNRUN_DISTINCT_CABINET_NOT_BOOK_RETRY')
    return a


def account(bodies,authority_bytes,*,destination,now):
    scope.successor_authority(authority_bytes);old=predecessor(bodies)
    need(type(destination)is str and destination.endswith('-cabinet-unrun-night-source-four01')
         and type(now)in(int,float)and math.isfinite(now)and now<scope.SUCCESSOR_STOP,
         'ONE_CABINET_BRANCH_BEFORE_ORIGINAL_SUCCESSOR_STOP')
    v=deepcopy(old)
    v.update(native_adopted=False,native_destination=destination,
        custody='CLOSED_CARRIED_TO_NATIVE_OWNER_PENDING_INDEPENDENT_READBACK',revision=old['revision']+1)
    v.pop('qualification_sha256',None)
    v[FIELD]={k:base64.b64encode(b).decode()for k,b in bodies.items()}
    scope.validate_account(v);return v


def validate_carry(v):
    if FIELD not in v:return
    bodies={k:base64.b64decode(b,validate=True)for k,b in v[FIELD].items()};old=predecessor(bodies)
    need(v['operations'][:len(old['operations'])]==old['operations']
         and all(v['charged_upper'][k]>=old['charged_upper'][k]for k in scope.CAPS)
         and v['section5_successor']==old['section5_successor']
         and v['native_destination'].endswith('-cabinet-unrun-night-source-four01'),
         'CABINET_CARRY_ALL_NATIVE_UNKNOWN_UPPERS_NO_NEW_POOL')
    for op in v['operations'][len(old['operations']):]:
        if op['kind']=='ATTEMPT':
            need(op['task']=='ROBOCASA_OPEN_CABINET'
                 and op['operation']in('ROBOCASA_RIGHT_FIRST01','ROBOCASA_LEFT_FIRST01'),
                 'CABINET_CARRY_CANNOT_RETRY_BOOK_OR_ADD_CONDITIONS')


def transfer(old_bytes,new_account_bytes,*,destination):
    """Pure old custody close; native caller must verify closed actors/lease."""
    need(hashlib.sha256(old_bytes).hexdigest()==PINS['account'],'EXACT_OLD_NATIVE_CUSTODY')
    v=json.loads(new_account_bytes);validate_carry(v)
    need(v['native_destination']==destination and v['native_adopted']is False,
         'ONE_UNADOPTED_NATIVE_CABINET_DESTINATION')
    old=json.loads(old_bytes);old.update(custody='CLOSED_CARRIED_TO_ONE_UNRUN_CABINET_BRANCH',
        custody_transferred_destination=destination,custody_transferred_account_sha256=hashlib.sha256(new_account_bytes).hexdigest(),
        revision=old['revision']+1)
    scope.validate_account(old);return old
