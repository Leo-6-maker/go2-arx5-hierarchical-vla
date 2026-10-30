"""One slower turn after the measured PORT04 relative-speed grasp failure."""
import base64,hashlib,json,math
from copy import deepcopy
from . import cabinet_port04 as previous,night_scope as scope

VERSION='ROBOCASA_ARX5_PORT05_SLOWER_TURN_REFERENCE'
FIELD='closed_cabinet_turn_speed_port05_carry'
PINS=dict(account='2ee24cc0795f6a00067427fc8e28c6a5e347dc38cdfeb7ec3de548cbd6dab885',
    lease='d952cd307e7644cd8c6a66dfaedd2c3dfd947c56e149341c807eea5c0b45c15b',
    terminal='185667a503009dccfa7edc2009c84fc0254287a0e8f2ad3780e4ca8b3ed3229b',
    episode_terminal='bd5d34649b03157da79df9646f0b5a2c277f6fe41a222d65e3d55cbf27c894a9',
    audit='5414c3e395360f0c3aaaefcc39a6737f7bd7527c044875e2de8d761d7a874db1')


def cards():
    result=previous.cards()
    for side,c in result.items():
        c.update(id='NIGHT30_ROBOCASA_'+side.upper()+'_PORT05_SLOW_TURN',port_revision=VERSION,
            turn_reference_limits=dict(max_linear_m_s=.03,max_angular_rad_s=.15))
    return result


def predecessor(bodies):
    scope.need(set(bodies)==set(PINS)and all(type(b)is bytes and hashlib.sha256(b).hexdigest()==PINS[k]
        for k,b in bodies.items()),'EXACT_CLOSED_PORT04_GRASP_SPEED_FAILURE')
    a,l,t,e,u=(json.loads(bodies[k])for k in('account','lease','terminal','episode_terminal','audit'))
    scope.validate_account(a)
    scope.need(a['native_adopted']and a['custody']=='ONE_NATIVE_CORE_SOURCE_OWNER'
        and a['charged_upper']['attempts']==5 and l['status']=='CLOSED'and l['all_children_terminal']
        and t['qualified_source_pilots']==0 and len(t['results'])==1 and t['results'][0]['OS_exit_status']==1
        and e['operation']=='ROBOCASA_RIGHT_PORT04_HINGE'and e['actual_integrations']==5950 and e['actual_RGB_pairs']==139
        and e['error']=='PermissionError: ACTUAL_GRASP_LOST_REQUIRES_NEW_RECOVERY_FROM_THIS_STATE'
        and u['original_native_safety_pass']is True and u['qualified_source_pilot']is False,
        'ONLY_ONE_SLOWER_REFERENCE_NOT_RELAXED_GRASP_OR_SAFETY')
    return a


def account(bodies,authority_bytes,*,destination,now):
    scope.successor_authority(authority_bytes);old=predecessor(bodies)
    scope.need(type(destination)is str and destination.endswith('-cabinet-port05-night-source-four01')
        and math.isfinite(now)and now<scope.SUCCESSOR_STOP,'ONE_SLOW_TURN_FIX_BEFORE_STOP')
    v=deepcopy(old);v.pop(previous.FIELD);v.pop('qualification_sha256',None)
    v.update(native_adopted=False,native_destination=destination,revision=old['revision']+1,
        custody='CLOSED_CARRIED_TO_NATIVE_OWNER_PENDING_INDEPENDENT_READBACK')
    v[FIELD]={k:base64.b64encode(b).decode()for k,b in bodies.items()}
    scope.validate_account(v);return v


def transfer(old_bytes,new_account_bytes,*,destination):
    scope.need(hashlib.sha256(old_bytes).hexdigest()==PINS['account'],'EXACT_OLD_PORT04_CUSTODY')
    v=scope.validate_account(json.loads(new_account_bytes))
    scope.need(FIELD in v and v['native_destination']==destination and not v['native_adopted'],
        'ONE_UNADOPTED_PORT05_SLOW_TURN_DESTINATION')
    old=json.loads(old_bytes);old.update(custody='CLOSED_CARRIED_TO_ONE_SLOW_TURN_PORT05',
        custody_transferred_destination=destination,custody_transferred_account_sha256=hashlib.sha256(new_account_bytes).hexdigest(),
        revision=old['revision']+1)
    scope.validate_account(old);return old
