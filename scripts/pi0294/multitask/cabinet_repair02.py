"""One explicit geometry repair after the closed PORT01 initial collision.

The original cards remain unchanged. This source-port development version
changes only placement, common home, grasp and released withdrawal reference.
Robot/driver/fixture primitives, safety and independent score stay frozen.
"""
import base64,hashlib,json,math
from copy import deepcopy
import numpy as np
from .robocasa_cabinet_cards import cards as original_cards
from . import night_scope as scope

VERSION='ROBOCASA_ARX5_PORT02_GEOMETRY_REPAIR'
FIELD='closed_cabinet_geometry_repair02_carry'
PINS=dict(account='33b195275c1078221ebc9dee4dccf7dcf063666bb3b234df55b3f3a2661161bc',
    lease='5261f2a5ea196ee2484bd401e1d576df7e8d0d5172f04c3ae38a33727a5e2b70',
    terminal='3f8eb186864be93683b0b9879681e9fc9bb8d0c6a3438af6d59f99f524e6c9e2',
    stop='fb6a5cdeafd653c4cfa9e52dc6f13021c65e0be92feb170d9beb83acc1afa3b3')


def cards():
    result=original_cards();home=np.eye(4)
    home[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]];home[:3,3]=[.30,0.,.59]
    t=math.sqrt(.5)
    for side,c in result.items():
        shift=.20 if side=='right'else-.20
        c.update(id='NIGHT30_ROBOCASA_'+side.upper()+'_PORT02_GEOMETRY_REPAIR',
            port_revision=VERSION,initial_tcp=home.tolist(),withdraw_tcp_rotation_world=home[:3,:3].tolist())
        c['cabinet']['cabinet_world'][1][3]+=shift
        c['door']['hinge_world'][1][3]+=shift;c['source_base_card']['goal_xy'][1]+=shift
        c['door']['object_to_tcp_grasp']=[[0.,-1.,0.,0.],[t,0.,t,0.],[-t,0.,t,0.],[0.,0.,0.,1.]]
    return result


def checked_card(card,side):
    if side not in('left','right'):raise ValueError('EXACT_CABINET_SIDE')
    if card.get('port_revision')=='ROBOCASA_ARX5_PORT04_MEASURED_HINGE_ANCHOR':
        from .cabinet_port04 import cards as boundary_cards
        if card!=boundary_cards()[side]:raise PermissionError('EXACT_SEPARATE_PORT04_MEASURED_HINGE_ANCHOR')
        return True
    if card.get('port_revision')=='ROBOCASA_ARX5_PORT03_FIXED_CAMERA_HOME':
        from .cabinet_camera_port03 import cards as camera_cards
        if card!=camera_cards()[side]:raise PermissionError('EXACT_SEPARATE_PORT03_FIXED_CAMERA_HOME')
        return True
    repair=card.get('port_revision')==VERSION
    if card!=(cards()if repair else original_cards())[side]:
        raise PermissionError('EXACT_PORT01_OR_SEPARATE_PORT02_CABINET_CARD')
    return repair


def attempt_operation(card,side):
    repair=checked_card(card,side)
    suffix='_PORT04_HINGE'if card.get('port_revision')=='ROBOCASA_ARX5_PORT04_MEASURED_HINGE_ANCHOR'else'_PORT03_CAMERA'if card.get('port_revision')=='ROBOCASA_ARX5_PORT03_FIXED_CAMERA_HOME'else'_REPAIR02'if repair else'_FIRST01'
    return 'ROBOCASA_'+side.upper()+suffix


def predecessor(bodies):
    scope.need(set(bodies)==set(PINS)and all(type(b)is bytes and hashlib.sha256(b).hexdigest()==PINS[k]
        for k,b in bodies.items()),'EXACT_CLOSED_CABINET_INITIAL_FAILURE_FOR_ONE_REPAIR')
    a,l,t,s=(json.loads(bodies[k])for k in('account','lease','terminal','stop'))
    scope.validate_account(a)
    scope.need(a['native_adopted']is True and a['custody']=='ONE_NATIVE_CORE_SOURCE_OWNER'
        and a['charged_upper']['attempts']==2 and l['status']=='CLOSED'and l['all_children_terminal']
        and t['qualified_source_pilots']==0 and len(t['results'])==1
        and t['results'][0]['side']=='right'and t['results'][0]['OS_exit_status']==1
        and s['operation']=='ROBOCASA_RIGHT_FIRST01'
        and s['reason']=='PermissionError: ACTUAL_NATIVE_INITIAL_SAFETY_OR_UNINTENDED_GRASP',
        'ONE_EXACT_CLOSED_INITIAL_GEOMETRY_FAILURE_NOT_UNKNOWN_OR_HEALTHY')
    return a


def account(bodies,authority_bytes,*,destination,now):
    scope.successor_authority(authority_bytes);old=predecessor(bodies)
    scope.need(type(destination)is str and destination.endswith('-cabinet-repair02-night-source-four01')
        and math.isfinite(now)and now<scope.SUCCESSOR_STOP,'ONE_COLD_EVIDENCED_GEOMETRY_REPAIR_BEFORE_STOP')
    v=deepcopy(old);v.pop('closed_book_unrun_cabinet_carry');v.pop('qualification_sha256',None)
    v.update(native_adopted=False,native_destination=destination,revision=old['revision']+1,
        custody='CLOSED_CARRIED_TO_NATIVE_OWNER_PENDING_INDEPENDENT_READBACK')
    v[FIELD]={k:base64.b64encode(b).decode()for k,b in bodies.items()}
    scope.validate_account(v);return v


def transfer(old_bytes,new_account_bytes,*,destination):
    scope.need(hashlib.sha256(old_bytes).hexdigest()==PINS['account'],'EXACT_OLD_CABINET_CUSTODY')
    v=scope.validate_account(json.loads(new_account_bytes))
    scope.need(FIELD in v and v['native_destination']==destination and not v['native_adopted'],
        'ONE_UNADOPTED_GEOMETRY_REPAIR_DESTINATION')
    old=json.loads(old_bytes);old.update(custody='CLOSED_CARRIED_TO_ONE_EVIDENCED_CABINET_REPAIR02',
        custody_transferred_destination=destination,custody_transferred_account_sha256=hashlib.sha256(new_account_bytes).hexdigest(),
        revision=old['revision']+1)
    scope.validate_account(old);return old
