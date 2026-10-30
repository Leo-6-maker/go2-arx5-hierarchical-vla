"""One evidenced hinge-reference correction after the closed PORT03 grasp."""
import base64,hashlib,json,math
from copy import deepcopy
from . import cabinet_camera_port03 as previous,night_scope as scope
from .native_hinge_reference import VERSION as REFERENCE_VERSION

VERSION='ROBOCASA_ARX5_PORT04_MEASURED_HINGE_ANCHOR'
FIELD='closed_cabinet_hinge_port04_carry'
PINS=dict(account='014f387d4392f3bed2a6c94974f939eb7bbaace62c84e44324d0e2904e08ebf4',
    lease='749f8d8f2d6cc17de25f95e521b91fe2fd5014d0e54a8ad4a810b3d10136226a',
    terminal='eac473cb56e05ff1681974a8e3397f6e317af6bd6ef462d89daf24046d8824a6',
    episode_terminal='1d591b8135857eb4f44029d22ad63028d173af279d7872a82ea3d8521eb2aed6')


def cards():
    result=previous.cards()
    for side,c in result.items():
        c.update(id='NIGHT30_ROBOCASA_'+side.upper()+'_PORT04_MEASURED_HINGE_ANCHOR',
            port_revision=VERSION,hinge_boundary_reference=REFERENCE_VERSION)
    return result


def predecessor(bodies):
    scope.need(set(bodies)==set(PINS)and all(type(b)is bytes and hashlib.sha256(b).hexdigest()==PINS[k]
        for k,b in bodies.items()),'EXACT_CLOSED_PORT03_HINGE_REFERENCE_FAILURE')
    a,l,t,e=(json.loads(bodies[k])for k in('account','lease','terminal','episode_terminal'))
    scope.validate_account(a)
    scope.need(a['native_adopted']and a['custody']=='ONE_NATIVE_CORE_SOURCE_OWNER'
        and a['charged_upper']['attempts']==4 and l['status']=='CLOSED'and l['all_children_terminal']
        and t['qualified_source_pilots']==0 and len(t['results'])==1 and t['results'][0]['OS_exit_status']==1
        and e['operation']=='ROBOCASA_RIGHT_PORT03_CAMERA'and e['actual_integrations']==5900 and e['actual_RGB_pairs']==138
        and e['error']=='ValueError: DECLARED_HINGE_LIMITS_AND_REFERENCE_SPEEDS_REQUIRED',
        'ONE_CLOSED_REFERENCE_GATE_FAILURE_NOT_UNKNOWN_OR_SAFETY_BYPASS')
    return a


def account(bodies,authority_bytes,*,destination,now):
    scope.successor_authority(authority_bytes);old=predecessor(bodies)
    scope.need(type(destination)is str and destination.endswith('-cabinet-port04-night-source-four01')
        and math.isfinite(now)and now<scope.SUCCESSOR_STOP,'ONE_MEASURED_BOUNDARY_FIX_BEFORE_STOP')
    v=deepcopy(old);v.pop(previous.FIELD);v.pop('qualification_sha256',None)
    v.update(native_adopted=False,native_destination=destination,revision=old['revision']+1,
        custody='CLOSED_CARRIED_TO_NATIVE_OWNER_PENDING_INDEPENDENT_READBACK')
    v[FIELD]={k:base64.b64encode(b).decode()for k,b in bodies.items()}
    scope.validate_account(v);return v


def transfer(old_bytes,new_account_bytes,*,destination):
    scope.need(hashlib.sha256(old_bytes).hexdigest()==PINS['account'],'EXACT_OLD_PORT03_CUSTODY')
    v=scope.validate_account(json.loads(new_account_bytes))
    scope.need(FIELD in v and v['native_destination']==destination and not v['native_adopted'],
        'ONE_UNADOPTED_PORT04_REFERENCE_DESTINATION')
    old=json.loads(old_bytes);old.update(custody='CLOSED_CARRIED_TO_ONE_MEASURED_HINGE_PORT04',
        custody_transferred_destination=destination,custody_transferred_account_sha256=hashlib.sha256(new_account_bytes).hexdigest(),
        revision=old['revision']+1)
    scope.validate_account(old);return old
