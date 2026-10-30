"""One fixed-home correction for the closed PORT02 wrist-view rejection."""
import base64,hashlib,json,math
from copy import deepcopy
from . import cabinet_repair02 as previous,night_scope as scope

VERSION='ROBOCASA_ARX5_PORT03_FIXED_CAMERA_HOME'
FIELD='closed_cabinet_camera_port03_carry'
PINS=dict(account='a01769f8f49b73a05166837314d5cb3f45ea76f17f0e04912c7961847542e62f',
    lease='ad536698b7fb3a916fdf226ab38fd8f47dc18001141052b3284791fc9ace2a18',
    terminal='00073f699854b814100c1c881d8c70bc35c8ad8c3e3ae0607afcbdc19caafb68',
    episode_terminal='3d885ed7d0869e9e1b6b5957546a6bc79b55e1e0bbf8de635ae8d984b7aacced')


def cards():
    result=previous.cards();t=math.sqrt(.5)
    for side,c in result.items():
        c.update(id='NIGHT30_ROBOCASA_'+side.upper()+'_PORT03_FIXED_CAMERA_HOME',port_revision=VERSION,
            initial_tcp=[[0.,-t,t,.30],[0.,t,t,.30 if side=='right'else-.30],[-1.,0.,0.,.59],[0.,0.,0.,1.]])
    # This changes only fixed declared initial poses. Original camera frames,
    # FOV and PORT02 withdrawal/fixture/grasp/score remain unchanged.
    return result


def predecessor(bodies):
    scope.need(set(bodies)==set(PINS)and all(type(b)is bytes and hashlib.sha256(b).hexdigest()==PINS[k]
        for k,b in bodies.items()),'EXACT_CLOSED_PORT02_CAMERA_FAILURE_FOR_ONE_FIX')
    a,l,t,e=(json.loads(bodies[k])for k in('account','lease','terminal','episode_terminal'))
    scope.validate_account(a)
    scope.need(a['native_adopted']and a['custody']=='ONE_NATIVE_CORE_SOURCE_OWNER'
        and a['charged_upper']['attempts']==3 and l['status']=='CLOSED'and l['all_children_terminal']
        and t['qualified_source_pilots']==0 and len(t['results'])==1 and t['results'][0]['OS_exit_status']==1
        and e['operation']=='ROBOCASA_RIGHT_REPAIR02'and e['actual_integrations']==0 and e['actual_RGB_pairs']==1
        and e['error']=='PermissionError: QUALIFICATION_OBJECT_OUTSIDE_DECLARED_DUAL_CAMERA_VIEW',
        'CLOSED_CAMERA_REJECTION_NOT_PHYSICAL_FAILURE_OR_UNKNOWN')
    return a


def account(bodies,authority_bytes,*,destination,now):
    scope.successor_authority(authority_bytes);old=predecessor(bodies)
    scope.need(type(destination)is str and destination.endswith('-cabinet-port03-night-source-four01')
        and math.isfinite(now)and now<scope.SUCCESSOR_STOP,'ONE_FIXED_HOME_CAMERA_CORRECTION_BEFORE_STOP')
    v=deepcopy(old);v.pop(previous.FIELD);v.pop('qualification_sha256',None)
    v.update(native_adopted=False,native_destination=destination,revision=old['revision']+1,
        custody='CLOSED_CARRIED_TO_NATIVE_OWNER_PENDING_INDEPENDENT_READBACK')
    v[FIELD]={k:base64.b64encode(b).decode()for k,b in bodies.items()}
    scope.validate_account(v);return v


def transfer(old_bytes,new_account_bytes,*,destination):
    scope.need(hashlib.sha256(old_bytes).hexdigest()==PINS['account'],'EXACT_OLD_PORT02_CUSTODY')
    v=scope.validate_account(json.loads(new_account_bytes))
    scope.need(FIELD in v and v['native_destination']==destination and not v['native_adopted'],
        'ONE_UNADOPTED_PORT03_CAMERA_DESTINATION')
    old=json.loads(old_bytes);old.update(custody='CLOSED_CARRIED_TO_ONE_FIXED_CAMERA_PORT03',
        custody_transferred_destination=destination,custody_transferred_account_sha256=hashlib.sha256(new_account_bytes).hexdigest(),
        revision=old['revision']+1)
    scope.validate_account(old);return old
