"""Small pure contracts shared by execution and its CPU boundary fixtures."""
import copy
import numpy as np

def route_mode(short_status,*,native_safety=False,operator=False):
    if native_safety:return 'NOT_RUN_SAFETY_STOP'
    if short_status=='PASS_SHORT_MILESTONES_AND_INTEGRITY':return 'AUTO'
    return 'ASSISTED' if operator else 'NOT_AVAILABLE_OPERATOR'

def slot_cost(kind,slot,counters,existing=()):
    if kind+'/'+str(slot) in existing:raise PermissionError('MPS2_FIRST_INTERNAL_SLOT_PRESERVED')
    definitions={
        'IK':(12,dict(goal_IK_calls=1,CLIK_reserved=16),dict(goal_IK_calls=2664,CLIK_reserved=42624)),
        'RRT':(8,dict(RRT_calls=1),dict(RRT_calls=1776)),
        'DENSE':(2,dict(dense_reserved_points=1001),dict(dense_reserved_points=221221))}
    if kind not in definitions:raise ValueError('MPS2_SLOT_KIND')
    bound,cost,limits=definitions[kind]
    if not 0<=slot<bound:raise PermissionError('MPS2_INTERNAL_SLOT_CAP')
    for name,n in cost.items():
        if counters.get(name,0)+n>limits[name]:raise PermissionError('MPS2_GLOBAL_'+name)
    return cost

def first_valid(rows):
    return next((r for r in rows if r['status']=='PASS'),None)

class Ghost:
    def __init__(self,actual):
        self.actual=copy.deepcopy(actual);self.target=np.array(actual['tcp'],copy=True);self.edits=[];self.opening=1.
    def edit(self,translation,rotation,*,opening=None,actor='HUMAN'):
        from .mps2_geometry import rodrigues
        if actor!='HUMAN':raise PermissionError('MPS2_REAL_HUMAN_REQUIRED')
        t=np.asarray(translation,dtype=float);r=np.asarray(rotation,dtype=float)
        if len(self.edits)>=128:raise PermissionError('MPS2_HUMAN_EDIT_CAP')
        if t.shape!=(3,) or r.shape!=(3,) or not np.isfinite(np.r_[t,r]).all():raise ValueError('MPS2_FINITE_GHOST_EDIT')
        if np.linalg.norm(t)>.02+1e-12 or np.linalg.norm(r)>.15+1e-12:raise ValueError('MPS2_GHOST_STEP_CAP')
        if opening is not None and (not np.isfinite(opening) or not 0<=opening<=1):raise ValueError('MPS2_GHOST_OPENING')
        self.target[:3,3]+=t
        if np.linalg.norm(r)>0:self.target[:3,:3]=rodrigues(r,np.linalg.norm(r))@self.target[:3,:3]
        if opening is not None:self.opening=float(opening)
        self.edits.append(dict(translation=t.tolist(),rotation=r.tolist(),opening=self.opening,actor=actor))
        return self.target.copy()
    def command(self):return dict(target=self.target.tolist(),opening=self.opening,action_type='EEF_REFERENCE_ONLY')

def validate_command(value):
    if set(value)-{'target','opening','action_type'} or value.get('action_type')!='EEF_REFERENCE_ONLY':raise PermissionError('MPS2_NO_JOINT_ROOT_OBJECT_COMMAND')
    p=np.asarray(value['target'])
    if p.shape!=(4,4) or not np.isfinite(p).all():raise ValueError('MPS2_FINITE_TARGET')
    if not np.allclose(p[3],[0,0,0,1]) or not np.allclose(p[:3,:3].T@p[:3,:3],np.eye(3),atol=1e-8) or np.linalg.det(p[:3,:3])<0:raise ValueError('MPS2_TARGET_ROTATION')
    if not 0<=value['opening']<=1:raise ValueError('MPS2_OPENING')
    return p
