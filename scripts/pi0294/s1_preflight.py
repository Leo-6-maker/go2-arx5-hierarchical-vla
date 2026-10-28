"""Simulation-privileged prefix8 rejection; never alters live actions or state."""
import copy,hashlib,pickle,random
import numpy as np

VERSION='PI0294_R3_D0_S1_COPYDATA_V1'


def data_state(model,data):
    import mujoco
    kind=mujoco.mjtState.mjSTATE_INTEGRATION
    value=np.empty(mujoco.mj_stateSize(model,kind));mujoco.mj_getState(model,data,value,kind)
    return value


def fingerprint(engine):
    python={k:v for k,v in engine.__dict__.items() if k not in ('model','data','ik_data','renderer')}
    return hashlib.sha256(data_state(engine.model,engine.data).tobytes()
        +data_state(engine.model,engine.ik_data).tobytes()+pickle.dumps(python)
        +pickle.dumps(random.getstate())+pickle.dumps(np.random.get_state())).hexdigest()


def clone(engine):
    if hasattr(engine,'finger_driver') or engine.mode!='normal':raise ValueError('S1_ORIGINAL_D0_ONLY')
    shadow=object.__new__(type(engine))
    for name,value in engine.__dict__.items():
        if name=='model':setattr(shadow,name,value)  # read-only model shared, all data private
        elif name in ('data','ik_data'):setattr(shadow,name,copy.copy(value))
        elif name=='renderer':setattr(shadow,name,None)
        else:setattr(shadow,name,copy.deepcopy(value))
    for name in ('data','ik_data'):
        a,b=getattr(engine,name),getattr(shadow,name)
        for field in ('qpos','qvel','ctrl','qacc_warmstart','act','qfrc_applied','xfrc_applied','userdata'):
            if np.shares_memory(getattr(a,field),getattr(b,field)):raise ValueError('S1_DATA_ALIAS:'+name+'.'+field)
        if not np.array_equal(data_state(engine.model,a),data_state(shadow.model,b)):
            raise ValueError('S1_INTEGRATION_STATE_COPY_MISMATCH')
    return shadow


def observed_state(engine):
    return np.r_[data_state(engine.model,engine.data),data_state(engine.model,engine.ik_data),
        engine.arm_targets,engine.finger_targets,
        np.asarray(engine._ik_target).ravel(),engine._cached_ik_solution,
        np.asarray(engine._cached_ik_error),float(bool(engine.safety)),engine.steps]


def predict(engine,poses,opening,*,nonce,guard,charge):
    """Caller has performed the original prefix-start snapshot exactly once."""
    poses,opening=np.asarray(poses),np.asarray(opening)
    if poses.shape!=(8,4,4) or opening.shape!=(8,) or not np.isfinite(poses).all() or not np.isfinite(opening).all():
        raise ValueError('S1_EXACT_FINITE_PREFIX8_REQUIRED')
    if not isinstance(nonce,str) or not nonce:raise ValueError('S1_NONCE_REQUIRED')
    before=fingerprint(engine);shadow=clone(engine);rows=[];error=None
    digest=hashlib.sha256(poses.tobytes()+opening.tobytes()).hexdigest()
    charge(0,invocation=True)
    try:
        for i in range(200):
            guard()
            if i and i%25==0:shadow.snapshot()
            previous=shadow.steps
            try:shadow.step(poses[i//25],float(opening[i//25]))
            except Exception as exc:error=f'{type(exc).__name__}: {exc}'
            actual=shadow.steps-previous
            if actual!=1:raise RuntimeError('S1_SHADOW_NATIVE_STEP_ACCOUNTING_UNCERTAIN')
            charge(1,invocation=False)
            shadow.snapshot()  # exact original record_tick call, no rendering
            rows.append(observed_state(shadow))
            if error:break
        result=dict(version=VERSION,nonce=nonce,state_sha256=before,action_sha256=digest,
            decision='REJECT' if error or shadow.safety else 'PASS',
            error=error,predicted_native_safety=list(shadow.safety),shadow_integrations=len(rows),
            first_predicted_violation_offset_tick=len(rows) if shadow.safety else None,
            privileged_simulation_state=True,actions_modified=False,live_integrations=0)
        if fingerprint(engine)!=before:raise RuntimeError('S1_PREFLIGHT_CHANGED_LIVE_STATE_OR_RNG')
        result['live_state_unchanged']=True
        return result,np.asarray(rows)
    finally:shadow.close()


def bind_install(engine,poses,opening,result,nonce):
    digest=hashlib.sha256(np.asarray(poses).tobytes()+np.asarray(opening).tobytes()).hexdigest()
    if (result['decision']!='PASS' or result['nonce']!=nonce or result['state_sha256']!=fingerprint(engine)
            or result['action_sha256']!=digest):
        raise PermissionError('S1_STALE_STATE_NONCE_OR_ACTION_REJECT')
