"""Finite GEN1 t=0 inspection through the original scene and Core counters.

No Teacher, exporter, query, dynamics or model loader. Runtime/once allocation
validation precedes importing the native engine. CPU fixtures are not renders.
"""
from copy import deepcopy
import hashlib,json,time
from pathlib import Path
import xml.etree.ElementTree as ET

PACKET='plans/GEN1_V5_PREVIEW_V2.private.json'
PACKET_SHA='952cca224ad3622872d118ab09d779301637991c94417977333aace00d041200'
CAMERAS=('body_rgb','wrist_rgb')


def need(ok,why):
    if not ok:raise PermissionError('GEN1_STATIC_'+why)


def specifications(packet):
    from .spec import validate,VISUALS,digest
    need(packet.get('schema')=='GEN1_V5_PREVIEW_V2_BINDING_V1'
         and packet.get('version')=='GEN1_TRAIN_PREVIEW_V2_NEAR_015_018','EXACT_PREDECLARED_V2_PACKET')
    rows=packet['specs'];need(len(rows)==12,'SIX_GEOMETRIES_TWO_VISUALS')
    groups={}
    for spec in rows:
        validate(spec)
        need(spec['study_split']=='G_TRAIN' and spec['reset_profile']=='LEGACY_ABOVE_SOURCE'
             and spec['visual_profile']in VISUALS,'TRAIN_ONLY_ORIGINAL_RESET')
        key=spec['geometry_id'];by_visual=groups.setdefault(key,{})
        need(spec['visual_profile']not in by_visual,'NO_DUPLICATE_DESIGN')
        by_visual[spec['visual_profile']]=spec
    need(len(groups)==6 and all(set(v)=={'V0','V1'}for v in groups.values()),'EXACT_COMPLETE_FACTORIAL')
    for pair in groups.values():
        a,b=deepcopy(pair['V0']),deepcopy(pair['V1'])
        for row in (a,b):
            for k in ('design_hash','scene_variant_id','visual_profile'):row.pop(k)
        need(a==b,'TABLE_COLOR_IS_ONLY_FACTOR_WITHIN_GEOMETRY')
    return groups


def xml_invariants(xml):
    root=ET.fromstring(xml);table=root.find(".//geom[@name='table']")
    need(table is not None and 'rgba'in table.attrib,'EXACT_TABLE_COLOR_ATTRIBUTE')
    color=table.attrib.pop('rgba')
    return hashlib.sha256(ET.tostring(root)).hexdigest(),color


def zero_state(engine):
    import numpy as np
    from scripts.pi0294.s1_preflight import data_state
    need(engine.steps==0 and engine.data.time==0 and not engine.safety,'NO_INTEGRATION_OR_NATIVE_SAFETY_EVENT')
    state=np.r_[data_state(engine.model,engine.data),data_state(engine.model,engine.ik_data),
                engine.arm_targets,engine.finger_targets]
    need(np.isfinite(state).all(),'FINITE_REAL_NATIVE_INITIAL_STATE')
    return hashlib.sha256(state.tobytes()).hexdigest()


def inspect(card,asset_root,*,construct,guard,save_images=None):
    """Charge *before* each original native constructor, including failures."""
    from scripts.pi0294.scene import make_engine,rendered_field_of_view
    from scripts.pi0294.s1_preflight import fingerprint
    guard();construct();engine=None
    try:
        engine=make_engine(card,asset_root);guard();initial=zero_state(engine)
        snapshot=engine.snapshot();canonical=zero_state(engine)
        # snapshot calls original mj_forward; record that distinct operation
        # rather than claiming it cannot change warmstart/integration state.
        state=canonical;full_before=fingerprint(engine)
        images,geometry=engine.render(snapshot);guard()
        need(zero_state(engine)==state and fingerprint(engine)==full_before,'RENDER_MUTATED_NATIVE_STATE_OR_RNG')
        need(set(images)==set(geometry)==set(CAMERAS),'ORIGINAL_TWO_CAMERAS')
        import numpy as np
        need(all(images[n].shape==(256,256,3) and images[n].dtype==np.uint8 for n in CAMERAS),'ACTUAL_RGB256_UINT8')
        invariant,color=xml_invariants(engine.xml)
        row=dict(initial_before_snapshot_sha256=initial,canonical_t0_state_sha256=canonical,
            original_snapshot_changed_state=initial!=canonical,render_state_unchanged=True,
            integration_steps=0,time_s=0.,native_dimensions=dict(nq=engine.model.nq,nv=engine.model.nv,
                nu=engine.model.nu,nmocap=engine.model.nmocap,neq=engine.model.neq),
            original_initial_IK_error=list(engine.last_ik_error),initial_safety_valid=bool(snapshot['safety_valid']),
            table_rgba_xml=color,all_other_XML_sha256=invariant,cameras=geometry,
            field_of_view=rendered_field_of_view(images,geometry,card),
            raw_RGB_sha256={n:hashlib.sha256(images[n].tobytes()).hexdigest()for n in CAMERAS},
            arm_q=snapshot['arm_q'].tolist(),actual_initial_tcp=snapshot['tcp'].tolist(),
            dynamic_teacher='NOT_RUN',task_success_claim=False)
        if save_images is not None:row['saved_images']=save_images(images)
        return row
    finally:
        if engine is not None:engine.close()


def compare_reset(a,b,*,same_visual):
    fields=('canonical_t0_state_sha256','all_other_XML_sha256','native_dimensions','arm_q','actual_initial_tcp')
    need(all(a[k]==b[k]for k in fields),'NATIVE_RESET_OR_NONVISUAL_MECHANISM_CHANGED')
    # Geometry carries card-specific snapshot_id. Actual optical calibration,
    # projected points and ray visibility must still be exactly identical.
    for name in CAMERAS:
        aa,bb=deepcopy(a['cameras'][name]),deepcopy(b['cameras'][name])
        aa.pop('snapshot_id',None);bb.pop('snapshot_id',None)
        need(aa==bb,'CAMERA_CALIBRATION_OR_GEOMETRY_CHANGED')
    if same_visual:need(a['raw_RGB_sha256']==b['raw_RGB_sha256'],'DUPLICATE_RESET_RENDER_NOT_DETERMINISTIC')
    else:need(a['table_rgba_xml']!=b['table_rgba_xml'],'TABLE_COLOR_FACTOR_NOT_APPLIED')


def inspect_task_pair(spec,asset_root,*,construct,guard,save_images=None):
    """Three finite constructions: repeated T1 and independently reset T2.

    The caller must already hold its own registered native scope. This helper
    is not a launcher and does not expand the old30-constructor V5 static run.
    """
    from .spec import design_cards
    t1,t2=design_cards(spec)
    first=inspect(t1,asset_root,construct=construct,guard=guard,save_images=save_images)
    repeat=inspect(t1,asset_root,construct=construct,guard=guard)
    paired=inspect(t2,asset_root,construct=construct,guard=guard)
    compare_reset(first,repeat,same_visual=True)
    compare_reset(first,paired,same_visual=True)
    return dict(first=first,repeat=repeat,T2=paired,actual_task_pair_reset_checked=True,
                deterministic_repeat_checked=True,constructors=3,integrations=0)


def run(root):
    from scripts.pi0294.pv1.runtime import validate_runtime
    from scripts.pi0294.pv1.successor_core_context import GEN_ROLE
    from scripts.pi0292.durable import read,write,OwnerLock
    from scripts.pi0294.native_dataset import sha
    r=Path(root);manifest=validate_runtime(r,'4080-simulation')
    need(manifest['scope']==GEN_ROLE and manifest['execution']['sealed_inputs'].get(PACKET)==PACKET_SHA
         and sha(r/PACKET)==PACKET_SHA,'OWN_REGISTERED_STATIC_SCOPE_AND_FROZEN_PACKET')
    groups=specifications(read(r/PACKET));output=r/'GEN1_STATIC_RESULT.private.json'
    need(not output.exists() and not (r/'GEN1_STATIC_STARTED.private.json').exists(),'ONCE_STATIC_ATTEMPT_NO_SILENT_REPLAY')
    from scripts.pi0294.pv1.core_ledger import Ledger
    ledger=Ledger(r);guard=ledger.guard;lock=OwnerLock(r/'GEN1_STATIC_OWNER.lock');started=time.process_time()
    write(r/'GEN1_STATIC_STARTED.private.json',dict(source_sha=manifest['code']['vla_commit'],started_epoch=time.time()))
    result=dict(schema='GEN1_V5_NATIVE_STATIC_INSPECTION_V1',status='FAILED_PRESERVED',
        source_sha=manifest['code']['vla_commit'],packet_sha256=PACKET_SHA,variants=[],homes=[],
        expected_images=24,actual_images=0,integrations=0,Teacher_calls=0,GR00T_queries=0,
        C_D_training_data=False,visual_generalisation_proven=False,dynamic_teacher='NOT_RUN')
    try:
        # This is the first native import, after the full registered CPU gate.
        import mujoco
        from PIL import Image
        from .spec import design_cards
        from .validation import common_home_preparation
        assets=read(r/'INPUT_BINDINGS.private.json')['asset_root'];original_step=mujoco.mj_step
        def prohibited_step(*args,**kwargs):raise PermissionError('GEN1_ZERO_INTEGRATION_NO_MJ_STEP')
        mujoco.mj_step=prohibited_step
        try:
            for index,(geometry,pair)in enumerate(groups.items()):
                baseline=None
                for visual,spec in pair.items():
                    card=design_cards(spec)[0]
                    def save_images(images):
                        saved={}
                        for camera,image in images.items():
                            name=f'preview/{index:02d}_{visual}_{camera}.png';p=r/name;p.parent.mkdir(exist_ok=True,mode=0o700)
                            need(not p.exists(),'PRESERVE_EACH_ACTUAL_IMAGE');Image.fromarray(image).save(p);p.chmod(0o400)
                            saved[camera]=dict(relative=name,sha256=sha(p),bytes=p.stat().st_size);result['actual_images']+=1
                        return saved
                    first=inspect(card,assets,construct=lambda:ledger.charge('engine_starts',1),guard=guard,save_images=save_images)
                    again=inspect(card,assets,construct=lambda:ledger.charge('engine_starts',1),guard=guard)
                    compare_reset(first,again,same_visual=True)
                    if baseline is None:baseline=first
                    else:compare_reset(baseline,first,same_visual=False)
                    result['variants'].append(dict(geometry_id=geometry,visual=visual,first=first,
                        duplicate=again,deterministic_reset=True,table_only_invariance=True))
            anchor=design_cards(next(iter(groups.values()))['V0'])[0]
            for home in common_home_preparation()['candidates']:
                card=dict(anchor,initial_tcp=home['initial_tcp'])
                try:
                    a=inspect(card,assets,construct=lambda:ledger.charge('engine_starts',1),guard=guard)
                    b=inspect(card,assets,construct=lambda:ledger.charge('engine_starts',1),guard=guard)
                    compare_reset(a,b,same_visual=True)
                    result['homes'].append(dict(id=home['id'],status='FINITE_INITIAL_STATIC_CHECK_ONLY',
                        initial=a,duplicate=b,approach_path='NOT_RUN',selected=False,dynamic_qualified=False))
                except ValueError as exc:
                    # An unreachable TRAIN candidate is a retained negative,
                    # not a reason to choose a different unregistered home.
                    result['homes'].append(dict(id=home['id'],status='STATIC_INITIAL_IK_REJECT',
                        error=type(exc).__name__+': '+str(exc),selected=False,dynamic_qualified=False))
            need(result['actual_images']==24,'EXACT_TWENTY_FOUR_SAVED_IMAGES')
            result['status']='COMPLETE_STATIC_PREVIEWS_NOT_TEACHER_OR_POLICY_QUALIFICATION'
        finally:mujoco.mj_step=original_step
        return result
    except BaseException as exc:
        result['error']=type(exc).__name__+': '+str(exc);raise
    finally:
        result['actual_constructor_attempts']=read(ledger.path)['engine_starts']
        result['CPU_s']=time.process_time()-started;ledger.cpu(result['CPU_s'])
        write(output,result);output.chmod(0o400);lock.close()
