"""Numerical scene candidates for the frozen PR425 population."""
from copy import deepcopy
from hashlib import sha256
import json
from math import cos, hypot, radians, sin, sqrt
from pathlib import Path

GOAL = 'SLA-CORE-MULTIENV-DATA-V1'
POPULATION_SHA = 'ceb8b60269625c5211ed2c572a10f2b34c3755a0b321f08e857bc10bde9449b5'
COLORS = ('red', 'blue')
FIELDS = ('common_world_x_mm', 'common_world_y_mm', 'relative_parallel_mm',
          'relative_perp_mm', 'red_yaw_deg', 'blue_yaw_deg')


def unit(scene, index, field):
    payload = f"{GOAL}|{scene['id']}|{scene['scene_seed']}|{index}|{field}"
    return int.from_bytes(sha256(payload.encode('utf-8')).digest()[:8], 'big') / 2**64


def candidate(scene, index, card):
    if not 0 <= index < 64 or scene['stratum'] not in ('NEAR_NEG','NEAR_POS','FAR_NEG','FAR_POS'):
        raise ValueError('Unregistered geometry candidate')
    distance,side = scene['stratum'].split('_')
    bounds = [(-6,6), (-6,6), (-8,-4) if distance=='NEAR' else (4,8),
              (-6,-2) if side=='NEG' else (2,6), (-10,10), (-10,10)]
    values = {field: low+(high-low)*unit(scene,index,field)
              for field,(low,high) in zip(FIELDS,bounds,strict=True)}
    position = card['object_positions_world']
    ex,ey = (position['blue'][k]-position['red'][k] for k in (0,1))
    length = hypot(ex,ey)
    if length <= 0:
        raise ValueError('Frozen template red and blue positions coincide')
    ex,ey = ex/length,ey/length
    parallel,perp = values['relative_parallel_mm'],values['relative_perp_mm']
    dx,dy = parallel*ex-perp*ey,parallel*ey+perp*ex
    x,y = values['common_world_x_mm'],values['common_world_y_mm']
    shifts = {'red':(x-dx/2,y-dy/2), 'blue':(x+dx/2,y+dy/2)}
    if max(abs(v) for point in shifts.values() for v in point)>13:
        raise ValueError('Candidate left the preregistered 13 mm envelope')
    yaw_mode = scene['yaw_mode']
    if yaw_mode not in ('ZERO','INDEPENDENT_PM10_DEG'):
        raise ValueError('Unknown preregistered yaw mode')
    result = deepcopy(card)
    result['object_yaw_degrees'] = {}
    for name in COLORS:
        result['object_positions_world'][name] = [
            a+d/1000 for a,d in zip(position[name],(*shifts[name],0),strict=True)
        ]
        yaw = 0.0 if yaw_mode=='ZERO' else values[name+'_yaw_deg']
        result['object_yaw_degrees'][name] = yaw
        values[name+'_yaw_deg'] = yaw
    return dict(index=index,fields=values,world_offset_mm=shifts,card=result)


def near_duplicate(left, right):
    # All objects stand on the same support. SO(3) distance for these pure yaws is the wrapped yaw difference.
    lpos,rpos = (x['object_positions_world'] for x in (left,right))
    ly,ry = (x.get('object_yaw_degrees',{}) for x in (left,right))
    return all(sqrt(sum((a-b)**2 for a,b in zip(lpos[c],rpos[c],strict=True)))<=.002
               and abs((ly.get(c,0)-ry.get(c,0)+180)%360-180)<=2 for c in COLORS)


def boxes_overlap(card):
    # Separating-axis check for the two actual yawed 40 mm square footprints.
    axes = {}
    for color in COLORS:
        yaw = radians(card['object_yaw_degrees'][color])
        axes[color] = ((cos(yaw),sin(yaw)),(-sin(yaw),cos(yaw)))
    pos = card['object_positions_world']
    delta = [pos['blue'][i]-pos['red'][i] for i in (0,1)]
    dot = lambda a,b: a[0]*b[0]+a[1]*b[1]
    directions = [axis for color in COLORS for axis in axes[color]]
    return not any(abs(dot(delta,axis)) >= .02*sum(abs(dot(other,axis)) for other in directions)
                   for axis in directions)


def rotated_xml(template, base, card):
    """Update the same two passive free bodies and their stored reset quaternion."""
    from xml.etree import ElementTree as ET
    import numpy as np
    from go2arx5_vla.world_model.multitask_scene_v6 import OBJECTS, numbers, overlay_xml

    xml, binding = overlay_xml(template,base,card)
    root = ET.fromstring(xml)
    qs = {}
    for color in COLORS:
        angle = radians(card['object_yaw_degrees'][color])/2
        q = [cos(angle),0.,0.,sin(angle)]
        name = OBJECTS[color]['body']
        body = root.find(f"./worldbody/body[@name='{name}']")
        if body is None or body.find('freejoint') is None:
            raise ValueError('Expected the original passive cube free body')
        body.set('quat',numbers(q))
        qs[color] = q
    for key in root.findall('./keyframe/key'):
        values = np.fromstring(key.get('qpos'),sep=' ')
        if values.shape!=(41,):
            raise ValueError('Unexpected protected keyframe ABI')
        values[30:34] = qs['red']
        values[37:41] = qs['blue']
        key.set('qpos',numbers(values))
    return ET.tostring(root,encoding='unicode'),binding,qs


def load_population(path: Path):
    raw = path.read_bytes()
    if sha256(raw).hexdigest()!=POPULATION_SHA:
        raise ValueError('PR425 logical population changed')
    doc = json.loads(raw)
    if doc['goal']!=GOAL or len(doc['scenes'])!=88 or len(doc['collection'])!=528:
        raise ValueError('Frozen scene and condition count mismatch')
    if len({row['reset_seed'] for row in doc['scenes']})!=88:
        raise ValueError('New layouts share a reset identity')
    return doc


def compile_static(root, population, baseline_pin, old_native_pin, deadline):
    """Compile the new population against the fixed old compiler and protected scene identities."""
    import time
    import mujoco
    import numpy as np
    from numpy.testing import assert_allclose
    from go2arx5_vla.world_model.multitask_scene_v6 import OBJECTS, robot_contract
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from scripts.sla.hopper_parent_custody import digest, write_once
    from scripts.sla.hopper_site import pinned
    from scripts.wm02.multitask_scene_audit_v6 import asset_binding

    if mujoco.__version__!='3.2.7' or mujoco.get_mjcb_control() is not None:
        raise PermissionError('Cold geometry requires the frozen MuJoCo version without a controller')
    baseline = pinned(baseline_pin)
    legacy_hold = Path(baseline_pin['path']).parent/'population/COMPILED_POPULATION.private.json'
    inventory = pinned(baseline['split_inventory'])
    history = [*inventory['original_populations'],inventory['new_evaluation_population'],old_native_pin,
               {'path':str(legacy_hold),'sha256':'297b21cc67baa4b2cbdd78af68d244b550337b4cb9213fd27b26818e3958bec0'}]
    history.extend(ref for name,ref in baseline.items() if name.endswith('_population'))
    protected = set(inventory['protected_scene_hashes'])
    protected.update(c['sha256'] for c in pinned(inventory['C20_scene_catalog']))
    protected.update(c['scene']['sha256'] for c in pinned(inventory['V52_split'])['layouts'])
    occupied = {'F1':[], 'F2':[]}
    resets = set()
    for pin in {x['sha256']:x for x in history}.values():
        historic = pinned(pin)
        for previous in historic['clusters']:
            protected.add(previous['scene']['sha256'])
            resets.add(previous['reset'])
            card = previous.get('card',{})
            if set(card.get('object_positions_world',{}))==set(COLORS):
                occupied[previous['family']].append(card)
    templates = {}
    for family,ref in baseline['templates'].items():
        template = pinned(ref)
        template['asset_root'],_ = asset_binding(template)
        model = template['model_path']
        if digest(Path(model)) != template['frozen_files'][model]:
            raise ValueError('Frozen template hash changed')
        templates[family] = (template,mujoco.MjModel.from_binary_path(model))
    doc = load_population(population)
    rows = []
    for scene in doc['scenes']:
        if time.time()>=deadline or (root/'STOP').exists():
            raise TimeoutError('Geometry compilation reached its fixed wall-clock limit')
        if scene['reset_seed'] in resets:
            raise ValueError('New reset identity overlaps a registered layout')
        template,base = templates[scene['family']]
        attempted=[]
        compiled = None
        for index in range(64):
            proposal = candidate(scene,index,baseline['prototype_cards'][scene['family']])
            card = proposal['card']
            if any(near_duplicate(card,a) for a in occupied[scene['family']]):
                attempted.append(dict(index=index,status='NEAR_DUPLICATE'))
                continue
            # Check both rotated footprints with the old 5 mm support clearance.
            table = base.geom('task_table').id
            center,half = base.geom_pos[table],base.geom_size[table]
            fits=True
            for color in COLORS:
                yaw=radians(card['object_yaw_degrees'][color])
                footprint=.02*(abs(cos(yaw))+abs(sin(yaw)))+.005
                pos=np.asarray(card['object_positions_world'][color])
                fits &= bool(np.all(abs(pos[:2]-center[:2])+footprint<=half[:2]))
            if not fits:
                attempted.append(dict(index=index,status='ROTATED_SUPPORT_CLEARANCE'))
                continue
            if boxes_overlap(card):
                attempted.append(dict(index=index,status='ROTATED_FOOTPRINT_OVERLAP'))
                continue
            try:
                xml,bindings,quaternions = rotated_xml(template,base,card)
            except ValueError as exc:
                if str(exc) not in ('FULL_INITIAL_OBJECT_FOOTPRINT_ON_UNCHANGED_SUPPORT_REQUIRED',
                                    'TWO_INITIAL_BOXES_MUST_NOT_OVERLAP'):
                    raise
                attempted.append(dict(index=index,status=str(exc)))
                continue
            with forbid_integration(mujoco) as calls:
                model=mujoco.MjModel.from_xml_string(xml)
                bodies = {c:model.body(OBJECTS[c]['body']).id for c in COLORS}
                # Compare the protected original fields with only the declared cube orientation excluded.
                rq=bodies['red']
                actual_red=model.body_quat[rq].copy()
                model.body_quat[rq]=base.body_quat[base.body(OBJECTS['red']['body']).id]
                protected_robot=robot_contract(base,model)
                model.body_quat[rq]=actual_red
                for c,i in bodies.items():
                    joint=int(model.body_jntadr[i]);address=int(model.jnt_qposadr[joint])
                    assert_allclose(model.body_quat[i],quaternions[c],atol=1e-12,rtol=0)
                    assert_allclose(model.qpos0[address:address+3],card['object_positions_world'][c],atol=1e-12,rtol=0)
                    assert_allclose(model.qpos0[address+3:address+7],quaternions[c],atol=1e-12,rtol=0)
                    assert_allclose(model.key_qpos[:,address+3:address+7],
                                    np.tile(quaternions[c],(model.nkey,1)),atol=1e-12,rtol=0)
                    assert_allclose(model.geom_quat[model.geom(OBJECTS[c]['geom']).id],[1,0,0,0],atol=1e-12,rtol=0)
                out=root/(scene['id']+'.mjb')
                mujoco.mj_saveModel(model,str(out),None)
                if calls:
                    raise RuntimeError('Unexpected native integration during cold compilation')
                sha=digest(out)
            if sha in protected:
                attempted.append(dict(index=index,status='PROTECTED_GEOMETRY_HASH'))
                out.unlink()
                continue
            protected.add(sha)
            resets.add(scene['reset_seed'])
            occupied[scene['family']].append(card)
            attempted.append(dict(index=index,status='COMPILED'))
            compiled = dict(manifest=dict(path=str(out),sha256=sha),card=card,quaternion_wxyz=quaternions,
                            robot_fingerprint=protected_robot['fingerprint'],field_values=proposal['fields'],
                            scoring_geometry=bindings)
            break
        rows.append(dict(scene,compiled=compiled,candidates=attempted,status='COMPILED' if compiled else 'STATIC_UNBOUND'))
        write_once(root/(scene['id']+'.json'),rows[-1])
    result=dict(status='COLD_GEOMETRY_PASS',population_sha256=POPULATION_SHA,scenes=rows,
                conditions=len(doc['collection']),new_integrations=0,native_requests=0,model_generations=0)
    write_once(root/'RESULT.json',result)
    return result
