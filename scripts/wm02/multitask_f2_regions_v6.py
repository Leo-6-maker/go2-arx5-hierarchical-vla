"""Bounded F2 visual-region layout comparison using one saved task-start pose."""
import copy
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image

from scripts.pi0292.durable import write
from go2arx5_vla.world_model.multitask_camera_v6 import TwoObjectCameraSession
from go2arx5_vla.world_model.multitask_contact_v6 import actuation_source
from go2arx5_vla.world_model.multitask_scene_v6 import overlay_xml, robot_contract
from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.world_model.storage import digest
from .multitask_binding_audit_v6 import compile_candidate, pinned, recorded_prefix
from .multitask_scene_audit_v6 import asset_binding


# Both boxes and their mass/geometry stay fixed. Candidates move only the
# existing visual geoms and preserve left/right with respect to the +Y axis.
REGIONS = (
    (.280,.400,.198),
    (.310,.430,.190),
    (.350,.470,.190),
    (.370,.490,.155),
    (.390,.510,.198),
    (.335,.455,.155),
)


def run(spec, root):
    import mujoco
    if spec['physics'] or spec['new_physical_requests_budget'] or mujoco.get_mjcb_control() is not None:
        raise PermissionError('COLD_F2_VISUAL_LAYOUT_ONLY')
    outer, outer_identity = compile_candidate(spec['outer_build'],root,'task_actuation',actuation_source)
    outer.WM02OuterNext  # Resolve the real library export without invoking any step.
    item = spec['F2']
    base = json.loads(pinned(item['static_result']).read_bytes())
    row = next(r for r in base['records'] if r['family']=='F2')
    template = json.loads(pinned(row['template']).read_bytes())
    asset_root, _ = asset_binding(template)
    template['asset_root'] = asset_root
    original_path = template['model_path']
    if digest(original_path) != row['original_scene_sha256']:
        raise ValueError('OLD_F2_MJB_BINDING_CHANGED')
    original = mujoco.MjModel.from_binary_path(original_path)
    candidate_base = mujoco.MjModel.from_binary_path(str(pinned(row['model'])))
    for packet, packet_digest, prefix_sha in recorded_prefix(Path(item['recorded_journal'])):
        pass
    if packet_digest != item['query0_state_sha256'] or prefix_sha.hexdigest() != item['original_prefix_sha256']:
        raise ValueError('EXACT_SAVED_F2_QUERY0_AND_PREFIX_REQUIRED')

    results = []
    for index,(left_x,right_x,forward_y) in enumerate(REGIONS):
        card = copy.deepcopy(row['card'])
        card['zones']['left']['center_world_XY'] = [left_x,forward_y]
        card['zones']['right']['center_world_XY'] = [right_x,forward_y]
        folder = root / f'layout-{index:02d}'
        folder.mkdir()
        xml, geometry = overlay_xml(template,original,card)
        with forbid_integration(mujoco) as calls:
            model = mujoco.MjModel.from_xml_string(xml)
            contract = robot_contract(original,model)
            if not all(np.array_equal(getattr(model,name),getattr(candidate_base,name))
                       for name in ('key_qpos','key_qvel','key_ctrl','geom_contype','geom_conaffinity')):
                raise ValueError('NONCONTACT_MARKER_ONLY_LAYOUT_CHANGE_REQUIRED')
            new_file = folder/'SCENE.private.mjb'
            mujoco.mj_saveModel(model,str(new_file),None)
            clone = dict(packet,qpos=np.r_[packet['qpos'][:27],model.key_qpos[0,27:]],
                         qvel=np.r_[packet['qvel'][:26],np.zeros(12)],contacts=[])
            session = TwoObjectCameraSession(new_file)
            try:
                images, body, _, _ = session.render_memory(clone)
                masks = session.render_copy.visible_masks(clone)
                Image.fromarray(body).save(folder/'body_rgb.private.png')
                Image.fromarray(images['wrist_rgb']).save(folder/'wrist_rgb.private.png')
            finally:
                session.close()
            if calls:
                raise RuntimeError('COLD_REGION_AUDIT_ATTEMPTED_PHYSICS')
        roi = masks['body_rgb']
        # Rank only declared visual properties from one saved pose. Any
        # subsequent native outcome uses the preselected immutable scene.
        scores = [roi[k] for k in ('red','blue','left_region','right_region')]
        result = dict(index=index,zones=geometry['zones'],visible_pixels=masks,
            minimum_body_visible_pixels=min(scores),model=dict(path=str(new_file),sha256=digest(new_file)),
            geometry=geometry,robot_fingerprint=contract['fingerprint'],
            XML_sha256=hashlib.sha256(xml.encode()).hexdigest())
        (folder/'SCENE.private.xml').write_text(xml)
        results.append(result)
        write(root/'PROGRESS.private.json',dict(completed=len(results),new_integrations=0))
    best=max(results,key=lambda x:(x['minimum_body_visible_pixels'],-x['index']))
    return dict(exit_code=0,status='F2_COLD_VISUAL_PLACEMENT_COMPARISON_COMPLETE',records=results,
        selected_candidate=best if best['minimum_body_visible_pixels']>0 else None,
        comparison='maximize minimum red/blue/left/right body segmentation pixels, then earlier listed index',
        recorded_pose_sha256=packet_digest,outer_candidate=outer_identity)


def main(spec):
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    root=Path(spec['registered_root'])
    start,cpu=time.monotonic(),time.process_time()
    try:
        result=run(spec,root)
    except BaseException:
        result=dict(exit_code=1,status='PRESERVED_COLD_F2_REGION_TECHNICAL_FAILURE',first_failure=traceback.format_exc())
    result.update(source_sha=spec['source_sha'],wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,
        native_dispatch_qualified=False,new_native_requests=0,new_native_ticks=0,new_updates=0,new_model_generations=0)
    write(root/'RESULT.private.json',result)
    close(spec,result['exit_code'])
    return result['exit_code']
