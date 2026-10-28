"""Reviewable unexecuted scene/split list; no renderer, teacher or model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from object_transfer_teacher import GRASP_TCP_OFFSET_M, TIMEOUTS

# Fixed proposals, chosen before any physics or rendered/model scores.
# (object x shift, source y, tray y, tray center separation / 2)
LAYOUTS = [(-.010, -.180, .180, .140), (.010, -.170, .190, .150),
           (-.005, -.160, .200, .145), (.005, -.150, .210, .155),
           (-.012, -.175, .185, .148), (.012, -.165, .195, .152),
           (-.008, -.155, .205, .142), (.008, -.145, .215, .158),
           (-.006, -.185, .175, .146), (.006, -.175, .185, .154),
           (-.014, -.165, .195, .144), (.014, -.155, .205, .156),
           (-.011, -.172, .188, .149), (.011, -.162, .198, .151),
           (-.007, -.152, .208, .143), (.007, -.142, .218, .157)]
TRAY_WALL_HEIGHT_M = .04


def build():
    families, slots = [], []
    rotation = np.diag([1., -1., -1.]).tolist()
    for i, (shift, source_y, tray_y, sep) in enumerate(LAYOUTS):
        fid = f'OBJECT-TRANSFER-FAMILY-{i+1:03d}'
        split = 'DEV' if i < 4 else 'TRAIN' if i < 12 else 'TASK_VAL'
        positions = [[-.10+shift, source_y, .665], [.10+shift, source_y+.02, .665]]
        colors = ('red', 'blue') if i % 2 == 0 else ('blue', 'red')
        objects = {c: dict(instance_id=f'{fid}-object-{c}', position_m=p, dimensions_m=[.04, .03, .03], mass_kg=.05) for c, p in zip(colors, positions)}
        trays = dict(left=dict(interior_center_xy_m=[-sep, tray_y], floor_z_m=.65),
                     right=dict(interior_center_xy_m=[sep, tray_y+.01], floor_z_m=.65))
        for tray in trays.values():
            xy = np.asarray(tray['interior_center_xy_m'])
            tray['interior_bounds_m'] = [np.r_[xy-.08, .65].tolist(), np.r_[xy+.08, .65+TRAY_WALL_HEIGHT_M].tolist()]
        families.append(dict(family=fid, split=split, objects=objects, trays=trays,
                             table_bounds_m=[[-.6, -.6, .60], [.6, .6, .65]],
                             tray_interior_xy_m=[.16, .16], tray_wall_height_m=TRAY_WALL_HEIGHT_M, tray_wall_thickness_m=.008,
                             initial_tcp_position_m=[0, -.38, .85], initial_tcp_rotation=rotation, initial_gripper_width_m=.08))
        for c in ('red', 'blue'):
            for side in ('left', 'right'):
                source = np.array(objects[c]['position_m'])
                target = np.r_[trays[side]['interior_center_xy_m'], .665]
                bounds = np.asarray(trays[side]['interior_bounds_m'])
                half = np.asarray(objects[c]['dimensions_m'])/2
                assert np.all(target-half >= bounds[0]-1e-12) and np.all(target+half <= bounds[1]+1e-12)
                # Necessary lower bounds only: they omit contact dynamics and tracking lag.
                travel = float(np.linalg.norm(target-source))
                assert travel/.10+.2 < TIMEOUTS[4]
                assert (.08-.03)/.04+.2 < TIMEOUTS[2]
                assert .07/.04+.2 < TIMEOUTS[6]
                assert .12/.10+2. < TIMEOUTS[7]
                slots.append(dict(slot=len(slots), episode_id=f'{fid}-{c}-{side}', family=fid, split=split,
                                  target_color=c, target_tray=side,
                                  language=f'Move the {c} object into the {side} bin, release it, and withdraw.',
                                  ideal_transfer_min_s=travel/.10, attempts=1, replacement_allowed=False))
    assert len(families)==16 and len(slots)==64
    assert len({tuple(tuple(o['position_m']) for o in f['objects'].values()) for f in families})==16
    assert len({o['instance_id'] for f in families for o in f['objects'].values()})==32
    for f in families:
        p = [np.array(o['position_m']) for o in f['objects'].values()]
        assert np.linalg.norm(p[1]-p[0]) > .10
        assert f['trays']['left']['interior_center_xy_m'][0] < f['trays']['right']['interior_center_xy_m'][0]
    return dict(schema='OBJECT_TRANSFER_UNEXECUTED_PROPOSAL_DRAFT_V1', status='REVIEWABLE_PRE_PHYSICS_NOT_AUTHORIZED',
                charter_release='ed398fee17151cfec47864fed9d48ce149b28d61', charter_issue11=5994405991,
                launch_allowed=False, training_launch_allowed=False, scene_families=families, teacher_slots=slots,
                split_before_episodes_and_windows=True, commands_per_family=4, red_blue_are_not_distinct_behavior_families=True,
                teacher_data_filters_by_target_wbc_success=False, actor_inputs=['body_rgb','wrist_rgb','state_v1','language'],
                task_val_claim='new nearby layouts of the same cube shapes/colors; not blind TEST or unseen object classes',
                left_right_semantics='fixed proposed body camera image order; mobile/WBC task-frame binding remains separate',
                proposal_static_bounds='NECESSARY_ONLY_NOT_COLLISION_OR_OBSERVATION_QUALIFICATION',
                grasp_tcp_offset_m=GRASP_TCP_OFFSET_M,
                tray_score_geometry=dict(wall_height_m=TRAY_WALL_HEIGHT_M, full_object_height_m=.03,
                                         previous_wall_height_m=.02, previous_full_3D_containment_impossible=True,
                                         bounds_explicit_per_tray=True, physics_or_contact_qualified=False),
                proposed_camera=dict(body_eye_world_m=[0,-.50,1.0], body_look_at_world_m=[0,0,.70], body_fov_deg=55,
                                     wrist_eye_from_tcp_m=[.06,0,-.03], wrist_look_at_from_tcp_m=[0,0,GRASP_TCP_OFFSET_M], wrist_fov_deg=65,
                                     resolution=[96,96], actual_visibility='NOT_RUN', design_frozen_for_execution=False),
                proposed_budget=dict(teacher_episodes=64, task_limit_s=30, terminal_record_s=2,
                                     physics_dt_s=1/120, physics_ticks=245760, rgb_pairs=40960,
                                     model_queries=0, optimizer_updates=0, WBC_episodes=0, hardware_runs=0),
                actual_runs=dict(teacher_episodes=0, physics_steps=0, renders=0, model_queries=0, optimizer_updates=0),
                selected_checkpoint='NONE', deployment_selected_checkpoint='NONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    draft = build()
    draft['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(draft, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=draft['status'], families=16, teacher_slots=64, launch_allowed=False,
                         proposal_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest()), sort_keys=True))
