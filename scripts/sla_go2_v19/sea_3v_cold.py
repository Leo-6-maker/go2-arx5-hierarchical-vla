"""Check the frozen three-view mount against sealed Go2 geometry and the official ABI."""

import argparse
from itertools import product
from dataclasses import replace
import os
from pathlib import Path

import numpy as np

from sea_three_view import AUXILIARY_CAMERA, THREE_VIEW_V1, AuxCapture, ThreeViewStream, calibration, calibration_sha256, camera_copy, camera_local
from sea_observation import Frame, prepare_sensor
from source_custody import BASE, read, require, target_case, write_new
from demonstrated_grip import rows
from policy_query import tensor_sha

ROSTER = BASE / "reports/SLA_GO2_V19/SEA0/SEA0_PAIRED_ROSTER.json"


def hit(model, data, origin, point):
    import mujoco as mj

    delta = point-origin
    ray = delta / np.linalg.norm(delta)
    geom = np.zeros(1,dtype=np.int32)
    distance = mj.mj_ray(model,data,origin,ray,None,1,-1,geom)
    return distance, None if distance < 0 else model.geom(int(geom[0])).name


def project(world_from_camera, p):
    local = world_from_camera[:3,:3].T @ (p-world_from_camera[:3,3])
    z = -local[2]
    K = np.asarray(calibration()["K"])
    px = None if z <= 0 else [float(K[0,0]*local[0]/z+K[0,2]),float(-K[1,1]*local[1]/z+K[1,2])]
    return px, px is not None and 8 <= px[0] <= 632 and 8 <= px[1] <= 472


def box_visibility(model, data, optics, name):
    """Ray-read target surface samples, retaining center occlusion separately."""
    geom = data.geom(name)
    center = geom.xpos.copy()
    turn = geom.xmat.reshape(3,3)
    origin = optics[:3,3]
    box = [v for v in product((-1,0,1),repeat=3) if v != (0,0,0)]
    samples = center + (np.asarray(box)*model.geom(name).size) @ turn.T
    passed = []
    for p in samples:
        _,inside = project(optics,p)
        _,first = hit(model,data,origin,p)
        passed.append(inside and first == name)
    return dict(samples=len(samples),first_hits=sum(passed),center_first_geometry=hit(model,data,origin,center)[1])


def geometry(root, dest):
    import mujoco as mj
    from sea2_preflight import sealed_archive
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from demonstrated_grip import pin

    sealed = sealed_archive(root)
    out = []
    with forbid_integration(mj) as integrations:
        for a in read(ROSTER)["go2"]:
            case = target_case(root,a["layout"])
            native = case / read(sealed(case/"NATIVE_REQUEST.json"))["episode"]
            spec = read(sealed(case/"EXECUTION_SPEC.json"))
            raw = next(x for x in rows(sealed(native/"MOBILE_SAMPLES.private.jsonl"))
                       if x["native_end_step"] == a["tick"])
            path = Path(spec["model_path"])
            sha = spec["frozen_files"][str(path)]
            m,opt = camera_copy(pin(path,sha))
            d = mj.MjData(m)
            d.qpos[:] = raw["qpos"]
            mj.mj_fwdPosition(m,d)
            base = d.body("base_link")
            b = np.eye(4)
            b[:3,:3],b[:3,3] = base.xmat.reshape(3,3),base.xpos
            expected = b @ camera_local()
            require(np.allclose(d.cam_xpos[opt],expected[:3,3],atol=1e-9,rtol=0)
                    and np.allclose(d.cam_xmat[opt].reshape(3,3),expected[:3,:3],atol=1e-9,rtol=0),
                    "pose is not rigidly indexed to base")
            origin = expected[:3,3]
            pose = lambda n: d.geom(n).xpos.copy()
            objects = [(k,pose(k)) for k in ("task_cube_geom","task_blue_cube_geom")]
            tip = d.site("x5_ee").xpos.copy()
            target_points = objects + [("current_tool",tip),("20mm_above_proxy",objects[0][1]+[0,0,.02])]
            fields = {}
            for name, point in target_points:
                uv, frame = project(expected,point)
                dist,g = hit(m,d,origin,point)
                delta = np.linalg.norm(point-origin)-dist if dist >= 0 else None
                fields[name] = dict(pixel=uv,inside=frame,
                                    first_ray_geometry=g,hit_before_projection_mm=None if delta is None else float(1000*delta))
            minimum = min((v for axis in (*np.eye(3), *-np.eye(3))
                           for v,_ in (hit(m,d,origin,origin+axis),) if v>=0), default=float("inf"))
            surface = box_visibility(m,d,expected,"task_cube_geom")
            visible_proxy = surface["center_first_geometry"] == "task_cube_geom"
            out.append(dict(id=a["id"],tick=a["tick"],scene_sha256=sha,
                            pose_matches_base_se3=True,current_tool_in_fov=fields["current_tool"]["inside"],
                            center_ray_to_proxy_object=visible_proxy, lens_min_axis_ray_mm=1000*minimum,
                            proxy_surface=surface,proxy_targets=fields))
    require(not integrations,"geometry preflight called the physics integrator")
    require(len(out)==4,"wrong Go2 archive population")
    write_new(dest,dict(
        status="GEOMETRY_ONLY_CAMERA_MOUNT_EVALUATED", mount=calibration(), mount_sha256=calibration_sha256(),
        ready_for_one_render_check=all(a["proxy_surface"]["first_hits"] and a["current_tool_in_fov"]
                                       and a["lens_min_axis_ray_mm"]>20 for a in out),
        old_proxy_scene_rows=out, original_dynamic_model_changed=False, physics_ticks=0,rendered_images=0,model_forwards=0,
    ))


def expect_failure(action):
    try:
        action()
    except (ValueError,TypeError) as exc:
        return str(exc)
    raise AssertionError("invalid camera or clock admitted")


def processor(root,dest):
    from jax import default_backend
    from sea1_observation_preflight import packet, sampler_fields
    from sea1b_panel import registration, source_inputs, verified_processor

    require(default_backend()=="cpu","CPU processor check only")
    frozen, source = registration()
    prepare, _, identity = verified_processor(root,frozen)
    sources, _, virtual_sha = source_inputs(root,source)
    known = {x["id"]:x["tensors"] for x in read(BASE/"reports/SLA_GO2_V19/SEA1/OBSERVATION_INGRESS_ACTUAL.json")["source"]}
    require(all(sampler_fields(prepare(obs[0]))==known[name] for name,obs in sources.items()),"source bypass bytes changed")
    require(all(sampler_fields(prepare_sensor(packet(obs[0],"source",s["id"],s["frame"],"ROBODOJO_RECORDED_30HZ",
                                                    next(x["history_frames"] for x in read(ROSTER)["source"] if x["id"]==s["id"]),
                                                    left_present=True),prepare)[0])==known[s["id"]]
                for s in source["anchors"] for obs in (sources[s["id"]],)), "typed source route changed bytes")
    virtual = sources["S00"][1]["state"][:8]
    base = np.eye(4)
    grid = np.indices((480,640)).sum(axis=0)%256
    synthetic_rgb = np.stack((grid,grid//2,255-grid),axis=-1).astype(np.uint8)
    rgb = np.full((256,256,3),19,dtype=np.uint8)
    raw = ThreeViewStream("cold","Pick up the fruit, lift it above the table and keep holding it.",
                          virtual,[0.,0.,0.],base,virtual_sha)
    clock = "1"*64

    def make(tick, image=synthetic_rgb, name="cold"):
        return AuxCapture(Frame(name,AUXILIARY_CAMERA,tick,image),calibration_sha256(),0,clock)

    def observe(stream,tick,aux,identity=clock):
        stream.observe("cold",tick,.002*tick,rgb,rgb,base,base,base,[.042,.042],auxiliary=aux,sample_sha256=identity)

    observe(raw,0,make(0))
    observe(raw,500,make(500))
    key,o,receipt = raw.begin_query_with_receipt()
    actual_fields = sampler_fields(raw.prepared_latest(prepare)[0])
    require(o["images"]["cam_left_wrist"].shape==(480,640,3)
            and receipt["aux_rgb_sha256"]==tensor_sha(synthetic_rgb)
            and receipt["history_ticks"]==[0]*9
            and actual_fields["current_masks"]=={"base_0_rgb":True,"left_wrist_0_rgb":True,"right_wrist_0_rgb":True},
            "three-view data or clocks at the actual official processor differ")
    sensor = raw._sensor_locked(key)
    controls = dict(
        absent=expect_failure(lambda: observe(raw,1000,None)),
        stale=expect_failure(lambda: observe(raw,1000,make(500))),
        future=expect_failure(lambda: observe(raw,1000,make(1001))),
        foreign=expect_failure(lambda: observe(raw,1000,make(1000,name="other"))),
        substituted_state=expect_failure(lambda: observe(raw,1000,make(1000),"2"*64)),
        zero=expect_failure(lambda: make(500,np.zeros_like(synthetic_rgb))),
        external_camera=expect_failure(lambda: replace(sensor,left=replace(sensor.left,camera="external_oracle"))),
        wrong_slot=expect_failure(lambda: replace(sensor,left=replace(sensor.left,camera="cam_left_wrist"))),
        mask_off=expect_failure(lambda: prepare_sensor(sensor,prepare,left_mask_only=True)),
    )
    observe(raw,1500,make(1500))
    controls["incomplete_history"]=expect_failure(raw.policy_input)
    raw.stop("cold")
    controls["stopped"]=expect_failure(raw.policy_input)
    write_new(dest,dict(
        status="THREE_VIEW_COLD_INGRESS_VERIFIED_SYNTHETIC_AUX_RGB", mount_sha256=calibration_sha256(),
        official_identity=identity, unchanged_source_windows=4, processed_synthetic_rgb_fields=actual_fields,
        controls=controls, history_verified=receipt["history_ticks"], state_digest=receipt["input_fields"]["state"],
        model_forwards=0,physics_ticks=0,actual_rendered_images=0,
    ))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root",type=Path)
    cli.add_argument("mode",choices=("geometry","processor"))
    cli.add_argument("output",type=Path)
    args=cli.parse_args()
    os.umask(0o077)
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU cold only")
    {"geometry":geometry,"processor":processor}[args.mode](args.root.resolve(),args.output.resolve())
