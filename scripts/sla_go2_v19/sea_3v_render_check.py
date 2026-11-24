"""Acquire four authorized auxiliary RGB frames from sealed, paused Go2 states."""

import argparse
import os
from pathlib import Path
import socket
import time

import numpy as np

from sea_three_view import AUXILIARY_CAMERA, THREE_VIEW_V1, AuxRGBRenderer, calibration_sha256
from sea_observation import Frame, SensorObservation, prepare_sensor
from source_custody import BASE, digest, pin, read, require, target_case, target_obs, write_new
from policy_query import input_identity, tensor_sha

ROSTER = BASE / "reports/SLA_GO2_V19/SEA0/SEA0_PAIRED_ROSTER.json"
SLOTS = ("G00-00", "G00-01", "G01-00", "G01-01")
ID = "THREE-VIEW-V1-SEALED-RGB-01"


def typed(obs, episode, row, aux):
    return SensorObservation(
        episode,row["id"],row["tick"],"GO2_NATIVE_2MS",
        tuple(Frame(episode,"cam_high",t,x) for t,x in zip(row["history_ticks"],obs["images"]["cam_high"],strict=True)),
        Frame(episode,"cam_right_wrist",row["tick"],obs["images"]["cam_right_wrist"]),
        aux,obs["state"],obs["prompt"],THREE_VIEW_V1,
    )


def allowed(grant_path, sha, geometry_path, processor_path, output):
    grant = read(pin(grant_path,sha))
    geo = read(geometry_path)
    proc = read(processor_path)
    require(not output.exists() and socket.gethostname().split(".")[0]=="corelab-4080s"
            and grant.get("id")==ID and grant.get("status")=="APPROVED" and bool(grant.get("authorization_id"))
            and grant.get("slots")==list(SLOTS) and grant.get("rgb_cap")==grant.get("segmentation_cap")==4
            and grant.get("new_integrations")==grant.get("forwards")==0
            and grant.get("mount_sha256")==geo["mount_sha256"]==proc["mount_sha256"]==calibration_sha256()
            and grant.get("renderer_sha256")==digest(Path(__file__))
            and grant.get("geometry_sha256")==digest(geometry_path)
            and grant.get("processor_sha256")==digest(processor_path)
            and geo["ready_for_one_render_check"]
            and proc["status"]=="THREE_VIEW_COLD_INGRESS_VERIFIED_SYNTHETIC_AUX_RGB"
            and grant["valid_until_epoch"]>time.time(),"invalid bounded camera allocation")
    return grant


def render(root, output, geometry_path, processor_path, grant_path, grant_sha):
    import mujoco as mj
    from OpenGL import GL
    from go2arx5_vla.world_model.multitask_state_v6 import packets
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from sea2_preflight import sealed_archive

    grant = allowed(grant_path,grant_sha,geometry_path,processor_path,output)
    sealed = sealed_archive(root)
    population = read(ROSTER)["go2"]
    require([a["id"] for a in population]==list(SLOTS),"fixed archived population changed")
    output.mkdir(mode=0o700)
    write_new(output/"START.json",dict(panel=ID,slots=list(SLOTS),allocation_sha256=grant_sha,
                                      geometry_sha256=digest(geometry_path),processor_sha256=digest(processor_path)))
    captured = []
    try:
        with forbid_integration(mj) as integrations:
            for a in population:
                require(time.time()<grant["valid_until_epoch"] and not (output/"STOP.json").exists(),"render expired or stopped")
                name = a["id"]
                c = target_case(root,a["layout"])
                spec = read(sealed(c/"EXECUTION_SPEC.json"))
                ep = read(sealed(c/"NATIVE_REQUEST.json"))["episode"]
                obs = target_obs(pin(sealed(c/f"query-{name[-2:]}/INPUT.npz"),a["input_file_sha256"]))
                require(input_identity(obs)[0]==a["input_sha256"],"old authenticated images changed")
                p = next(r for r in packets(sealed(c/ep/"state.bin.journal")) if r["step"]==a["tick"])
                require(abs(.002*p["step"]-p["time_s"])<1e-8,"image and native tick differ")
                scene = Path(spec["model_path"])
                renderer = AuxRGBRenderer(scene,spec["frozen_files"][str(scene)])
                try:
                    driver = GL.glGetString(GL.GL_RENDERER).decode()
                    require(any(k in driver.lower() for k in ("llvmpipe","softpipe")),"CPU-only graphics context required")
                    write_new(output/f"{name}.RGB_ENTERED.json",dict(id=name,tick=a["tick"]))
                    aux = renderer.capture(ep,p)
                    sensor = typed(obs,ep,a,aux.frame)
                    full = sensor.legacy_input()
                    identity, fields = input_identity(full)
                    _, old = input_identity(obs)
                    require(all(fields[k]==old[k] for k in ("cam_high","cam_right_wrist","state","prompt"))
                            and fields["cam_left_wrist"]!=old["cam_left_wrist"],
                            "aux route changed original image history, state or instruction")
                    write_new(output/f"{name}.SEG_ENTERED.json",dict(id=name,tick=a["tick"]))
                    renderer.renderer.enable_segmentation_rendering()
                    renderer.renderer.update_scene(renderer.data,camera=renderer.optical)
                    seg = renderer.renderer.render()
                    def pixels(name):
                        return int(((seg[:,:,0]==renderer.model.geom(name).id)
                                    & (seg[:,:,1]==int(mj.mjtObj.mjOBJ_GEOM))).sum())
                    audit = {k:pixels(k) for k in ("task_cube_geom","task_blue_cube_geom","task_left_pad","task_right_pad","task_table")}
                finally:
                    renderer.close()
                target = output/f"{name}.private.npz"
                with target.open("xb") as sink:
                    np.savez(sink, head=full["images"]["cam_high"], right=full["images"]["cam_right_wrist"],
                             aux=full["images"]["cam_left_wrist"],state=full["state"],prompt=np.array(full["prompt"]),
                             episode=np.array(ep))
                captured.append(dict(id=name,tick=a["tick"],scene_sha256=renderer.scene_sha256,
                                     camera=AUXILIARY_CAMERA, calibration_sha256=aux.calibration_sha256,
                                     sample_sha256=aux.native_state_sha256,rgb_sha256=tensor_sha(aux.frame.rgb),
                                     input_sha256=identity,payload_sha256=digest(target),renderer=driver,
                                     auxiliary_capture_wall_s=aux.render_wall_s,pixel_visibility=audit,
                                     original_body_wrist_and_history_fingerprints_unchanged=True))
            require(not integrations,"offline camera called integration")
    finally:
        rgb = len(list(output.glob("*.RGB_ENTERED.json")))
        seg = len(list(output.glob("*.SEG_ENTERED.json")))
        write_new(output/"RENDER_CLOSE.json",dict(
            panel=ID, rows=captured,entered_rgb=rgb,entered_segmentation=seg,
            rendered_slots=len(captured), model_forwards=0,physics_ticks=0,optimizer_updates=0,
            started_owner_released=True,renderers_closed=True,
        ))


def ingress(root, render_output, output):
    from jax import default_backend
    from sea1b_panel import registration, verified_processor
    from sea1_observation_preflight import sampler_fields

    require(default_backend()=="cpu","official processor check uses CPU")
    manifest = read(render_output/"RENDER_CLOSE.json")
    require(manifest["renderers_closed"] and manifest["rendered_slots"]==4
            and manifest["entered_rgb"]==manifest["entered_segmentation"]==4
            and manifest["physics_ticks"]==manifest["model_forwards"]==0,"incomplete camera result")
    frozen,_ = registration()
    prepare,_,processor = verified_processor(root,frozen)
    a = {r["id"]:r for r in read(ROSTER)["go2"]}
    receipts=[]
    for row in manifest["rows"]:
        item=a[row["id"]]
        with np.load(pin(render_output/f"{row['id']}.private.npz",row["payload_sha256"]),allow_pickle=False) as z:
            raw=dict(images=dict(cam_high=z["head"],cam_right_wrist=z["right"],cam_left_wrist=z["aux"]),
                     state=z["state"],prompt=str(z["prompt"]))
            aux=Frame(str(z["episode"]),AUXILIARY_CAMERA,item["tick"],z["aux"])
            sensor=typed(raw,str(z["episode"]),item,aux)
        result,identity=prepare_sensor(sensor,prepare)
        require(identity==row["input_sha256"] and aux.rgb.shape==(480,640,3) and all(result["image_mask"].values()),
                "real auxiliary slot not admitted by the official processor")
        as_original={**raw,"images":{**raw["images"],"cam_left_wrist":np.zeros((480,640,3),np.uint8)}}
        before,after=map(sampler_fields,(prepare(as_original),result))
        require(input_identity(as_original)[0]==item["input_sha256"]
                and before["current"]["left_wrist_0_rgb"]!=after["current"]["left_wrist_0_rgb"]
                and all(before["current"][k]==after["current"][k] for k in ("base_0_rgb","right_wrist_0_rgb"))
                and all(before[k]==after[k] for k in ("memory","current_masks","state","token_ids","token_mask")),
                "adding the physical capture changed another sampler field")
        receipts.append(dict(**row,actual_sampler_fields=after))
    write_new(output,dict(
        status="FOUR_ARCHIVED_PAUSES_WITH_THREE_VIEW_RGB_VERIFIED",panel=ID,processor_identity=processor,
        exact_mount_sha256=calibration_sha256(),render_close_sha256=digest(render_output/"RENDER_CLOSE.json"),
        allocation_sha256=read(render_output/"START.json")["allocation_sha256"],rows=receipts,
        rgb_frames=4,audit_segmentation_frames=4,new_native_starts=0,physics_ticks=0,model_forwards=0,optimizer_updates=0,
    ))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root",type=Path)
    sub=cli.add_subparsers(dest="mode",required=True)
    r=sub.add_parser("render")
    for n in ("output","geometry","processor","grant"):
        r.add_argument(n,type=Path)
    r.add_argument("sha")
    p=sub.add_parser("ingress")
    p.add_argument("render",type=Path)
    p.add_argument("output",type=Path)
    a=cli.parse_args()
    os.umask(0o077)
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU offscreen only")
    if a.mode=="render":
        render(a.root.resolve(),a.output,a.geometry,a.processor,a.grant,a.sha)
    else:
        ingress(a.root.resolve(),a.render,a.output)
