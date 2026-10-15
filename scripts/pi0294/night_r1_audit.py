"""R1 independent saved replay; numerical, contact and scoring predicates unchanged."""
import hashlib,json,struct
from pathlib import Path
import numpy as np
from .task_capture_runtime import read,sha
from .task_state import ActualState,parse_state
from .task_clock import FROZEN_FIELDS
from .task_contacts import allowed
from .night_r1_runtime import FOLDER
from .base_intent import native_document
from .component_teacher_audit import installed_parity
from .night_r1_source import load,sample,with_entry
from .mp_client import Client as OriginalClient
from go2arx5_vla.trajectory.so3 import geodesic_angle

def lines(path):return [json.loads(l) for l in Path(path).read_text().splitlines()] if Path(path).exists() else []

def physics(root,episode):
    import mujoco
    from PIL import Image
    from scripts.pi0292.durable import write
    from go2arx5_vla.target_sim.scorer import BOOL_FIELDS,score_episode
    root=Path(root);p=root/FOLDER;folder=p/'episodes'/episode;output=folder/'INDEPENDENT_CPU_AUDIT.json'
    if output.exists():return read(output)
    result=read(folder/'RESULT.json');card=read(folder/'CARD.json');saved=lines(folder/'samples.jsonl');start=read(folder/'TASK_START.json')
    model=mujoco.MjModel.from_binary_path(str(folder/'task-scene.private.mjb'));actual=ActualState(model);packets={};micro=[];object_xyz=[];tcp500=[];times500=[];illegal=[];count=0
    with (folder/'state.bin.journal').open('rb') as f:
        while raw:=f.read(4):
            size,=struct.unpack('<I',raw);packet=parse_state(f.read(size));count+=1;assert packet['step']==count
            relative=count-start['native_step']
            if relative<0:continue
            state=actual.update(packet)
            if relative%25==0:packets[relative//25]=packet
            if relative>0:
                micro.append([int(state['robot_object_contact']),int(state['native_grasp']),int(state['left_object_contact'] and state['right_object_contact']),state['object_pose'][2,3],int(not state['safety_valid'])])
                object_xyz.append(state['object_pose'][:3,3].copy());tcp500.append(state['tcp'].copy());times500.append(relative*.002)
            for contact in packet['contacts']:
                ids=[contact['geom0'],contact['geom1']];names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) for g in ids]
                if not allowed(*names):
                    illegal.append(dict(step=count,task_sample_time_s=relative*.002,native_contact_source_time_s=packet['contact_time_s']-start['time_s'],
                        geom_ids=ids,geom_names=names,body_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) for g in ids],
                        distance_m=float(contact['distance_m']),normal_force_N=float(contact['wrench'][0])))
    intents=[r for r in lines(folder/'committed_intents.jsonl') if r['phase']=='SUBMITTED_REQUIRES_INSTALL_ACK'];acks=[r for r in lines(folder/'committed_intents.jsonl') if r['phase']=='INSTALL_ACK']
    grip=[r for r in lines(folder/'committed_intents.jsonl') if r['phase']=='GRIPPER_PREFIX20HZ'];planner=lines(folder/'planner.jsonl');accepted={r['ref_id']:r for r in planner if r['kind']=='accept'}
    assert len(intents)==len(acks)==len(accepted)
    arrays={k:[] for k in BOOL_FIELDS};tcp=[];objects=[];v=[];w=[];rgb_hashes={};camera_checks=0;reference=[];h40=[]
    for intent in intents:
        seq=intent['sequence'];request_folder=p/'requests'/episode/f'{seq:04d}';assert sha(request_folder/'INPUT.private.json')==intent['persistent']['certificate']['input_sha256']
        with np.load(request_folder/'H40.private.npz',allow_pickle=False) as f:h40.append(f['poses'].copy())
    for i,s in enumerate(saved):
        packet=packets[i];assert s['native_step']==packet['step'] and all(np.array_equal(s[k],packet[k]) for k in ('qpos','qvel','ctrl'))
        state=actual.update(packet)
        for k in ('tcp','object_pose','object_linear_velocity','object_angular_velocity'):assert np.allclose(state[k],s[k],atol=1e-12,rtol=0)
        for k in BOOL_FIELDS:assert bool(state[k])==bool(s[k]);arrays[k].append(bool(state[k]))
        calibration=read(folder/'images'/f'{i:06d}-calibration.json')
        for camera in ('body_rgb','wrist_rgb'):
            assert calibration[camera]['timestamp_s']==packet['time_s'];image=folder/'images'/f'{i:06d}-{camera}.png'
            with Image.open(image) as pixels:assert pixels.size==(256,256)
            rgb_hashes[image.name]=sha(image);camera_checks+=1
        if i:assert s['active_intent_sequence']==intents[(i-1)//8]['sequence'] and s['executed_prefix_index']==(i-1)%8
        tcp.append(state['tcp'].copy());objects.append(state['object_pose'].copy());v.append(state['object_linear_velocity'].copy());w.append(state['object_angular_velocity'].copy())
        reference.append(h40[0][0] if i==0 else h40[(i-1)//8][i-intents[(i-1)//8]['index']])
    first_state=actual.update(packets[0]);entry_binding=read(folder/'ACTUAL_ENTRY_BINDING.json');warm=p/'requests'/(episode+'_WARMUP')/'0000';warm_result=read(warm/'FINAL.private.json');route=read(warm/f"ROUTE-{warm_result['route']:02d}.private.json")
    client=OriginalClient(root,actual);entry=dict(time=np.asarray(route['time'])*warm_result['certificate']['uniform_time_scale'],poses=client.fk_path(first_state,np.asarray(route['position'])))
    if card.get('entry_timing_version')=='QUANTIZED_NATIVE_SLERP_R1':
        from .night_r1_entry import retime
        entry=retime(entry,np.asarray(route['position']))
        assert entry['entry_timing']==entry_binding['entry_timing']
    record=next(r for r in read(p/'PROGRAMS.json')['rows'] if r['id']==entry_binding['source_id']);program=load(root,record);combined=with_entry(program,entry)
    with np.load(folder/'ACTUAL_COMBINED_PROGRAM.npz',allow_pickle=False) as original_combined:
        assert all(np.array_equal(combined[k],original_combined[k]) for k in combined)
    previous=np.zeros(3)
    for seq,(intent,ack) in enumerate(zip(intents,acks)):
        packet=packets[seq*8];wire=accepted[intent['wire_sha256']];a=np.asarray(intent['action_EEF13']);document=json.loads(wire['raw'])
        assert hashlib.sha256(wire['raw'].encode()).hexdigest()==intent['wire_sha256'] and ack['wire_sha256']==intent['wire_sha256'] and ack['native_step']==packet['step']
        rebuilt=native_document(a,np.asarray(intent['actual_BASE_mount']),base_valid=True,previous_velocity=previous)
        assert all(document[k]==rebuilt[k] for k in ('ee','base'));previous=a[7,10:13].copy()
        assert np.array_equal(a[:8],intent['scheduled_prefix_EEF13']) and intent['canonical_safety_gate']['decision']!='REJECT'
        expected,opening=sample(combined,seq*.4+np.arange(40)*.05)
        assert np.allclose(h40[seq],expected,atol=1e-12,rtol=0)
        from .task_gripper_contract import teacher_physical_fraction
        assert np.array_equal(a[:,9],teacher_physical_fraction(opening))
        assert intent['source_mpc_s']==round(packet['time_s']*1e9)/1e9 and 0<=wire['receiver_mpc_s']-wire['source_mpc_s']<=.025
    from .night_r1_terminal import gripper_tail
    terminal_gripper=gripper_tail(grip,len(saved),count,start['native_step'],
        bool(result['native_stops']) or result['status']=='VALID_NATIVE_SAFETY_FAIL_TASK_VERSION_STOPPED')
    for r in grip:
        i=r['index'];intent=intents[i//8];assert r['sequence']==i and r['prefix_index']==i-intent['index'] and r['opening']==intent['action_EEF13'][i%8][9]
        assert abs(r['source_mpc_s']-packets[i]['time_s'])<1e-10
    pauses=lines(folder/'pause_intervals.jsonl');assert all(all(r['before'][k]==r['after'][k] for k in FROZEN_FIELDS) for r in pauses)
    parity=installed_parity(planner);times=np.asarray([s['time_s'] for s in saved]);assert np.allclose(times,np.arange(len(saved))*.05,atol=1e-10,rtol=0)
    tcp=np.asarray(tcp);obj=np.asarray(objects);reference=np.asarray(reference);micro=np.asarray(micro).reshape(-1,5);object_xyz=np.asarray(object_xyz).reshape(-1,3)
    arrays={k:np.asarray(v,dtype=bool) for k,v in arrays.items()};arrays['robot_object_contact']=np.asarray([s['robot_object_contact'] for s in saved],dtype=bool);arrays.update(time_s=times,tcp_world=tcp,object_pose=obj,object_linear_velocity=np.asarray(v),object_angular_velocity=np.asarray(w))
    complete=bool(len(saved)==int(card['duration_s'])*20+1 and not result['native_stops'] and result['status']=='RAW_TASK_COMPLETE_REQUIRES_INDEPENDENT_CPU_AUDIT')
    arrays.update(contact_time_s=np.asarray(times500),contact_audit=micro)
    from .night_score import score as night_score
    ranges=raw_ranges(root,episode)
    quality=night_score(arrays,card,complete=complete,high_rate_ranges_PASS=ranges['all_raw_samples_within_original_ranges'])
    error=np.linalg.norm(reference[:,:3,3]-tcp[:,:3,3],axis=1);angle=geodesic_angle(reference[:,:3,:3],tcp[:,:3,:3]);displacement=np.linalg.norm(object_xyz[:,:2]-obj[0,:2,3],axis=1)
    output_arrays=dict(arrays,reference_tcp_world=reference,tcp_position_tracking_error_m=error,tcp_SO3_tracking_error_rad=angle,contact_time_s=np.asarray(times500),contact_audit=micro,actual_object_xyz_500Hz=object_xyz,actual_TCP_500Hz=np.asarray(tcp500))
    np.savez_compressed(folder/'ACTUAL_PHYSICS.npz',**output_arrays);write(folder/'RGB_HASHES.json',rgb_hashes)
    from .night_r1_terminal import verdict
    summary=dict(status=verdict(result['native_stops'],ranges['all_raw_samples_within_original_ranges'],quality['persistent_success']),
        runtime_source_sha=result['source_sha'],episode=episode,observations=len(saved),installed_chunks=len(intents),native_integrations=count,task_integrations=count-start['native_step'],task_prefix_s=float(times[-1]),
        complete_32s=bool(complete and card['duration_s']==32),complete_declared_duration=complete,duration_domain=quality['duration_domain'],quality=quality,high_rate_ranges=ranges,actual_object_contact_ticks=int(micro[:,0].sum()),actual_grasp_ticks=int(micro[:,1].sum()),
        illegal_contact_records=len(illegal),first_illegal_contact=illegal[0] if illegal else None,last_illegal_contact=illegal[-1] if illegal else None,
        original_native_stop=result['native_stops'],pending_planning_stop=result.get('stop_reason',result.get('error')),final_source_phase=result.get('final_stage'),
        initial_actual_TCP_xyz=tcp[0,:3,3].tolist(),final_actual_TCP_xyz=tcp[-1,:3,3].tolist(),final_same_tick_reference_xyz=reference[-1,:3,3].tolist(),
        max_actual_TCP_displacement_m=float(np.linalg.norm(tcp[:,:3,3]-tcp[0,:3,3],axis=1).max()),max_actual_object_horizontal_displacement_m=float(displacement.max(initial=0)),
        tracking_XYZ_initial_m=float(error[0]),tracking_XYZ_final_m=float(error[-1]),tracking_XYZ_median_m=float(np.median(error)),tracking_SO3_final_rad=float(angle[-1]),
        image_files_verified=camera_checks,actual_RGB_pairs=len(saved),source_reference_values_replayed_exact=True,source_gripper20Hz_labels_exact=True,
        installed_reference=parity,pause_history_unchanged=True,actual_state_and_FK_replay=True,terminal_gripper_commands=terminal_gripper,
        interpretation='ORIGINAL_SAFETY_AND_TASK_VERDICT; NO_RETRY',new_physics=0,new_IK=0,new_RRT=0,new_render=0,new_neural_queries=0,selected='NONE',deployment='NONE')
    write(output,summary);return summary

def video(root,episode):
    from PIL import Image,ImageDraw
    import subprocess
    from scripts.pi0292.durable import write
    root=Path(root);folder=root/FOLDER/'episodes'/episode;target=folder/'ACTUAL_NIGHT_RESULT.mp4';receipt=folder/'ACTUAL_VIDEO_RECEIPT.json'
    if receipt.exists():assert sha(target)==read(receipt)['video_sha256'];return read(receipt)
    assert not target.exists();saved=lines(folder/'samples.jsonl');audit=read(folder/'INDEPENDENT_CPU_AUDIT.json');log=(folder/'VIDEO_ENCODE.log').open('x')
    process=subprocess.Popen(['/usr/bin/ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','512x352','-r','20','-i','pipe:0',
        '-an','-c:v','libx264','-threads','1','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE,stderr=log)
    try:
        for i,s in enumerate(saved):
            frame=Image.new('RGB',(512,352),'black')
            for x,camera in ((0,'body_rgb'),(256,'wrist_rgb')):
                with Image.open(folder/'images'/f'{i:06d}-{camera}.png') as pixels:frame.paste(pixels,(x,96))
            draw=ImageDraw.Draw(frame);draw.text((6,4),'ACTUAL FULL Go2+ARX5 | TF-NIGHT | NON REALTIME',fill='white')
            draw.text((6,20),audit['status'],fill=(255,140,80))
            draw.text((6,36),'t=%.2fs | %d/%d installed | contact=%s'%(s['time_s'],min(audit['installed_chunks'],(i+7)//8),audit['installed_chunks'],s['robot_object_contact']),fill='white')
            draw.text((6,52),str(audit.get('pending_planning_stop'))[:78],fill='white')
            draw.text((6,68),'Original WBC / safety / task score unchanged',fill='white')
            draw.text((6,82),'BODY RGB',fill='white');draw.text((262,82),'WRIST RGB',fill='white');process.stdin.write(frame.tobytes())
        process.stdin.close();assert process.wait(timeout=60)==0
    finally:
        if process.poll() is None:process.kill();process.wait()
        log.close()
    value=dict(status='PASS_ACTUAL_SAVED_RGB_VIDEO',frames=len(saved),physical_prefix_s=audit['task_prefix_s'],playback_Hz=20,non_realtime=True,
        label=audit['status'],video_sha256=sha(target),bytes=target.stat().st_size,source_RGB_hashes_sha256=sha(folder/'RGB_HASHES.json'),
        public=False,new_physics=0,new_render=0,new_model_queries=0)
    write(receipt,value);return value

def contact_summary(root,episode):
    """Read force-bearing task contacts from saved journal; never step or solve."""
    import mujoco
    from .task_state import parse_state
    from scripts.pi0292.durable import write
    root=Path(root);p=root/FOLDER;e=p/'episodes'/episode;out=e/'TASK_CONTACT_RECONCILIATION.json'
    if out.exists():return read(out)
    m=mujoco.MjModel.from_binary_path(str(e/'task-scene.private.mjb'));start=read(e/'TASK_START.json');cube=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'task_cube_geom');pads={mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'task_'+side+'_pad'):side for side in ['left','right']}
    events=[];steps=[];table_body=[];root0=None;root_end=None;contacts_all=[]
    with (e/'state.bin.journal').open('rb') as f:
        while raw:=f.read(4):
            n,=struct.unpack('<I',raw);packet=parse_state(f.read(n))
            if packet['step']<start['native_step']:continue
            if root0 is None:root0=packet['qpos'][:3].copy()
            root_end=packet['qpos'][:3].copy();this=False
            for c in packet['contacts']:
                ids=[c['geom0'],c['geom1']];names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(i)) for i in ids]
                if 'task_table' in names and any(int(m.geom_bodyid[i])==mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'base_link') for i in ids):table_body.append(packet['step'])
                if cube in ids and any(i in pads for i in ids):
                    row=dict(step=packet['step'],recorded_task_time_s=(packet['step']-start['native_step'])*.002,contact_source_task_time_s=packet['contact_time_s']-start['time_s'],names=names,pad_side=next(pads[i] for i in ids if i in pads),normal_force_N=float(c['wrench'][0]),distance_m=float(c['distance_m']))
                    contacts_all.append(row)
                    if c['wrench'][0]>=.05:events.append(row);this=True
            if this:steps.append(packet['step'])
    run=maximum=0;previous=None
    for step in steps:
        run=run+1 if previous is not None and step==previous+1 else 1;maximum=max(maximum,run);previous=step
    output=dict(status='PASS_INDEPENDENT_SAVED_TASK_CONTACT_RECONCILIATION',actual_task_object_contact=bool(events),force_threshold_original_N=.05,
        force_bearing_contact_ticks=len(steps),cumulative_contact_duration_s=len(steps)*.002,max_contiguous_contact_duration_s=maximum*.002,contact_records=len(events),
        first_force_contact=events[0] if events else None,last_force_contact=events[-1] if events else None,max_normal_force_N=max([r['normal_force_N'] for r in events],default=0.),
        deepest_pad_object_distance_m=min([r['distance_m'] for r in contacts_all],default=None),sides=sorted({r['pad_side'] for r in events}),
        table_base_link_contact_ticks=len(set(table_body)),root_translation_delta_xyz_m=(root_end-root0).tolist(),root_translation_displacement_m=float(np.linalg.norm(root_end-root0)),
        actual_grasp_ticks=read(e/'INDEPENDENT_CPU_AUDIT.json')['actual_grasp_ticks'],persistent_task_success=read(e/'INDEPENDENT_CPU_AUDIT.json')['quality']['persistent_success'],
        contact_is_not_grasp_or_success=True,new_physics=0,new_IK=0,new_RRT=0,new_render=0,new_neural_queries=0)
    write(out,output);return output


def raw_ranges(root,episode):
    import mujoco
    from scripts.pi0292.durable import write
    from .night_r1_plan_runtime import reserve_static,finish_static
    folder=Path(root)/FOLDER/'episodes'/episode;out=folder/'ALL_JOINT_RANGE_AUDIT.json'
    if out.exists():return read(out)
    model=mujoco.MjModel.from_binary_path(str(folder/'task-scene.private.mjb'));joints=np.flatnonzero(model.jnt_limited)
    lows=np.full(len(joints),np.inf);highs=-lows.copy();crossings=np.zeros(len(joints),dtype=int);first=[None]*len(joints);last=[None]*len(joints);steps=0
    name='RAW_RANGE_AUDIT_'+episode;reserve_static(root,name,66503)
    try:
        with (folder/'state.bin.journal').open('rb') as f:
            while header:=f.read(4):
                size,=struct.unpack('<I',header);packet=parse_state(f.read(size));steps+=1;q=packet['qpos'][model.jnt_qposadr[joints]]
                lows=np.minimum(lows,q);highs=np.maximum(highs,q);bad=(q<model.jnt_range[joints,0])|(q>model.jnt_range[joints,1])
                for i in np.flatnonzero(bad):
                    crossings[i]+=1;first[i]=packet['step'] if first[i] is None else first[i];last[i]=packet['step']
        rows=[dict(joint=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,int(j)),range=model.jnt_range[j].tolist(),min_q=float(lows[i]),max_q=float(highs[i]),crossing_samples=int(crossings[i]),first_crossing_step=first[i],last_crossing_step=last[i]) for i,j in enumerate(joints)]
        value=dict(status='PASS' if not crossings.any() else 'RAW_MECHANICAL_RANGE_FAIL',native_packets=steps,limited_joints=len(joints),rows=rows,all_raw_samples_within_original_ranges=not bool(crossings.any()),new_physics=0,new_IK=0,new_RRT=0)
        write(out,value);return value
    finally:finish_static(root,name,steps,'COMPLETE')
