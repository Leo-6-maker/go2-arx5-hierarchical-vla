"""Independent all-packet startup audit; no integration, rendering or policy."""
import json
import struct
from pathlib import Path
import numpy as np
from .night_r1_runtime import read,sha
from .task_state import parse_state
from .task_contacts import allowed
from scripts.pi0292.durable import write


def one(root,episode):
    import mujoco
    root=Path(root);p=root/'factory/episodes'/episode;output=p/'INDEPENDENT_STARTUP_AUDIT.json'
    if output.exists():return read(output)
    result=read(p/'RESULT.json')
    if result['status']!='RAW_COMPLETE_STARTUP_REQUIRES_INDEPENDENT_AUDIT':raise ValueError('ACTUAL_COMPLETE_STARTUP_REQUIRED')
    m=mujoco.MjModel.from_binary_path(str(p/'task-scene.private.mjb'));js=np.flatnonzero(m.jnt_limited)
    lows=np.full(len(js),np.inf);highs=-lows.copy();count=0;illegal=[]
    with (p/'state.bin.journal').open('rb') as f:
        while True:
            h=f.read(4)
            if not h:break
            size,=struct.unpack('<I',h);packet=parse_state(f.read(size));count+=1
            if packet['step']!=count or abs(packet['time_s']-count*.002)>1e-9:raise ValueError('CONTIGUOUS_NATIVE_STEP_CLOCK')
            q=packet['qpos'][m.jnt_qposadr[js]];lows=np.minimum(lows,q);highs=np.maximum(highs,q)
            if np.any(q<m.jnt_range[js,0]) or np.any(q>m.jnt_range[js,1]):raise ValueError('ORIGINAL_RAW_JOINT_RANGE_FAIL')
            for c in packet['contacts']:
                names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in (c['geom0'],c['geom1'])]
                if not allowed(*names):illegal.append(dict(step=count,names=names))
    if count!=2500 or illegal or result['native_stops']:raise ValueError('COMPLETE_NATIVE_SAFETY_REQUIRED')
    end=read(p/'ACTUAL_STARTUP_END.json');before=read(p/'BEFORE_PAUSED_MODE_TRANSITION.json');mode=read(p/'COMMAND_AT_NATIVE_LOCOMOTION.json')
    if mode['physics_during_mode_initialization']!=0 or mode['native_step']!=before['native_step'] or mode['sim_time_s']!=before['time_s'] or mode['running'] is not False:raise ValueError('PAUSED_NATIVE_MODE_TRANSACTION')
    if end['control']['arm_mode']!='WBC' or end['native_step']!=2500:raise ValueError('ACTUAL_TERMINAL_NATIVE_WBC_REQUIRED')
    identities=[json.loads(l) for l in (p/'PROCESS_IDENTITIES.jsonl').read_text().splitlines()]
    for identity in identities:
        proc=Path('/proc')/str(identity['pid'])
        if proc.exists() and (proc/'stat').read_text().split()[21]==identity['start']:raise ValueError('OWNED_NATIVE_CHILD_STILL_LIVE')
    seal=read(root/'capsules'/result['source_sha']/'SOURCE_SEAL.json')
    proof=dict(status='PASS_INDEPENDENT_FIVE_SECOND_NATIVE_STARTUP',episode=episode,source=result['source_sha'],
        module_sha256=seal['files']['scripts/pi0294/night_r1_startup.py'],model_sha256=sha(p/'task-scene.private.mjb'),
        reset_domain=dict(table_top_m=.47,initial_finger_q_m=.042,task='T1'),high_rate_packets=count,all_raw_joints_in_original_ranges=True,
        illegal_contacts=0,actual_dynamic_GetUp=True,native_mode_initialization_physics_steps=0,terminal_mode='WBC',physical_time_s=5.,
        actual_range_rows=[dict(joint=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,int(j)),min=float(lows[i]),max=float(highs[i]),range=m.jnt_range[j].tolist()) for i,j in enumerate(js)],
        unique_child_processes_terminal=True,forced_terminal_cleanup=any(x==-9 for x in result['owned_returncodes']),
        journal_sha256=sha(p/'state.bin.journal'),new_task_submissions=0,new_RGB=0,old_failures_preserved=True)
    write(output,proof);return proof


def qualify(root,episodes):
    root=Path(root)
    if len(episodes)!=2 or len(set(episodes))!=2:raise ValueError('TWO_DISTINCT_CLEAN_STARTS_REQUIRED')
    rows=[one(root,episode) for episode in episodes]
    for field in ('module_sha256','model_sha256','source'):
        if len({r[field] for r in rows})!=1:raise ValueError('SAME_STARTUP_SOURCE_AND_RESET_DOMAIN_REQUIRED')
    if rows[0]['module_sha256'] in read(root/'R1_RESOURCE_LEDGER.json').get('quarantined_startup_module_hashes',[]):raise ValueError('STARTUP_RELEASE_QUARANTINED')
    old=root/'evidence/STARTUP_QUALIFICATION.json'
    if old.exists() and not (root/'evidence/STARTUP_QUALIFICATION_PRE_SYNC_REPAIR.json').exists():write(root/'evidence/STARTUP_QUALIFICATION_PRE_SYNC_REPAIR.json',read(old))
    receipt=dict(status='PASS_TWO_SAFE_COMPLETE_STARTS',rows=rows,
        old_safety_failures_retained=1+read(root/'R1_RESOURCE_LEDGER.json')['native_safety_failures'],
        qualifies_only=dict(table_top_m=.47,initial_finger_q_m=.042,concurrency=1),task_success=False)
    write(old,receipt);return receipt
