"""Finite fresh source teacher episodes; original gate and native recorder reused."""
import argparse,fcntl,json,os,time
from pathlib import Path
import numpy as np
from contract import ROOT,INSTRUCTIONS,SPLITS,guard,sha,write
import scene,recorder
from scorer import score
from teacher import Teacher


def original_gate(command,state,now,slot):
    from go2arx5_vla.contracts.trajectory import TrajectoryChunkV1
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix,matrix_to_quat
    rotations=rot6d_rows_to_matrix(command[:,3:9])
    chunk=TrajectoryChunkV1(chunk_id=slot,source='PI0285',observation_timestamp=now,
        point_timestamps=now+np.arange(40)*.05,eef_position=command[:,:3],eef_orientation=matrix_to_quat(rotations),
        source_frame='PI0276_MANISKILL_WORLD',tcp_semantic_id='PI0276_PANDA_HAND_TCP',validity_start=now,validity_end=now+1.95,gripper=command[:,-1])
    return TrajectorySafetyGateV1().evaluate(chunk,now=now,achieved_position=state[:3],achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(state[None,3:9]))[0]).as_dict()


def step(ep,command,phase,teacher):
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
    target=np.eye(4);target[:3,3]=command[:3];target[:3,:3]=rot6d_rows_to_matrix(command[None,3:9])[0]
    action=recorder.native_action(target,ep.root,float(command[-1]))
    ep.step(action,phase=phase,command=command,teacher=teacher)
    assert np.max(abs(recorder.matrix(ep.arm._target_pose)-np.linalg.inv(ep.root)@target))<1e-5


def episode(task,split,seed):
    slot=f'{task}-{split}-{seed}-TEACHER';parent=recorder.RAW/'episodes'/slot
    roots={parent,ROOT/'source/episodes'/slot}
    valid=[p for directory in roots for p in directory.glob('attempt-*/RESULT.json') if json.loads(p.read_text())['status'].startswith('VALID')]
    if valid:
        assert len(valid)==1;return json.loads(valid[0].read_text())
    assert not any(list(directory.glob('attempt-*')) for directory in roots),'Incomplete attempts need exact owner audit before technical replacement'
    ep=recorder.Episode(slot,seed,'pd_ee_pose','PI0285PickPlaceStack-v1',dict(layout_id=seed))
    began=time.time();status='VALID_SOURCE_COMPLETE';gates=[]
    try:
        write(ep.folder/'EPISODE_CARD.json',dict(task=task,split=split,layout=seed,instruction=INSTRUCTIONS[task],scene=ep.env.layout_spec,teacher_sha256=sha(ROOT/'code/teacher.py'),code_seal_sha256=sha(ROOT/'design/CODE_SEAL.json'),NON_REALTIME=True,policy_inputs='RGB_ACTUAL_EEF_OPENING_LANGUAGE_ONLY',new_WBC=0))
        red=ep.states[-1]['cubeA_pose'][:3].copy();green=ep.states[-1]['cubeB_pose'][:3].copy()
        neutral=ep.observe()['state_v1'].copy();neutral[-1]=1.
        for _ in range(20):step(ep,neutral,-1,False)
        teacher=Teacher(ep,task,ep.env.layout_spec['zone_xy'])
        for i in range(640):
            observation=ep.observe()
            command,phase=teacher.query(i*.05)
            gate=original_gate(np.tile(command,(40,1)),observation['state_v1'],ep.control*.05,slot+f'-{i}')
            gates.append(dict(control=i,decision=gate['decision'],issues=gate.get('issues',[])))
            if gate['decision']=='REJECT':status='VALID_SOURCE_GATE_REJECT';break
            step(ep,command,phase,True)
        data={k:np.stack([v[k] for v in ep.states]) for k in ep.states[0]}
        resultscore=score(data,task,ep.env.layout_spec['zone_xy'],red,green,status,ep.control)
        write(ep.folder/'TEACHER_EVENTS.json',dict(events=teacher.events,within_episode_recovery_attempts=teacher.attempts,teacher_failed=teacher.failed,command_limits=dict(linear_speed=.15,angular_speed=1.)))
        write(ep.folder/'GATE_RECORDS.json',gates);write(ep.folder/'TASK_SCORE_CPU.json',resultscore)
        result=ep.finish(status)
        result.update(task=task,split=split,layout=seed,instruction=INSTRUCTIONS[task],folder=str(ep.folder),task_score=resultscore,persistent_success=resultscore['persistent_success'],elapsed_wall_s=time.time()-began,teacher_passes=1,planner_calls=0,model_queries=0,NON_REALTIME=True,domain='NEW_COMMON_SOURCE_PANDA_NOT_OLD_BENCHMARK',selected='NONE',deployment='NONE')
        write(ep.folder/'RESULT.json',result)
        return result
    except BaseException as exc:
        ep.finish('ENGINEERING_INCOMPLETE',repr(exc));raise


def run(split):
    guard();assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    assert split in ('PILOT','TRAIN','DEV'),'Confirmation is never a teacher dataset'
    lock=(ROOT/'runtime/SOURCE_GPU0.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();write(ROOT/'runtime/SOURCE_RUNNING.json',dict(pid=os.getpid(),start_epoch=start,split=split))
    tasks=list(INSTRUCTIONS)
    if split!='PILOT':
        prior=json.loads((ROOT/'source/evidence/PILOT.json').read_text());tasks=[task for task in tasks if prior['tasks'][task]['admitted']]
    rows=[]
    try:
        for seed in range(SPLITS[split][0],SPLITS[split][1]+1):
            for task in tasks:
                rows.append(episode(task,split,seed));write(ROOT/'source/evidence'/f'{split}_PROGRESS.json',dict(rows=rows))
        counts={task:dict(success=sum(v['persistent_success'] for v in rows if v['task']==task),requested=SPLITS[split][1]-SPLITS[split][0]+1) for task in tasks}
        for task,v in counts.items():v['admitted']=v['success']>=dict(PILOT=7,TRAIN=56,DEV=14)[split]
        result=dict(status='FIXED_TEACHER_PANEL_COMPLETE',split=split,tasks=counts,rows=rows,new_model_queries=0,new_WBC=0)
        write(ROOT/'source/evidence'/f'{split}.json',result);print(json.dumps(dict(status=result['status'],split=split,tasks=counts)),flush=True)
    finally:
        write(ROOT/'runtime/SOURCE_CLOSED.json',dict(pid=os.getpid(),split=split,start_epoch=start,end_epoch=time.time()))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=('PILOT','TRAIN','DEV'),required=True);a=p.parse_args();run(a.split)
