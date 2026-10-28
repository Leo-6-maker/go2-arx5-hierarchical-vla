"""One finite CPU check: contracts, mocked ordering and source-bound pure math."""
import ast,copy,hashlib,importlib,json,multiprocessing,sys,tempfile,time
from pathlib import Path
import numpy as np
from cpu_dataset import RunConfig,Consumption,ItemLedger,RELEASE
from scene_specs import cards,static_check
from validate_scene_specs import validate as scene_validate
from validate_episode_schema import policy_packet,validate,qualified_bridge
from collection_dryrun import command,run
from perf_audit import audit

def rejects(call,error=ValueError):
    try:call()
    except error:return
    raise AssertionError('Invalid input admitted')

def main():
    start=time.perf_counter();cpu_start=time.process_time();code=Path(__file__).parent;repo=code.parent.parent
    for p in code.glob('*.py'):
        compile(ast.parse(p.read_text()),str(p),'exec')
        if p.name!=Path(__file__).name:importlib.import_module(p.stem)
    assert 'sapien' not in sys.modules and 'mani_skill' not in sys.modules and 'gr00t' not in sys.modules
    config=RunConfig('root','assets',(),({'path':'source','sha256':'0'*64},),'1'*64)
    boundaries=[]
    for step in (0,4,128,4096,8191,8192):
        index=config.next_micro(step);cursor=Consumption(index);original=range(index,min(index+32,config.micro_stop))
        expected=[hashlib.sha256(f'fixed-sample-label-mask-visit-{i}'.encode()).hexdigest() for i in original]
        consumed=[]
        for i in original:cursor.consume(i);consumed.append(hashlib.sha256(f'fixed-sample-label-mask-visit-{i}'.encode()).hexdigest())
        assert consumed==expected and cursor.checkpoint()['next_micro']==index+len(expected)
        boundaries.append(dict(update=step,calls=len(expected),status='MOCK_CPU_CURSOR_PASS',actual_TRAIN_items=0))
    c=Consumption();c.consume(0);rejects(c.checkpoint);rejects(lambda:c.consume(2));rejects(lambda:config.next_micro(-1))
    # Prefetch is a reservation, not actual main-process consumption.
    with tempfile.TemporaryDirectory() as folder:
        ledger=ItemLedger(Path(folder)/'calls.jsonl',multiprocessing.get_context('spawn'))
        ledger.reserve(0);ledger.reserve(1);ledger.reserve(2);cursor=Consumption();cursor.consume(0);cursor.consume(1)
        assert cursor.checkpoint()['next_micro']==2 and ledger.counter.value==3
        assert ItemLedger(Path(folder)/'calls.jsonl',multiprocessing.get_context('spawn')).counter.value==3
    rows=cards();assert scene_validate(rows)['cards']==456
    bad=copy.deepcopy(rows);bad[1]['split']='TRAIN';rejects(lambda:scene_validate(bad))
    rejects(lambda:scene_validate(rows,[50000]));assert len([r for r in rows if r['split']=='TEST' and r['camera'] is None])==96
    plan=command(RELEASE,False,rows);assert len(plan['episodes'])==48
    calls=[]
    rejects(lambda:command(RELEASE,True,rows,lambda *args:calls.append(args)),PermissionError);assert not calls
    quality=run(plan['episodes'][:1],lambda *args:dict(status='VALID_QUALITY_FAIL'),1);assert len(quality)==1
    technical=run(plan['episodes'][:1],lambda *args:dict(status='ENGINEERING_FAIL'),1);assert len(technical)==2
    rgb=np.zeros((40,96,96,3),np.uint8);state=np.tile([0,0,0,1,0,0,0,1,0,1.],(40,1));tcp=np.tile(np.eye(4),(40,1,1))
    episode=dict(scene_id='fixture',parent_group='DEV-50000',split='DEV',initial_state_sha256='0'*64,observation_episode_id='same',command_episode_id='same',
        control_time_s=np.arange(40)*.05,frame_time_s=np.arange(40)*.05,state_v1=state,body_rgb=rgb,wrist_rgb=rgb,
        commanded_eef_gripper=state.copy(),installed_tcp=tcp,achieved_tcp=tcp,instruction='fixed',
        evidence=dict(source_commit='62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3',target_WBC_calls=0,contacts=[],native_predicates=[],analysis_phase=[],camera_identity='CPU_SYNTHETIC_FIXTURE',opening_command_and_measurement_separate=True))
    assert validate(episode)['genuine_H40_windows']==1
    bad=dict(episode,command_episode_id='stale');rejects(lambda:validate(bad))
    rejects(lambda:validate(dict(episode,evidence={})));rejects(lambda:config.next_micro(True))
    rejects(lambda:policy_packet(dict(body_rgb=rgb[0],wrist_rgb=rgb[0],state_v1=state[0],language='fixed',goal=[0,0,0])))
    report_dir=repo/'reports/preparation/post_pi0276'
    if (report_dir/'ACTUAL_TRAIN_CONFIG.yaml').exists():
        perf=audit(report_dir/'ACTUAL_TRAIN_CONFIG.yaml',report_dir/'ACTUAL_TRAINER_STATE.json',repo/'scripts/pi0276_execution/train.py',report_dir/'TRAINING_ARGS_PICKLE_AUDIT.txt')
        assert perf['resolved_loader_settings']['dataloader_pin_memory'] is True and perf['resolved_loader_settings']['per_device_train_batch_size']==1
    bridge=qualified_bridge(repo/'scripts/pi0276_execution/source_worker.py',repo/'reports/preparation/post_pi0276/source/mani_skill/utils/geometry/rotation_conversions.py')
    from scipy.spatial.transform import Rotation
    root=np.eye(4);root[:3,3]=[-.615,0,0];world=np.eye(4);world[:3,:3]=Rotation.from_euler('XYZ',[.34,-.23,1.12]).as_matrix();world[:3,3]=[.2,-.1,.3]
    action=bridge['native_action'](world,root,.25)
    assert np.allclose(action[:3],world[:3,3]-root[:3,3]) and abs(action[-1]+.5)<1e-6
    assert np.max(abs(Rotation.from_euler('XYZ',action[3:6]).as_matrix()-world[:3,:3]))<1e-5
    rejects(lambda:bridge['native_action'](world,root,1.10001),AssertionError)
    assert 'sapien' not in sys.modules and 'mani_skill' not in sys.modules and 'gr00t' not in sys.modules
    report=dict(status='CPU_STATIC_PASS',resume_boundaries=boundaries,actual_TRAIN_items=0,synthetic_item_reservations=3,
        real_processor_hash_equivalence='NOT_RUN',GPU_optimizer_resume='NOT_RUN',collection_execute_rejected_before_executor=True,
        new_neural_calls=0,new_native_calls=0,new_teacher_calls=0,new_render_calls=0,new_WBC_calls=0,wall_s=time.perf_counter()-start,CPU_seconds=time.process_time()-cpu_start)
    if sys.platform=='linux':
        import resource
        report['process_peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        assert report['process_peak_RSS_bytes']<64*2**30
    print(json.dumps(report));return report

if __name__=='__main__':main()
