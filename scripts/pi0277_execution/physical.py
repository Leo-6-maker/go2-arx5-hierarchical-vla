"""Unchanged source-Panda execution, explicitly rebound to PI0277 counters and lanes."""
import argparse,ast,importlib.util,json,sys,types
from pathlib import Path
import numpy as np
from evaluation_card import ROOT,OLD,RAW,OLD_RAW,read,sha,write,freeze

def runtime():
    """Import physics hooks only in the actual physical execution process."""
    card=read(ROOT/'design/EVALUATION_CARD.json')
    run=types.ModuleType('run');run.ROOT=ROOT;run.sha=sha;run.write=write;sys.modules['run']=run
    for name in ('ManiSkill','project-src'):
        target=ROOT/name
        if not target.exists():target.symlink_to(OLD/name,target_is_directory=True)
        assert target.resolve()==(OLD/name).resolve()
    original_index=OLD/'design/SOURCE_FILESET.json';dest=ROOT/'design/SOURCE_FILESET.json'
    if not dest.exists():dest.write_bytes(original_index.read_bytes())
    assert dest.read_bytes()==original_index.read_bytes()
    spec=importlib.util.spec_from_file_location('pi0277_native_source',OLD/'code/source_worker.py')
    source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
    source.ROOT=ROOT;source.RAW=RAW;source.LIMITS=card['physics_limits']
    old_episode=source.Episode;original_retained=source.retained
    class Episode(old_episode):
        def __init__(self,slot,seed,mode):
            fixed=any(slot==f'{lane}-{seed}-{point}' and seed in seeds for lane,seeds in card['physical'].items() for point in card['policies'])
            if not fixed:
                permit=read(ROOT/'evidence/CONFIRMATION_ADMISSION.json')
                assert permit['status']=='PASS_REGISTERED_BASIC_TASK_MILESTONE_BOTH_FINALS_FROZEN'
                assert seed in card['confirmation_reserved_seeds'] and slot in [f'CONFIRM-{seed}-{p}' for p in ('joint','eef',*card['policies'])]
            paths=list((RAW/'episodes').glob('*/attempt-*/START.json'))
            parent=RAW/'episodes'/slot
            if parent.exists():assert len(paths)-len({p.parent.parent.name for p in paths})<8,'PI0277_TECHNICAL_START_CAP'
            super().__init__(slot,seed,mode)
    source.Episode=Episode
    def retained(slot):
        existing=original_retained(slot)
        if existing:return existing
        # The inherited client asks for TEST/TRAIN EEF initial states. Only those
        # exact immutable originals are routed to the predecessor; models stay local.
        if slot.endswith('-eef') and slot.startswith(('TEST-','TRAIN-')):
            seed=int(slot.split('-')[1])
            if seed in card['confirmation_reserved_seeds']:return original_retained(f'CONFIRM-{seed}-eef')
            candidates=[p for p in (OLD_RAW/'episodes'/slot).glob('attempt-*/RESULT.json') if read(p)['status'].startswith('VALID')]
            assert len(candidates)==1,slot
            return candidates[0].parent,read(candidates[0])
        return None
    text=(OLD/'code/evaluate.py').read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='policy_episode')
    scope=dict(ROOT=ROOT,RAW=RAW,np=np,json=json,write=write,retained=retained,Episode=Episode,
        native_action=source.native_action,pose_gate=source.pose_gate,matrix=source.matrix,flatten=source.flatten)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(OLD/'code/evaluate.py'),'exec'),scope)
    freeze(ROOT/'evidence/NATIVE_EXECUTOR_REUSE.json',dict(status='SOURCE_BOUND_UNCHANGED_PHYSICAL_CLIENT',
        source_worker_sha256=sha(OLD/'code/source_worker.py'),policy_episode_source_sha256=sha(OLD/'code/evaluate.py'),
        extracted_policy_ast_sha256=__import__('hashlib').sha256(ast.dump(node).encode()).hexdigest(),
        changed=['output_root','new_budget_limits','eight_technical_replacements','old_expert_initial_state_lookup'],
        controller_gate_dynamics_cameras_unchanged=True,source_fileset_sha256=sha(original_index)))
    return source,scope['policy_episode']

def main(point):
    from evaluate_fixed import Server
    assert point in ('CONTROL4096','GRIP4096')
    assert read(ROOT/'evidence'/f'POLICY_ADMISSION_{point}.json')['status']=='PASS_PI0277_FIXED_POLICY_IDENTITY'
    source,episode=runtime();card=read(ROOT/'design/EVALUATION_CARD.json')
    noises={(r['lane'],r['condition'],r['index']):r for r in read(ROOT/'design/EVALUATION_NOISE.json')['rows']}
    server=Server(point);rows=[]
    try:
        for lane,seeds in card['physical'].items():
            for seed in seeds:
                folder,result=episode(seed,lane,server,noises)
                rows.append(dict(lane=card['lane_labels'][lane],stored_lane=lane,seed=seed,point=point,path=str(folder),result_sha256=sha(folder/'RESULT.json'),status=result['status']))
    finally:server.close()
    assert len(rows)==48
    freeze(ROOT/'evidence'/f'PHYSICAL_COMPLETE_{point}.json',dict(status='ALL_FIXED_PHYSICAL_CONDITIONS_RETAINED',rows=rows,
        quality='PENDING_INDEPENDENT_CPU_SCORING',NON_REALTIME=True,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))

if __name__=='__main__':
    import fcntl
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('CONTROL4096','GRIP4096'),required=True);a=p.parse_args()
    with (ROOT/'runtime/quality_owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(a.point)
