"""Summarize sealed DEV arrays only. No solver, renderer, model or physics calls."""
import json
from pathlib import Path
import numpy as np
from pi0260_r2 import OLD,ROOT,read,write,sha

def phase(task,t):return ('approach' if t<4 else 'arc' if t<16 else 'hold') if task=='M3' else ('motion' if t<6 else 'hold')

def run():
    reg=read(OLD/'design/registry.json')
    limits=np.array(read(Path('/home/liuyu-nfs/Projects/go2-arx5/PI0255/evidence/resolved_native_config_cpu.json'))['joint_limits'])
    ik=[];visible=[]
    for p in sorted((OLD/'proposals').glob('*/*/geometry_result.json')):
        result=read(p)
        if result['status']!='VALID_GEOMETRY_FAIL':continue
        seq=result['sequence'];ledger=read(p.with_name('solver_ledger.json'))['attempts']
        failures=[x for x in ledger if not x['accepted']]
        assert failures
        def detail(x):
            q=np.array(x['candidate']);e=np.array(x['residual'])
            return dict(time=x['t1'],phase=phase(seq['task'],x['t1']),depth=x['depth'],position_residual_m=float(np.linalg.norm(e[:3])),
                rotation_residual_rad=float(np.linalg.norm(e[3:])),residual=x['residual'],candidate=q.tolist(),
                lower_margins_rad=(q-limits[:,0]).tolist(),upper_margins_rad=(limits[:,1]-q).tolist(),
                projected_candidates=x['projected_candidates'],last_accepted_start=x['start'],reason=x['checks'])
        with np.load(p.with_name('geometry.npz')) as z:
            last=dict(time=float(z['last_accepted_internal_time']),joints=z['last_accepted_internal_joints'].tolist())
        ik.append(dict(sequence=seq['sequence_id'],task=seq['task'],first_rejected_trial=detail(failures[0]),
            terminal_bounded_failure=detail(failures[-1]),last_accepted=last,
            ledger_sha256=sha(p.with_name('solver_ledger.json')),array_sha256=sha(p.with_name('geometry.npz')),
            pixels_depth='NOT_RECORDED: incomplete family did not enter rendering'))
    for i in (5,6,14,18):
        family=f'pi0260-family-{i:03d}';replay=read(OLD/'review_DEV'/(family+'.json'));rows=[]
        for seq in replay['sequences']:
            if not seq.get('failed_cues'):continue
            first=seq['failed_cues'][0];meta=read(OLD/(first['frame']+'.json'));depth={}
            with np.load(OLD/(first['frame']+'.npz')) as z:
                for cue in first['failures']:
                    sid=meta['scene_ids'][cue]
                    for stream in ('body','wrist'):
                        values=z[stream+'_depth'][z[stream+'_segmentation']==sid]
                        depth[cue+'_'+stream]=dict(pixels=len(values),negative_camera_axis_metres=values.tolist(),
                                                   missing=None if len(values) else 'NO_VISIBLE_PIXELS')
                joints=z['arm_joints'].tolist()
            rows.append(dict(sequence=seq['sequence'],fraction=seq['visibility_fraction'],first_time=first['time'],
                phase=phase(seq['task'],first['time']),first_failure=first,recorded_depth=depth,joints=joints,
                solver_residual_at_image='NOT_RECORDED_IN_FRAME: full sequence geometry replay already passed',
                array_sha256=sha(OLD/(first['frame']+'.npz'))))
        visible.append(dict(family=family,sequences=rows))
    assert len(ik)==35 and len(visible)==4 and sum(len(x['sequences']) for x in visible)==8
    report=dict(status='SAVED_ARRAY_DIAGNOSTICS_COMPLETE',old_valid_science_FAIL_preserved=True,IK_failures=ik,
        visibility_failed_families=visible,new_IK=0,new_render=0,new_model_queries=0,
        inference='Bounded DLS failures do not prove global infeasibility. Small endpoint raster counts fail the original four-pixel gate.',
        source_manifest_sha256=sha(OLD/'handoff/DEV_v1/manifest.json'))
    write(ROOT/'evidence/old_DEV_failure_diagnostics.json',report)
    print(json.dumps(dict(status=report['status'],IK_sequences=len(ik),visibility_families=len(visible))))

if __name__=='__main__':run()
