"""Small positive/negative CPU checks; explicitly not a model or resume canary."""
from cpu_common import *
import copy
import resource
import numpy as np
from next_run_draft import validate_panels,reject_launch,proposed_rotation_contract


def main():
    restrict()
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def forbidden(*a,**kw):raise RuntimeError('NO_MODEL_OPTIMIZER')
    original_call=torch.nn.Module.__call__
    def transform_only(module,*a,**kw):
        if type(module).__module__.startswith('torchvision.transforms.') and not list(module.parameters()):
            return original_call(module,*a,**kw)
        return forbidden()
    torch.nn.Module.__call__=transform_only;torch.optim.Optimizer.__init__=forbidden
    import pi0260_r2_codec as codec
    checks=[]
    def negative(name,fn):
        try:fn()
        except (AssertionError,ValueError,PermissionError,RuntimeError):checks.append(dict(name=name,result='EXPECTED_REJECT'))
        else:raise AssertionError(name+' unexpectedly accepted')
    toy=dict(min=[0.,0.],max=[1.,0.],q01=[0.,0.],q99=[1.,0.]);x=np.array([.3,.06]);encoded=codec.normalize(x,toy)
    assert encoded[1]==119999 and np.allclose(codec.unnormalize(encoded,toy),x,rtol=0,atol=1e-17)
    checks.append(dict(name='toy_near_constant_large_invertible_affine',result='PASS',scope='CONSTRUCTED_NOT_PROJECT_BOUNDS'))
    coupled=np.vstack([np.r_[np.eye(3)[i],np.eye(3)[i]]*sign for i in range(3) for sign in (-1,1)])
    assert np.linalg.matrix_rank(coupled)==3 and np.linalg.matrix_rank(np.eye(6))==6
    checks.append(dict(name='local_coupled_rank3_independent_rank6',result='PASS'))
    fit=read(OUT/'SEEN_TRAIN_112_MANIFEST.json');factors=read(OUT/'FACTOR_304_MANIFEST.json');panel=validate_panels(fit,factors);checks.append(dict(name='actual_panels',result='PASS'))
    for kind in ('missing','duplicate','heldout'):
        bad=copy.deepcopy(fit)
        if kind=='missing':bad['rows'].pop()
        elif kind=='duplicate':bad['rows'][1]=bad['rows'][0]
        else:bad['rows'][0]['split']='NEW_SIM_EVAL'
        negative(kind,lambda:validate_panels(bad,factors))
    negative('launch_without_authority',lambda:reject_launch({'launch_allowed':False}))
    negative('flag_alone_not_authority',lambda:reject_launch({'launch_allowed':True},'pretend-GO'))
    def physical_contract(unit,frame):assert unit=='m/rad' and frame=='world-left/row6D'
    negative('wrong_physical_units',lambda:physical_contract('mm/deg','world-left/row6D'))
    negative('wrong_pose_frame',lambda:physical_contract('m/rad','body-right/column6D'))
    noise_checks=[]
    for n in fit['existing_noise']:
        assert sha(Path(n['path']))==n['file_sha256']
        with np.load(n['path']) as z:x=z[n['key']]
        assert x.shape==(1,40,132) and x.dtype==np.float32
        noise_checks.append(dict(file=n['path'],key=n['key'],file_sha256=n['file_sha256'],array_sha256=digest(x),dtype=str(x.dtype),shape=list(x.shape)))
    for filename in sorted({r['noise_file'] for r in factors['conditions']}):
        own=[r for r in factors['conditions'] if r['noise_file']==filename];assert sha(Path(filename))==own[0]['noise_file_sha256']
        with np.load(filename) as z:
            for key in sorted({r['noise_key'] for r in own}):
                x=z[key];assert x.shape==(1,40,132) and x.dtype==np.float32
                noise_checks.append(dict(file=filename,key=key,file_sha256=own[0]['noise_file_sha256'],array_sha256=digest(x),dtype=str(x.dtype),shape=list(x.shape)))
    checks.append(dict(name='existing_noise_packets_hash_key_dtype_shape',result='PASS'))
    stat=read(SOURCE/'actual_statistics.json');changed=proposed_rotation_contract(stat)
    assert changed!=stat and stat==read(SOURCE/'actual_statistics.json')
    for block in ('state','relative_action'):
        for f in ('min','max','q01','q99'):
            a=np.asarray(stat['new_embodiment'][block]['eef_9d'][f]);b=np.asarray(changed['new_embodiment'][block]['eef_9d'][f]);assert np.array_equal(a[...,:3],b[...,:3])
    assert changed['new_embodiment']['action']==stat['new_embodiment']['action']
    checks.append(dict(name='draft_only_rotation_blocks_no_live_mutation',result='PASS'))
    # Actual full processor/collator, strictly no model construction or learned forward.
    sys.path.insert(0,str(DATA/'runtime'))
    from pi0260_r3_inputs import build,batch,identity,inventory
    processor=build(stat,BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot')
    metadata=read(OUT/'TRAIN_NUMERIC_INDEX.json');codec_proc=codec.processor(stat);boundaries=[]
    selected=[next(r for r in fit['rows'] if r['task']==task and r['command']==0 and r['variant']==v) for task in ('M1','M3') for v in (0,1)]
    for row in selected:
        folder=RAW/'S2'/row['family'];before={n:sha(folder/n) for n in ('proposals.npz','policy_frames.npz')}
        with np.load(folder/'proposals.npz') as z:s=z['state_v1'][row['index']];a=z['raw_action'][row['index']]
        with np.load(folder/'policy_frames.npz') as z:packet={k:z[k][row['frame_index']] for k in ('body_rgb','wrist_rgb')}
        packet.update(state_v1=s,language=row['language']);value=batch(processor,packet,a)['inputs'];condition={k:v for k,v in value.items() if k not in ('action','action_mask')}
        assert identity(condition)==row['condition_sha256'];cs,ca,cm=codec.encode(codec_proc,s,a)
        assert np.array_equal(value['state'].numpy().reshape(cs.shape),cs) and np.array_equal(value['action'].numpy().reshape(ca.shape),ca)
        assert np.array_equal(value['action_mask'].numpy().reshape(cm.shape),cm)
        boundaries.append(dict(family=row['family'],index=row['index'],condition_sha256=identity(condition),inventory=inventory(value),raw_processed_exact=True))
        assert before=={n:sha(folder/n) for n in before}
        def bind(unit= row['unit'],frame=row['frame_key'],label=digest(a)):
            assert (unit,frame,label)==(row['unit'],row['frame_key'],row['raw_label_sha256'])
        for name,args in [('wrong_unit',{'unit':-1}),('wrong_frame',{'frame':'wrong'}),('wrong_state_label',{'label':'wrong'})]:negative(name,lambda args=args:bind(**args))
    negative('extra_hidden_condition',lambda:batch(processor,dict(packet,teacher_phase=0),a))
    negative('accidental_neural_forward',lambda:torch.nn.Linear(1,1)(torch.ones(1)))
    negative('weight_load_guard',lambda:open(str(PREP/'forbidden.safetensors'),'rb'))
    negative('renderer_or_PBS_subprocess_guard',lambda:__import__('subprocess').Popen(['qsub','forbidden']))
    snap=read(OUT/'SOURCE_SNAPSHOT.json')
    def complete(part):assert part['completed']==64 and part['eligible_for_complete_point_analysis']
    negative('partial_point',lambda:complete(dict(completed=63,eligible_for_complete_point_analysis=False)))
    reject=next(p for p in sorted((RAW/'offline/R4096/new').glob('*/record.json')) if read(p)['gate']['decision']=='REJECT')
    before=sha(reject);record=read(reject);assert record['gate']['decision']=='REJECT' and sha(reject)==before
    checks.append(dict(name='original_gate_REJECT_retained_unchanged',result='PASS',source_sha256=before))
    assert all(sha(Path(p))==v['sha256'] for p,v in snap['source_identities'].items())
    checks.append(dict(name='active_source_identities_unchanged',result='PASS'))
    write('PROCESSOR_BOUNDARY.json',dict(rows=boundaries,stats_sha256=sha(SOURCE/'actual_statistics.json'),build_source_sha256=sha(DATA/'runtime/pi0260_r2_inputs.py'),full_plan_reused_sha256=sha(R1/'evidence/FULL_TRAIN_INPUT_PLAN.json'),**ZERO))
    write('PROPOSED_EXISTING_NOISE_IDENTITIES.json',noise_checks)
    write('CPU_CHECKS.json',dict(status='PASS_STATIC_AND_REAL_PROCESSOR_CPU_CHECKS_NOT_MODEL_CANARY',checks=checks,dry_run=panel,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,**ZERO));print('PASS_CPU_CHECKS',len(checks),flush=True)


if __name__=='__main__':main()
