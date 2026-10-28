"""Freeze the 64 designated saved AGG-state proposals without labels or queries."""
import json
from pathlib import Path
from support import BASE, OLD, ROOT, RAW, read, sha, write


def main():
    import os
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30));os.nice(19)
    import numpy as np
    import torch
    from quality_worker import build, batch, identity
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    original=torch.nn.Module.__call__
    def guard(self,*a,**kw):
        if type(self).__module__.startswith('torchvision.transforms'):return original(self,*a,**kw)
        raise RuntimeError('NO_LEARNED_CPU_QUERY')
    def forbidden(*a,**kw):raise RuntimeError('NO_CPU_OPTIMIZER')
    torch.nn.Module.__call__=guard;torch.optim.Optimizer.__init__=forbidden
    assert not torch.cuda.is_available()
    import hashlib
    scene=BASE/'PI0267/design/SCENE_CARD.json';card=OLD/'design/EVALUATION_CARD.json'
    families=[f['family_id'] for f in read(scene)['families'] if f['split']=='NEW_SIM_EVAL'][:2]
    assert len(families)==2
    stats=OLD/'contracts/ROT/source/dataset/statistics.json';proc=build(read(stats),BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot')
    binding=read(OLD/'model_bindings/AGG4096.json');oldraw=RAW.with_name('PI0270');rows=[]
    for family in families:
        for task in ('M1','M3'):
            for command in (0,1):
                for backend in ('E0','E1'):
                    case=f'{family}-{task}-c{command}-{backend}';path=oldraw/'virtual/AGG4096'/case/'result.json'
                    rec=read(path);assert rec['validity']=='VALID' and rec['point']=='AGG4096'
                    assert rec['evaluation_card_sha256']==sha(card) and sha(path.parent/'trajectory.npz')==rec['arrays_sha256']
                    queries={q['replan']:q for q in rec['queries']};assert len(queries)==len(rec['queries'])
                    for index in (0,25,50,99):
                        row=dict(input_id=f'{case}-q{index:03d}',case=case,family=family,task=task,command=command,backend=backend,
                                 query_index=index,designated_timestamp_s=.4*index,source_result=str(path),source_result_sha256=sha(path),
                                 DIAGNOSIS_ONLY=True,never_train_or_blind_TEST=True)
                        if index not in queries:
                            assert rec['stop_reason'] is not None
                            row.update(status='MISSING_AFTER_STOP',stop_reason=rec['stop_reason']);rows.append(row);continue
                        q=queries[index];request=q['query'];obs=Path(request['input']);prediction=Path(request['output'])
                        assert q['timestamp']==.4*index and sha(obs)==q['observation_sha256'] and sha(prediction)==q['prediction_sha256']
                        with np.load(obs,allow_pickle=False) as z:
                            assert float(z['timestamp'])==.4*index
                            packet=dict(body_rgb=z['body_rgb'].copy(),wrist_rgb=z['wrist_rgb'].copy(),state_v1=z['state_v1'].copy(),language=request['language'])
                            fields={k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype),sha256=hashlib.sha256(z[k].tobytes()).hexdigest()) for k in z.files}
                        policy=batch(proc,packet)['inputs'];assert 'action' not in policy and 'action_mask' not in policy
                        condition=identity(policy);receipt=read(prediction.with_suffix('.receipt.json'))
                        assert receipt['binding']==binding and receipt['input_sha256']==condition and receipt['output_sha256']==sha(prediction)
                        with np.load(request['noise_file'],allow_pickle=False) as z:noise=z[request['noise_name']].copy()
                        assert noise.dtype==np.float32 and noise.shape==(1,40,132) and np.isfinite(noise).all()
                        assert hashlib.sha256(noise.tobytes()).hexdigest()==receipt['noise_sha256']
                        actual=torch.from_numpy(noise).to(torch.bfloat16).float().numpy()
                        with np.load(prediction,allow_pickle=False) as z:
                            assert np.array_equal(z['noise'],noise) and np.array_equal(z['actual_initial_noise'],actual)
                        row.update(status='AVAILABLE_EXACT_SAVED_AGG_INPUT_NOISE',request=request,observation_sha256=sha(obs),
                                   observation_fields=fields,condition_sha256=condition,AGG_prediction=str(prediction),
                                   AGG_prediction_sha256=sha(prediction),AGG_receipt_sha256=sha(prediction.with_suffix('.receipt.json')),
                                   source_noise_file_sha256=sha(request['noise_file']),source_noise_sha256=receipt['noise_sha256'],
                                   actual_bf16_initial_noise_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                                   camera_per_frame_matrix='UNOBSERVED_NOT_SAVED',source_renderer_identity='BOUND_BY_PREDECESSOR_QUALITY_FILESET',
                                   teacher_label_status='NOT_CALLED_A0_REQUIRED')
                        rows.append(row)
    assert len(rows)==64 and len({r['input_id'] for r in rows})==64
    sources=[dict(path=str(p),sha256=sha(p)) for p in (scene,card,stats,OLD/'design/QUALITY_EXECUTION_FILESET.json',Path(__file__))]
    write(ROOT/'design/VISITED_STATE_CARD.json',dict(schema='PI0271_FIXED_64_SAVED_AGG_STATE_PROPOSALS_V1',rows=rows,source_files=sources,
          families=families,model_queries=0,teacher_calls=0,new_episodes=0,renders=0,A0_still_required=True,
          selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print('PASS_FIXED_64_SAVED_AGG_INPUT_NOISE_PROPOSALS_NO_MODEL_OR_TEACHER_CALL',flush=True)


if __name__=='__main__':main()
