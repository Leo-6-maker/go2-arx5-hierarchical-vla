"""Matched real-input/free-generation probes using the original decode once."""
import json,random,time
from pathlib import Path
import numpy as np


def measure(root,label,instance,data):
    import torch
    from scripts.pi0294.e1.model import payload
    from scripts.pi0292.eval_contract import error_metrics
    from scripts.pi0294.e1.artifacts import atomic_npz
    from scripts.pi0292.durable import write
    from scripts.pi0294.e1.journal import Journal
    from .storage import digest
    root=Path(root);folder=root/label;folder.mkdir(exist_ok=False)
    journal=Journal(folder/'journal',folder.name)
    # Repeated queries for diagnostics must not consume the optimization stream.
    py=random.getstate();np_state=np.random.get_state()
    from .task_learning_probes_v1 import selections
    selected=selections(data)
    continuous=bool(selected and selected[0][1]is not None)
    records=[]
    try:
        with torch.random.fork_rng(devices=[0]):
            for i,phase in selected:
                row,anchor=data.rows[i]
                state,expert,images,text=data.raw(i)
                seed=(2026110900 if continuous else 2026110800)+len(records)
                file=folder/(row['id']+(f'-a{anchor:02d}'if continuous else'')+'.npz')
                def generate():
                    started=time.monotonic()
                    result=instance.generate(payload(state,images,text),seed)
                    inference_wall_s=time.monotonic()-started
                    atomic_npz(file,physical_H40=result['physical_H40'],normalized=result['normalized'],
                        expert_policy_H40=expert,actual_state10=state,noise_seed=np.asarray(seed,dtype=np.int64))
                    return dict(noise=result['noise'],sha256=digest(file),
                        input_encode_infer_single_decode_wall_s=inference_wall_s,
                        errors=error_metrics(result['physical_H40'],expert,state))
                actual=journal.once('Q'+str(len(records)).zfill(4),
                    dict(dataset_sha256=data.manifest_sha,episode=row['id'],anchor=anchor,
                         seed=seed,model_identity=instance.identity),generate)
                records.append(dict(episode=row['id'],split=data.split,anchor=anchor,seed=seed,
                    command=row['command'],**actual,Processor_decode_calls=1,
                    model_global_step=instance.identity['global_step'],model_identity=dict(instance.identity)))
                if continuous:
                    records[-1].update(phase_audit_only=phase,
                        genuine_recovery_audit_only=row.get('genuine_recovery_success')is True)
    finally:
        random.setstate(py);np.random.set_state(np_state)
    result=dict(label=label,records=records,actual_VLA_queries=len(records),
        observations='REAL_ADMITTED_TARGET_'+data.split,
        baseline_label=('SAME_TARGET_PROCESSOR_ZERO_STEP_PARENT' if instance.identity['global_step']==0
                        else 'TARGET_CANDIDATE_WITH_SAME_TARGET_PROCESSOR'),
        original_source_B0_Processor_changed=False,optimizer_stream_RNG_preserved=True)
    if continuous:
        from .task_learning_probes_v1 import contract
        result['selection_contract']=contract()
    write(folder/'RESULT.json',result)
    return records
