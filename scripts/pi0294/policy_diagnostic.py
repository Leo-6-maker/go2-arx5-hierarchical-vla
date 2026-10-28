"""Real frozen TRAIN observations and expert controls, only inside own PBS."""
import argparse
import base64
import hashlib
import io
from pathlib import Path

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0292.eval_contract import FRAME, TCP, expert_decode, gate_action, error_metrics
from .continuation import RUN, PARENT, LIMITS
from .native_dataset import NativeDataset, sha
from .resident import ResidentModel


def diagnostic_plan(panel, manifest):
    if panel['run_uuid'] != PARENT or manifest['statistics_sha256'] != 'fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7':
        raise PermissionError('ORIGINAL_PANEL_AND_FROZEN_STATS_REQUIRED')
    episodes = {r['id']:r for r in manifest['episodes']}
    cards = panel['panels']['M512_TRAIN']
    if len(cards) != 4 or len({c['original_attempt_id'] for c in cards}) != 4:
        raise PermissionError('ORIGINAL_FOUR_TRAIN_DEVELOPMENT_EPISODES')
    selected = []
    for card in cards:
        row = episodes[card['original_attempt_id']]
        if row['split'] != 'TRAIN' or row['task'] != card['task']:
            raise PermissionError('NO_DEV_OR_TASK_SUBSTITUTION')
        anchors = sorted({values[0] for values in row['stage_anchors'].values() if values})
        # At most five actual stage starts plus the final full-H40 anchor.
        anchors = sorted(set(anchors+[600]))[:6]
        for anchor in anchors:
            if not 0 <= anchor < row['windows']:
                raise PermissionError('ACTUAL_SEALED_FULL_WINDOW_REQUIRED')
            for noise_index in (0,1):
                key=f"PI0294/R1/OFFLINE/{card['scene_group']}/{anchor}/{noise_index}"
                seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')%(2**63)
                selected.append(dict(episode_id=row['id'],anchor=anchor,noise_index=noise_index,seed=seed,
                    scene_group=card['scene_group'],task=card['task']))
    if not 0 < len(selected) <= 48:
        raise PermissionError('MAXIMUM_48_OFFLINE_H40_DRAWS')
    return dict(schema='PI0294_R1_FIXED_TRAIN_DIAGNOSTIC_PLAN_V1',run_uuid=RUN,
                draws=selected,privileged_stage_for_sampling_only=True,
                model_inputs='OBSERVATION_ONLY_RGB_LANGUAGE_MEASURED_EEF',not_closed_loop=True)


def payload(state, images, language, *, snapshot):
    from PIL import Image
    encoded=[]
    for camera, array in zip(('external_rgb','wrist_rgb'),images):
        stream=io.BytesIO();Image.fromarray(array).save(stream,format='PNG');data=stream.getvalue()
        encoded.append(dict(camera=camera,png_base64=base64.b64encode(data).decode(),sha256=hashlib.sha256(data).hexdigest()))
    return dict(snapshot_uuid=snapshot,language=language,state_eef10=state.tolist(),images=encoded,frame=FRAME,tcp=TCP)


def run(args):
    root=Path(args.root);request=read(args.request)
    from .neural import verify_allocation
    verify_allocation(root,'resident',args.request)
    owner=OwnerLock(root/'NEURAL_OWNER.lock')
    output=Path(request['directory'])
    if output.exists() or not output.resolve().is_relative_to(root.resolve()):
        raise PermissionError('NEW_REGISTERED_DIAGNOSTIC_OUTPUT_REQUIRED')
    import uuid
    import numpy as np
    # ResidentModel verifies native allocation before importing or loading NN.
    identity=read(request['identity'])
    if identity['global_step'] != 512 or request['resident_operation'] != 'TRAIN_DIAGNOSTIC':
        raise PermissionError('M512_DIAGNOSTIC_ONLY')
    b=read(args.bindings);lp=root/'HOPPER_RESOURCE_LEDGER.json'
    l=read(lp);l['model_loads']+=1;write(lp,l)
    model=ResidentModel(root,Path(args.request),b,request['checkpoint'],identity)
    plan=read(request['diagnostic_plan'])
    if sha(request['diagnostic_plan']) != request['diagnostic_plan_sha256'] or plan['run_uuid'] != RUN or not 0<len(plan['draws'])<=48:
        raise PermissionError('SEALED_BOUNDED_DIAGNOSTIC_DRAW_PLAN_REQUIRED')
    output.mkdir(parents=True,exist_ok=False)
    dataset=NativeDataset(b['dataset'],b['source'],b['cosmos'],raw_bindings=read(b['raw_bindings']))
    dataset.initialize()
    indices={(r['id'],a):i for i,(r,a) in enumerate(dataset.rows)}
    results=[]
    try:
        for n,draw in enumerate(plan['draws']):
            state,expert,images,text=dataset.raw(indices[(draw['episode_id'],draw['anchor'])])
            obs=payload(state,images,text,snapshot=str(uuid.uuid4()))
            l=read(lp)
            if l['generation_queries']>=LIMITS['generation_queries']:
                raise PermissionError('GLOBAL_R1_QUERY_BUDGET')
            l['generation_queries']+=1;write(lp,l)
            write(output/f'INTENT-{n:03d}.json',dict(draw=draw,query_index=l['generation_queries'],no_resampling=True))
            generated=model.generate(obs,draw['seed'])
            decoded,control=expert_decode(model.processor,state,expert)
            generated_gate=gate_action(generated['physical_H40'],state,draw['anchor']/20,str(uuid.uuid4()))
            expert_gate=gate_action(decoded,state,draw['anchor']/20,str(uuid.uuid4()))
            arrays=output/f'{n:03d}.npz'
            np.savez_compressed(arrays,state=state,expert_absolute=expert,expert_decoded=decoded,
                generated_normalized=generated['normalized'],generated_absolute=generated['physical_H40'])
            value=dict(draw=draw,arrays_sha256=sha(arrays),expert_control=control,expert_gate=expert_gate,
                generated_gate=generated_gate,errors=error_metrics(generated['physical_H40'],expert,state),
                expert_codec_errors=error_metrics(decoded,expert,state),noise=generated['noise'],
                physics_integrations=0,extra_correction=False,model_has_teacher_labels=False)
            write(output/f'{n:03d}.json',value);results.append(value)
        value=dict(status='ACTUAL_M512_OFFLINE_GENERATION_NOT_CLOSED_LOOP_OR_SKILL',run_uuid=RUN,
            draws=len(results),expert_positive_control='PASS' if all(r['expert_gate']['decision'] in ('ACCEPT','WARN') for r in results) else 'FAIL',
            expert_rejections=sum(r['expert_gate']['decision']=='REJECT' for r in results),
            model_rejections=sum(r['generated_gate']['decision']=='REJECT' for r in results),
            plan_sha256=request['diagnostic_plan_sha256'],selected='NONE',deployment='NONE')
        write(output/'RESULT.json',value)
        return value
    finally:
        owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','request','bindings'):p.add_argument('--'+name,required=True)
    run(p.parse_args())
