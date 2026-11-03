"""Finite single-GPU FP32 WM01 fit and full checkpoint reload."""
import argparse
import json
from pathlib import Path
import time

RECIPE = dict(members=3, hidden_layers=3, width=512, effective_batch=256,
    lr=.0003, weight_decay=.00001, gradient_clip=1., horizon=4,
    state_loss_weight=1., rollout_loss_weight=1., event_loss_weight=.1,
    preexecution_event_sampling='observed prevalence; termination logit only; no physical target',
    precision='float32', smoke_updates=200, initial_updates=5000)


def fit(spec, data, output, *, device, endpoint, resume=None):
    import os
    import copy
    import random
    import numpy as np
    import torch
    from go2arx5_vla.world_model.replay import Replay
    from go2arx5_vla.world_model.model import Ensemble, multistep_loss
    from go2arx5_vla.world_model.checkpoint import save, restore
    from go2arx5_vla.world_model.storage import write_json, digest
    if endpoint not in (200, 5000) or device != 'cuda':
        raise ValueError('Native W1 endpoint requires one actual GPU at 200 or 5000 updates')
    if spec.get('reward', {}).get('status') != 'VALIDATED_FROZEN':
        raise ValueError('Freeze and validate the real-state reward before optimization')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(1103); np.random.seed(1103); torch.manual_seed(1103)
    replay = Replay(data, split='TRAIN')
    identity = dict(manifest_sha256=replay.manifest_sha256, source_sha=replay.source_sha,
                    complete_macros=replay.complete_macros, windows=len(replay.keys))
    mean, scale = replay.normalizer()
    model = Ensemble(mean, scale).to(device)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.0003, weight_decay=.00001) for m in model.members]
    bootstrap = [np.random.default_rng(1103+member*100003) for member in range(3)]
    start = 0
    if resume:
        state = restore(resume, model, optimizers, bootstrap, replay_identity=identity, device=device)
        if state['recipe'] != RECIPE:
            raise ValueError('Scientific recipe changed on resume')
        start = state['updates']
    if start >= endpoint:
        raise ValueError('Resume endpoint must advance the existing run')
    began, cpu = time.monotonic(), time.process_time()
    torch.cuda.reset_peak_memory_stats()
    write_json(output/'FIT_IDENTITY.json', dict(data=identity, recipe=RECIPE,
        source_sha=spec['source_sha'], parameters=sum(p.numel() for p in model.parameters()),
        device=torch.cuda.get_device_name(), GPU_uuid=spec.get('GPU_uuid'),
        event_positive_counts=replay.event_counts, partial_macros=replay.partial_macros,
        event_only_preexecution_rejections=len(replay.rejected_states),
        normalization_split='TRAIN', initialization_seed=1103, independent_bootstrap=True))
    snapshots = []

    def update():
        losses = []
        sample_ids = []
        for member, optimizer, rng in zip(model.members, optimizers, bootstrap):
            values, ids = replay.sample(rng, 256)
            tensors = [torch.from_numpy(v).to(device) for v in values]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = multistep_loss(model, member, *tensors)
            if len(replay.rejected_states):
                rejected_ids = rng.integers(0,len(replay.rejected_states),size=256)
                rs,ra,rg = [torch.from_numpy(v[rejected_ids]).to(device) for v in
                            (replay.rejected_states,replay.rejected_actions,replay.rejected_goals)]
                _, rejection_logits = member(rs,ra,rg,model.mean,model.scale)
                rejection_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    rejection_logits[:,-1],torch.ones(256,device=device))
                _,real_logits=member(tensors[0][:,0],tensors[1][:,0],tensors[2],model.mean,model.scale)
                real_terminal_loss=torch.nn.functional.binary_cross_entropy_with_logits(real_logits[:,-1],tensors[4][:,0,-1])
                prevalence=len(replay.rejected_states)/(replay.complete_macros+len(replay.rejected_states))
                loss=loss+.1*.25*prevalence*(rejection_loss-real_terminal_loss)
                parts['rejection_event'] = rejection_loss.detach()
            if not torch.isfinite(loss):
                raise RuntimeError('NONFINITE_WM_LOSS_PRESERVED')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(member.parameters(), 1.)
            if not torch.isfinite(norm):
                raise RuntimeError('NONFINITE_WM_GRADIENT_PRESERVED')
            optimizer.step()
            losses.append({**{k:float(v) for k,v in parts.items()}, 'total':float(loss.detach()), 'grad_norm':float(norm)})
            sample_ids.append(ids)
        return losses, sample_ids

    for step in range(start+1, endpoint+1):
        if time.time() >= spec['deadline_epoch'] or (Path(spec['registered_root'])/'STOP.json').exists():
            save(output/f'interrupted-{step-1}.pt', model, optimizers, bootstrap, updates=step-1,
                 replay_identity=identity, source_sha=spec['source_sha'], recipe=RECIPE)
            raise TimeoutError('FINITE_WM_TRAINING_ENDPOINT')
        losses, _ = update()
        if step % 10 == 0:
            with (output/'UPDATES.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(updates=step, epoch=time.time(), losses=losses))+'\n')
                stream.flush()
        if step in (200, endpoint):
            checkpoint = output/f'checkpoint-{step}.pt'
            save(checkpoint, model, optimizers, bootstrap, updates=step, replay_identity=identity,
                 source_sha=spec['source_sha'], recipe=RECIPE)
            # Exercise the *next* actual update twice, restore the saved endpoint
            # afterwards; diagnostic updates have separate explicit costs.
            expected_losses, expected_ids = update()
            expected = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            restore(checkpoint, model, optimizers, bootstrap, replay_identity=identity, device=device)
            actual_losses, actual_ids = update()
            maximum = max(float((expected[k]-v.detach().cpu()).abs().max()) for k,v in model.state_dict().items())
            ids_equal = all(np.array_equal(a,b) for a,b in zip(expected_ids, actual_ids))
            if maximum != 0 or not ids_equal or actual_losses != expected_losses:
                raise RuntimeError('ACTUAL_WM_NEXT_UPDATE_RESTORE_MISMATCH')
            restore(checkpoint, model, optimizers, bootstrap, replay_identity=identity, device=device)
            snapshots.append(dict(updates=step, checkpoint_sha256=digest(checkpoint),
                next_update_max_abs_error=maximum, sample_indices_equal=ids_equal,
                diagnostic_ensemble_updates=2, actual_member_optimizer_steps=6))
    torch.cuda.synchronize()
    result = dict(source_sha=spec['source_sha'], replay=identity, start_update=start,
        optimizer_updates=endpoint, actual_new_ensemble_updates=endpoint-start,
        member_optimizer_steps=(endpoint-start)*3,
        restore_diagnostic_ensemble_updates=len(snapshots)*2,
        actor_updates=0, new_native_physics_ticks=0, checkpoints=snapshots,
        CPU_s=time.process_time()-cpu, GPU_wall_s=time.monotonic()-began,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        event_calibration='TERMINAL_HEAD_UNCALIBRATED_IF_NO_POSITIVE_NATIVE_TERMINALS')
    write_json(output/'RESULT.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--endpoint', type=int, choices=(200, 5000), default=200)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(args.spec, args.run_id, args.profile, args.output, gpu=True) as spec:
        print(json.dumps(fit(spec, args.data, args.output, device='cuda', endpoint=args.endpoint, resume=args.resume)), flush=True)


if __name__ == '__main__': main()
