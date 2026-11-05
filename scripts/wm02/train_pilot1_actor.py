"""Run an admitted local four-action Actor job with full save/restore evidence."""
import argparse
import json
from pathlib import Path


def task_definition(spec,prior,data,meta,train_ids):
    """A coverage experiment must keep the original task and score frozen."""
    if spec.get('frozen_task'):
        from go2arx5_vla.world_model.storage import digest
        path=Path(spec['frozen_task'])
        if digest(path)!=spec['frozen_task_sha256']:raise ValueError('FROZEN_ACTOR_TASK_CHANGED')
        task=json.loads(path.read_text())
        if task.get('schema')!='LOCAL_EEF_REACH_PILOT_V1'or prior and task!=prior['task']:
            raise ValueError('ORIGINAL_LOCAL_PILOT_TASK_REQUIRED')
        return task
    if prior:return prior['task']
    from go2arx5_vla.world_model.pilot1_actor import task_from_train
    return task_from_train(data,meta,train_ids)


def run(spec, output):
    import hashlib, random, time
    import numpy as np
    import torch
    from go2arx5_vla.world_model.pilot1_actor import (Actor, TorchARX, examples, export_arrays, numpy_logits, task_from_train, features, observed_feedback_loss)
    from go2arx5_vla.world_model.target_arx import CausalARX
    from go2arx5_vla.world_model.target_replay import load
    from go2arx5_vla.world_model.target_replay_v2 import select_samples
    from go2arx5_vla.world_model.neural_training import rng_state, restore_rng, tree_error, memory
    from go2arx5_vla.world_model.runtime import cuda_uuid
    from go2arx5_vla.world_model.storage import digest, write_json
    started, cpu = time.monotonic(), time.process_time()
    device = torch.device('cuda:0')
    props = torch.cuda.get_device_properties(0)
    if torch.cuda.device_count() != 1 or cuda_uuid(props.uuid) != spec['GPU_uuid']:
        raise PermissionError('ONE_ACTUAL_LOCAL_GPU_REQUIRED')
    torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes'] / props.total_memory)
    torch.set_num_threads(len(spec['cpus']))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if digest(spec['replay']) != spec['replay_sha256'] or digest(spec['ARX']) != spec['ARX_sha256']:
        raise ValueError('FROZEN_REAL_REPLAY_OR_ARX_CHANGED')
    data, meta = load(spec['replay'])
    train_ids, dev_ids = (select_samples(data, meta, split) for split in ('TRAIN', 'DEV'))
    with np.load(spec['ARX'], allow_pickle=False) as arrays:
        teacher = CausalARX.from_arrays(meta, arrays)
    prior = None
    if spec.get('resume'):
        if digest(spec['resume']) != spec['resume_sha256']:
            raise ValueError('FROZEN_ACTOR_CHECKPOINT_CHANGED')
        prior = torch.load(spec['resume'], map_location='cpu', weights_only=True)
    task = task_definition(spec,prior,data,meta,train_ids)
    write_json(output / 'TASK.private.json', task)
    # Equivalent differentiability check uses existing TRAIN histories only.
    ix = train_ids[:4]
    values = dict(prefix=data['state'][ix, :9], goal=data['goal'][ix], origin=data['window_start_s'][ix],
                  age=data['source_age'][ix, 8], remaining=teacher.remaining(data, ix))
    reference = teacher.predict(data, ix)
    errors = {}
    for dtype, label in ((torch.float64, 'FP64'), (torch.float32, 'FP32')):
        port = TorchARX(teacher, dtype).to(device)
        tensors = {k: torch.as_tensor(v, dtype=dtype, device=device) for k, v in values.items()}
        prediction = port(**tensors)
        errors[label] = {str(h): float(np.max(np.abs(prediction.detach().cpu().numpy()[:, h-1] - reference[:, h-1]))) for h in (1, 8, 16)}
        errors[label+'_by_state_group']={name:{str(h):float(np.max(np.abs(prediction.detach().cpu().numpy()[:,h-1,sl]-reference[:,h-1,sl])))for h in(1,8,16)}for name,sl in teacher.layout.items()}
        if label == 'FP64':
            if max(errors[label].values()) > 1e-10:
                raise ValueError('FP64_ARX_EQUIVALENCE_FAILED')
            tensors['goal'] = tensors['goal'].clone().requires_grad_(True)
            scalar = port(**tensors)[0, -1, teacher.layout['tcp_position_m'].start]
            gradient = torch.autograd.grad(scalar, tensors['goal'])[0][0, 0].item()
            numeric = []
            for sign in (-1, 1):
                changed = dict(tensors); changed['goal'] = tensors['goal'].detach().clone()
                changed['goal'][0, 0] += sign * 1e-7
                numeric.append(port(**changed)[0, -1, teacher.layout['tcp_position_m'].start].item())
            difference = (numeric[1] - numeric[0]) / 2e-7
            errors['model_gradient'] = dict(autograd=gradient, finite_difference=difference,
                absolute_error=abs(gradient-difference), native_physical_gradient_validated=False)
            if abs(gradient-difference) > 1e-5:
                raise ValueError('MODEL_AUTOGRAD_FINITE_DIFFERENCE_FAILED')
    write_json(output / 'ARX_EQUIVALENCE.json', errors)
    x, labels, scores = examples(teacher, data, meta, train_ids, task)
    dx, dlabels, dscores = examples(teacher, data, meta, dev_ids, task)
    random.seed(36001); np.random.seed(36001); torch.manual_seed(36001)
    model = Actor(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=1e-4)
    x, labels, scores = [torch.as_tensor(v, dtype=torch.float32, device=device) for v in (x, labels, scores)]
    dx, dlabels, dscores = [torch.as_tensor(v, dtype=torch.float32, device=device) for v in (dx, dlabels, dscores)]
    feedback=None
    if spec.get('feedback_recipe'):
        if spec['feedback_recipe']!='SUCCESS_CLONE_FAILURE_UNLIKELIHOOD_V1'or not prior:
            raise ValueError('EXPLICIT_RESUMED_ACTOR_FEEDBACK_RECIPE_REQUIRED')
        ids=np.asarray([i for i in train_ids if meta['episodes'][int(data['episode_index'][i])].get('pilot1_feedback')],dtype=np.int64)
        if not len(ids):raise ValueError('ACTUAL_NEW_TRAIN_OUTCOMES_REQUIRED')
        observed=[meta['episodes'][int(data['episode_index'][i])]['pilot1_feedback']for i in ids]
        fx=features(data,meta,ids,np.asarray([o['goal_offset_xy_m']for o in observed]))
        feedback=(torch.as_tensor(fx,dtype=torch.float32,device=device),
                  torch.as_tensor([o['action']for o in observed],dtype=torch.long,device=device),
                  torch.as_tensor([o['local_XY_success']for o in observed],dtype=torch.bool,device=device))
        write_json(output/'ACTUAL_FEEDBACK_AUDIT.json',dict(TRAIN_episodes=len(ids),successes=sum(o['local_XY_success']for o in observed),failures=sum(not o['local_XY_success']for o in observed),recipe=spec['feedback_recipe'],weight=spec['feedback_weight'],counterfactual_oracle=False,full_manipulation_successes=0))
    cursor, previous_updates = 0, 0
    if prior:
        if (prior['schema'] != 'WM02_PILOT1_ACTOR_CHECKPOINT_V1' or prior['ARX_sha256'] != spec['ARX_sha256']
                or any(prior['metadata'][k] != meta[k] for k in ('state_fields', 'joints', 'robot_fingerprint'))):
            raise ValueError('STRICT_PILOT1_ACTOR_RESUME_IDENTITY')
        model.load_state_dict(prior['model']); optimizer.load_state_dict(prior['optimizer'])
        restore_rng(prior['rng']); cursor = prior['cursor']; previous_updates = prior['cumulative_updates']
    else:
        model.fit_normalizer(x)
    def update(m, opt, at):
        ids = torch.arange(at, at + spec['batch_size'], device=device) % len(x)
        logits = m(x[ids]); loss = -(labels[ids] * torch.log_softmax(logits, -1)).sum(-1).mean()
        if feedback is not None:loss=loss+spec['feedback_weight']*observed_feedback_loss(m(feedback[0]),feedback[1],feedback[2])
        opt.zero_grad(set_to_none=True); loss.backward()
        gradient = float(torch.sqrt(sum(p.grad.square().sum() for p in m.parameters())))
        opt.step()
        return at + spec['batch_size'], float(loss.detach()), gradient
    def metrics(xx, yy, ss):
        with torch.no_grad():
            logits = model(xx); chosen = logits.argmax(-1); target = yy.argmax(-1)
            regret = ss.max(-1).values - ss.gather(1, chosen[:, None]).squeeze(1)
            return dict(examples=len(xx), teacher_argmax_agreement=float((chosen == target).float().mean()),
                        mean_model_regret_m=float(regret.mean()), selected_action_counts=torch.bincount(chosen, minlength=4).tolist(),
                        soft_cross_entropy=float(-(yy * torch.log_softmax(logits, -1)).sum(-1).mean()),
                        physical_control_benefit='NOT_ESTABLISHED_BY_DISTILLATION')
    def save(step):
        value = dict(schema='WM02_PILOT1_ACTOR_CHECKPOINT_V1', source_sha=spec['source_sha'], model=model.state_dict(),
                     optimizer=optimizer.state_dict(), rng=rng_state(), cursor=cursor, cumulative_updates=previous_updates+step,
                     current_job_updates=step, task=task, metadata=meta, ARX_sha256=spec['ARX_sha256'],
                     replay_sha256=spec['replay_sha256'], dimension=model.mean.numel())
        torch.save(value, output / f'checkpoint-{step}.pt')
    write_json(output / 'BEFORE_UPDATE.json', dict(TRAIN=metrics(x, labels, scores), DEV=metrics(dx, dlabels, dscores)))
    save(0)
    event0, event1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats(); event0.record()
    for step in range(1, spec['endpoint'] + 1):
        if time.time() >= spec['deadline_epoch'] or any((Path(spec[k]) / 'STOP.json').exists() for k in ('registered_root', 'campaign_root')):
            raise TimeoutError('PILOT1_FINITE_WORKER_OR_STOP')
        cursor, loss, gradient = update(model, optimizer, cursor)
        with (output / 'OPTIMIZER_STEPS.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(update=step, cumulative_update=previous_updates+step, cursor=cursor, loss=loss, gradient_norm=gradient)) + '\n')
        if step in set(spec['checkpoints']) | {spec['endpoint']-1, spec['endpoint']}:
            save(step)
            write_json(output / f'METRICS-{step}.json', dict(TRAIN=metrics(x, labels, scores), DEV=metrics(dx, dlabels, dscores)))
    event1.record(); torch.cuda.synchronize()
    checkpoint = output / f"checkpoint-{spec['endpoint']}.pt"
    expected_rng = rng_state()
    clone = Actor(model.mean.numel()).to(device)
    co = torch.optim.AdamW(clone.parameters(), lr=spec['learning_rate'], weight_decay=1e-4)
    saved = torch.load(output / f"checkpoint-{spec['endpoint']-1}.pt", map_location='cpu', weights_only=True)
    clone.load_state_dict(saved['model']); co.load_state_dict(saved['optimizer']); restore_rng(saved['rng'])
    next_cursor, _, _ = update(clone, co, saved['cursor'])
    restore_error = max(tree_error(model.state_dict(), clone.state_dict()), tree_error(optimizer.state_dict(), co.state_dict()), tree_error(expected_rng, rng_state()))
    if restore_error != 0 or next_cursor != cursor:
        raise ValueError('FULL_ACTOR_CONTINUATION_NOT_EXACT')
    endpoint_saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    endpoint_error = max(tree_error(model.state_dict(), endpoint_saved['model']), tree_error(optimizer.state_dict(), endpoint_saved['optimizer']))
    if endpoint_error != 0 or endpoint_saved['cursor'] != cursor:
        raise ValueError('ACTOR_ENDPOINT_RELOAD_DIFFERS')
    portable = export_arrays(model)
    path = output / 'ACTOR_NUMPY.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream, **portable)
    with torch.no_grad():
        held = dx[:256].cpu().numpy().astype(np.float64); expected = model(dx[:256]).cpu().numpy()
    actual = numpy_logits(portable, held)
    if np.max(np.abs(actual-expected)) > 5e-5 or not np.array_equal(actual.argmax(-1), expected.argmax(-1)):
        raise ValueError('PORTABLE_NATIVE_ACTOR_OUTPUT_DIFFERS')
    result = dict(status='SMALL_ACTOR_TRAINED_AND_RELOADABLE', source_sha=spec['source_sha'],
                  checkpoint_sha256=digest(checkpoint), portable_actor_sha256=digest(path), task_sha256=digest(output/'TASK.private.json'),
                  replay_sha256=spec['replay_sha256'], ARX_sha256=spec['ARX_sha256'], main_Actor_updates=spec['endpoint'],
                  prior_Actor_updates=previous_updates, restore_probe_optimizer_calls=1, complete_restore_error=restore_error,
                  endpoint_reload_error=endpoint_error,
                  portable_max_logit_error=float(np.max(np.abs(actual-expected))), TRAIN_windows=len(train_ids), DEV_windows=len(dev_ids),
                  real_feedback_examples=0 if feedback is None else len(feedback[0]),feedback_recipe=spec.get('feedback_recipe'),
                  TRAIN=metrics(x, labels, scores), DEV=metrics(dx, dlabels, dscores), WM_updates=0, native_integrations=0,
                  native_policy_evaluation='NOT_YET_EXECUTED', GPU_uuid=spec['GPU_uuid'], GPU_peak=memory(device),
                  CUDA_elapsed_ms=event0.elapsed_time(event1), CPU_s=time.process_time()-cpu, wall_s=time.monotonic()-started)
    write_json(output / 'RESULT.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('spec', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(args.spec, args.output.name, 'wm02-pilot1-neural', args.output, gpu=True) as spec:
        print(json.dumps(run(spec, args.output)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
