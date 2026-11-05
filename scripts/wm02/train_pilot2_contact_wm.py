"""Admitted small direct-contact WM comparison with exact sampler restoration."""
import argparse
import gzip
import json
import random
import time
from pathlib import Path


def run(spec, output):
    import numpy as np
    import torch
    from go2arx5_vla.world_model.pilot2_contact_model import replay, DirectContactWM, metrics
    from go2arx5_vla.world_model.actor_sampling import ExactSampler
    from go2arx5_vla.world_model.neural_training import rng_state, restore_rng, tree_error, memory
    from go2arx5_vla.world_model.storage import digest, write_json
    from go2arx5_vla.world_model.runtime import cuda_uuid
    start, cpu = time.monotonic(), time.process_time(); device = torch.device(spec['device'])
    if spec['kind'] != 'CONTACT_WM' or spec.get('contact_recipe') != 'WM02_PILOT2_DIRECT_CONTACT_V1':
        raise ValueError('DECLARED_PILOT2_DIRECT_CONTACT_RECIPE_REQUIRED')
    if type(spec.get('endpoint')) is not int or spec['endpoint'] < 2:
        raise ValueError('FINITE_ENDPOINT_WITH_RESTORE_PREDECESSOR_REQUIRED')
    torch.set_num_threads(len(spec['cpus'])); torch.use_deterministic_algorithms(True)
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        if torch.cuda.device_count() != 1 or cuda_uuid(props.uuid) != spec['GPU_uuid']:
            raise PermissionError('ONE_ACTUALLY_REGISTERED_GPU_REQUIRED')
        torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes']/props.total_memory)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]; begin.record()
    episodes = []
    for item in spec['bundles']:
        if digest(item['path']) != item['sha256']: raise ValueError('SEALED_REAL_CONTACT_BUNDLE_CHANGED')
        bundle = json.loads(gzip.decompress(Path(item['path']).read_bytes()))
        if (bundle['schema'] != 'WM02_PILOT2_SUPPORTED_ACTUAL_BUNDLE_V1'
                or bundle['complete_MPC_WBC_hidden_state'] is not False):
            raise ValueError('ACTUAL_SUPPORTED_HISTORY_NOT_FULL_SIMULATOR_REQUIRED')
        episodes.extend(bundle['episodes'])
    data = replay(episodes); train, dev = [np.flatnonzero(data['split']==s) for s in ('TRAIN', 'DEV')]
    train = train[data['valid'][train].any(1)]
    if not len(train) or not len(dev): raise ValueError('NONEMPTY_WHOLE_PARENT_TRAIN_DEV_REQUIRED')
    tensor = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)
    history = tensor(data['history']); plan = tensor(data['plan']); blind = tensor(data['blind_plan'])
    canonical = torch.cat((history, plan), -1); y, event = tensor(data['delta_mm']), tensor(data['events'])
    valid = tensor(data['valid']); arms = {}; total = 0
    frequency = np.zeros((len(data['horizons']), 4))
    for j in range(len(data['horizons'])):
        ix = train[data['valid'][train, j]]
        if not len(ix): raise ValueError('REAL_TRAIN_FORECASTS_FOR_EVERY_DECLARED_HORIZON_REQUIRED')
        frequency[j] = data['events'][ix, j].mean(0)
    constant = np.broadcast_to(frequency, data['events'].shape)
    persistence = np.repeat(data['current_events'][:, None], len(data['horizons']), 1)
    cv = data['cv_mm'][:, None] * np.asarray(data['horizons'])[None, :, None]
    baselines = {name: {s: metrics(delta, probability, data, ix) for s, ix in [('TRAIN', train), ('DEV', dev)]}
        for name, delta, probability in [('PERSISTENCE', np.zeros_like(data['delta_mm']), persistence),
            ('CONSTANT_VELOCITY', cv, persistence), ('TRAIN_EVENT_FREQUENCY', np.zeros_like(cv), constant)]}
    write_json(output/'BASELINES.json', baselines)
    identities = [dict(run_id=e['run_id'], parent=e['parent'], split=e['split']) for e in episodes]
    prior_roots = spec.get('resume_arms', {})
    for arm, future in [('ACTION_CONDITIONED', plan), ('NO_FUTURE_PROGRAM', blind)]:
        random.seed(spec['seed']); np.random.seed(spec['seed']); torch.manual_seed(spec['seed'])
        model = DirectContactWM(history.shape[1], plan.shape[1], len(data['horizons']), spec['hidden']).to(device)
        model.fit_normalizer(canonical[train]); opt = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=1e-4)
        sampler = ExactSampler(len(train), 'SEEDED_GLOBAL_SHUFFLE', spec['seed']); previous_updates = 0
        if prior_roots:
            item = prior_roots[arm]
            if digest(item['path']) != item['sha256']: raise ValueError('PRIOR_CONTACT_CHECKPOINT_CHANGED')
            prior = torch.load(item['path'], map_location='cpu', weights_only=True)
            if (prior['schema'] != 'WM02_PILOT2_DIRECT_CONTACT_CHECKPOINT_V1' or prior['arm'] != arm
                    or prior['horizons'] != list(data['horizons']) or prior['seed'] != spec['seed']
                    or any(e not in identities for e in prior['episode_identities'])):
                raise ValueError('CONTACT_CONTINUATION_MUST_RETAIN_OLD_PARENT_SPLITS_AND_EPISODES')
            model.load_state_dict(prior['model']); opt.load_state_dict(prior['optimizer'])
            previous_updates = prior['updates']; restore_rng(prior['rng'])
            if prior['bundles'] == spec['bundles']: sampler.load_state_dict(prior['sampler'])
            elif spec.get('data_expansion_normalizer') != 'FROZEN_PRIOR_TRAIN_NORMALIZER_NEW_SHUFFLE':
                raise ValueError('NEW_DATA_CONTINUATION_NORMALIZER_AND_SAMPLER_MUST_BE_EXPLICIT')
        x = torch.cat((history, future), -1); folder = output/arm; folder.mkdir()
        def update(m, o, s):
            ix = torch.as_tensor(train[s.take(spec['batch_size'])], device=device)
            pred = m(x[ix]); mask = valid[ix]
            state_loss = (pred[..., :6]-y[ix]).square().mean(-1)
            event_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred[..., 6:], event[ix], reduction='none').mean(-1)
            value = ((state_loss+event_loss)*mask).sum()/mask.sum()
            o.zero_grad(set_to_none=True); value.backward()
            if not torch.isfinite(value) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in m.parameters()):
                raise FloatingPointError('NONFINITE_CONTACT_LOSS_OR_GRADIENT')
            o.step(); return float(value.detach())
        def measure(step):
            with torch.no_grad():
                pred = model(x); delta = pred[..., :6].cpu().numpy(); prob = pred[..., 6:].sigmoid().cpu().numpy()
            v = {s: metrics(delta, prob, data, ix) for s, ix in [('TRAIN', train), ('DEV', dev)]}
            write_json(folder/f'METRICS-{step}.json', v); return v
        def save(step):
            path = folder/f'checkpoint-{step}.pt'
            torch.save(dict(schema='WM02_PILOT2_DIRECT_CONTACT_CHECKPOINT_V1', arm=arm, seed=spec['seed'],
                source_sha=spec['source_sha'], model=model.state_dict(), optimizer=opt.state_dict(),
                rng=rng_state(), sampler=sampler.state_dict(), updates=step, bundles=spec['bundles'],
                episode_identities=identities, horizons=list(data['horizons']),
                normalization='OWN_TRAIN_ONLY_OR_EXPLICIT_FROZEN_PRIOR',
                full_state_simulator=False), path)
            return path
        measure(previous_updates)
        for step in range(1, spec['endpoint']+1):
            if time.time() >= spec['deadline_epoch'] or (Path(spec['registered_root'])/'STOP.json').exists():
                raise TimeoutError('FINITE_CONTACT_WORKER_ENDPOINT_OR_STOP')
            loss = update(model, opt, sampler); total += 1
            with (folder/'OPTIMIZER_STEPS.jsonl').open('a') as log:
                log.write(json.dumps(dict(update=previous_updates+step, new_update=step,
                    loss=loss, sample_draws=sampler.draws))+'\n')
            if step == spec['endpoint']-1 or step % 500 == 0 or step == spec['endpoint']:
                save(previous_updates+step)
            if step % 500 == 0 and step != spec['endpoint']: measure(previous_updates+step)
        final_step = previous_updates+spec['endpoint']; final_path = save(final_step) if not (folder/f'checkpoint-{final_step}.pt').exists() else folder/f'checkpoint-{final_step}.pt'
        saved_rng = rng_state()
        clone = DirectContactWM(history.shape[1], plan.shape[1], len(data['horizons']), spec['hidden']).to(device)
        co = torch.optim.AdamW(clone.parameters(), lr=spec['learning_rate'], weight_decay=1e-4)
        prior = torch.load(folder/f'checkpoint-{final_step-1}.pt', map_location='cpu', weights_only=True)
        clone.load_state_dict(prior['model']); co.load_state_dict(prior['optimizer']); restore_rng(prior['rng'])
        cs = ExactSampler(len(train), 'SEEDED_GLOBAL_SHUFFLE', spec['seed']); cs.load_state_dict(prior['sampler'])
        write_json(folder/'RESTORE_PROBE_STARTED.json', dict(probe_optimizer_calls_requested=1))
        update(clone, co, cs)
        write_json(folder/'RESTORE_PROBE_EXECUTED.json', dict(actual_optimizer_calls=1))
        error = max(tree_error(model.state_dict(), clone.state_dict()), tree_error(opt.state_dict(), co.state_dict()),
            tree_error(sampler.state_dict(), cs.state_dict()), tree_error(saved_rng, rng_state()))
        endpoint = torch.load(final_path, map_location='cpu', weights_only=True)
        reload = max(tree_error(endpoint['model'], model.state_dict()), tree_error(endpoint['optimizer'], opt.state_dict()),
            tree_error(endpoint['sampler'], sampler.state_dict()))
        if error != 0 or reload != 0: raise ValueError('COMPLETE_DIRECT_CONTACT_RESTORE_DIFFERS')
        arms[arm] = dict(main_WM_updates=spec['endpoint'], prior_WM_updates=previous_updates,
            restore_probe_optimizer_calls=1, full_restore_error=error, endpoint_reload_error=reload,
            checkpoint_sha256=digest(final_path), last_loss=loss, metrics=measure(final_step))
    elapsed = None
    if device.type=='cuda': end.record(); torch.cuda.synchronize(); elapsed = begin.elapsed_time(end)/1000
    result = dict(status='DIRECT_MULTIHORIZON_CONTACT_WM_TRAINED_AND_RESTORED', source_sha=spec['source_sha'],
        bundles=spec['bundles'], main_WM_updates=total, restore_probe_optimizer_calls=2, Actor_updates=0,
        native_integrations=0, TRAIN_anchors=len(train), DEV_anchors=len(dev),
        TRAIN_parents=sorted({e['parent'] for e in episodes if e['split']=='TRAIN'}),
        DEV_parents=sorted({e['parent'] for e in episodes if e['split']=='DEV'}),
        horizons_atoms=list(data['horizons']), arms=arms, baselines=baselines,
        invalid_or_fallback_crossing_forecasts_excluded=int(data['valid'].size-data['valid'].sum()),
        TEST_read=False, free_rollout_qualified=False, native_policy_gain_evaluated=False,
        GPU_elapsed_event_s=elapsed, GPU_memory=memory(device), CPU_s=time.process_time()-cpu,
        wall_s=time.monotonic()-start)
    write_json(output/'RESULT.json', result); return result


def main(argv=None):
    from go2arx5_vla.world_model.runtime import admitted_run
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(argv)
    with admitted_run(a.spec, a.output.name, 'wm02-pilot1-neural', a.output, gpu=True) as s:
        print(json.dumps(run(s, a.output)))
    return 0


if __name__=='__main__': raise SystemExit(main())
