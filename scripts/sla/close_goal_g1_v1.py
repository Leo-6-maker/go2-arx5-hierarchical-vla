"""Bounded, zero-update complete-sampler factorial for the reviewed CLOSE goal."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from scripts.pi0292.durable import OwnerLock, write
from go2arx5_vla.world_model.mfree_unattended_ledger_v1 import read, sha
from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze

ROUTES = ('cached_actual', 'cached_highest', 'full_actual', 'full_highest')


def wait_resource_owner(spec):
    """Finite existing-device capacity queue, without claiming another owner's GPU."""
    import subprocess
    root=Path(spec['registered_root'])
    while time.time()<spec['deadline_epoch']-300 and not (root/'STOP.json').exists():
        try:lock=OwnerLock(spec['neural_owner_lock'])
        except BlockingIOError:lock=None
        free_host=int(next(l.split()[1]for l in Path('/proc/meminfo').read_text().splitlines()
                           if l.startswith('MemAvailable:')))*1024
        query=subprocess.run(['nvidia-smi','--id='+spec['GPU_uuid'],'--query-gpu=memory.free',
                              '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        free_gpu=int(query.stdout.strip())*1024**2 if query.returncode==0 else None
        ready=lock is not None and free_gpu is not None and free_host>=spec['minimum_host_available_bytes'] and free_gpu>=spec['allocator_limit_bytes']+1024**3
        if ready:return lock
        if lock:lock.close()
        write(root/'goal_state.json',dict(schema='SLA_CLOSE_SAFE_MOBILE_GOAL_STATE_V1',phase='WAIT_RESOURCE',
            epoch=time.time(),pid=os.getpid(),source_sha=spec['source_sha'],plan_sha=spec['goal_plan_sha'],
            free_host_bytes=free_host,free_gpu_bytes=free_gpu,canonical_owner_available=lock is not None,
            new_optimizer_updates=0,new_training_visits=0,new_complete_generations=0,new_physical_requests=0,
            goal_met=False,deadline_epoch=spec['deadline_epoch']))
        time.sleep(30)
    raise TimeoutError('FINITE_G1_RESOURCE_QUEUE_DEADLINE_OR_STOP')


def width_metrics(prediction, expert):
    """Mechanical width; crossing is scored only for an annotated in-window event."""
    width = .002 + .040 * np.asarray(prediction)[:, 9]
    target = .002 + .040 * np.asarray(expert)[:, 9]
    crossing = lambda x: next((int(k) for k, v in enumerate(x) if v <= .022), None)
    truth_cross = crossing(target) if target[0] > .022 and target[-1] <= .022 else None
    predicted_cross = crossing(width)
    return dict(width_MAE_m=float(abs(width-target).mean()),
                terminal_width_error_m=float(abs(width[-1]-target[-1])),
                predicted_width_change_m=float(width[-1]-width[0]),
                expert_width_change_m=float(target[-1]-target[0]),
                expert_crossing_in_window=truth_cross, predicted_first_crossing=predicted_cross,
                crossing_error_s=None if truth_cross is None or predicted_cross is None
                else .05*abs(truth_cross-predicted_cross),
                missed_in_window_crossing=truth_cross is not None and predicted_cross is None,
                peak_finger_speed_m_s=float(abs(np.diff(width)/.05).max()))


def summarize(rows):
    groups = {}
    for route in ROUTES:
        items = [r for r in rows if r['route'] == route]
        if not items:
            continue
        close = [r for r in items if r['window']['phase'] == 'CLOSE']
        other = [r for r in items if r['window']['phase'] != 'CLOSE']
        events = [r for r in items if r['width']['expert_crossing_in_window'] is not None]
        groups[route] = dict(predictions=len(items), H40_pass=sum(r['envelope']['original_full_H40_pass'] for r in items),
            CLOSE_count=len(close), CLOSE_pass=sum(r['envelope']['original_full_H40_pass'] for r in close),
            nonCLOSE_count=len(other), nonCLOSE_pass=sum(r['envelope']['original_full_H40_pass'] for r in other),
            terminal_width_MAE_m=float(np.mean([r['width']['terminal_width_error_m'] for r in items])),
            annotated_crossing_count=len(events), crossing_within100ms=sum(
                r['width']['crossing_error_s'] is not None and r['width']['crossing_error_s'] <= .10000000001 for r in events),
            missed_crossings=sum(r['width']['missed_in_window_crossing'] for r in events),
            latency_after_first_call_s=[r['synchronized_wall_s'] for r in items[1:] if r.get('synchronized_wall_s') is not None])
    return groups


def choose_global(summary):
    """Predeclared TRAIN-only ranking. No per-window or per-noise route switches."""
    base = summary['cached_actual']
    eligible = [r for r in ROUTES if summary[r]['nonCLOSE_pass'] >= base['nonCLOSE_pass']]
    # The original path wins ties; no numerical change without a CLOSE gain.
    winner = max(eligible, key=lambda r: (summary[r]['CLOSE_pass'], -ROUTES.index(r)))
    return winner if summary[winner]['CLOSE_pass'] > base['CLOSE_pass'] else 'cached_actual'


def make_sampler(graph, fields, *, cached, precision):
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx

    def call(active, frozen, observation, noise):
        with jax.default_matmul_precision(precision):
            model = nnx.merge(graph, active, frozen)
            prefix = fields['prefix_field'](model, observation) if cached else None
            initial = (noise, jnp.asarray(1., jnp.float32), jnp.asarray(0, jnp.int32),
                       jnp.zeros((10,)+noise.shape, noise.dtype),
                       jnp.zeros((10,)+noise.shape, noise.dtype), jnp.zeros((10,), jnp.float32))

            def step(carry):
                x_t, tau, i, states, velocities, times = carry
                velocity = (fields['cache_field'](model, *prefix, x_t, tau) if cached else
                            fields['official_field'](model, observation, x_t,
                                                     jnp.broadcast_to(tau, observation.state.shape[0])))
                return (x_t + (-1.0/10)*velocity, tau + (-1.0/10), i+1,
                        states.at[i].set(x_t), velocities.at[i].set(velocity), times.at[i].set(tau))

            return jax.lax.while_loop(lambda c: c[1] >= .05, step, initial)

    return jax.jit(call)


def semantic_masks(model, obs, noise):
    import jax.numpy as jnp
    from openpi.models import model as upstream
    from openpi.models.pi0 import make_attn_mask
    obs = upstream.preprocess_observation(None, obs, train=False)
    prefix, pm, pa = model.embed_prefix(obs)
    suffix, sm, sa, _ = model.embed_suffix(obs, noise, jnp.ones((1,), jnp.float32))
    mask = jnp.concatenate([pm, sm], axis=1)
    full = make_attn_mask(mask, jnp.concatenate([pa, sa]))
    size = prefix.shape[1]
    cached_suffix = jnp.concatenate([jnp.broadcast_to(pm[:, None, :], (1, suffix.shape[1], size)),
                                    make_attn_mask(sm, sa)], axis=-1)
    positions = jnp.cumsum(mask, axis=1)-1
    return dict(prefix_cannot_see_suffix=jnp.logical_not(full[:, :size, size:]).all(),
                prefix_attention_equal=jnp.array_equal(full[:, :size, :size], make_attn_mask(pm, pa)),
                suffix_attention_equal=jnp.array_equal(full[:, size:, :], cached_suffix),
                prefix_positions_equal=jnp.array_equal(positions[:, :size], jnp.cumsum(pm, axis=1)-1),
                suffix_positions_equal=jnp.array_equal(positions[:, size:], jnp.sum(pm, axis=-1)[:, None]+jnp.cumsum(sm, axis=-1)-1))


def run(spec):
    import jax
    import jax.numpy as jnp
    import jaxlib
    import flax.nnx as nnx
    from openpi.models import pi0
    from openpi_client.image_tools import resize_with_pad
    from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields, PI0_SHA, difference
    from scripts.sla.export_mfree_c20_v1 import _reference
    from go2arx5_vla.sla_adapter import batch_v2
    from go2arx5_vla.sla_adapter.mfree_training_v1 import MobileDataset
    from go2arx5_vla.sla_adapter.provider_mfree_v1 import Provider
    from go2arx5_vla.sla_adapter.model_v1 import observation
    from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec
    from go2arx5_vla.sla_adapter.codec_v1 import pad32

    root = Path(spec['registered_root']); previous = Path(spec['previous_campaign'])
    old = Path(spec['C20_root']); started = time.time(); counters = dict(new_complete_generations=0,
        new_extra_field_evaluations=0, reused_complete_predictions=0, optimizer_updates=0,
        optimization_visits=0, physical_requests=0, native_integrations=0)
    ledger = root/'effects.jsonl'

    def state(phase, **more):
        write(root/'goal_state.json', dict(schema='SLA_CLOSE_SAFE_MOBILE_GOAL_STATE_V1', phase=phase,
            epoch=time.time(), source_sha=spec['source_sha'], plan_sha=spec['goal_plan_sha'],
            counters=counters, old_C20_updates=1000, old_C20_visits=32000,
            quarantine='F1_F2_AND_STUDENTS_AND_M_ASSIST_RETAINED', goal_met=False, **more))

    def effect(kind, **details):
        with ledger.open('a') as handle:
            handle.write(json.dumps(dict(epoch=time.time(), kind=kind, **details))+'\n')
            handle.flush(); os.fsync(handle.fileno())

    state('NUMERIC_DIAGNOSIS_LOAD_MODEL')
    assert sha(inspect.getfile(pi0)) == PI0_SHA
    fields, extracted = extracted_fields(batch_v2, pi0)
    (root/'EXTRACTED_FIELDS.private.py').write_text(extracted)
    bank = read(previous/'diagnostic/WINDOW48_PLAN.private.json')['windows']
    selected = read(previous/'vector-field-v3/FIELD_PLAN.private.json')['windows']
    assert len(selected) == 16 and sum(w['phase']=='CLOSE' for w in selected) == 8
    assert len({w['index'] for w in selected}) == 16
    write(root/'FIXED_INPUT_PLAN.private.json', dict(factorial=selected, confirmation=bank, noises=[0,1],
        noise_seed_base=2026111200, routes=ROUTES, steps=10,
        selection='MAX_CLOSE_WITH_NONCLOSE_NO_REGRESSION_TIE_ORIGINAL', extra_field_cases='NOISE0_D_TRAJECTORY_STEPS3_AND7_16_OBSERVATIONS_THREE_OTHER_ROUTES'))
    jax.config.update('jax_compilation_cache_dir', str(root/'cache'))
    model_result = read(old/'candidate1000/RESULT.private.json')
    dataset = MobileDataset(old/'dataset', sha(old/'dataset/MANIFEST.private.json'),
                            resize_with_pad=resize_with_pad, max_cache_bytes=512*1024**2)
    provider = Provider(model_result['package']['path'], manifest_sha256=model_result['package']['sha256'],
        parent_params=spec['raw_parent'], parent_manifest_path=spec['parent_manifest']['path'])
    actual_precision = jax.config.jax_default_matmul_precision
    assert actual_precision is None, 'REVIEWED_ACTUAL_DEFAULT_PRECISION_CHANGED'
    samplers = {route: make_sampler(provider.graph, fields, cached=route.startswith('cached'),
                 precision='highest' if route.endswith('highest') else actual_precision) for route in ROUTES}
    field_functions = {}
    for route in ROUTES:
        def factory(route):
            def call(active, frozen, obs, x, tau):
                with jax.default_matmul_precision('highest' if route.endswith('highest') else actual_precision):
                    m = nnx.merge(provider.graph, active, frozen)
                    if route.startswith('cached'):
                        pref = fields['prefix_field'](m, obs)
                        return fields['cache_field'](m, *pref, x, tau)
                    return fields['official_field'](m, obs, x, jnp.broadcast_to(tau, (1,)))
            return jax.jit(call)
        field_functions[route] = factory(route)
    rows = []; trajectories = {}; source_rows = {}
    previous_rows = read(previous/'diagnostic/TRAIN48_PREDICTIONS.private.json')
    prior = {(r['index'], r['noise']): r for r in previous_rows}

    def context(w):
        index = w['index']; ep, anchor = dataset.rows[index]
        state10, expert, _, _ = dataset.raw(index)
        frame = dataset.frames[ep['id']]['T_policy_from_mount']; codec = GripLogitCodec(frame)
        if ep['id'] not in source_rows:
            capture = read(ep['original_live_capture']['path'])
            source_rows[ep['id']] = read(capture['labels']['path'])
        native_spec = read(Path(ep['native_result']['path']).parent/'EXECUTION_SPEC.json')
        obs, _ = observation(dataset[index], provider.tokenizer)
        return state10, expert, codec, obs, np.asarray(native_spec['policy_from_world']), _reference(source_rows[ep['id']][anchor])

    def decode(codec, state10, array):
        return codec.decode(state10, pad32(provider.statistics['actions'].unnormalize(array[0, :, :7])))

    def record(w, noise_id, route, *, stage, reused=False):
        key = (w['index'], noise_id, route)
        if key in trajectories:
            return trajectories[key]
        state10, expert, codec, obs, frame, reference = context(w)
        if reused:
            saved = prior[(w['index'], noise_id)]
            assert sha(saved['prediction']['path']) == saved['prediction']['sha256']
            with np.load(saved['prediction']['path'], allow_pickle=False) as z:
                normalized = z['normalized'].copy(); physical = z['physical_H40'].copy()
            trace = None; latency = None; counters['reused_complete_predictions'] += 1
        else:
            if (root/'STOP.json').exists() or time.time() > spec['deadline_epoch']-180:
                raise TimeoutError('FINITE_G1_STOP_BEFORE_NEXT_COMPLETE_GENERATION')
            assert counters['new_complete_generations'] < 320
            state(stage, next_window=w['index'], noise=noise_id, route=route)
            effect('COMPLETE_GENERATION_STARTED', index=w['index'], noise=noise_id, route=route)
            noise = jax.random.normal(jax.random.PRNGKey(2026111200+noise_id), (1,40,32))
            start = time.perf_counter(); out = samplers[route](provider.active, provider.frozen, obs, noise)
            jax.block_until_ready(out); latency = time.perf_counter()-start
            normalized, final_tau, steps, xs, vs, times = map(np.asarray, out)
            assert int(steps) == 10 and np.isfinite(normalized).all()
            counters['new_complete_generations'] += 1
            physical = decode(codec, state10, normalized)
            width = np.stack([.002+.040*decode(codec, state10, x)[:,9] for x in xs])
            trace = dict(x_tau=xs, vector_field=vs, flow_times=times, width_from_x_tau_m=width,
                         final_tau=final_tau)
            effect('COMPLETE_GENERATION_COMMITTED', index=w['index'], noise=noise_id, route=route,
                   seconds=latency, Euler_steps=10)
        file = root/'data'/f"{w['index']:04d}-n{noise_id}-{route}.npz"
        with file.open('xb') as h:
            np.savez(h, normalized=normalized, physical_H40=physical, **(trace or {}))
        row = dict(window=w, noise=noise_id, route=route, stage=stage, reused=reused,
                   output=dict(path=str(file), sha256=sha(file)),
                   normalized_sha256=hashlib.sha256(normalized.tobytes()).hexdigest(),
                   envelope=analyze(physical, state10, frame, reference), width=width_metrics(physical, expert),
                   synchronized_wall_s=latency, allocator=jax.devices()[0].memory_stats())
        if route=='cached_actual' and not reused:
            with np.load(prior[(w['index'],noise_id)]['prediction']['path'], allow_pickle=False) as z:
                row['original_provider_cache_comparison'] = difference(normalized, z['normalized'])
        rows.append(row); trajectories[key] = row
        write(root/'ALL_SAMPLER_RECORDS.private.json', rows)
        state(stage, completed_records=len(rows))
        return row

    # Semantics use identical real tensors, not shape-only comparisons.
    _, _, _, first_obs, _, _ = context(selected[0])
    def masks(a, f, obs):
        return semantic_masks(nnx.merge(provider.graph,a,f), obs, jnp.zeros((1,40,32),jnp.float32))
    semantic = jax.jit(masks)(provider.active, provider.frozen, first_obs)
    semantic = {k:bool(v) for k,v in semantic.items()}
    assert all(semantic.values()), 'FULL_CACHE_ATTENTION_OR_POSITION_SEMANTIC_MISMATCH'
    write(root/'SEMANTIC_ACTUAL.private.json', dict(checks=semantic,
        cache_reset='NEW_PREFIX_FOR_EVERY_COMPLETE_SAMPLER_CALL_NEVER_SHARED_ACROSS_OBSERVATIONS_OR_MODELS',
        versions=dict(jax=jax.__version__, jaxlib=jaxlib.__version__, device=str(jax.devices()[0])),
        XLA_FLAGS=os.environ.get('XLA_FLAGS'), actual_precision=actual_precision,
        official_pi0_sha256=PI0_SHA, extra_field_evaluations=0))
    for w in selected:
        for noise_id in (0,1):
            for route in ROUTES:
                record(w, noise_id, route, stage='NUMERIC_FACTORIAL_128')
    factorial = summarize(rows); chosen = choose_global(factorial)
    write(root/'FACTORIAL_SUMMARY.private.json', dict(routes=factorial, chosen_global_route=chosen))
    local = []; accumulated = []
    for w in selected:
        state10, expert, codec, obs, _, _ = context(w)
        for noise_id in (0,1):
            arrays = {}
            for route in ROUTES:
                with np.load(trajectories[(w['index'],noise_id,route)]['output']['path'],allow_pickle=False) as z:
                    arrays[route] = {k:z[k].copy() for k in ('x_tau','vector_field','flow_times','normalized','physical_H40')}
            baseline = arrays['cached_actual']
            for route in ROUTES[1:]:
                current = arrays[route]
                accumulated.append(dict(index=w['index'],noise=noise_id,route=route,
                    same_initial_noise=bool(np.array_equal(current['x_tau'][0],baseline['x_tau'][0])),
                    first_same_state_field_difference=difference(current['vector_field'][0],baseline['vector_field'][0]),
                    per_step_state_max_abs=[float(abs(x-y).max()) for x,y in zip(current['x_tau'],baseline['x_tau'])],
                    final_normalized=difference(current['normalized'],baseline['normalized']),
                    final_width_max_difference_m=float(.040*abs(current['physical_H40'][:,9]-baseline['physical_H40'][:,9]).max()),
                    legal_before=trajectories[(w['index'],noise_id,'cached_actual')]['envelope']['original_full_H40_pass'],
                    legal_after=trajectories[(w['index'],noise_id,route)]['envelope']['original_full_H40_pass']))
            if noise_id == 0:
                reference = arrays['full_highest']
                for k in (3,7):
                    for route in ROUTES[:-1]:
                        assert counters['new_extra_field_evaluations'] < 96
                        x = reference['x_tau'][k]; tau = reference['flow_times'][k]
                        value = np.asarray(field_functions[route](provider.active,provider.frozen,obs,jnp.asarray(x),jnp.asarray(tau)))
                        counters['new_extra_field_evaluations'] += 1
                        clean_left = x-tau*value; clean_right=x-tau*reference['vector_field'][k]
                        a=decode(codec,state10,clean_left); b=decode(codec,state10,clean_right)
                        local.append(dict(index=w['index'],noise=0,step=k,route=route,tau=float(tau),
                            same_D_state_field_difference=difference(value,reference['vector_field'][k]),
                            clean_width_speed_max_difference_m_s=float(abs(np.diff(.040*(a[:,9]-b[:,9]))/.05).max())))
    write(root/'SAME_STATE_LOCAL96.private.json',local)
    write(root/'ACCUMULATED_COMPARISONS.private.json',accumulated)
    for w in bank:
        for noise_id in (0,1):
            record(w,noise_id,'cached_actual',stage='NUMERIC_CONFIRM48',reused=True)
            if chosen!='cached_actual':record(w,noise_id,chosen,stage='NUMERIC_CONFIRM48')
    confirm = [trajectories[(w['index'],n,r)] for w in bank for n in (0,1)
               for r in dict.fromkeys(['cached_actual',chosen])]
    confirmation = summarize(confirm)
    picked = confirmation[chosen]
    offline_goal = (picked['CLOSE_pass']>=18 and picked['nonCLOSE_pass']==76
                    and picked['missed_crossings']==0 and picked['crossing_within100ms']==picked['annotated_crossing_count']
                    and all(r['width']['terminal_width_error_m']<=.001 for r in confirm
                            if r['route']==chosen and r['width']['expert_crossing_in_window'] is not None))
    result = dict(schema='SLA_CLOSE_GOAL_G1_COMPLETE_SAMPLER_ACTUAL_V1',exit_code=0,
        factorial=factorial,confirmation=confirmation,chosen_global_route=chosen,
        numerical_route_meets_offline_targets=offline_goal,G2_needed=not offline_goal,
        numeric_conclusion='TRAIN_GLOBAL_ROUTE_GAIN' if chosen!='cached_actual' else 'BOUNDED_NUMERICAL_DIFFERENCES_WITHOUT_CLOSE_GAIN',
        counters=counters, original_provider_comparison=[r['original_provider_cache_comparison'] for r in rows
            if 'original_provider_cache_comparison' in r], semantic_checks=semantic,
        elapsed_wall_s=time.time()-started, CPU_s=time.process_time(),
        GPU_model=str(jax.devices()[0]), memory=jax.devices()[0].memory_stats(),
        result_pins={p.name:sha(p) for p in root.glob('*.private.json')},
        physical_admission=False,goal_met=False,training_submitted=False)
    write(root/'RESULT.private.json',result)
    state('READY_FOR_REVIEW',G2_needed=not offline_goal,chosen_global_route=chosen,result_sha256=sha(root/'RESULT.private.json'))
    return result


if __name__=='__main__':
    import sys
    s=read(sys.argv[2]); root=Path(s['registered_root']); lock=None
    try:lock=wait_resource_owner(s);run(s); code=0
    except BaseException:
        write(root/'RESULT.private.json',dict(exit_code=1,failure=traceback.format_exc(),
            last_state=read(root/'goal_state.json') if (root/'goal_state.json').exists() else None,
            physical_requests=0,optimizer_updates=0,goal_met=False)); code=1
    finally:
        if lock:lock.close()
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    close(s,code)
    raise SystemExit(code)
