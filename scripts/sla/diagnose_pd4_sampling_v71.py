"""Exactly64 new PD4 RGB20 predictions; reuse the original64 RGB10 outputs.

One existing SLA parent stays resident. Both28-array packages and their own
statistics are verified before use; no optimizer, teacher or native world is
constructed. This diagnostic never changes the formal10-step policy.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import statistics
import time

import numpy as np

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.pd4_config_v71 import MANIFESTS, STATISTICS

OPERATION = 'SLA_PD4_FIXED500_SAMPLER20_DIAGNOSTIC_V71'
PROBES = [0,61,124,185,22,82,148,208,32,93,2,81,147,213,23,57]
BASELINE_METRICS = {
    'PD4_EXECUTED_REF':'95965f8b446fc2b3c13c07264a0a37a8c025546394288e884432fad38984a74b',
    'PD4_FINAL_PLAN':'411345bb7dda0acb81d15bb42a60395b206529e52f0cb1bca57b3ac84c26f5c4',
}


def read(path):
    return json.loads(Path(path).read_bytes())


def admission(spec):
    if (spec.get('operation') != OPERATION or spec.get('scope') != OPERATION
            or spec.get('physics') is not False or spec.get('native_execution_admitted') is not False
            or spec.get('optimizer_updates') != 0 or spec.get('planned_generations') != 64
            or spec.get('sampling_steps') != 20 or spec.get('original_sampler_steps') != 10
            or spec.get('original_generations_reused') != 64
            or spec.get('per_window_best_selection') is not False
            or spec.get('selected_indices') != PROBES or spec.get('noise_ids') != [0,1]):
        raise PermissionError('EXACT64_RGB20_ONLY_ORIGINAL_RGB10_REUSED_NO_TRAINING_OR_NATIVE')


def switch_package(provider, pin):
    """Same graph/frozen parent, verified new active leaves and frozen stats."""
    import jax
    from go2arx5_vla.sla_adapter.codec_v1 import Quantiles
    from go2arx5_vla.sla_adapter.provider_v1 import active_rows
    from go2arx5_vla.sla_adapter.provider_v3 import contract
    package = Path(pin['path'])
    assert sha(package/'MANIFEST.private.json') == pin['sha256']
    manifest = read(package/'MANIFEST.private.json'); contract(manifest)
    before = provider.manifest
    assert (manifest['step'], manifest['cursor'], manifest['sampler_steps']) == (500,16000,10)
    assert manifest['parent_manifest'] == before['parent_manifest']
    assert manifest['tokenizer_sha256'] == before['tokenizer_sha256']
    assert manifest['bindings']['frozen_fingerprint'] == provider.inventory['frozen_fingerprint']
    assert manifest['bindings'].get('parent_host_cast_v7',False) == before['bindings'].get('parent_host_cast_v7',False)
    assert sha(package/'STATISTICS.json') == manifest['statistics_sha256']
    assert sha(package/'paligemma_tokenizer.model') == manifest['tokenizer_sha256']
    stats = read(package/'STATISTICS.json')
    provider.manifest = manifest
    try:
        provider.validate_statistics_metadata(stats)
    finally:
        provider.manifest = before
    rows, structure = active_rows(dict(arrays=manifest['adapter_arrays']), provider.active)
    assert len(rows) == 28
    values = []
    for row in rows:
        path = package/row['file']
        assert path.resolve().is_relative_to(package.resolve()) and sha(path) == row['sha256']
        value = np.load(path,allow_pickle=False)
        assert list(value.shape) == row['shape'] and str(value.dtype) == row['dtype']
        values.append(jax.device_put(value))
    active, = jax.tree_util.tree_unflatten(structure,values)
    jax.block_until_ready(active)
    provider.active, provider.manifest, provider.package = active, manifest, package
    provider.statistics = {k:Quantiles(stats[k]['q01'],stats[k]['q99']) for k in ('state','actions')}
    return dict(adapter_arrays=28,package_manifest_sha256=pin['sha256'],
        statistics_sha256=manifest['statistics_sha256'],frozen_parent_reloaded=False,optimizer_loaded=False)


def input_digest(state, images, text, frame):
    digest = hashlib.sha256()
    for value in (state,*images,frame):
        value = np.asarray(value)
        digest.update(str((value.shape,str(value.dtype))).encode());digest.update(value.tobytes())
    digest.update(text.encode())
    return digest.hexdigest()


def fit(row):
    return bool(row['EEF_full_H40_pass']and row['xyz_RMSE_m']<=.002
        and row['mean_SO3_rad']<=.01 and row['nominal_grip_MAE']<=.02)


def paired_classification(before,after):
    return 'BOTH_PASS'if before and after else'ONLY10_PASS'if before else'ONLY20_PASS'if after else'BOTH_FAIL'


def run(spec):
    admission(spec)
    from go2arx5_vla.sla_adapter.site_v2 import validate
    site = validate(spec)
    from go2arx5_vla.sla_adapter.provider_pd4_v71 import Provider
    from go2arx5_vla.sla_adapter.pd4_training_v71 import PD4Dataset
    from go2arx5_vla.sla_adapter import model_v1
    from openpi_client.image_tools import resize_with_pad
    from scripts.sla.train_s16_offline_v7 import prediction_metrics
    from scripts.sla.diagnose_eef_margin_v71 import compare
    timing_spec=importlib.util.spec_from_file_location('sla_sampling_timer_isolated',
        Path(__file__).with_name('sampling_timing_v71.py'))
    timing_module=importlib.util.module_from_spec(timing_spec)
    timing_spec.loader.exec_module(timing_module)
    SamplingTimer=timing_module.SamplingTimer
    from scripts.pi0292.durable import write
    from scripts.pi0294.e1.artifacts import atomic_npz
    import jax
    import flax.nnx as nnx
    root, owner = Path(spec['registered_root']), Path(spec['PD4_owner'])
    assert sha(owner/'RESULT.private.json') == spec['PD4_result_sha256']
    terminal = read(owner/'RESULT.private.json')
    assert terminal['exit_code'] == 0 and read(owner/'control/LEASE.json')['status'] == 'CLOSED'
    assert (terminal['actual_optimizer_updates'],terminal['actual_training_visits']) == (1000,32000)
    arms = {a['view']:a for a in terminal['completed_arms']};assert set(arms) == set(MANIFESTS)
    assert jax.__version__ == '0.5.3'
    jax.config.update('jax_compilation_cache_dir', spec['compilation_cache_dir'])
    started = time.time();rows=[];restores=[];inputs={};noise_hashes={}
    attempts=loads=0;timer=None;provider=None
    try:
        for view in MANIFESTS:
            old = arms[view];candidate=Path(old['root'])/'candidate500'
            assert sha(Path(old['root'])/'RESULT.private.json') == old['result_sha256']
            original_spec=read(Path(old['root'])/'EXECUTION_SPEC.json')
            assert original_spec['parent_params'] == spec['parent_params']
            assert sha(candidate/'PREDICTIONS32.private.json') == BASELINE_METRICS[view]
            baseline=read(candidate/'PREDICTIONS32.private.json')
            assert [(r['index'],r['noise']) for r in baseline] == [(i,n) for i in PROBES for n in (0,1)]
            data=PD4Dataset(original_spec['dataset'],MANIFESTS[view],resize_with_pad=resize_with_pad,
                            max_cache_bytes=spec['frame_cache_bytes'])
            assert data.manifest['statistics']['sha256'] == STATISTICS[view]
            if provider is None:
                write(root/'MODEL_LOAD_INTENT.private.json',dict(package=old['package'],retries=False))
                provider=Provider(old['package']['path'],manifest_sha256=old['package']['sha256'],
                    parent_params=spec['parent_params'],parent_manifest_path=spec['parent_manifest']['path'])
                jax.block_until_ready((provider.active,provider.frozen));loads=1
                write(root/'MODEL_LOADED.private.json',dict(actual_SLA_parent_loads=1,teacher_loaded=False,optimizer_loaded=False))
                graph=provider.graph
                function=jax.jit(lambda active,frozen,obs,noise:nnx.merge(graph,active,frozen).sample_actions(
                    jax.random.PRNGKey(0),obs,noise=noise,num_steps=20))
                timer=SamplingTimer(function,jax_module=jax,model_module=model_v1)
                provider.infer=timer
                restores.append(dict(view=view,adapter_arrays=28,package_manifest_sha256=old['package']['sha256'],
                    statistics_sha256=STATISTICS[view],frozen_parent_reloaded=False,optimizer_loaded=False))
            else:
                restores.append(dict(view=view,**switch_package(provider,old['package'])))
            assert provider.manifest['statistics_sha256'] == STATISTICS[view]
            folder=root/view;folder.mkdir()
            for reference in baseline:
                if time.time()>=spec['deadline_epoch'] or (root.parent/'STOP.json').exists():
                    raise TimeoutError('FINITE_DIAGNOSTIC_ENDPOINT_OR_STOP_NO_RETRY')
                index,noise=reference['index'],reference['noise']
                state,_,images,text=data.raw(index);ep,anchor=data.rows[index]
                frame=data.frames[ep['id']]['T_policy_from_mount']
                digest=input_digest(state,images,text,frame)
                if index in inputs:assert digest == inputs[index],'PAIRED_ACTUAL_RGB_STATE_LANGUAGE_AND_FRAME'
                else:inputs[index]=digest
                attempts+=1;assert attempts<=64
                with(root/'GENERATION_INTENTS.private.jsonl').open('a')as f:
                    f.write(json.dumps(dict(view=view,index=index,noise=noise,sampling_steps=20,
                        attempt=attempts,prior_completed=len(rows),automatic_retry=False))+'\n')
                begin=time.perf_counter()
                with timer.prediction() as timing:
                    value=provider.predict(state,images,text,frame,noise_id=noise)
                prediction_wall=time.perf_counter()-begin
                if noise in noise_hashes:assert noise_hashes[noise] == timing['initial_noise_sha256']
                else:noise_hashes[noise]=timing['initial_noise_sha256']
                normalized,physical=value['normalized'],value['physical_H40']
                assert normalized.shape == (1,40,32) and physical.shape == (40,10)
                assert timing['observation_calls'] == 1 and timing['input_H2D_calls'] == 2
                name=f'{index:04d}-{noise}.npz';atomic_npz(folder/name,normalized=normalized,physical_H40=physical)
                metrics=prediction_metrics(data,index,normalized,physical)
                detail=compare(physical,data.labels[ep['id']]['nominal_policy_H40'][anchor],state)
                assert detail['EEF_pass'] == metrics['EEF_full_H40_pass']
                row=dict(view=view,index=index,noise=noise,sampling_steps=20,input_sha256=digest,
                    file=str((folder/name).relative_to(root)),
                    normalized_sha256=hashlib.sha256(normalized.tobytes()).hexdigest(),
                    physical_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),
                    baseline10=reference,**metrics,boundary_analysis=detail,
                    timing=dict(timing,predict_total_wall_s=prediction_wall))
                rows.append(row)
                with(root/'GENERATIONS.private.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                print(json.dumps(dict(completed=len(rows),planned=64,view=view,index=index,noise=noise,
                    EEF_pass=metrics['EEF_full_H40_pass'])),flush=True)
            del data;gc.collect()
        assert attempts == len(rows) == timer.submitted == timer.completed == 64
        summaries={}
        for view in MANIFESTS:
            group=[r for r in rows if r['view']==view]
            summaries[view]=dict(predictions=32,baseline10_EEF_pass=sum(r['baseline10']['EEF_full_H40_pass']for r in group),
                diagnostic20_EEF_pass=sum(r['EEF_full_H40_pass']for r in group),
                baseline10_fit_pass=sum(fit(r['baseline10'])for r in group),diagnostic20_fit_pass=sum(fit(r)for r in group),
                EEF_pair_classification=dict(Counter(paired_classification(r['baseline10']['EEF_full_H40_pass'],r['EEF_full_H40_pass'])for r in group)),
                fit_pair_classification=dict(Counter(paired_classification(fit(r['baseline10']),fit(r))for r in group)),
                median_timing_s={k:statistics.median(r['timing'][k]for r in group)for k in(
                    'input_H2D_s','inference_device_completed_s','output_D2H_s','predict_total_wall_s')},
                total_compilation_s=sum(r['timing']['compilation_s']for r in group),
                latency_scope='20_STEP_ONLY_WITH_DEVICE_COMPLETION; ORIGINAL10_COMPONENT_TIMING_UNAVAILABLE')
        result=dict(schema=OPERATION,status='FIXED64_RGB20_COMPLETE_DIAGNOSTIC_ONLY',exit_code=0,
            generation_attempts=attempts,model_executions_submitted=timer.submitted,
            model_executions_completed=timer.completed,actual_generations=len(rows),
            actual_SLA_parent_loads=loads,actual_compilations=timer.compilations,restores=restores,
            original64_RGB10_outputs_reused=True,original64_RGB10_rerun=False,
            fixed_indices=PROBES,noise_ids=[0,1],actual_initial_noise_sha256=noise_hashes,
            matched_raw_inputs_across_views=16,formal_sampler_steps=10,diagnostic_sampler_steps=20,
            normalized_losses_compared=False,matched10_device_timing_available=False,
            optimizer_updates=0,training_visits=0,inference_window_reads=64,physical_requests=0,physics_integrations=0,
            full_WBC_state_restored=False,native_execution_admitted=False,per_window_best_selection=False,
            extra_warmup_generations=0,automatic40step_extension=False,automatic_retry=False,
            source_prediction_metrics_sha256=BASELINE_METRICS,summaries=summaries,
            site=site,rows=rows,wall_s=time.time()-started)
        write(root/'RESULT.private.json',result)
        return result
    finally:
        write(root/'COUNTERS.private.json',dict(generation_attempts=attempts,
            model_executions_submitted=timer.submitted if timer else 0,
            model_executions_completed=timer.completed if timer else 0,
            completed_prediction_records=len(rows),actual_SLA_parent_loads=loads,
            model_load_completion_unknown=(root/'MODEL_LOAD_INTENT.private.json').exists()and loads==0,
            optimizer_updates=0,training_visits=0,physical_requests=0,physics_integrations=0))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('spec',type=Path);a=p.parse_args()
    result=run(read(a.spec));print(json.dumps({k:v for k,v in result.items()if k not in('rows','site')}))
