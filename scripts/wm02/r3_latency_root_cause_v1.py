"""Read-only R3_LATENCY_ROOT_CAUSE_V1 reconstruction; never imports a model.

Monotonic durations, journal wall-clock spans and filesystem write-time proxies
are deliberately separate. Missing internal timestamps stay None. Archive-copy
mtimes are never used as inference/consumer timestamps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = 'R3_LATENCY_ROOT_CAUSE_V1'
COMPONENTS = (
    'src/go2arx5_vla/sla_adapter/provider_time_shift7_v1.py',
    'src/go2arx5_vla/sla_adapter/provider_time_v1.py',
    'src/go2arx5_vla/sla_adapter/provider_v3.py',
    'src/go2arx5_vla/sla_adapter/model_v1.py',
    'src/go2arx5_vla/sla_adapter/dataset_v1.py',
    'src/go2arx5_vla/sla_adapter/codec_v1.py',
    'src/go2arx5_vla/sla_adapter/batch_v2.py',
    'src/go2arx5_vla/sla_adapter/gripper_v2.py',
    'src/go2arx5_vla/sla_adapter/gripper_time_v1.py',
    'src/go2arx5_vla/sla_adapter/gripper_time_shift7_v1.py',
    'src/go2arx5_vla/sla_adapter/time_shift7_training_v1.py',
    'scripts/sla/close_goal_g1_v1.py',
    'scripts/sla/mfree_c20_vectorfield_v3.py')


def percentiles(values):
    values=[float(x) for x in values if x is not None]
    return dict(n=len(values),p50=None,p95=None,p99=None,max=None) if not values else dict(
        n=len(values),**dict(zip(('p50','p95','p99','max'),map(float,np.percentile(values,[50,95,99,100])))))


def original_write_proxy(evidence, emitted):
    """Only SHA-verified original live files, never the archived copy times."""
    metadata={r['file']:r for r in evidence['source_file_metadata']}
    def stamp(name):
        item=metadata.get(name,{})
        if not item.get('original_sha_matches'):
            return None
        return item['original_stat']['mtime_ns']/1e9
    first=stamp('receiver/emitted-0001-00.json')
    install=stamp('receiver/install-0001.json')
    final=stamp('receiver/live.json')
    span=None if install is None or final is None else final-install
    return dict(kind='ORIGINAL_FILE_WRITE_TIMES_NOT_MONOTONIC_FUNCTION_OR_PUBLICATION_EVENTS',
        install_write_to_final_live_write_s=span,
        first_atom_write_to_final_live_write_s=None if first is None or final is None else final-first,
        consume_only_conditional_enclosure_s=span if emitted==10 and span is not None and span>=0 else None,
        enclosure_excludes_finish_final_fsync=True,
        assumptions=['Stable historical wall clock','Filesystem write-time order reflects original writes'],
        first_atom_publication_monotonic_s=None,
        first_atom_status='NOT_EMITTED' if emitted==0 else 'EMITTED_BUT_WALL_PUBLICATION_TIME_NOT_RECORDED')


def reconstruct(rows, evidence):
    if len(rows)!=32 or [r['index']for r in rows]!=list(range(32)):
        raise ValueError('EXACT_SEALED32_REQUIRED_NO_REPLACEMENTS')
    indexed={e['index']:e for e in evidence}
    points=[]
    for row in rows:
        e=indexed[row['index']];q=e['documents']['REQUEST.private.json']
        available=q['created_monotonic']-row['input_available_to_request_s']
        capture=available-row['file_delivery_s'];dispatch=row['dispatch_monotonic']
        response=dispatch+row['IPC_to_response_s']
        terminal=available+row['input_available_to_terminal_s']
        if abs(capture-row['scheduled_monotonic']-row['release_lateness_s'])>1e-8:
            raise ValueError('CAPTURE_CLOCK_RECONSTRUCTION_MISMATCH')
        provider=sum(row['model_branch_s']);worker=e['documents']['RESPONSE.json']['worker_wall_s']
        durations=dict(file_delivery_s=available-capture,available_to_request_s=q['created_monotonic']-available,
            request_to_dispatch_s=dispatch-q['created_monotonic'],provider_call_s=provider,
            worker_outside_provider_s=worker-provider,transport_publication_poll_residual_s=row['IPC_to_response_s']-worker,
            response_to_install_return_or_rejection_s=terminal-response,
            available_to_terminal_s=terminal-available,capture_to_terminal_s=terminal-capture,
            seal_enqueue_s=row['seal_enqueue_s'])
        nested=sum(durations[k]for k in ('available_to_request_s','request_to_dispatch_s','provider_call_s',
            'worker_outside_provider_s','transport_publication_poll_residual_s','response_to_install_return_or_rejection_s'))
        if abs(nested-durations['available_to_terminal_s'])>1e-8 or min(durations.values())< -1e-8:
            raise ValueError('NONNEGATIVE_NESTED_PHASE_CONSERVATION_REQUIRED')
        journal=e['reservation'];stat=e['reservation_stat']
        points.append(dict(index=row['index'],family=row['family'],phase=row['phase_score_only'],noise=row['noise'],
            cold=row['cold'],ACK=bool(row['ACK']),emitted_mock_atoms=row['emitted_mock_atoms'],
            TTL_rejected=row['TTL_rejected'],durations=durations,
            events=dict(capture=capture,available=available,request=q['created_monotonic'],dispatch=dispatch,
                response=response,install_return_or_rejection=terminal,model_completed=None,
                model_completion_causal_range=(dispatch+provider,response)),
            internal_unmeasured=dict(tokenizer_s=None,host_to_device_completion_s=None,GPU_kernel_s=None,
                synchronization_s=None,physical_decode_s=None,compile_only_s=None,
                install_publication_monotonic_s=None,mock_consume_and_finish_s=None,
                progress_records_durable_write_s=None,goal_state_durable_write_s=None),
            reservation=dict(INTENT_durable_and_noop_wall_s=journal['completed_epoch']-journal['created_epoch'],
                created_to_final_rename_ctime_proxy_s=stat['ctime_ns']/1e9-journal['created_epoch'],
                complete_record_final_directory_fsync_s=None,capacity_wait_s=row['reservation_wait_capacity_s'],
                wall_span_is_not_full_reserve_monotonic_duration=True),
            mock_write_proxy=original_write_proxy(e,row['emitted_mock_atoms']),cycle_to_next=None))
    for i,p in enumerate(points[:-1]):
        nxt=points[i+1];post=nxt['events']['capture']-p['events']['install_return_or_rejection']
        response=p['events']['install_return_or_rejection']-p['events']['dispatch']
        next_input=nxt['events']['dispatch']-nxt['events']['capture']
        total=nxt['events']['dispatch']-p['events']['dispatch']
        seal=p['durations']['seal_enqueue_s'];residual=post-seal
        if min(post,residual)<-1e-8 or abs(total-response-seal-residual-next_input)>1e-8:
            raise ValueError('CYCLE_CONSERVATION_REQUIRED')
        p['cycle_to_next']=dict(dispatch_interval_s=total,dispatch_to_terminal_s=response,
            post_terminal_to_next_capture_s=post,seal_enqueue_s=seal,
            unresolved_postresponse_excluding_seal_s=residual,next_capture_to_dispatch_s=next_input,
            mock_consume_and_finish_s=None,other_and_progress_and_next_reservation_s=None,
            intentional_release_wait_upper_bound_s=max(0.,rows[i+1]['scheduled_monotonic']-p['events']['install_return_or_rejection']),
            no_intentional_release_wait_provable=p['events']['install_return_or_rejection']>=rows[i+1]['scheduled_monotonic'])
    return points


def source_alignment(bundle):
    specs=[bundle['R1_spec'],bundle['R3_worker_spec']]
    hashes={key:{r['file']:r['sha256']for r in value['python_files']}for key,value in bundle['sources'].items()}
    components=[dict(file=name,R1_sha256=hashes['R1'].get(name),R3_sha256=hashes['R3'].get(name),
        identical=hashes['R1'].get(name) is not None and hashes['R1'].get(name)==hashes['R3'].get(name))for name in COMPONENTS]
    # Effective R1 launch overrides are checked against the preserved launcher
    # separately; spec alone is not represented as a process environment dump.
    env1=dict(specs[0]['native_environment'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','SSH_AUTH_SOCK','DISPLAY'):env1[key]=''
    env2=bundle['R3_spec']['gpu_environment']
    keys=('XLA_FLAGS','JAX_PLATFORMS','XLA_PYTHON_CLIENT_MEM_FRACTION','JAX_DEFAULT_MATMUL_PRECISION',
          'JAX_COMPILATION_CACHE_DIR','JAX_ENABLE_COMPILATION_CACHE','XLA_PYTHON_CLIENT_ALLOCATOR',
          'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LD_LIBRARY_PATH','LD_PRELOAD')
    return dict(components=components,effective_declared_environment=[dict(key=key,R1=env1.get(key),R3=env2.get(key),same=env1.get(key)==env2.get(key))for key in keys],
        environment_scope='Effective declared launch recipe; inherited user-manager environment was not captured',
        same_python_executable=specs[0]['python']==specs[1]['python'],same_GPU_uuid=specs[0]['GPU_uuid']==specs[1]['GPU_uuid'],
        same_configured_CPU_affinity=specs[0]['cpus']==specs[1]['cpus'],configured_logical_CPUs=len(specs[0]['cpus']),
        R3_controller_and_GPU_child_share_parent_CPU_affinity=True,
        hardware_clock_thermal_utilization_and_offCPU_time='NOT_RECORDED_DURING_ORIGINAL_RUNS',
        same_package=specs[0]['SHIFT7_package']==specs[1]['SHIFT7_package'],package_manifest_sha256=specs[0]['SHIFT7_package']['sha256'],
        same_parent_manifest=specs[0]['parent_manifest']==specs[1]['parent_manifest'],
        parent_manifest_sha256=specs[0]['parent_manifest']['sha256'],
        JIT_cache=dict(declared_paths_differ=True,explicit_binding_in_original_R1_R3_entrypoints=False,
            historical_effective_cache_configuration='NOT_CAPTURED',persistent_cache_misses_or_retraces='NOT_CAPTURED',
            hot_in_process_JIT_cache_not_equivalent_to_persistent_disk_cache=True))


def analyze(bundle, metadata):
    points=reconstruct(bundle['R3_rows'],bundle['requests']);hot=points[1:];transitions=points[1:-1]
    origin=points[0]['events']['capture']
    matched=[]
    for r in bundle['matched']:
        equal=all(v['bit_exact']for v in r['input_arrays'].values())and r['language_equal']and r['previous_reference_equal']
        matched.append(dict(R3_index=r['R3_index'],R1_entry=r['R1_entry'],episode=r['episode'],anchor=r['anchor'],noise=r['noise'],
            provider_inputs_bit_exact=equal,inputs=r['input_arrays'],outputs=r['outputs'],
            R1_provider_s=r['R1_provider_s'],R3_provider_s=r['R3_provider_s'],difference_s=r['R3_provider_s']-r['R1_provider_s'],
            ratio=r['R3_provider_s']/r['R1_provider_s'],causal_matched_experiment=False))
    for p in points:
        e=p['events']
        p['age']=dict(available_to_request_s=e['request']-e['available'],available_to_install_return_or_rejection_s=e['install_return_or_rejection']-e['available'],
            request_to_install_return_or_rejection_s=e['install_return_or_rejection']-e['request'],
            model_completion_since_available_range_s=[v-e['available']for v in e['model_completion_causal_range']],
            model_completion_exact_s=None,first_atom_exact_age_s=None,
            first_atom_status=p['mock_write_proxy']['first_atom_status'])
        p['timeline_since_cold_capture_s']={k:(None if v is None else [x-origin for x in v]if isinstance(v,tuple)else v-origin)for k,v in e.items()}
        del p['events']
    summaries=dict(hot_request_durations={key:percentiles([p['durations'][key]for p in hot])for key in hot[0]['durations']},
        warm30_cycles={key:percentiles([p['cycle_to_next'][key]for p in transitions])for key in (
            'dispatch_interval_s','dispatch_to_terminal_s','post_terminal_to_next_capture_s','seal_enqueue_s',
            'unresolved_postresponse_excluding_seal_s','next_capture_to_dispatch_s')},
        warm30_without_intentional_release_wait=sum(p['cycle_to_next']['no_intentional_release_wait_provable']for p in transitions),
        hot_reservation_INTENT_wall_s=percentiles([p['reservation']['INTENT_durable_and_noop_wall_s']for p in hot]),
        hot_reservation_created_to_rename_ctime_proxy_s=percentiles([p['reservation']['created_to_final_rename_ctime_proxy_s']for p in hot]),
        full10_original_write_enclosure_proxy_s=percentiles([p['mock_write_proxy']['consume_only_conditional_enclosure_s']for p in hot]),
        full10_count=sum(p['emitted_mock_atoms']==10 for p in points),zero_atom_count=sum(p['emitted_mock_atoms']==0 for p in points),
        completed32_first_atom_timestamps_recorded=0,accepted_input_available_age_at_least400ms_indices=[p['index']for p in points if p['ACK']and p['age']['available_to_install_return_or_rejection_s']>=.4],
        archive_capacity_wait_s=sum(p['reservation']['capacity_wait_s']for p in points))
    return dict(schema=SCHEMA,scope='READ_ONLY_FORENSICS_NOT_A_NEW_EXECUTION_OR_RESCORING',
        sealed_publication_sha='3497840724c6bfcf54ec4317a5ed605290310b14',
        historical_R1_source=bundle['R1_spec']['source_sha'],historical_R3_source=bundle['R3_spec']['source_sha'],
        historical_R1_result_sha256='0321115e51307b6e16d2444d5494c359b71cd8c7cd0cecae442fcd65cb08a3c0',
        historical_R3_result_sha256='28a2869e4ab0169382c9c640432d4ced5aea7b421a9635e09744ec41e354a5dc',
        effects=dict(new_generations=0,new_updates=0,new_training_visits=0,new_native_requests=0,new_integrations=0),
        historical_R3_status_unchanged='BLOCKED_REALTIME',historical_guard_and_quarantine_unchanged=True,
        original_live_requests_SHA_verified=sum(e['original_live_exists']and all(f['original_exists']and f.get('original_sha_matches',False)for f in e['source_file_metadata'])for e in bundle['requests']),
        source_alignment=source_alignment(bundle),currently_observed_runtime_metadata=metadata.get('runtime_metadata'),
        declared_cache_directories_now=metadata.get('declared_cache_dirs'),matched_existing_conditions=matched,
        summaries=summaries,requests=points,
        identifiability=dict(exact='Original same-host monotonic endpoint and duration algebra',
            wall_clock='Journal created/completed epochs; not complete reservation duration',
            proxy='SHA-verified original file write times; not publication/consume event timers',
            unavailable=['GPU kernel and transfer timing','Tokenizer and decode internal phase timing',
                'Exact model completion timestamp','Actual install publication and first-atom timestamps',
                'Separate mock consume+finish and root progress write timings','Historical device clocks and offCPU scheduling']),
        global_causal_root_cause='NOT_IDENTIFIED_WITH_EXISTING_TIMESTAMPS',no_new_GPU_scope_granted=True,native_readmission=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bundle');p.add_argument('metadata');p.add_argument('output');args=p.parse_args()
    raw=Path(args.bundle).read_bytes();result=analyze(json.loads(raw),json.loads(Path(args.metadata).read_bytes()))
    result['private_input_bundle_sha256']=hashlib.sha256(raw).hexdigest()
    with Path(args.output).open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
