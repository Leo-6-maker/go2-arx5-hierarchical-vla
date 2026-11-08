"""Run a registered response cohort or frozen four-arm DEV panel with existing workers."""
import argparse, json, os, resource, subprocess, time
from contextlib import closing
from pathlib import Path


def run(input_path):
    from scripts.pi0292.durable import OwnerLock, write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.night10.native_relay import snapshot, once
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model.fusion_scope import validate, SCOPE
    from go2arx5_vla.world_model.fusion_identification import validate as validate_panel
    from go2arx5_vla.world_model.target_parallel import validate_layout
    inp = json.loads(input_path.read_text()); stage = input_path.parent
    evaluation = inp.get('study_kind')=='FROZEN_FOUR_ARM_DEV'
    base = json.loads(Path(inp['base_spec']).read_text()); source = Path(inp['source_root'])
    campaign = Path(base['campaign_root']); runs = Path(inp['output_parent'])
    lease_path = stage/'control/LEASE.json'; lease = json.loads(lease_path.read_text())
    if lease.get('status') != 'ACTIVE' or lease.get('input_sha256') != digest(input_path):
        raise PermissionError('FRESH_REGISTERED_FUSION_IDENTIFICATION_LEASE_REQUIRED')
    if (stage/'STARTED.private.json').exists(): raise FileExistsError('RECONCILE_EXISTING_FUSION_COHORT')
    results = []; qualifications = []; proofs = {}; code = 1; failure = None
    start = time.monotonic(); cpu = time.process_time(); usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    j = Journal(campaign/'control/journal', campaign.name)

    def new_root(root):
        root.mkdir()
        for name in ('control', 'data', 'cache', 'logs', 'tmp'): (root/name).mkdir()

    def child(root, row, slot, batch=None):
        new_root(root); scene = next(s for s in catalog['scenes'] if s['scene_id'] == row['scene_id'])
        episode = dict(inp['episode_contract'], **row, run=str(root))
        write(root/'EPISODE_MANIFEST.private.json', episode)
        s = dict(base, source_sha=inp['source_sha'], source_root=str(source), source_files=inp['source_files'],
            registered_root=str(root), role='WM02_'+root.name.upper().replace('-', '_'),
            registry_snapshot=str(root/'REGISTRY_SNAPSHOT.private.json'), lease=str(root/'control/LEASE.json'),
            deadline_epoch=min(time.time()+600, inp['deadline_epoch']), cpus=list(range(4*slot, 4*slot+4)),
            ROS_DOMAIN_ID=220+slot, episode_manifest=str(root/'EPISODE_MANIFEST.private.json'),
            episode_manifest_sha256=digest(root/'EPISODE_MANIFEST.private.json'), model_path=scene['model_path'])
        for name in ('scene_qualification', 'scene_qualification_sha256', 'parallel_batch_spec', 'slots'): s.pop(name, None)
        s['frozen_files'] = dict(base['frozen_files']); s['frozen_files'].pop(base['model_path'], None)
        s['frozen_files'][scene['model_path']] = scene['model_sha256']
        if row['purpose'] != 'SCENE_QUALIFICATION':
            proof = proofs[row['scene_id']]
            s.update(previous_stage=str(proof), previous_stage_sha256=digest(proof),
                scene_qualification=str(proof), scene_qualification_sha256=digest(proof))
        if batch: s['parallel_batch_spec'] = str(batch/'EXECUTION_SPEC.json')
        for name in ('data', 'cache', 'logs', 'tmp'): s[name] = str(root/name)
        write(root/'EXECUTION_SPEC.json', s)
        return s

    def grant(specs, key):
        reg, pin = snapshot(base['registry'])
        for s in specs:
            reg['entries'].append(dict(role=s['role'], path=s['registered_root'], run_uuid=Path(s['registered_root']).name,
                source_sha=s['source_sha'], owner=s['owner'], executor_node=s['host'], scope=SCOPE,
                pilot1_binding=s['pilot1_binding'], fusion_binding=s['fusion_binding'], local_lease=s['lease']))
        once(Path(base['registry']), reg, key+'-REGISTER', expected_before_sha256=pin)
        for s in specs:
            r = Path(s['registered_root']); write(r/'REGISTRY_SNAPSHOT.private.json', reg)
            write(Path(s['lease']), dict(status='ACTIVE', run_uuid=r.name, spec_sha256=digest(r/'EXECUTION_SPEC.json')))

    def worker(s, parallel=False):
        root = Path(s['registered_root']); env = dict(os.environ, **base['native_environment'])
        env.update(PYTHONPATH=str(source/'src')+':'+str(source)+':'+env.get('PYTHONPATH', ''),
            CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1', TMPDIR=str(root/'tmp'), ROS_LOG_DIR=str(root/'logs'),
            XDG_CACHE_HOME=str(root/'cache'), XDG_CONFIG_HOME=str(root/'cache'))
        argv = ['taskset', '-c', '0-7' if parallel else '0-3', s['python'], '-B', '-m',
            'scripts.wm02.pilot1_parallel_worker' if parallel else 'scripts.wm02.pilot1_native_worker',
            '--spec', str(root/'EXECUTION_SPEC.json')]
        def execute():
            with (root/'logs/worker-out.log').open('xb') as out, (root/'logs/worker-err.log').open('xb') as err:
                p = subprocess.Popen(argv, cwd=source, env=env, stdout=out, stderr=err)
                while p.poll() is None:
                    write(stage/'PROGRESS.private.json', dict(current_run=root.name, pid=p.pid,
                        completed_pairs=len(results), completed_qualifications=len(qualifications), updated_epoch=time.time()))
                    if time.time() >= s['deadline_epoch']: write(root/'STOP.json', dict(reason='FINITE_FUSION_IDENTIFICATION_ENDPOINT'))
                    time.sleep(.5)
            return dict(argv=argv, exit_code=p.returncode)
        return j.once(root.name.upper()+'-RUN', dict(spec_sha256=digest(root/'EXECUTION_SPEC.json')), execute)

    def observed(s, row):
        root = Path(s['registered_root']); n = root/'native-01'
        paths = dict(native=n/'RESULT.json', decision=n/'FUSION_DECISION.private.json', score=n/'FUSION_SCORE.private.json')
        values = {k:json.loads(p.read_text()) if p.exists() else {} for k,p in paths.items()}
        native, decision, score = [values[k] for k in ('native', 'decision', 'score')]
        passed = bool(native.get('status') == 'NATIVE_STAGE_PASSED' and native.get('shutdown_clean')
            and not native.get('physics_stops') and native.get('actual_target_integrations') == 4103)
        valid = bool(passed and decision.get('native_step') == 3703
            and (decision.get('selected_action') in range(4) if evaluation else decision.get('selected_action') == row['nominal_action'])
            and decision.get('nominal_action')==row['nominal_action'] and decision.get('future_response_observed')is False
            and decision.get('method') == row['method'] and decision.get('source_sha') == inp['source_sha']
            and decision.get('WM_sha256')==base.get('response_numpy_sha256',base['ARX_sha256'])
            and decision.get('static_model_sha256')==base['train_mean_response_model_sha256']
            and score.get('parent') == row['parent'] and score.get('split') == row['split']
            and score.get('selected_action')==decision.get('selected_action')
            and native.get('native_binary_sha256') == inp['native_binary_sha256'])
        return dict(run=root.name, row=row, attempted=n.exists(), native_passed=passed, evidence_valid=valid,
            native_ticks=native.get('actual_target_integrations', 0), score=score,
            evidence_sha256={k:digest(p) for k,p in paths.items() if p.exists()})

    try:
        if inp.get('study_kind')not in (None,'BALANCED_RESPONSE','FROZEN_FOUR_ARM_DEV'):
            raise ValueError('EXPLICIT_FUSION_STUDY_KIND_REQUIRED')
        if (not Path(__file__).resolve().is_relative_to(source.resolve()) or inp['max_native_slots'] != 2
                or time.time()+600 >= inp['deadline_epoch']): raise PermissionError('FINITE_QUALIFIED_LOCAL_TWO_SLOT_COHORT_REQUIRED')
        for name, pin in inp['source_files'].items():
            if digest(source/name) != pin: raise ValueError('IMMUTABLE_FUSION_SOURCE_CHANGED:'+name)
        if digest(inp['panel']) != inp['panel_sha256']: raise ValueError('PREREGISTERED_RESPONSE_PANEL_CHANGED')
        plan=json.loads(Path(inp['panel']).read_text())
        if evaluation:
            from go2arx5_vla.world_model.fusion_evaluation import validate as validate_panel
            from go2arx5_vla.world_model.fusion_response_inference import validate_asset
            validate_asset(base)
        rows = validate_panel(plan)
        validate(dict(base, registered_root=str(stage), deadline_epoch=inp['deadline_epoch']))
        catalog = json.loads(Path(base['scene_allowlist']).read_text())
        write(stage/'STARTED.private.json', dict(pid=os.getpid(), source_sha=inp['source_sha'], started_epoch=time.time()))
        # Each new physical scene is qualified alone before its first paired use.
        for at in range(0, len(rows), 2):
            if time.time()+600 >= inp['deadline_epoch']: raise TimeoutError('NO_TIME_FOR_NEXT_FINITE_RESPONSE_PAIR')
            group = rows[at:at+2]; scene_id = group[0]['scene_id']
            if scene_id not in proofs:
                root = runs/(inp['run_prefix']+'-q-'+scene_id.lower()); key = root.name.upper()
                qrow = dict(group[0], purpose='SCENE_QUALIFICATION', method='B0_DIRECT', nominal_action=0, desired_delta_xy_m=[0., 0.])
                def register_qual():
                    with closing(OwnerLock(campaign/'OWNER.lock')), closing(OwnerLock(base['physical_owner_lock'])):
                        s = child(root, qrow, 0); grant([s], key); return dict(spec_sha256=digest(root/'EXECUTION_SPEC.json'))
                j.once(key+'-SPEC', dict(panel_sha256=inp['panel_sha256'], row=qrow), register_qual)
                s = json.loads((root/'EXECUTION_SPEC.json').read_text()); wk = worker(s); result = observed(s, qrow)
                result['worker'] = wk; qualifications.append(result); write(stage/'QUALIFICATIONS.private.json', qualifications)
                prefix = root/'native-01/STEADY_PREFIX_RESULT.private.json'
                if (wk['exit_code'] or not result['evidence_valid'] or not prefix.exists()
                        or not json.loads(prefix.read_text()).get('qualified')):
                    raise RuntimeError('NATIVE_SCENE_QUALIFICATION_FAILURE_PRESERVED_NO_RETRY')
                proofs[scene_id] = root/'native-01/RESULT.json'
            root = runs/(inp['run_prefix']+'-pair-%02d' % (at//2)); key = root.name.upper()
            def register_pair():
                with closing(OwnerLock(campaign/'OWNER.lock')), closing(OwnerLock(base['physical_owner_lock'])):
                    new_root(root); children = [child(runs/(root.name+'-slot-'+str(i)), row, i, root) for i,row in enumerate(group)]
                    s = dict(base, schema='WM02_PILOT1_TWO_FREE_FEEDBACK_BATCH_V1' if group[0]['split']=='TRAIN' else 'WM02_PILOT1_TWO_FREE_EVALUATION_BATCH_V1',
                        profile='wm02-auto1-target-batch', registered_root=str(root), role='WM02_'+key.replace('-', '_'),
                        registry_snapshot=str(root/'REGISTRY_SNAPSHOT.private.json'), lease=str(root/'control/LEASE.json'),
                        deadline_epoch=min(time.time()+600, inp['deadline_epoch']), cpus=list(range(8)),
                        source_sha=inp['source_sha'], source_root=str(source), source_files=inp['source_files'],
                        native_runs_root=str(runs), parallel_qualification=inp['parallel_qualification'],
                        parallel_qualification_sha256=inp['parallel_qualification_sha256'],
                        slots=[dict(spec=str(Path(c['registered_root'])/'EXECUTION_SPEC.json'),
                            spec_sha256=digest(Path(c['registered_root'])/'EXECUTION_SPEC.json')) for c in children])
                    for name in ('data', 'cache', 'logs', 'tmp'): s[name] = str(root/name)
                    write(root/'EXECUTION_SPEC.json', s); grant(children+[s], key)
                    try: validate_layout(s, children)
                    except BaseException:
                        from scripts.wm02.pilot1_parallel_worker import close_proven_unstarted_children
                        close_proven_unstarted_children(root/'EXECUTION_SPEC.json', s, root)
                        v=json.loads(Path(s['lease']).read_text())
                        v.update(status='CLOSED', worker_started=False, closed_epoch=time.time(),
                            reason='PAIR_PREFLIGHT_REJECTED_BEFORE_WORKER', worker_exit_code=None)
                        write(Path(s['lease']), v)
                        raise
                    return dict(spec_sha256=digest(root/'EXECUTION_SPEC.json'))
            j.once(key+'-SPEC', dict(panel_sha256=inp['panel_sha256'], rows=group), register_pair)
            s = json.loads((root/'EXECUTION_SPEC.json').read_text()); wk = worker(s, True)
            bp = root/'batch-01/RESULT.json'; batch = json.loads(bp.read_text()) if bp.exists() else {}
            children = [observed(json.loads(Path(slot['spec']).read_text()), row) for slot,row in zip(s['slots'],group)]
            result = dict(run=root.name, worker=wk, batch=batch, children=children,
                whole_pair_eligible=not wk['exit_code'] and batch.get('status')=='TWO_ENVIRONMENT_NATIVE_PILOT_PASSED' and all(c['evidence_valid'] for c in children))
            results.append(result); write(stage/'RESULTS.private.json', results)
            if not result['whole_pair_eligible']: raise RuntimeError('NATIVE_RESPONSE_PAIR_FAILURE_PRESERVED_NO_SCORE_RETRY')
        code = 0
    except BaseException as error: failure = type(error).__name__+': '+str(error)
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        analysis_sha=None
        if evaluation and code==0:
            from go2arx5_vla.world_model.pilot2_evaluation import analyze
            observations=[dict(c['row'],attempted=c['attempted'],native_passed=c['native_passed'],evidence_valid=c['evidence_valid'],
                safety_stop=not c['native_passed'],task_success=c['score'].get('local_XY_success',False),
                XY_error_m=c['score'].get('actual_XY_error_m'),objective_m=c['score'].get('objective_m'))for b in results for c in b['children']]
            try:
                analysis=analyze(plan,observations,minimum_gain_m=inp['minimum_practical_improvement_m'],
                    baseline=plan['baseline'],seed=366502,bootstrap=10000)
                write(stage/'ANALYSIS.private.json',analysis);analysis_sha=digest(stage/'ANALYSIS.private.json')
            except BaseException as error:code=1;failure='ANALYSIS_FAILED_AFTER_NATIVE: '+type(error).__name__+': '+str(error)
        write(stage/'RESULT.private.json', dict(schema='WM02_FUSION_FOUR_ARM_DEV_COHORT_RESULT_V1'if evaluation else'WM02_FUSION_BALANCED_RESPONSE_COHORT_RESULT_V1',
            source_sha=inp['source_sha'], exit_code=code, failure=failure, qualification_results=qualifications, results=results,
            actual_native_ticks=sum(q['native_ticks'] for q in qualifications)+sum(c['native_ticks'] for b in results for c in b['children']),
            actual_response_evaluations=sum(c['attempted'] for b in results for c in b['children']),
            planned_evaluations=192 if evaluation else 48, planned_qualifications=6, analysis_sha256=analysis_sha,
            WM_updates=0, Actor_updates=0, GR00T_updates=0,
            VLA_inferences=0, TEST_read=False, exact_fork=False, wall_s=time.monotonic()-start,
            CPU_s=time.process_time()-cpu, child_CPU_s=after.ru_utime+after.ru_stime-usage.ru_utime-usage.ru_stime))
        lease.update(status='CLOSED', worker_exit_code=code, closed_epoch=time.time()); write(lease_path, lease)
    return code


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input', type=Path, required=True)
    return run(p.parse_args(argv).input)


if __name__=='__main__': raise SystemExit(main())
