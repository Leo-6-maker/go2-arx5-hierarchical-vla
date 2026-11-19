"""Regression checks for exact matched exposure and long-run optimizer state."""
import numpy as np
import pytest

from go2arx5_vla.sla_adapter import scale20k_checkpoint as checkpoint
from go2arx5_vla.sla_adapter.scale20k_contract import (
    DOMAINS, PLAN, PRIVATE_ROOTS, RECIPE, bank_sha256, cursors, learning_rates, make_adamw, parameter_partition, visits,
)


def test_shared_learning_rate_table_and_only_admitted_training_across_cursors():
    rates = learning_rates()
    assert len(rates) == 20000 and rates[499] == 2e-5 and rates[-1] == 2e-6
    assert np.all(np.diff(rates[:500]) > 0) and np.all(np.diff(rates[499:]) < 0)
    assert cursors('SCALE-MIX20K',5000) == dict(target=160000,robodojo=240000,libero=240000)
    assert cursors('SCALE-TGT5K',5000) == dict(target=160000,robodojo=0,libero=0)
    rows = [dict(episode='a',task='short',split='TRAIN',admitted=True,window_count=1),
            dict(episode='b',task='long',split='TRAIN',admitted=True,window_count=30)]
    bank = visits(rows,'target')
    assert len(bank) == 640000 and (bank==0).sum() == 320000 and bank.max() == 30
    np.testing.assert_array_equal(bank,visits(rows,'target'))
    with pytest.raises(ValueError,match='UNIQUE_ADMITTED_TRAIN'):
        visits([rows[0],dict(rows[1],split='SOURCE_DEV')],'target')


def test_private_moments_and_decay_and_next_update_survive_checkpoint(tmp_path):
    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')
    pytest.importorskip('optax')
    init,apply = make_adamw(beta1=.9,beta2=.95,eps=1e-8,weight_decay=.01,clip_norm=1.)
    params = {name:(jnp.asarray([.4,1.2],dtype=jnp.float32),) for name in ('shared',*DOMAINS)}
    optimizer = init(params)
    rng = jax.random.PRNGKey(4)
    gradient = {name:(jnp.asarray([.5,1.1],dtype=jnp.float32),) for name in params}
    target,states,_ = apply(params,optimizer,{key:gradient[key] for key in ('shared','target')},2e-5)
    for source in DOMAINS[1:]:
        assert target[source] is params[source] and states[source] is optimizer[source]
    active,state,_ = apply(params,optimizer,gradient,2e-5)
    arm,digest = 'SCALE-MIX20K','4'*64
    binding = dict(recipe=RECIPE,plan_sha=PLAN,arm=arm,learning_rates_sha256=bank_sha256(learning_rates()),
        parent_manifest_sha256=digest,partition_sha256=digest,initial_state_sha256=digest,optimizer_sha256=digest,
        data_manifest_sha256=dict.fromkeys(DOMAINS,digest),visit_bank_sha256=dict.fromkeys(DOMAINS,digest))
    pin = checkpoint.save(tmp_path/'ready',arm=arm,step=1,parameters=active,optimizer=state,rng=rng,binding=binding)
    restored,step,offsets = checkpoint.restore(pin,arm=arm,parameters=params,optimizer=optimizer,rng=rng,binding=binding)
    assert step == 1 and offsets == dict(target=32,robodojo=48,libero=48)
    np.testing.assert_array_equal(restored['rng'],rng)
    left = apply(active,state,gradient,2e-6)[:2]
    right = apply(restored['parameters'],restored['optimizer'],gradient,2e-6)[:2]
    for a,b in zip(jax.tree.leaves(left),jax.tree.leaves(right),strict=True):
        np.testing.assert_array_equal(a,b)
    from go2arx5_vla.sla_adapter.runtime_v1 import read_pin

    inference_args = dict(arm=arm, parameters=params, abstract_optimizer=jax.eval_shape(init, params),
                          rng=rng, binding=binding)
    weights = checkpoint.restore_inference(pin, **inference_args)
    for expected, observed in zip(jax.tree.leaves(active), jax.tree.leaves(weights), strict=True):
        np.testing.assert_array_equal(expected, observed)
    manifest = read_pin(dict(path=str(tmp_path/'ready/COMPLETE.json'), sha256=pin['sha256']))
    optimizer_row = next(record for record in manifest['arrays'] if record['key'].startswith("['optimizer']"))
    member = tmp_path/'ready'/optimizer_row['file']
    member.write_bytes(member.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='ARRAY_IDENTITY_MISMATCH'):
        checkpoint.restore(pin,arm=arm,parameters=params,optimizer=optimizer,rng=rng,binding=binding)
    with pytest.raises(ValueError,match='ARRAY_IDENTITY_MISMATCH'):
        checkpoint.restore_inference(pin, **inference_args)


def test_upstream_mlp_lora_members_are_active_alongside_attention_lora():
    llm = ('PaliGemma','llm','layers')
    mlp = [(*llm,'mlp',name) for name in ('gating_einsum_lora_a','gating_einsum_lora_b','linear_lora_a','linear_lora_b')]
    attention = (*llm,'attn','q_einsum','lora_b')
    full_expert = (*llm,'mlp_1','linear')
    frozen = [(*llm,'mlp','linear'),('PaliGemma','img','kernel')]
    private = [(root,'kernel') for root in PRIVATE_ROOTS]
    groups = parameter_partition([*frozen,*mlp,attention,full_expert,*private])
    assert set(groups['vlm_lora']) == {*mlp,attention}
    assert groups['action_expert'] == (full_expert,)
    assert set(groups['frozen']) == set(frozen)


def test_core_successor_binds_only_a_closed_hash_verified_training_segment(tmp_path, monkeypatch):
    from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
    from scripts.pi0292.durable import write
    from scripts.sla import scale20k_core_prefix as core

    arm = 'SCALE-MIX20K'
    previous = tmp_path/'old'
    checkpoint_root = previous/'train/step00200'
    checkpoint_root.mkdir(parents=True)
    manifest = checkpoint_root/'COMPLETE.json'
    write(manifest, dict(step=200))
    result_file = previous/'train/RESULT.private.json'
    write(result_file,dict(scope=RECIPE,source_sha='f'*40,arm=arm,mode='train',status='FIXED_SEGMENT_SAVED',
        exit_code=0,global_step=200,global_cursors=cursors(arm,200),new_native_requests=0,new_native_ticks=0,
        checkpoint=dict(path=str(checkpoint_root),sha256=sha(manifest))))
    prior = dict(path=str(result_file),sha256=sha(result_file))
    (previous/'control').mkdir()
    lease = previous/'control/LEASE.json'
    write(lease,dict(status='CLOSED',unit='wm02-old.service'))
    source = previous/'source'
    (source/'scripts/sla').mkdir(parents=True)
    (source/'scripts/sla/train_scale20k.py').touch()
    write(previous/'RESULT.private.json',dict(stages=dict(train=prior)))
    config = dict(arm=arm,source_sha='f'*40,previous_train_result=prior)
    monkeypatch.setattr(core.subprocess,'check_output',lambda *args,**kwargs:'0\n')
    assert core.closed_predecessor(config,tmp_path/'new') == (prior,source,200)
    monkeypatch.setattr(core.subprocess,'check_output',lambda *args,**kwargs:'975\n')
    with pytest.raises(PermissionError,match='STILL_OWNED'):
        core.closed_predecessor(config,tmp_path/'new')


def test_core_offline_does_not_consume_formal_budget_on_the_technical_prefix(tmp_path, monkeypatch):
    from scripts.pi0292.durable import write
    from scripts.sla import scale20k_core_offline as offline

    result_path = tmp_path/'RESULT.private.json'
    training_path = tmp_path/'SPEC.private.json'
    write(result_path, dict(checkpoint={'path': 'endpoint', 'sha256': '4'*64}))
    write(training_path, dict(planned_step=500))
    prior = offline.pin(result_path)
    config = dict(arm='SCALE-MIX20K', step=500, training_source_sha='a'*40,
                  training_spec=offline.pin(training_path))
    monkeypatch.setattr(offline, 'closed_predecessor', lambda config, root: (prior, None, 200))
    with pytest.raises(ValueError, match='SEALED_FORMAL_TRAINING_STEP'):
        offline.offline_endpoint(config, tmp_path/'evaluation')
    monkeypatch.setattr(offline, 'closed_predecessor', lambda config, root: (prior, None, 500))
    assert offline.offline_endpoint(config, tmp_path/'evaluation')[0]['sha256']=='4'*64
    write(training_path, dict(planned_step=200))
    with pytest.raises(ValueError, match='ORIGINAL_TERMINAL_TRAINING_SPEC_REQUIRED'):
        offline.offline_endpoint(config, tmp_path/'evaluation')


def test_scale_portable_transport_keeps_original_training_card_and_checks_all_arrays(tmp_path, monkeypatch):
    import io

    from scripts.pi0292.durable import write
    from scripts.sla import scale20k_portable_endpoint as transport

    source = tmp_path/'vanda'
    stage = source/'train'
    state = stage/'step00500'
    state.mkdir(parents=True)
    arm, step, ident = transport.ARM, 500, 'a'*64
    job, launcher, train_source = '123.site', 'b'*40, 'c'*40
    binding = dict(recipe=RECIPE,plan_sha=PLAN,arm=arm,learning_rates_sha256=bank_sha256(learning_rates()),
        parent_manifest_sha256=ident,partition_sha256=ident,initial_state_sha256=ident,optimizer_sha256=ident,
        data_manifest_sha256=dict.fromkeys(DOMAINS,ident),visit_bank_sha256=dict.fromkeys(DOMAINS,ident))
    rows = []
    for i in range(140):
        name = f'{i:06d}.npy'
        file = state/name
        np.save(file,np.array([i],np.int32),allow_pickle=False)
        rows.append(dict(file=name,sha256=transport.pin(file)['sha256']))
    complete = state/'COMPLETE.json'
    write(complete,dict(schema=checkpoint.SCHEMA,binding=binding,step=step,cursors=cursors(arm,step),
                        training_full_generations=step*4,physics=False,arrays=rows))
    endpoint = dict(path=str(state),sha256=transport.pin(complete)['sha256'])
    write(stage/'MODEL_AT_START.private.json',dict(binding=binding))
    write(stage/'SPEC.private.json',dict(arm=arm,source_sha=train_source,plan_sha=PLAN,recipe=RECIPE,
        global_stop_step=step,physics=False,hardware_allowed=False,dataset='/original/vanda/dataset'))
    terminal = stage/'RESULT.private.json'
    write(terminal,dict(arm=arm,scope=RECIPE,mode='train',source_sha=train_source,exit_code=0,
        status='FIXED_SEGMENT_SAVED',global_step=step,global_cursors=cursors(arm,step),
        new_native_requests=0,new_native_ticks=0,checkpoint=endpoint))
    write(source/'RESULT.private.json',dict(source_sha=launcher,training_source_sha=train_source,
        status='FIXED_SEGMENT_SAVED',PBS_JOBID=job,stages=dict(train=transport.pin(terminal))))
    write(source/'EXECUTION.private.json',dict(arm=arm,plan_sha=PLAN,source_sha=launcher,job_id=job,root=str(source)))
    monkeypatch.setattr(transport,'scheduler_terminal',lambda cfg:dict(id=job,state='F',exit_code=0))
    stream = io.BytesIO()
    transport.export_bundle(source,stream)
    destination = tmp_path/'core'
    custody = transport.receive_bundle(destination,io.BytesIO(stream.getvalue()))
    assert custody['checkpoint']['sha256'] == endpoint['sha256']
    assert (destination/'train/SPEC.private.json').read_bytes() == (stage/'SPEC.private.json').read_bytes()
    assert transport.local_custody(destination,custody['transfer']) == custody
    (state/'000001.npy').write_bytes(b'damaged')
    rejected = io.BytesIO()
    with pytest.raises(ValueError,match='TRANSFER_INPUT_HASH_MISMATCH'):
        transport.export_bundle(source,rejected)
    assert rejected.getvalue() == b''


def test_scale_relocated_training_requires_clock_and_all_dataset_pins():
    from copy import deepcopy

    from scripts.sla.evaluate_scale20k import relocated_training

    def pin(digest):
        return dict(path='original',sha256=digest*64)

    domains = {domain:dict(manifest=pin('a')) for domain in DOMAINS}
    domains['target'].update(admission=pin('b'),time_audit=pin('c'))
    train = dict(data=domains,parent_manifest=pin('d'),tokenizer=pin('e'),parent_params='origin',
                 dataset='origin',optimizer=dict(beta1=.9))
    site = deepcopy(train)
    site['dataset'] = 'core'
    site['optimizer']['beta1'] = .5
    assert relocated_training(train,site)['optimizer'] == dict(beta1=.9)
    assert relocated_training(train,site)['dataset'] == 'core'
    site['data']['target']['time_audit']['sha256'] = '0'*64
    with pytest.raises(ValueError,match='RELOCATED_ASSET_HASH'):
        relocated_training(train,site)


def test_vanda_continuation_requires_the_original_complete_pbs_prefix(tmp_path, monkeypatch):
    import json

    from scripts.pi0292.durable import write
    from scripts.sla import scale20k_vanda_prefix as vanda

    old = tmp_path/'prefix'
    stage = old/'train'
    state = stage/'step00200'
    state.mkdir(parents=True)
    worker = tmp_path/'old-source'
    files = {}
    for relative in ('scripts/sla/train_scale20k.py','src/go2arx5_vla/sla_adapter/scale20k_training.py'):
        f = worker/relative
        f.parent.mkdir(parents=True)
        f.write_text(relative)
        files[relative] = vanda.sha(f)
    source = 'a'*40
    (worker/'SOURCE_SHA.private').write_text(source+'\n')
    files_path = tmp_path/'SOURCE_FILES.json'
    write(files_path, files)
    job_id = '123.server'
    mode = 'R'

    def qstat(*_args, **_kwargs):
        return json.dumps(dict(Jobs={job_id:dict(job_state=mode,Exit_status=0,Job_Owner='worker@site',
                                                 project='project',Job_Name='slatgt20k')}))

    monkeypatch.setattr(vanda.subprocess, 'check_output', qstat)
    model_path = stage/'MODEL_AT_START.private.json'
    write(model_path,dict(binding={'fixed':True}))
    monkeypatch.setattr(vanda, 'read_complete', lambda *args: (state,dict(step=200,arrays=[None]*140)))
    spec = stage/'SPEC.private.json'
    write(spec,dict(source_sha=source,arm=vanda.ARM,global_stop_step=200,microbatch=dict.fromkeys(DOMAINS,1)))
    worker_result = stage/'RESULT.private.json'
    expected = cursors(vanda.ARM,200)
    write(worker_result,dict(scope=RECIPE,mode='train',arm=vanda.ARM,source_sha=source,exit_code=0,
        status='FIXED_SEGMENT_SAVED',global_step=200,global_cursors=expected,new_updates=200,
        new_visits=expected,new_model_generations=800,new_native_requests=0,new_native_ticks=0,
        checkpoint=dict(path=str(state),sha256='4'*64)))
    parent = old/'RESULT.private.json'
    result_pin = vanda.pin(worker_result)
    write(parent,dict(PBS_JOBID=job_id,source_sha=source,stages=dict(train=result_pin)))
    receipt = dict(train_result=result_pin,campaign_result=vanda.pin(parent),training_spec=vanda.pin(spec),
        model_at_start=vanda.pin(model_path),worker_root=str(worker),worker_files=vanda.pin(files_path),job_id=job_id)
    config = dict(user='worker',project='project',job_id='124.server',continuation=receipt)
    ledger = stage/'UPDATES.private.jsonl'
    ledger.write_text(''.join(json.dumps(dict(step=k,cursors=cursors(vanda.ARM,k),
        domains=dict(target=dict(training_full_generations=4))))+'\n' for k in range(1,201)))
    new = tmp_path/'next'
    with pytest.raises(PermissionError,match='MUST_HAVE_FINISHED'):
        vanda.completed_prefix(config,new)
    mode = 'F'
    assert vanda.completed_prefix(config,new)[0] == result_pin
    f.write_text('changed training worker')
    with pytest.raises(ValueError,match='ORIGINAL_TRAINER_BYTES_CHANGED'):
        vanda.completed_prefix(config,new)


def test_core_successor_imports_verified_original_worker_and_uses_original_cache(tmp_path):
    import os
    from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
    from scripts.pi0292.durable import write
    from scripts.sla.scale20k_core_prefix import training_launch_context

    previous = tmp_path/'old'
    worker, launcher, cache = previous/'source', tmp_path/'new/source', previous/'cache'
    stage = previous/'train'
    for folder in (worker, launcher, cache, stage):
        folder.mkdir(parents=True, exist_ok=True)
    spec = stage/'SPEC.private.json'
    cfg = dict(root=str(launcher.parent), source_sha='0'*40, GPU_uuid='same-GPU')
    write(spec, dict(cfg, compilation_cache_dir=str(cache)))
    write(previous/'PROGRESS.private.json', {'spec_sha256':sha(spec)})
    files = ('scripts/sla/train_scale20k.py','src/go2arx5_vla/sla_adapter/scale20k_training.py')
    for name in files:
        member = worker/name
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_text(name)
    cfg['training_source_files'] = {name:sha(worker/name) for name in files}
    prior = {'path':str(stage/'RESULT.private.json')}
    extra = tmp_path/'unmodified_site_dependencies'
    env = dict(PYTHONPATH=os.pathsep.join(map(str,(launcher, launcher/'src',extra))), CUDA_VISIBLE_DEVICES='same-GPU')
    child, selected_cache = training_launch_context(cfg, launcher, worker, prior, env)
    assert child['PYTHONPATH'].split(os.pathsep) == list(map(str,(worker,worker/'src',extra)))
    assert child['CUDA_VISIBLE_DEVICES'] == env['CUDA_VISIBLE_DEVICES'] and selected_cache == cache
    (worker/files[0]).write_text('changed')
    with pytest.raises(ValueError,match='ORIGINAL_WORKER_SOURCE_CHANGED'):
        training_launch_context(cfg, launcher, worker, prior, env)


def test_scale_native_worker_binds_checkpoint_stage_and_never_takes_source_weights(tmp_path, monkeypatch):
    from go2arx5_vla.sla_adapter import scale20k_package as package

    digest = 'd'*64
    arm, step = 'SCALE-MIX20K', 5000
    keys = ("['parameters']['shared']['expert']", "['parameters']['target']['io']",
            "['parameters']['robodojo']['io']", "['optimizer']['shared']")
    rows = [dict(key=key, file=f'{i:06d}.npy', dtype='float32', sha256=digest) for i, key in enumerate(keys)]
    target = package.checkpoint_parameters(dict(arrays=rows))
    target = [dict(row, file='parameters/'+row['file']) for row in target]
    assert [row['key'] for row in target] == list(keys[:2])
    meta = dict.fromkeys(('checkpoint_sha256','offline_evaluation_sha256','offline_restore_sha256','statistics_sha256',
        'tokenizer_sha256','frozen_fingerprint','partition_sha256','initial_state_sha256','panel_sha256',
        'target_data_manifest_sha256','target_admission_sha256'), digest)
    manifest = dict(meta, schema=package.SCHEMA, plan_sha=PLAN, recipe=RECIPE, arm=arm, step=step,
        cursors=cursors(arm, step), codec=package.CODEC, arrays=target, parent_manifest=dict(sha256=digest, path='parent'),
        inference=dict(sampler='ten_cached_flow_steps', matmul='highest', cameras='body_and_left_wrist',
                       horizon=40, interval_s=.05, state_token_dropout=False), hardware_allowed=False,
        native_admission=False, optimizer_required=False, source_data_required=False, teacher_required=False)
    package.contract(manifest)
    with pytest.raises(ValueError, match='GO2_ONLY'):
        package.contract(dict(manifest, step=200, cursors=cursors(arm, 200)))
    with pytest.raises(ValueError, match='GO2_ONLY'):
        package.contract(dict(manifest, arrays=[*target, dict(rows[2], file='parameters/'+rows[2]['file'])]))

    monkeypatch.setattr(package, 'read_package', lambda *a: manifest)
    name = 'SCALE-MIX5K'
    kwargs = dict(scope=package.SCOPE, plan_commit=PLAN, physics=False, new_physical_requests_budget=0,
        packages={name: dict(path=str(tmp_path), sha256=digest)}, parent_manifest=manifest['parent_manifest'],
        parent_params='parent/model/params', request_bindings_by_arm={name: dict(arm=name, policy_package_sha256=digest,
                                                                                 statistics_sha256=digest)})
    wrong = dict(kwargs, packages={arm: kwargs['packages'][name]},
        request_bindings_by_arm={arm: dict(kwargs['request_bindings_by_arm'][name], arm=arm)})
    with pytest.raises(ValueError, match='ENDPOINT_AND_REQUEST_BINDING'):
        package.Pool(wrong)
    with pytest.raises(PermissionError, match='NONINTEGRATING'):
        package.Pool(dict(kwargs, physics=True))


def test_scale_native_keeps_dev_and_final_confirmation_conditions_distinct():
    from scripts.wm02.multitask_panel_suffix import budgets
    from go2arx5_vla.world_model import scale20k_native as scale
    from go2arx5_vla.world_model.multitask_score_v6 import INSTRUCTIONS

    clusters, rows = [], {}
    for stage in scale.CAPS:
        rows[stage] = []
        for index,family in enumerate(('F1','F1','F2','F2')):
            name, reset = f'{stage}-{index}', len(clusters)
            clusters.append(dict(stage=stage,id=name,family=family,reset=reset,scene={'sha256':name}))
            rows[stage].extend(dict(id=f'{name}-{instruction}', cluster=name, family=family,
                                    reset=reset, instruction=instruction) for instruction in INSTRUCTIONS)
    population = dict(schema=scale.POPULATION_SCHEMA,plan=scale.PLAN,clusters=clusters,conditions=rows,
                      scene_instruction_independent=True,old_TEST_reused=False)
    controls = [scale.conditions(population,scale.DEV,{arm}) for arm in scale.ENDPOINTS]
    assert all(budgets(cases)==scale.CAPS[scale.DEV] for cases in controls)
    assert all([r for r,_ in cases]==rows[scale.DEV] for cases in controls)
    confirmation = scale.conditions(population,scale.CONFIRM,{'SCALE-MIX20K'})
    assert {r['cluster'] for r,_ in confirmation}.isdisjoint(r['cluster'] for r in rows[scale.DEV])
    with pytest.raises(ValueError,match='ONE_FROZEN_ENDPOINT'):
        scale.conditions(population,scale.CONFIRM,{'SCALE-MIX5K'})
    with pytest.raises(ValueError,match='ONE_FROZEN_ENDPOINT'):
        scale.conditions(population,scale.DEV,set(scale.ENDPOINTS))


def test_scale_native_does_not_admit_core_replay_with_altered_physical_output(monkeypatch):
    from go2arx5_vla.world_model import scale20k_native as scale
    from go2arx5_vla.world_model import multitask_scope_v6 as scope

    recorded = [dict(ordinal=i,index=i,noise_id=i+10,normalized_sha256='n',decoded_sha256='p')
                for i in range(96)]
    evaluation = dict(exit_code=0,status='ENDPOINT_PREDICTIONS_SEALED',arm='SCALE-MIX20K',step=5000,optimization_updates=0,
        checkpoint={'sha256':'ck'},panel={'sha256':'panel'},
        new_model_generations=288,rows=dict(target=recorded),old_TEST_access=False,new_native_requests=0)
    core = dict(schema=scale.RGB_SCHEMA,arm='SCALE-MIX5K',exit_code=0,passed=True,package_sha256='pkg',
        evaluation_sha256='eval',site=dict(host='corelab-4080s'),new_model_generations=16,new_native_requests=0,
        new_native_ticks=0,optimizer_updates=0,rows=[dict(recorded[i],same_original_H40_and_seam_decision=True)
                                                     for i in scale.RESTORE_ORDINALS])
    manifest = dict(arm='SCALE-MIX20K',step=5000,offline_evaluation_sha256='eval',checkpoint_sha256='ck',panel_sha256='panel')
    monkeypatch.setattr(scope,'read_pin',lambda pin: {'eval':evaluation,'core':core}[pin['sha256']])
    monkeypatch.setattr(scale,'read_package',lambda _path,_sha:manifest)
    identity = ('SCALE-MIX5K',dict(path='package',sha256='pkg'),dict(sha256='eval'),dict(sha256='core'))
    assert scale.endpoint(*identity)==(manifest,evaluation)
    core['rows'][0]['decoded_sha256'] = 'changed'
    with pytest.raises(ValueError,match='OUTPUT_OR_ORIGINAL_GUARD_CHANGED'):
        scale.endpoint(*identity)
