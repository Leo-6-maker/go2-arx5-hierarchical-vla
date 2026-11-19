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
