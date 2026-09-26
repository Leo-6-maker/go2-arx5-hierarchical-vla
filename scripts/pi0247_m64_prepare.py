#!/usr/bin/env python3
"""Seal M64 from the unique qualified U2000 and existing 64-source table."""
import argparse
import copy
from pathlib import Path

import pi0247_lu_train as run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--u-card', type=Path, required=True)
    p.add_argument('--fit-recipe', type=Path, required=True)
    p.add_argument('--paired-verdict', type=Path, required=True)
    p.add_argument('--stage-b-card', type=Path, required=True)
    p.add_argument('--repo-code', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execution-commit', required=True)
    a = p.parse_args()
    parent = run._verify_card(a.u_card, 'U')
    fit, paired = run.read_json(a.fit_recipe), run.read_json(a.paired_verdict)
    uroot = a.u_card.parent
    manifest_path = uroot/'training_manifest.json'
    manifest = run.read_json(manifest_path)
    if (run.sha256_file(manifest_path) != fit['training_manifest_sha256']
            or manifest.get('status') != 'PASS_TRAINING_INTEGRITY'
            or manifest.get('global_step') != 2000
            or manifest.get('run_card_sha256') != run.sha256_file(a.u_card)):
        raise run.LUError('qualified U training manifest identity mismatch')
    stage_b = run.read_json(a.stage_b_card)
    if (run.sha256_file(a.stage_b_card) != 'efe9b20a21a791dd781217460db6f83283c77702f1ab41d331483b0991b4450e'
            or run.read_json(a.stage_b_card.with_name('pi0243_run_card.seal.json'))['sha256'] != run.sha256_file(a.stage_b_card)):
        raise run.LUError('original 64-source Stage-B card mismatch')
    old16 = stage_b['fixed16_source_ids']
    if set(old16) != set(paired['source_episode_ids']) or set(old16) != set(fit['noise_identity']['B0']):
        raise run.LUError('qualified old16 source support changed')
    if a.output.exists() or a.output.is_symlink():
        raise run.LUError('refusing to reseal an existing M64 root')
    prefix = 'arms/PI0247/r2-qualified-U-M64-20260926'
    hf = run._hf_preflight(run.REPO_ID, Path(parent['archive']['token_file']), [prefix], check_empty=True)
    a.output.mkdir(parents=True)
    (a.output/'logs').mkdir()
    view = a.output/'initializer_U2000'
    view.mkdir()
    receipt_path = uroot/'durable_receipts/checkpoint-2000.durable_receipt.json'
    receipt = run.read_json(receipt_path)
    binding = fit['checkpoint_binding']
    if (receipt['status'] != 'DURABLE_VERIFIED'
            or run.sha256_file(receipt_path) != binding['durable_receipt_sha256']
            or receipt['hf_manifest_commit_id'] != binding['hf_manifest_commit_id']
            or receipt['source_record']['inventory']['tree_sha256'] != binding['tree_sha256']
            or manifest['checkpoints']['2000'] != binding):
        raise run.LUError('qualified U2000 full checkpoint receipt mismatch')
    expected = {e['path']:e for e in receipt['source_record']['inventory']['entries']}
    for name in ('config.json', 'model.safetensors.index.json', 'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'):
        source = uroot/'training'/name
        if source.stat().st_size != expected[name]['bytes'] or run.sha256_file(source) != expected[name]['sha256']:
            raise run.LUError(f'U2000 final weights differ from pinned full checkpoint: {name}')
        (view/name).symlink_to(source)
    ct = fit['processor_identity']['frozen_C_T']
    for name in ('processor_config.json', 'statistics.json', 'embodiment_id.json'):
        source = uroot/'training/processor'/name
        expected_sha = (ct['processor_config_sha256'] if name == 'processor_config.json'
                        else ct['T_card_base_statistics_sha256'] if name == 'statistics.json'
                        else next(x['sha256'] for x in parent['initializer']['model_files'] if x['path'] == name))
        if run.sha256_file(source) != expected_sha:
            raise run.LUError(f'frozen C_T processor sidecar mismatch: {name}')
        (view/name).symlink_to(source)
    for name, expected_sha in (('stats.json', ct['NEW_EMBODIMENT_stats_sha256']), ('relative_stats.json', ct['relative_stats_sha256'])):
        if run.sha256_file(Path(parent['paths']['train_dataset'])/'meta'/name) != expected_sha:
            raise run.LUError(f'frozen C_T TRAIN statistics mismatch: {name}')
    view_receipt_path = a.output/'initializer_view_receipt.json'
    run.write_json(view_receipt_path, {
        'status':'PASS_U2000_PINNED_MODEL_VIEW', 'revision':binding['hf_manifest_commit_id'],
        'checkpoint_tree_sha256':binding['tree_sha256'], 'model_files':run._model_files(view),
        'processor_identity':fit['processor_identity'], 'source_card_sha256':run.sha256_file(a.u_card),
    })
    card = copy.deepcopy(parent)
    card.update(schema='PI0247_M64_RUN_CARD_V1', arm='U', family=f'PI0247_M64_{run.M64_RECIPE}_V1',
                execution_authorization=run.R2_EXECUTION_AUTHORIZATION,
                canonical_issue_comment=run.R2_CANONICAL_COMMENT, pointer_pr_comment=run.R2_POINTER_COMMENT,
                execution_update_commit=run.R2_AUTHORITY_COMMIT, selected_checkpoint='NONE',
                qualified_recipe=run.M64_RECIPE, route='AUTO_M64',
                stage={'name':'M64','steps':5000,'checkpoint_steps':[1000,2500,5000],
                       'primary_screen_steps':[0,1000,2500,5000],'primary_noise':'B0','final_noise':'B1'},
                support={'count':64,'old16_source_ids':old16}, cohort=stage_b['cohort'], schedule=stage_b['schedule'])
    card['loss_contract'] = fit['loss_contract']
    assert manifest['rollout_audit']['parity_nonselection_rollouts'] == 2
    card['nonselection_parity_budget'] = {'campaign_limit': 8, 'prior_U_rollouts': 2,
                                        'M64_planned_rollouts': 2, 'total_after_M64': 4}
    card['paths'].update(base_model_path=str(view), output_root=str(a.output), output_base=str(a.output),
                         repo_code_root=str(a.repo_code),
                         p_run_card_path=str(Path(parent['paths']['base_model_path']).parents[1]/'run_card.json'))
    card['initializer'] = {
        'path':str(view), 'family':parent['family'], 'durable_receipt_path':str(receipt_path),
        'durable_receipt_sha256':binding['durable_receipt_sha256'],
        'durable_revision':binding['hf_manifest_commit_id'], 'screen_revision':binding['hf_manifest_commit_id'],
        'screen_manifest_sha256':fit['final_screen_report_sha256'],
        'full_checkpoint_tree_sha256':binding['tree_sha256'], 'full_checkpoint_bytes':binding['total_bytes'],
        'model_files':run._model_files(view),
    }
    card['optimizer'].update(steps=5000, checkpoint_steps=[1000,2500,5000])
    card['archive'].update(path_prefix=prefix, checkpoint_steps=[1000,2500,5000])
    card['hf_preflight'] = hf
    source = card['source']
    source.pop('child_code_patch', None)
    source.update(execution_code_commit=a.execution_commit,
                  parent_resolved_config_path=str(uroot/'training/resolved_config.json'),
                  parent_resolved_config_sha256=run.sha256_file(uroot/'training/resolved_config.json'),
                  parent_run_card_path=str(a.u_card), parent_run_card_sha256=run.sha256_file(a.u_card),
                  fit_recipe_manifest_path=str(a.fit_recipe), fit_recipe_manifest_sha256=run.sha256_file(a.fit_recipe),
                  paired_verdict_path=str(a.paired_verdict), paired_verdict_sha256=run.sha256_file(a.paired_verdict),
                  stage_b_card_path=str(a.stage_b_card), stage_b_card_sha256=run.sha256_file(a.stage_b_card),
                  training_manifest_path=str(manifest_path), training_manifest_sha256=run.sha256_file(manifest_path),
                  initializer_view_receipt_path=str(view_receipt_path), initializer_view_receipt_sha256=run.sha256_file(view_receipt_path))
    code = source['execution_code_sha256']
    code.pop('pi0247_lu_train_h200.pbs')
    for name in ('pi0247_lu_train.py','pi0247_u_rollout_overlay.py','pi0228_train_calib_checkpoint_selection.py'):
        code[name] = run.sha256_file(a.repo_code/'scripts'/name)
    code['pi0247_m64_train_h200.pbs'] = run.sha256_file(a.repo_code/'hpc/pbs/pi0247_m64_train_h200.pbs')
    source['preparation_script_sha256'] = run.sha256_file(Path(__file__))
    run.STEPS, run.CHECKPOINTS = run.M64_STEPS, run.M64_CHECKPOINTS
    shared, runtime = Path(card['paths']['code_root']), Path(card['paths']['training_runtime_root'])
    run._verify_runtime_bindings(card['paths'], source['runtime_bindings'])
    fixed = run.load_module(shared/'scripts/pi0241_fixed16_learning.py', 'pi0247_m64_fixed')
    core = run.load_module(runtime/'pi0246_gate_pass_train.py', 'pi0247_m64_core')
    fixed.MAX_STEPS, fixed.CHECKPOINT_STEPS = run.STEPS, run.CHECKPOINTS
    core.STEPS, core.CHECKPOINTS = run.STEPS, run.CHECKPOINTS
    _, _, summary = core.build_runtime_config(card['paths'], a.output/'training', fixed)
    run._check_summary(summary, parent['resolved_config'], str(view))
    card['resolved_config'] = summary
    run._verify_m64_sources(card)
    run._validate_initializer_runtime(card)
    run._verify_code(card, a.repo_code, shared, runtime)
    capacity = run._validate_storage(card)
    path = a.output/'run_card.json'
    run.write_json(path, card)
    run.write_json(a.output/'run_card.seal.json', {'sha256':run.sha256_file(path)})
    run.write_json(a.output/'preparation.json', {'status':'READY_NOT_STARTED','selected_checkpoint':'NONE',
                   'run_card_sha256':run.sha256_file(path),'capacity':capacity,'resolved_config':summary})
    print({'status':'READY_NOT_STARTED','card':str(path),'sha256':run.sha256_file(path)})


if __name__ == '__main__':
    main()
