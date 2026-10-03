"""Freeze the full new quality path only after actual data/CPU/final-model admission."""
from pathlib import Path
from admission import BASE, ROOT, OLD, read, sha, write


def main():
    audit = read(ROOT/'evidence/MAIN_TRAINING_AUDIT.json')
    assert audit['status'] == 'PASS_AGG4096_REUSED_CONTROL_TRAINING_INTEGRITY' and audit['optimizer_updates'] == 4096
    assert {(c['arm'], c['step']) for c in audit['complete_checkpoints']} == {('AGG',2048),('AGG',4096)}
    assert read(ROOT/'evidence/AGG4096/FROZEN_SCOPE.json')['status'] == 'PASS_FIXED_FINAL_MODEL_VIEW_PRECISION_FROZEN_SCOPE'
    cpu = read(ROOT/'evidence/EVALUATION_CPU_CHECK.json')
    assert cpu['status'] == 'PASS_PI0270_ACTUAL_NEWFIT192_AND_ORIGINAL_EXECUTOR_GATE' and len(cpu['checks']) == 192
    assert cpu['source_virtual_sha256'] == sha(ROOT/'code/quality_virtual.py')
    actual = {r['input_id']: r for r in audit['NEW_FIT192_actual_consumption']}
    fit = read(ROOT/'design/NEW_FIT192.json'); assert len(fit['rows']) == 192
    for row in fit['rows']:
        assert actual[row['input_id']]['actual_seen']
        assert all(actual[row['input_id']][k] == row[k] for k in ('condition_sha256', 'labels_masks_sha256'))
    from quality_source_check import check
    source_check = check(ROOT/'code', OLD/'code')
    assert source_check['status'] == 'PASS_PREPARED_QUALITY_SOURCE_NO_QUERY'
    paths = [ROOT/'code'/name for name in ('quality_gate.py','quality_worker.py','quality_virtual.py','quality_components.py',
             'quality_offline.py','newfit.py','quality_evaluate.py','quality_cpu_check.py','quality_source_check.py','seal_quality.py',
             'quality_report.py','trajectory_replay.py','video.py')]
    paths.extend(ROOT/'design'/name for name in ('EVALUATION_CARD.json','FIXED_OFFLINE_CARD.json','COMPONENT_CARD.json',
                 'PAIRING_PLAN.json','RECOVERY_MAPPING.json','NEW_FIT192.json','NEWFIT_QUERY_CARD.json'))
    paths.extend(ROOT/'evidence'/name for name in ('EVALUATION_EXACT_REUSE.json','EVALUATION_CPU_CHECK.json','MAIN_TRAINING_AUDIT.json',
                 'AGG/checkpoint-4096.full_readback.json','AGG4096/FROZEN_SCOPE.json'))
    paths.extend(ROOT/'model_bindings'/(point+'.json') for point in ('AGG4096','PREFIX4096'))
    paths.extend([ROOT/'contracts/ROT/source/dataset/statistics.json', BASE/'PI0267/design/SCENE_CARD.json'])
    inherited = OLD/'design/QUALITY_EXECUTION_FILESET.json'; paths.append(inherited)
    for row in read(inherited)['files']:
        p = Path(row['path']); assert sha(p) == row['sha256'], p
        if not p.is_relative_to(OLD): paths.append(p)
    for row in fit['rows']:
        for k in ('input','noise_file'): paths.append(Path(row[k]))
    write(ROOT/'design/QUALITY_EXECUTION_FILESET.json', dict(status='FROZEN_PI0270_BEFORE_NEW_QUALITY',
          files=[dict(path=str(p),sha256=sha(p)) for p in sorted(set(paths))],
          source_check=source_check, reused_qualified_fileset_sha256=sha(inherited),
          standard_new_rows=968, NEWFIT_new_rows=384, new_main=64, new_component_proposals=32,
          model_queries_by_seal=0, selected_checkpoint='NONE', deployment_selected_checkpoint='NONE'))
    print('FROZEN_PI0270_ACTUAL_QUALITY_FILESET', len(set(paths)), flush=True)


if __name__ == '__main__': main()
