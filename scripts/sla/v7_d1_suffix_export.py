"""Cold-export the admitted D1 suffix and read every window with the R3 reader."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v65_recovery_data import dataset, source_episode

CASE = 'V7_D1_EARLY_TRAIN_15'
SEAL = '12819abb1a0285b05c5255668ba9b2fa0c723785c54977fe9d23a98efa981b53'
RECONCILIATION = 'd165dc95a7d134feac6767533077b0ac5961b002cefcca7d5ceb2e0c2d0b546c'
PROTECTED = 'a1dfa32f329cb91d474e20488cbeeafa3dc41858f32dacd6cd593dbed61bbb26'


def export(site, output):
    from openpi_client.image_tools import resize_with_pad
    from scripts.sla.core_hopper_native_scope import accepted
    from scripts.sla.hopper_site import pinned

    require(not output.exists(), 'Preserve the previous cold data version')
    sealed = site / 'V7-D1-EARLY-SEALED-02'
    close = read(pin(sealed / 'RESULT.json', RECONCILIATION))
    require(close['complete_reconciliation'] and close['terminal_cases'] == close['train_recovery_count'] == 1
            and close['actual_forwards'] == 43 and close['native_ticks'] == 41453,
            'Require the single closed, consumption-proven native qualification')
    output.mkdir(mode=0o700)
    entry = source_episode(site, output, 0, '15', 100, SEAL, name=CASE,
                           seal_path=sealed / CASE / 'ACTUAL.json')
    protected = read(pin(site / 'v63-place-core-code-01/PROTECTED_POPULATION.json', PROTECTED))
    excluded = {row['scene']['sha256'] for row in protected['clusters']
                if row.get('split') in ('QUAL', 'DEV', 'VAL', 'CONF', 'TEST')}
    require(entry['scene_sha256'] not in excluded, 'A protected scene cannot supply training labels')
    worker = read(pin(site / 'ADAPT0250-NATIVE-PLACE-01/worker/EXECUTION_SPEC.json',
                      '361709e35c2bf427d5644e7af37a925014ab52861c075d4782749afd371d6bfc'))
    _, _, receiver = accepted(worker)
    target = pinned(receiver['input'])['target_D1']
    statistics = dict(state=pinned(target['statistics'])['state'], actions=pinned(receiver['action'])['actions'])
    manifest = dict(schema='V7_D1_EARLY_NATIVE_RECOVERY_COLD_V1', episodes=[entry], windows=100,
                    state_statistics_sha256=target['statistics']['sha256'],
                    action_statistics_sha256=receiver['action']['sha256'],
                    protected_population_sha256=PROTECTED, native_reconciliation_sha256=RECONCILIATION,
                    independent_early_takeover_starts=1, new_independent_layouts=0,
                    active_A93_R3_curriculum_changed=False, optimizer_visits=0)
    write(output / 'MANIFEST.json', manifest)
    reader = dataset(output, manifest, statistics, resize_with_pad)
    phases, supported, full, fingerprints = Counter(), 0, 0, hashlib.sha256()
    views = dict(base_0_rgb=True, left_wrist_0_rgb=True, right_wrist_0_rgb=False)
    for i in range(len(reader)):
        sample, support = reader[i], reader.support(i)
        require(sample['image_mask'] == views and sample['actions'].shape == (40, 32)
                and sample['state'].shape == (32,) and support[:10].all()
                and np.isfinite(sample['actions']).all() and np.isfinite(sample['state']).all()
                and not sample['actions'][~support].any(), 'Real reader RGB, state, mask or action changed')
        for field in (sample['state'], sample['actions'], support,
                      *(sample['image'][name] for name, kept in views.items() if kept)):
            fingerprints.update(field.tobytes())
        fingerprints.update(sample['prompt'].encode())
        phases[entry['expert_stage_audit_only'][i]] += 1
        supported += int(support.sum()); full += int(support.all())
    require(len(reader) == 100 and full == 97 and supported == 3940,
            'The actual suffix support denominator changed')
    receipt = dict(manifest_sha256=sha(output / 'MANIFEST.json'), windows=len(reader),
                   full_H40_windows=full, supported_action_points=supported, phase_windows=dict(phases),
                   all_rows_reader_sha256=fingerprints.hexdigest(), RGB_reader='UNCHANGED_V65_RECOVERY_DATASET',
                   current_training_consumption=0, new_optimizer_updates=0, new_native_integrations=0,
                   C1_two_early_starts_met=False)
    write(output / 'READER-ACTUAL.json', receipt)
    print(output, sha(output / 'READER-ACTUAL.json'), flush=True)


def export_new(site, sealed, reconciliation_sha, case_name, seal_sha, output):
    """Read a separately closed successful D1 start; retain the original export untouched."""
    from openpi_client.image_tools import resize_with_pad
    from scripts.sla.core_hopper_native_scope import accepted
    from scripts.sla.hopper_site import pinned

    require(not output.exists() and Path(case_name).name == case_name,
            'Use a fresh cold dataset and an exact new case identity')
    close = read(pin(sealed/'RESULT.json', reconciliation_sha))
    completed = [r for r in close['cases'] if r['state'] == 'CLOSED']
    require(close['complete_reconciliation'] and len(completed) == close['terminal_cases'] == close['train_recovery_count'] == 1
            and completed[0]['id'] == case_name and completed[0]['recovery_admitted']
            and not completed[0]['student_task_success'], 'Require one real successful Teacher suffix, not student success')
    seal = read(pin(sealed/case_name/'ACTUAL.json', seal_sha))
    windows = seal['training_windows']
    require(windows > 0 and seal['actual_recovery'] and not seal['optimizer_visits']
            and seal['actual_teacher_atoms'] == 10*windows, 'The actual suffix denominator changed')
    output.mkdir(mode=0o700)
    entry = source_episode(site, output, 0, case_name.rsplit('_',1)[-1], windows, seal_sha,
                           name=case_name, seal_path=sealed/case_name/'ACTUAL.json')
    protected = read(pin(site/'v63-place-core-code-01/PROTECTED_POPULATION.json', PROTECTED))
    excluded = {r['scene']['sha256'] for r in protected['clusters'] if r.get('split') in ('QUAL','DEV','VAL','CONF','TEST')}
    require(entry['scene_sha256'] not in excluded, 'Protected scenes cannot supply early training corrections')
    worker = read(pin(site/'ADAPT0250-NATIVE-PLACE-01/worker/EXECUTION_SPEC.json',
                      '361709e35c2bf427d5644e7af37a925014ab52861c075d4782749afd371d6bfc'))
    _,_,receiver = accepted(worker)
    target = pinned(receiver['input'])['target_D1']
    stats = dict(state=pinned(target['statistics'])['state'],actions=pinned(receiver['action'])['actions'])
    manifest = dict(schema='V7_D1_EARLY_NATIVE_RECOVERY_COLD_V1',episodes=[entry],windows=windows,
        state_statistics_sha256=target['statistics']['sha256'],action_statistics_sha256=receiver['action']['sha256'],
        protected_population_sha256=PROTECTED,native_reconciliation_sha256=reconciliation_sha,
        independent_early_takeover_starts=1,new_independent_layouts=0,
        active_A93_R3_curriculum_changed=False,optimizer_visits=0)
    write(output/'MANIFEST.json',manifest)
    reader = dataset(output,manifest,stats,resize_with_pad)
    phases,supported,full,fingerprints = Counter(),0,0,hashlib.sha256()
    views = dict(base_0_rgb=True,left_wrist_0_rgb=True,right_wrist_0_rgb=False)
    for i in range(len(reader)):
        sample,support = reader[i],reader.support(i)
        require(sample['image_mask'] == views and sample['actions'].shape == (40,32)
                and sample['state'].shape == (32,) and support[:10].all()
                and np.isfinite(sample['actions']).all() and np.isfinite(sample['state']).all()
                and not sample['actions'][~support].any(), 'Cold real reader changed RGB, mask, state or valid labels')
        for value in (sample['state'],sample['actions'],support,
                      *(sample['image'][name] for name,kept in views.items() if kept)):
            fingerprints.update(value.tobytes())
        fingerprints.update(sample['prompt'].encode())
        phases[entry['expert_stage_audit_only'][i]] += 1
        supported += int(support.sum());full += int(support.all())
    require(len(reader) == windows and dict(phases) == seal['phase_windows'] and full == windows-3
            and supported == 40*windows-60, 'The supported H40 tail must be actual consumed commands')
    require(all(phases[p] for p in ('PREGRASP','DESCEND','CLOSE','LIFT')), 'This is not a complete early correction suffix')
    proof = dict(manifest_sha256=sha(output/'MANIFEST.json'),windows=windows,full_H40_windows=full,
        supported_action_points=supported,phase_windows=dict(phases),all_rows_reader_sha256=fingerprints.hexdigest(),
        RGB_reader='UNCHANGED_V65_RECOVERY_DATASET',current_training_consumption=0,
        new_optimizer_updates=0,new_native_integrations=0,independent_early_start_case=case_name)
    write(output/'READER-ACTUAL.json',proof)
    print(output,'MANIFEST',sha(output/'MANIFEST.json'),'READER',sha(output/'READER-ACTUAL.json'),flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('site', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--sealed', type=Path)
    cli.add_argument('--reconciliation-sha')
    cli.add_argument('--case-name')
    cli.add_argument('--seal-sha')
    args = cli.parse_args()
    if args.sealed is None:
        require(not any((args.reconciliation_sha,args.case_name,args.seal_sha)), 'Incomplete new-suffix arguments')
        export(args.site,args.output)
    else:
        require(all((args.reconciliation_sha,args.case_name,args.seal_sha)), 'Pin all new-suffix authority references')
        export_new(args.site,args.sealed,args.reconciliation_sha,args.case_name,args.seal_sha,args.output)
