"""Finite final-only quality queries; closed old results never re-enter a worker."""
import fcntl
import os
import subprocess
from pathlib import Path
from quality_gate import BASE, ROOT, read, sha, admitted

RAW = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270')


def complete(paths, point, card, component=False):
    for path in paths:
        if not path.exists(): return False
        row = read(path); assert row['point'] == point
        if path.name == 'record.json':
            assert row['status'] in ('VALID', 'PREQUERY_EXCLUDED_KEEP_DENOMINATOR') and row['card_sha256'] == sha(card)
            if row.get('tensor'): assert sha(row['tensor']) == row['tensor_sha256']
        else:
            assert row['validity'] in (('VALID','PREQUERY_EXCLUDED') if component else ('VALID',))
            assert row['evaluation_card_sha256'] == sha(card)
            if row['validity'] == 'VALID': assert sha(path.parent/'trajectory.npz') == row['arrays_sha256']
    return True


def main():
    (ROOT/'runtime').mkdir(exist_ok=True)
    # ponytail: one finite evaluator owns the existing append-only budgets.
    with (ROOT/'runtime/quality.lock').open('a') as lock, (ROOT/'evidence/collection.lock').open('a') as collection:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(collection, fcntl.LOCK_EX | fcntl.LOCK_NB)
        admitted()
        assert read(ROOT/'evidence/COLLECTION_PROCESS_COMPLETE.json')['status'] == 'ALL_FIXED_COLLECTION_FAMILY_PROCESSES_EXIT_ZERO'
        modern = BASE/'PI0250/runtime/venv/bin/python'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(ROOT/'code'))
        env.pop('LD_PRELOAD', None); env.pop('LD_LIBRARY_PATH', None)
        def run(python, script, arguments, environment, log):
            admitted()
            with (ROOT/'evidence'/log).open('ab') as stream:
                subprocess.run([str(python), '-u', str(ROOT/'code'/script), *arguments],
                               env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        offline_path = ROOT/'design/FIXED_OFFLINE_CARD.json'; offline = read(offline_path)
        records = [RAW/'offline/AGG4096'/domain/r['uid']/'record.json'
                   for domain, rows in [('fit',offline['fit_rows']),('factor',offline['factor_rows']),('legacy',offline['legacy_rows'])] for r in rows]
        assert len(records) == 968
        if not complete(records, 'AGG4096', offline_path):
            run(modern, 'quality_offline.py', ['--point','AGG4096'], env, 'quality-offline-AGG4096.log')
        assert complete(records, 'AGG4096', offline_path)
        fit_path = ROOT/'design/NEW_FIT192.json'; fit = read(fit_path)
        for point in ('PREFIX4096','AGG4096'):
            records = [RAW/'newfit'/point/r['input_id']/'record.json' for r in fit['rows']]; assert len(records) == 192
            if not complete(records, point, fit_path):
                run(modern, 'newfit.py', ['--point',point], env, 'quality-newfit-'+point+'.log')
            assert complete(records, point, fit_path)
        from collect import native_environment
        native, native_env = native_environment()
        families = [f for f in read(BASE/'PI0267/design/SCENE_CARD.json')['families'] if f['split']=='NEW_SIM_EVAL']
        assert len(families) == 8
        virtual_card = ROOT/'design/EVALUATION_CARD.json'
        for family in families:
            fid = family['family_id']
            records = [RAW/'virtual/AGG4096'/f'{fid}-{task}-c{c}-{backend}'/'result.json'
                       for task in ('M1','M3') for c in (0,1) for backend in ('E0','E1')]
            if not complete(records, 'AGG4096', virtual_card):
                run(native, 'quality_virtual.py', ['--point','AGG4096','--family',fid], native_env, 'quality-main-'+fid+'.log')
            assert complete(records, 'AGG4096', virtual_card)
        component = read(ROOT/'design/COMPONENT_CARD.json')
        for family in dict.fromkeys(r['family'] for r in component['rows']):
            records = [RAW/'component/AGG4096'/r['case']/'result.json' for r in component['rows'] if r['family']==family]
            assert len(records) == 16
            if not complete(records, 'AGG4096', virtual_card, component=True):
                run(native, 'quality_components.py', ['--point','AGG4096','--family',family], native_env, 'quality-component-'+family+'.log')
            assert complete(records, 'AGG4096', virtual_card, component=True)
        from admission import write
        write(ROOT/'evidence/QUALITY_QUERY_LANES_COMPLETE.json', dict(status='ALL_FIXED_NEW_QUALITY_QUERY_LANES_COMPLETE',
              new_standard_logical_rows=968, new_NEWFIT_logical_rows=384, new_main_results=64, new_component_proposals=32,
              old_offline_reused=1936, old_main_reused=128, old_components_reused=64,
              CPU_replay_analysis_video_HF_and_terminal_return_still_required=True,
              selected_checkpoint='NONE', deployment_selected_checkpoint='NONE'))
        print('ALL_FIXED_NEW_QUALITY_QUERY_LANES_COMPLETE', flush=True)


if __name__ == '__main__': main()
