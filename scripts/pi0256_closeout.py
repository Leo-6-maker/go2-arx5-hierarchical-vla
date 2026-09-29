"""Executed preparation scorecard and bounded own-evidence HF archive."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import sys
import tarfile
import numpy as np
import pi0256_preparation as prep
from pi0255_motion3 import sha,write

ROOT=prep.ROOT


def legacy_fidelity():
    rows=[]
    for i in range(6):
        a=ROOT/'episodes'/('pi0256-prep-A%02d'%i);b=ROOT.parent/'PI0255/episodes'/('pi0255-family-%03d-M1-c0'%i)/'attempt-00'
        reset=json.loads((a/'post_reset.json').read_text())
        with np.load(b/'reset_state.npz',allow_pickle=False) as old:
            match={k:bool(np.array_equal(np.asarray(reset[k],np.float32),old[k])) for k in ('root','dof')}
        x=[json.loads(v) for v in (a/'raw_ticks.jsonl').read_text().splitlines()];y=[json.loads(v) for v in (b/'raw_ticks.jsonl').read_text().splitlines()]
        assert len(x)==len(y)
        pose=all(u['post_step']['TCP_world']==v['achieved_world'] and u['post_step']['quaternion_xyzw']==v['achieved_quaternion'] for u,v in zip(x,y))
        actors=all(u['policy']==v['WBC_policy'] for u,v in zip(x,y));times=all(u['time_s']==v['actual_sim_time_s'] for u,v in zip(x,y))
        assert all(match.values()) and pose and actors and times
        rows.append(dict(family=i,reset_matches=match,all_recorded_TCP_quaternion_values_match=pose,all_actor_outputs_match=actors,
            all_timestamps_match=times,ticks=len(x),historical_missing_actor_inputs_not_reconstructed=True))
    write(ROOT/'evidence/legacy_A_fidelity.json',dict(status='PASS_ALL_SIX_LEGACY_A_REPRODUCTIONS',families=rows))


def closeout():
    legacy_fidelity()
    old=json.loads((ROOT/'evidence/preparation_replay.json').read_text())
    fixed=json.loads((ROOT/'evidence/preparation_replay_tcp_sync.json').read_text())
    assert old['groups']['A']['valid']==6 and old['groups']['B']['valid']==6 and fixed['groups']['B']['valid']==6
    assert len(old['pairs'])==6 and len(fixed['pairs'])==6
    assert fixed['groups']['B']['passed']<6 and fixed['groups']['C']['valid']==0
    cases=old['cases']+[x for x in fixed['cases'] if x['group']=='B']
    assert len(cases)==18 and all(x['validity']=='VALID' and not x['success'] for x in cases)
    budget=json.loads((ROOT/'physical_budget.json').read_text())
    assert budget['starts']==budget['preparation_starts']==18
    assert sum(x['control_steps'] for x in cases)==budget['steps']
    for row in cases:
        out=ROOT/'episodes'/row['case']
        ticks=[json.loads(x) for x in (out/'raw_ticks.jsonl').read_text().splitlines()]
        clean=[x for x in ticks if x['clean']]
        row['predicate_fail_ticks']={k:sum(x[k]>limit for x in clean) for k,limit in
            [('speed_mps',.02),('angular_speed_radps',.20),('position_error_m',.035),('rotation_error_rad',.10)]}
        with np.load(out/'actor_inputs/0000-arm.npz',allow_pickle=False) as d:row['first_arm_history_shape']=list(d['history'].shape)
        row['reset_native_TCP_to_first_actor_change_m']=float(np.linalg.norm(np.asarray(json.loads((out/'post_reset.json').read_text())['TCP_world'])-ticks[0]['pre_step']['TCP_world']))
    summary=dict(schema='PI0256_EXECUTED_PREPARATION_CLOSEOUT_V1',status='PREPARATION_V2_BLOCKED',
        original_A=old['groups']['A'],initial_B=old['groups']['B'],source_corrected_B=fixed['groups']['B'],C='NOT_RELEASED',
        counters=dict(native_physics_starts=18,native_control_steps=budget['steps'],native_WBC_actor_calls=2*(budget['steps']-18),
                      task_physics_starts=0,task_phase_ticks=0,GR00T_queries=0,GR00T_heads=0,optimizer_updates=0,TRAIN_episodes=0,MT_VAL_episodes=0),
        pairs=dict(initial_B=old['pairs'],source_corrected_B=fixed['pairs']),cases=cases,
        scientific_stop='frozen shared preparation requires B6/6 then C6/6; effective B fails, no new reference/pose/threshold search',
        CPU_native_canonical=json.loads((ROOT/'evidence/native_canonical_cpu_replay.json').read_text()),
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE',historical_PI0254_PI0255_unchanged=True)
    write(ROOT/'evidence/preparation_stage_summary.json',summary)
    (ROOT/'reports').mkdir(exist_ok=True)
    lines=['# PI0256 executed preparation result','',
        '**PREPARATION_V2_BLOCKED — frozen upstream admission FAIL.**','',
        '| Cohort | Valid | Qualified | Outcome |','|---|---:|---:|---|',
        '| A legacy |6|0|3 gate rejects,3 native timing aborts|',
        '| B initial package |6|0|6 preparation timeouts|',
        '| B source-corrected cache package |6|0|Frozen preparation criteria not met|',
        '| C fresh |0|0|Not released|','',
        '| Case | End s | Clean XYZ RMS m | Clean SO3 RMS rad | Max stable s | Soft joint violation ticks |','|---|---:|---:|---:|---:|---:|']
    for x in cases:lines.append('|%s|%.3f|%.6f|%.6f|%.3f|%d|'%(x['case'],x['end_time_s'],x['clean_XYZ_RMS_m'],x['clean_SO3_RMS_rad'],x['stable_interval_max_s'],x['soft_joint_violation_ticks']))
    lines+=['','RMS values summarize clean recorded ticks; they are not the continuous, conjunctive qualification predicate.',
        'All first-valid failures are retained. No result was retried for a better score. No PASS is inherited across cache versions.',
        'Six old-family A/B pairs and six A/corrected-B pairs match pre-first-physics root, DOF and Python/NumPy/Torch CPU/CUDA RNG identities.',
        'Post-intervention reference/history/actor differences are package effects; no unique internal causal attribution is claimed.','',
        'Actual alias and source proof: camera fetch overwrites a derived EEF cache that shares raw rigid-state storage.',
        'The constructor-only cache-copy correction preserves raw state and makes the measured TCP stable across camera reads.',
        'Its actual constructor assertions passed. The native WBC source, weights, gains, dynamics, reset distribution, gate and reference formula remained unchanged.',
        'The frozen native soft joint predicate includes gripper bounds; hard URDF margins are separately retained. No bound or predicate was relaxed.','',
        'Corrected B clean-tick XYZ RMS ranges10.564–23.149mm; SO3 RMS ranges0.651731–1.490608rad.',
        'Rotation error exceeds0.10rad on249–250 of250 ticks per corrected condition; angular speed exceeds0.20rad/s on229–248 ticks.',
        'Translation speed exceeds0.02m/s on200–245 ticks; the original soft joint violation flag is true on all250 ticks in each condition.',
        'The failure is therefore not solely a soft-bound diagnostic. There is no0.5s qualified interval.',
        'The source correction makes reset-to-first-actor camera TCP change exactly0 in all six fixed conditions.',
        'Native measured TCP/SO3 versus independent URDF/root/DOF FK errors are retained in the CPU case summaries; no wrong-quaternion/frame substitute explains the observed scale.','',
        'The original elapsed deadline remains5s including reset. Native float32 clock crossing can produce a final timeout observation at5.02s;',
        'that observation cannot qualify outside the deadline. It is included in the actual control-step count.','',
        'All raw/canonical references, actual current observations/histories, actor outputs, pre/post-step state, native flags and pre-autoreset snapshots are retained.',
        'Independent CPU reconstruction verifies full-file hashes, actual history bytes, pose/speed/SO3/stability decisions and pair identities.',
        'Independent actual-native CPU canonical reconstruction reports its numerical differences without widening any scientific threshold.','',
        'Counters: '+json.dumps(summary['counters'],sort_keys=True)+'.','',
        'P2 expert pilots/collection, actual C_MT/tokenizer/collator/resume work, real-model canaries, Base/G6000 and offline model evaluation are NOT executed.',
        'There is no multitask dataset, trained model, language-generalization result or task-phase feasibility conclusion from this run.',
        'This is a preparation-package negative result on the registered conditions under the frozen stability/safety conjunction.',
        'No remaining budget transfers to new poses, references, seeds, losses, tasks or checkpoints. selected/deployment remain NONE.',
        'PI0254 U quality FAIL/R zero-update BLOCKED and original PI0255 36 preparation failures are unchanged.',
        'No learned-policy WBC, TEST/DEV/F, hardware, broad cleanup, mail/monitor or PR merge.','',
        'The NodeLocal IO move occurred only after a current attempt finished;1391 files were byte-verified and the original NFS copy remains.',
        'No healthy worker was edited/restarted and completed cells were reused; quota remains UNKNOWN.']
    (ROOT/'reports/CLOSEOUT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=summary['status'],counters=summary['counters'])),flush=True)


def archive():
    from huggingface_hub import HfApi,hf_hub_url
    import requests
    sys.path.insert(0,str(ROOT.parent/'PI0255/runtime'))
    from pi0255_archive_pilot import digest,save
    summary=json.loads((ROOT/'evidence/preparation_stage_summary.json').read_text())
    assert summary['status']=='PREPARATION_V2_BLOCKED'
    out=ROOT/'archive/preparation-v1';out.mkdir(parents=True,exist_ok=True)
    files=[]
    for directory in ('episodes','design'):
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file())
    files.extend(p for p in (ROOT/'evidence').glob('*') if p.suffix in ('.json','.npz') and p.name!='preparation_HF_public_receipt.json')
    files.extend(p for p in (ROOT/'runtime').glob('pi025*.py') if p.is_file())
    files.extend((ROOT/'runtime').glob('execution_seal*.json'))
    files.extend(ROOT/n for n in ('physical_starts.jsonl','attempt_results.jsonl','physical_budget.json'))
    import shutil
    notice=ROOT/'evidence/WBC_LICENSE_NOTICE.txt'
    shutil.copy2(ROOT.parent/'PI0255/evidence/WBC_LICENSE_NOTICE.txt',notice)
    files.append(notice)
    # Own synthetic outputs and public project code only; no simulator/actor/model/token payload.
    files=sorted(set(files),key=lambda p:p.relative_to(ROOT).as_posix())
    assert all(p.resolve().is_relative_to(ROOT.resolve()) and not p.is_symlink() for p in files)
    assert all(p.suffix in ('.json','.jsonl','.npz','.py','.txt') for p in files)
    entries=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in files]
    tree=hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    manifest=dict(schema='PI0256_PREPARATION_PUBLIC_MEMBERS_V1',files=entries,files_count=len(entries),bytes=sum(x['bytes'] for x in entries),
                  tree_sha256=tree,scope='own synthetic observations/numerical evidence/public project code; no private source/assets/weights/token',counters=summary['counters'])
    save(out/'MEMBER_MANIFEST.json',manifest)
    path=out/'PI0256_preparation_A_B_source_correction.tar.gz'
    if not path.exists():
        with path.open('wb') as f,gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w|') as tar:
            for p,row in zip(files,entries):
                info=tarfile.TarInfo(row['path']);info.size=row['bytes'];info.mode=0o644;info.mtime=0
                with p.open('rb') as data:tar.addfile(info,data)
    repo='leoyuer/pi0246-gate-pass-campaign';prefix='evidence/PI0256/preparation-A-B-source-correction-v1-20260929'
    api=HfApi(token=Path('/home/liuyu-nfs/.config/huggingface/token').read_text().strip());assert not api.model_info(repo).private
    commit_file=ROOT/'archive/preparation-upload-commit.json'
    if commit_file.exists():
        c=json.loads(commit_file.read_text());assert c['tree']==tree and c['archive_sha256']==digest(path);revision=c['revision']
    else:
        revision=api.upload_folder(repo_id=repo,repo_type='model',folder_path=str(out),path_in_repo=prefix,commit_message='PI0256 bounded preparation18 first-valid failures and source correction evidence').oid
        save(commit_file,dict(revision=revision,tree=tree,archive_sha256=digest(path),repo=repo,prefix=prefix))
    session=requests.Session();session.trust_env=False;observed=[]
    for source in (out/'MEMBER_MANIFEST.json',path):
        target=ROOT/'archive'/('anonymous-'+source.name);h=hashlib.sha256();size=0
        with session.get(hf_hub_url(repo,filename=prefix+'/'+source.name,revision=revision),stream=True,timeout=(30,180)) as response:
            response.raise_for_status();assert 'Authorization' not in response.request.headers
            with target.open('wb') as f:
                for b in response.iter_content(8<<20):f.write(b);h.update(b);size+=len(b)
        assert size==source.stat().st_size and h.hexdigest()==digest(source)
        observed.append(dict(path=source.name,bytes=size,sha256=h.hexdigest(),anonymous=True))
    expected={x['path']:x for x in entries};members=[]
    with tarfile.open(ROOT/'archive'/('anonymous-'+path.name),mode='r|gz') as tar:
        for item in tar:
            assert item.isfile() and not item.issym();row=expected.pop(item.name);h=hashlib.sha256();size=0
            with tar.extractfile(item) as f:
                for b in iter(lambda:f.read(1<<20),b''):h.update(b);size+=len(b)
            assert size==row['bytes'] and h.hexdigest()==row['sha256'];members.append(row)
    assert not expected
    receipt=dict(status='PASS_PUBLIC_ANONYMOUS_FULL_FILE_AND_MEMBER_READBACK',repo_id=repo,prefix=prefix,revision=revision,
        tree_sha256=tree,source_files=len(entries),source_bytes=manifest['bytes'],HF_files=observed,members=members,anonymous=True,
        model_weights_uploaded=False,private_source_or_assets_uploaded=False,GR00T_queries=0)
    save(ROOT/'evidence/preparation_HF_public_receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('status','revision','tree_sha256','source_files','source_bytes')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',action='store_true');a=p.parse_args()
    archive() if a.archive else closeout()
