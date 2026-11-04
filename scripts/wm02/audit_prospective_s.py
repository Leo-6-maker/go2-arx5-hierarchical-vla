"""Audit completed private S streams only; no MuJoCo import, inference or integration."""
import argparse,json,time,hashlib,subprocess
from pathlib import Path
import numpy as np
from go2arx5_vla.world_model.storage import digest,write_json
from go2arx5_vla.world_model.paired_replay import read
from go2arx5_vla.world_model.contracts import validate_atom
from go2arx5_vla.world_model.transitions import validate_record

def audit(root,output):
    start=time.monotonic();cpu=time.process_time();root=Path(root);output=Path(output)
    if output.resolve().is_relative_to(root.resolve())or json.loads((root.parent/'LEASE.json').read_text())['status']!='CLOSED':raise ValueError('Terminal source data and separate new audit output required')
    output.mkdir(exist_ok=False);result=json.loads((root/'RESULT.json').read_text());pins={str(p.relative_to(root)):digest(p)for p in sorted(root.rglob('*'))if p.is_file()};rows_count=ticks=unsafe=terminals=0;models={};restore_pins={};verified=0
    for p in sorted(root.rglob('*.pkl.gz')):
        if 'restore'in p.name:continue
        v=read(p)
        if 'rows'not in v:continue
        rows=v['rows'];previous=None
        for row in rows:
            validate_atom(row);validate_record(row,native_dt=.002)
            if row['split']!='DEV'or row['domain']!='S_FIXED_BASE_DIAGNOSTIC':raise ValueError('New validation must remain DEV source, never FIT or target')
            if previous and(abs(previous['t1']-row['t0'])>1e-8 or previous['parent_id']!=row['parent_id']):raise ValueError('Stream clock/parent break')
            h=row['context']['model_sha256'];parent=row['parent_id']
            if models.setdefault(parent,h)!=h:raise ValueError('Physical model changed within parent')
            ref=row['control_restore_ref'];q=Path(ref['path'])
            if str(q)not in restore_pins:restore_pins[str(q)]=digest(q)
            if restore_pins[str(q)]!=ref['sha256']:raise ValueError('Actual complete restore reference differs')
            point=row['native_progress']['point']
            if not np.array_equal(row['reference_compiled'][point],row['reference_installed']):raise ValueError('Installed H40 point differs')
            for i,x in enumerate(row['reference_consumed']):
                if not np.array_equal(x['eef10'],row['reference_installed'])or abs(x['time_s']-row['t0']-.002*i)>1e-8:raise ValueError('Actual2ms command consumption differs')
            rows_count+=1;ticks+=row['physics_ticks'];unsafe+=not bool(row['state_after']['safety_valid']);terminals+=bool(row['terminal']);previous=row
    # Recommendations were fsynced before any effect; verify original64-bit
    # physical proposal hashes against separately executed online records.
    for p in sorted(root.glob('scene-*/*/BEFORE_ANY_BRANCH.json')):
        v=json.loads(p.read_text());pin=digest(p)
        for method,choice in v['methods'].items():
            q=p.parent/('online-'+method+'.pkl.gz');rows=read(q)['rows'];raw=np.asarray(v['physical_raw'][choice['index']],dtype='<f8')
            if hashlib.sha256(raw.tobytes()).hexdigest()!=v['raw_sha256'][choice['index']]:raise ValueError('Original proposal hash differs')
            if p.stat().st_mtime_ns>q.stat().st_mtime_ns:raise ValueError('Recommendation file appeared after its actual stream')
            for k,row in enumerate(rows):
                expected=raw[k]if k<8 else np.r_[np.zeros(6),raw[-1,6]]
                if not np.array_equal(expected,row['action_raw']):raise ValueError('Executed command differs from frozen recommendation')
            if any(x['selection_sha256']!=pin for x in json.loads((p.parent/'RESULT.json').read_text())['online'].values()):raise ValueError('Phase lost before-effect identity')
            verified+=1
    if ticks!=result['live_physics_ticks']or rows_count!=result['actual_transitions']:raise ValueError('Terminal transition count differs from actual streams')
    write_json(output/'STREAM_MANIFEST.private.json',dict(files=pins));manifest=digest(output/'STREAM_MANIFEST.private.json')
    audit_sha=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[2]),'rev-parse','HEAD'],text=True).strip()
    report=dict(status='SAVED_S_STREAM_AUDIT_COMPLETE',audit_source_sha=audit_sha,audit_code_sha256=digest(__file__),source_run_sha=result['source_sha'],original_RESULT_sha256=digest(root/'RESULT.json'),stream_manifest_sha256=manifest,stream_manifest_files=len(pins),actual_transition_records=rows_count,actual_live_ticks=ticks,reported_shadow_ticks=result['shadow_ticks'],parents=len(models),scene_model_sha256=models,verified_before_effect_method_streams=verified,complete_restore_refs_checked=len(restore_pins),unsafe_live_rows=unsafe,terminal_live_rows=terminals,domain='S_FIXED_BASE_ONLY',split='DEV_NEVER_FIT',new_physics=0,model_updates=0,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start);write_json(output/'AUDIT.json',report);return report

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);print(json.dumps(audit(a.root,a.output),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
