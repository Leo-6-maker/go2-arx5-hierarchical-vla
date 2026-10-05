"""Fixed-scope research return from complete tables; no new evaluation or qualification."""
from quality_gate import ROOT,RAW,read,sha
from quality_worker import write
from compare import saved_table

def main():
    out=RAW/'delivery';verdict=read(out/'PRIMARY_VERDICT.json');comparison=read(out/'INTERVENTION_COMPARISON.json')
    assert verdict['status']=='PRIMARY_EVALUATION_COMPLETE' and comparison['status']=='COMPLETE_FIXED_LOCAL_FIELD_AND_ORIGINAL_START_COMPARISON'
    cpu=read(out/'CPU_REPLAY.json');assert cpu['status']=='PASS_PI0274_ALL_NEW_S4_DECODE_AND_TRAJECTORIES'
    rows=saved_table(out/'VIRTUAL192.csv');assert sha(out/'VIRTUAL192.csv')==cpu['tables']['VIRTUAL192.csv']['sha256']
    lines=['# PI0274 — Pose equilibrium and rotational recovery','',
        'Execution completion and quality qualification are separate. One new EQUIL5120 arm; closed AGG4096/BROAD5120 outputs reused.',
        'All results are development virtual EEF experiments. E0/E1 are mathematical executors; no WBC, dynamics, physical transfer, blind TEST or real-time claim.',
        'Formal selected_checkpoint=NONE; deployment_selected_checkpoint=NONE.','',
        '## Original-start task, reliability and safety','',
        '| Point | Task/backend | First completion /16 | Reliable /16 | Reliable commands /8 | Collisions | Gate refusals |',
        '|---|---|---:|---:|---|---:|---:|']
    totals={}
    for point in ('AGG4096','BROAD5120','EQUIL5120'):
        totals[point]={}
        for task in ('M1','M3'):
            own=[r for r in rows if r['point']==point and r['task']==task];assert len(own)==32
            totals[point][task]=dict(original=sum(r['success'] for r in own),reliable=sum(r['reliable'] for r in own),collisions=sum(r['dangerous_collision'] for r in own))
            for backend in ('E0','E1'):
                group=[r for r in own if r['backend']==backend];assert len(group)==16
                commands=[sum(r['reliable'] for r in group if r['command']==c) for c in (0,1)]
                lines.append(f"| {point} | {task}/{backend} | {sum(r['success'] for r in group)} | {sum(r['reliable'] for r in group)} | {commands[0]}, {commands[1]} | {sum(r['dangerous_collision'] for r in group)} | {sum(r['gate_refusals'] for r in group)} |")
    lines+=['','## Fixed local field: per draw, world-left, reference only','',
        '| Point | Task | Center XYZ median (mm) | Center SO3 median (rad) | Toward-center angular increments /576 | rho>1 matrices /96 | largest singular value>1 /96 |',
        '|---|---|---:|---:|---:|---:|---:|']
    field={}
    for point in ('BROAD5120','EQUIL5120'):
        proof=read(out/f'FIELD_{point}.json');assert proof['observed_rows']==2496
        field[point]=proof['summary']
        for r in proof['summary']:
            lines.append(f"| {point} | {r['task']} | {1000*r['center_XYZ_median_m'] if r['center_XYZ_median_m'] is not None else 'UNDEFINED'} | {r['center_SO3_median_rad']} | {r['toward_center_increment']} | {r['sampled_matrices_spectral_radius_above_one']} | {r['sampled_matrices_largest_singular_value_above_one']} |")
    lines+=['','These are finite-scale response estimates from two fixed noise draws and 24 TRAIN families. They are not analytic Jacobians or stochastic/WBC stability certificates. Matrix denominators remain 96 per task; incomplete/rejected neighborhoods do not acquire eigenvalues.',
        'Finite REJECT references remain in physical-error tables but were never executed. Zero-denominator center ratios are undefined. Translation-only field poses were excluded from supplemental training.',
        'The same-input teacher map and its actual projection metadata are retained for every cell. exp(-0.8) applies only when the actual teacher projection is terminal.','',
        '## Matched-case gains, losses and retention','',
        '| Comparator | Metric | Task/backend | Retained | Lost | Gained | Both fail |','|---|---|---|---:|---:|---:|---:|']
    for r in comparison['matched_case_strata']:
        c=r['counts'];lines.append(f"| {r['before']} | {r['label']} | {r['task']}/{r['backend']} | {c.get('RETAINED',0)} | {c.get('LOST',0)} | {c.get('GAINED',0)} | {c.get('BOTH_FAIL',0)} |")
    lines+=['','## Qualification and interpretation','']
    for point in ('BROAD5120','EQUIL5120'):
        lines.append(f"- {point}: original={verdict['ORIGINAL_VIRTUAL_QUALIFICATION'][point]['PASS']}; reliable={verdict['RELIABLE_VIRTUAL_QUALIFICATION'][point]['PASS']}; safe={verdict['full_point_safe'][point]}; original ROT LEGACY retained={verdict['legacy_retention_vs_original_ROT2048'][point]}.")
    delta={task:{key:totals['EQUIL5120'][task][key]-totals['BROAD5120'][task][key] for key in ('original','reliable','collisions')} for task in ('M1','M3')}
    lines += ['',f"EQUIL minus BROAD original/reliable/collision counts: {delta}.",
        'Local field improvement alone does not establish repair: it must be read with original-start M1 post-arrival SO3 departures, complete 35–40 s holding, M3 entry/order, safety and legacy.',
        'If M1 improves while M3 remains unqualified, the result is a targeted M1 advance. If local maps improve without reliable M1 holding, goal-local coverage is insufficient to solve the complete failure.',
        'Any qualified candidate is virtual research only. No automatic structural feedback intervention or physical transfer is authorized.','',
        '## Provenance and retained failures','',
        'A0 preserves all 128 BROAD/TARGET trajectories and onset/visibility evidence; new EQUIL onset covers all 64 original-start episodes. Same-tick reference, installed reference and achieved pose remain separate.',
        'All fixed panel rows, REJECT/nonfinite/collision failures and component exclusions are preserved. No quality retries, new seeds or outcome-based selection are permitted.',
        'CPU replay checks saved four-step BF16 Euler arithmetic, actual noise, FP32 decoder boundary and metrics. Gate, scorer, renderer and executor sources are shared frozen implementations; this is not an independent physics backend.',
        'The first frozen EVAL family ALL8 video uses saved actual RGB/timestamps and includes all failures. No rerendering or score-based clip selection.',
        'Detailed tables: OFFLINE2904, NEW_FIT576, LATE_CONTEXT2368, VIRTUAL192, COMPONENT96, PAIRED_FIELD2496 and MATCHED_CASE_GAIN_LOSS256. S4FIT local rows are historical context, not a new candidate.',
        'Final artifacts, literal cumulative budgets, two complete Trainer readbacks and dataset/output/summary revisions are closed by the final audit and terminal canonical.','']
    path=out/'FINAL_RESEARCH_RETURN.md';text='\n'.join(lines)
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    write(out/'RESEARCH_RETURN_RECEIPT.json',dict(status='FIXED_FIELD_AND_ORIGINAL_START_RETURN',report_sha256=sha(path),
        task_totals=totals,EQUIL_minus_BROAD=delta,field=field,quality_verdict_sha256=sha(out/'PRIMARY_VERDICT.json'),
        comparison_sha256=sha(out/'INTERVENTION_COMPARISON.json'),selected_checkpoint='NONE',deployment_selected_checkpoint='NONE',
        model_queries=0,teacher_calls=0,renders=0,physics=0))
    print('FIXED_FIELD_AND_ORIGINAL_START_RESEARCH_RETURN',delta,flush=True)
if __name__=='__main__':main()
