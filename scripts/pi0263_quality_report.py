"""Render the completed fixed experiment from saved CPU tables; no model calls."""
import collections
import csv
import hashlib
import json
import statistics
from pathlib import Path

from pi0261_quality_report import table

ROOT = Path(__file__).resolve().parents[1] / 'reports/PI0263'
OUT = ROOT / 'final'
POINTS = ('P2000', 'F2000', 'X2000')


def load_csv(name):
    with (OUT / (name + '.csv')).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    q = json.loads((OUT / 'QUALIFICATION.json').read_text())
    cpu = json.loads((OUT / 'CPU_REPLAY.json').read_text())
    audit = json.loads((ROOT / 'MAIN_TRAINING_AUDIT.json').read_text())
    assert cpu['status'] == 'PASS_INDEPENDENT_COMPUTATION_PATH_CPU_3456_REPLAY' and cpu['rows'] == 3456
    assert audit['status'] == 'PASS_BOTH_2000_MATCHED_TRAINING_INTEGRITY'
    assert all(not r['combined_pass'] for r in q['points']) and q['synthetic_research_candidate'] == 'NONE'
    assert q['selected_checkpoint'] == q['deployment_selected_checkpoint'] == 'NONE'
    legacy = load_csv('LEGACY_PANEL_ROWS')
    crossed = load_csv('CROSSED_PANEL_ROWS')
    assert len(legacy) == 2040 and len(crossed) == 1920
    for point in POINTS:
        assert sum(r['point'] == point for r in legacy) == 680
        assert sum(r['point'] == point for r in crossed) == 640
    lines = [
        '# PI0263 — completed F/X template counterbalancing pilot', '',
        '**Execution:** both arms completed 2000 updates. **Quality:** F2000 and X2000 are valid FAIL on both qualifications. '
        '**Synthetic research candidate / formal selected / deployment:** NONE / NONE / NONE.', '',
        'Authority: commit `1c574b6e6e9d6a7851e64ca57e89c468a4c9eb28`, '
        'Issue #11 canonical 5913659202, PR #77 pointer 5913663574. '
        'PI0261 full qualification FAIL and PI0262 COMPLETE/MIXED remain unchanged.', '',
        '## Scientific finding', '',
        'Counterbalanced X learned the newly supervised TRAIN T0/T1 combinations: original bindings are '
        '60/60 in both noises, other-T bindings 60/60 and 58/60; F retains 45/60 and 45/60 on other-T. '
        'This supports the benefit of the complete counterbalanced assignment policy in this setting. '
        'Its marginal template frequencies also differ; this does not isolate decorrelation at fixed marginals.', '',
        'Transfer is incomplete. X fails original M1 SEMANTIC/B1 at 8/12 (c0=3/6, c1=5/6). '
        'The new KIN-VAL criterion fails T1/B0=16/20, V0/B1=14/20 and V1/B0=16/20. '
        'All eight crossed numeric groups pass, so low geometric errors alone do not establish correct target choice. '
        'X improves five of eight robustness groups to passing versus F two, but the preregistered criterion requires all eight.', '',
        'F also improves multiple panels versus P. Both arms share another 2000 updates and the repeated-unit sampling design; '
        'their common gains cannot be credited uniquely to counterbalancing. X KIN-VAL V0/B1 drops below F (14 versus 15), '
        'and TRAIN V1/B0 drops below F (40 versus 42). The result remains MIXED rather than universal improvement.', '',
        'X T0/T1 TRAIN recombinations are now TRAIN-exposed. V0/V1 remain absent from optimization and statistics fitting, '
        'but this is DEVELOPMENT VALIDATION informed by prior research, not blind TEST or necessarily unseen base-model language. '
        'There are 30 TRAIN and 10 KIN-VAL geometry families; template/noise rows are correlated. '
        'No binomial independence, unique neural mechanism, physical generalization, WBC or hardware success is claimed.', '',
        '## Frozen execution and actual integrity', '',
        'Both arms start at PI0261 P2000 `875abdde2cc7c9e6858fac008e53f2f4b3375255`, '
        '`arms/PI0261/P/checkpoint-2000`. Plan SHA256 '
        '`602dc213ca91f2dcd5099bd97a45587fe5fe1e0d205243596d48e07c5f662617`. '
        '1000 original source units, 250 per M1/M3 × PAIR/REPLAY stratum, each visited twice; '
        'each arm 4000 examples, M1/M3 each 2000, 50% common-start and 50% ordinary replay. '
        'X M1 T0/T1 each 1000 examples. Source frequencies are inherited, not globally equalized.', '',
        'Actual 8000 microbatches verify source, color, labels, masks and nonlanguage inputs. '
        'Corresponding native epsilon/tau and S4 starting packets match within units and across arms. '
        'Both real continuous8 versus 4/save/fresh-process-restore/4 canaries pass with zero model, optimizer and scheduler differences. '
        'All four Euler steps retain gradients and modes restore; no expert prefix or padding intervention. '
        '559 frozen tensors remain byte-identical to P; all 471 trainable tensors changed. '
        'BF16 frozen backbone, FP32 trainable head and original BF16 inference are retained.', '',
        'Fresh AdamW LR1e-5, betas(.9,.999), eps1e-8, weight decay1e-5, clip1, warmup10/cosine2000, '
        'microbatch1/accumulation2. Inherited full-S4 objective: native + .4 position + .1 increment + .1 rotation + .1 peak-speed margin. '
        'Read-only final audit replays actual FP32 loss arithmetic exactly. The first reader incorrectly compared a double-precision sum; '
        'the reader was corrected without changing training, coefficients or tolerances.', '',
        'F main PBS640396 and X main PBS640622 finish exit0. PBS640356 failed at the CRLF shebang before Python '
        '(zero updates/heads); its logs remain. No quality retries, duplicate healthy jobs or scientific recovery updates.', '',
        '## Accounting', '',
        table(('Item', 'Actual', 'Limit / disposition'), [
            ('Main updates / microbatches', '4000 / 8000', '4000 / 8000'),
            ('Discarded canary updates', 32, 32), ('Completed technical recovery updates', 0, 256),
            ('All completed updates', 4032, 4288), ('Raw training heads', 72624, 100000),
            ('Unique logical rows P/F/X', '1152 / 1152 / 1152', 3456),
            ('Legacy / crossed memberships per point', '680 / 640', '168 overlap; 1152 unique'),
            ('Donor rows reused', 1152, '0 new donor queries'),
            ('New valid queries / inference heads', '2232 / 8928', '2432 / 9728 hard cap'),
            ('Exact forward reuse F/X', '36 / 36', 'labels individually rescored'),
            ('Invalid attempts / quality retries', '0 / 0', 'no spare-budget probes'),
            ('New scenes / renders / IK / stats fits', '0 / 0 / 0 / 0', 0),
            ('Physics / WBC / TEST / DEV-F / real hardware', '0 / 0 / 0 / 0 / 0', 0)]), '',
        'First valid outputs are retained. A sole producer performed the independent computation-path CPU replay of all 3456 '
        'saved rows; this is not external signoff. Original decoder, actual noise, labels, gate and denominators are verified. '
        'No checkpoint1000 quality query or alternate candidate search was performed.', '',
        'CPU receipt scope erratum: the completed replay checked3456 rows, but its final loop cursor X2000 '
        'was accidentally used to label the scope and donor-reuse count. The original receipt and computation code '
        'are retained under evidence/cpu_schema_erratum. A narrow check of the actual receipt expressions and '
        'all3456 record/tensor hashes corrected only the scope label and donor count. Metrics, provenance, '
        'quality FAIL and model-query totals are unchanged.', '',
        '## Original qualification — unchanged gates', '',
        'Four numeric groups each require median XYZ≤20mm, p95≤35mm, median SO3≤.050rad and zero REJECT. '
        'Eight semantic groups each require ≥10/12; ties and rejects are not correct.', '',
        table(('Point', 'Task', 'Noise', 'N', 'median XYZ mm', 'p95 XYZ mm', 'median SO3 rad', 'Reject', 'Pass'),
              [(r['point'], r['task'], r['noise'], r['n'], f"{r['median_xyz_m']*1000:.3f}",
                f"{r['p95_xyz_m']*1000:.3f}", f"{r['median_so3_rad']:.6f}", r['reject'], r['numeric_pass'])
               for r in q['legacy_groups'] if r['kind'] == 'KIN-VAL']), '',
        table(('Point', 'Task', 'Wording', 'Noise', 'Correct/12', 'c0/6', 'c1/6', 'Reject', 'Pass'),
              [(r['point'], r['task'], r['kind'], r['noise'], r['correct'], r['c0'], r['c1'], r['reject'], r['semantic_pass'])
               for r in q['legacy_groups'] if r['kind'] != 'KIN-VAL']), '',
        '## Complete crossed M1 results', '',
        'KIN-VAL: all eight template/noise groups must independently achieve ≥17/20 plus the numeric/zero-REJECT gates. '
        'TRAIN values are diagnostics. Correct c0/c1 and both-correct family counts remain separate.', '',
        table(('Point', 'Split', 'Template', 'Noise', 'Correct/N', 'c0', 'c1', 'Both families', 'median XYZ mm',
               'p95 XYZ mm', 'median SO3 rad', 'Reject', 'Numeric pass'),
              [(r['point'], r['split'], r['template'], r['noise'], f"{r['correct']}/{r['n']}", r['c0'], r['c1'],
                r['both_commands_correct'], f"{r['median_xyz_m']*1000:.3f}", f"{r['p95_xyz_m']*1000:.3f}",
                f"{r['median_so3_rad']:.6f}", r['reject'], r['numeric_pass']) for r in q['crossed_groups']]), '',
        'X V0/B1 KIN-VAL is c0=4/10, c1=10/10, both=4/10; V1/B0 is c0=10/10, c1=6/10, both=6/10. '
        'TRAIN V1/B0 is c0=30/30, c1=10/30. Pooling commands or choosing a favorable noise would hide these failures.', '',
        '## Original versus other TRAIN bindings', '',
        table(('Point', 'Binding', 'Noise', 'N', 'Correct', 'c0', 'c1', 'Reject'),
              [(r['point'],r['binding_relation'],r['noise'],r['n'],r['correct'],r['c0'],r['c1'],r['reject'])
               for r in load_csv('ORIGINAL_VS_OTHER_BINDINGS')]), '',
        '## Direction, bias, wording flips and noise boundaries', '',
        'Using the existing normalized signed target error s0/s1, g=(s1−s0)/2 and b=(s0+s1)/2. '
        'Both commands correct iff s0<0<s1, equivalently g>|b|. Positive ordering with one wrong target is not semantic success. '
        'These are output-space descriptions, not proven causal neural mechanisms.', '']
    targets = load_csv('TARGET_CHOICE_DECOMPOSITION')
    flips = load_csv('WORDING_FLIPS')
    noise = load_csv('NOISE_BOUNDARY_CROSSINGS')
    rows=[]
    for point in POINTS:
        for split in ('TRAIN','KIN-VAL'):
            for template in ('T0','T1','V0','V1'):
                for packet in ('B0','B1'):
                    sub=[r for r in targets if (r['point'],r['split'],r['template'],r['noise']) == (point,split,template,packet)]
                    counts=collections.Counter(r['category'] for r in sub)
                    rows.append((point,split,template,packet,len(sub),counts['both_correct'],counts['positive_ordering_bias_dominates'],
                                 counts['nonpositive_ordering'],f"{statistics.median(float(r['g']) for r in sub):.5f}",
                                 f"{statistics.median(float(r['b']) for r in sub):.5f}"))
    lines += [table(('Point','Split','Template','Noise','Families','Both correct','Positive ordering/wrong','Nonpositive ordering','median g','median b'),rows),'']
    rows=[]
    for point in POINTS:
        for split in ('TRAIN','KIN-VAL'):
            for a,b in (('T0','T1'),('V0','V1')):
                for packet in ('B0','B1'):
                    sub=[r for r in flips if (r['point'],r['split'],r['from_template'],r['to_template'],r['noise']) == (point,split,a,b,packet)]
                    rows.append((point,split,a+'→'+b,packet,len(sub),sum(r['correct_to_wrong']=='True' for r in sub),sum(r['wrong_to_correct']=='True' for r in sub)))
    lines += [table(('Point','Split','Wording change','Noise','Rows','Correct→wrong','Wrong→correct'),rows), '',
              'All six predefined template-pair comparisons, per-family signed margins and other wording flips are preserved in the raw CSVs.', '']
    rows=[]
    for point in POINTS:
        for split in ('TRAIN','KIN-VAL'):
            for template in ('T0','T1','V0','V1'):
                sub=[r for r in noise if (r['point'],r['split'],r['template']) == (point,split,template)]
                rows.append((point,split,template,len(sub),sum(r['c0_boundary_crossing']=='True' for r in sub),sum(r['c1_boundary_crossing']=='True' for r in sub)))
    lines += [table(('Point','Split','Template','Families','c0 B0/B1 crossing','c1 B0/B1 crossing'),rows), '',
              '### Every adverse X KIN-VAL family/template/noise outcome', '',
              table(('Family','Template','Noise','Category','s0','s1','g','b'),
                    [(r['family'],r['template'],r['noise'],r['category'],*[f"{float(r[k]):.6f}" for k in ('s0','s1','g','b')])
                     for r in targets if r['point']=='X2000' and r['split']=='KIN-VAL' and r['category']!='both_correct']), '',
              '## Color and command strata — complete denominators', '',
              'Command0 is not a fixed color. The underlying registry controls requested colors; original unequal color/family frequencies are retained.', '',
              table(('Point','Split','Template','Color','Command','Noise','N','Correct','Reject'),
                    [(r['point'],r['split'],r['template'],r['color'],r['command'],r['noise'],r['count'],r['correct'],r['reject'])
                     for r in load_csv('COLOR_COMMAND_GROUPS')]), '',
              '## M3, geometry, empty-language and O64 regression', '',
              'Original M3 semantic gates remain passing in both arms. X M3 PARAPHRASE/B1 falls from P/F 12/12 to 11/12 '
              'and remains above the original gate; this regression is retained. Empty-language rows are diagnostics, not a substitute semantic gate. '
              'O64 remains strongly failing: median XYZ about183–186mm and SO3 about1.09–1.10rad. No broad retention or generalist qualification is claimed.', '',
              table(('Point','Task','Kind','Noise','N','median XYZ mm','p95 XYZ mm','median SO3 rad','Reject','Numeric pass'),
                    [(r['point'],r['task'],r['kind'],r['noise'],r['n'],f"{float(r['median_xyz_m'])*1000:.3f}",
                      f"{float(r['p95_xyz_m'])*1000:.3f}",f"{float(r['median_so3_rad']):.6f}",r['reject'],r['numeric_pass'])
                     for r in load_csv('LEGACY_REGRESSION_GROUPS')]), '',
              '## Recoverable complete checkpoints', '',
              'All four archives are full16-file Trainer states, including optimizer/scheduler/RNG, each fixed-revision full-file readback PASS. '
              'Only model-only views were materialized on Corelab; these are not presented as resume states. No Trainer weights were relayed to PC.', '']
    rows=[]
    models=[]
    for arm in ('F','X'):
        for step in (1000,2000):
            r=json.loads((ROOT/f'{arm}{step}_FULL_READBACK.json').read_text())
            assert r['status']=='PASS_FULL_FILE_READBACK' and r['complete_trainer'] and len(r['files'])==16
            models.append(r)
            rows.append((arm,step,r['revision'],r['tree_sha256'],len(r['files'])))
    lines += [table(('Arm','Step','HF revision','Full Trainer tree SHA256','Files'),rows), '',
              'Repo: `leoyuer/pi0246-gate-pass-campaign`; prefixes `arms/PI0263/{F|X}/checkpoint-{1000|2000}`. '
              'Checkpoint1000 is a preserved recovery/milestone state, not a screened candidate.', '',
              '## Saved-array presentation and evidence', '',
              'The prospectively sealed18-chapter video contains720 decoded frames,20fps,36seconds,1280×720, '
              'using existing RGB/reference/prediction arrays only. It visibly states KINEMATIC RENDERED REFERENCE / OFFLINE PREDICTION / NO PHYSICS OR WBC. '
              'Representative frames from all18 chapters were visually inspected; this is not a claim to have manually inspected every frame. '
              'Video SHA256 `3eae05f4fb9461f0481abacad74ae7775e692101c80d3920b5bebe1217667dc9`.', '',
              'The public seal is recorded separately in archive/PUBLIC_RECEIPT.json after anonymous complete-file and every-tar-member readback. '
              'The computation closeout precedes that external receipt to avoid a circular archive hash. '
              'The final completion audit and GitHub canonical bind the receipt; this report does not infer upload success from a prepared archive.', '',
              'Machine-readable tables: QUALIFICATION.json, LEGACY_PANEL_ROWS.csv, CROSSED_PANEL_ROWS.csv, COLOR_COMMAND_GROUPS.csv, '
              'ORIGINAL_VS_OTHER_BINDINGS.csv, LEGACY_REGRESSION_GROUPS.csv, WORDING_FLIPS.csv, TARGET_CHOICE_DECOMPOSITION.csv and NOISE_BOUNDARY_CROSSINGS.csv. '
              'All3456 raw records/tensors, plans, source checks, canaries, training logs, actual RNG journals, graph/loss/gradient audits and HF receipts are retained.', '',
              '## Closed scientific scope', '',
              'Both authorized endpoints completed, valid negative qualification retained. No next model is admitted. '
              'Stop at STOP_FOR_INDEPENDENT_GENERALIZATION_OR_PHYSICAL_ADMISSION, selected/deployment NONE. '
              'Independent generalization or physical admission requires a new decision; no replay of old validation, new loss, extra seed, '
              'adapter, scene, IK, physics, WBC, TEST, DEV-F, hardware, broad cleanup, monitor, mail or merge is performed.', '']
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.suffix in ('.csv','.json')}
    lines += ['## Source table identities','',table(('Source','SHA256'),sources.items()),'']
    (OUT/'MODEL_QUALITY_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    closeout=dict(schema='PI0263_COMPUTE_AND_QUALITY_CLOSEOUT_V1',status='PI0263_COMPUTE_AND_QUALITY_COMPLETE_AWAIT_PUBLIC_READBACK',
        authority_commit='1c574b6e6e9d6a7851e64ca57e89c468a4c9eb28',main_updates={'F':2000,'X':2000},
        raw_completed_updates=4032,raw_training_heads=72624,technical_completed_update_reserve_used=0,
        unique_rows=3456,donor_reused=1152,new_attempts=2232,inference_heads=8928,exact_reuse=72,invalid_attempts=0,
        checkpoints=[dict(arm=r['arm'],step=r['step'],revision=r['revision'],tree_sha256=r['tree_sha256'],full_readback=r['status']) for r in models],
        qualification=q['points'],synthetic_research_candidate='NONE',selected_checkpoint='NONE',deployment_selected_checkpoint='NONE',
        cpu_status=cpu['status'],same_owner_independent_computation_path=True,external_signoff=False,
        public_seal_required='archive/PUBLIC_RECEIPT.json',stop='STOP_FOR_INDEPENDENT_GENERALIZATION_OR_PHYSICAL_ADMISSION',
        new_scenes=0,new_statistics_fits=0,physics_steps=0,WBC_steps=0,TEST=0,DEV_F=0,real_hardware=0,quality_retries=0)
    (OUT/'FINAL_CLOSEOUT.json').write_text(json.dumps(closeout,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(status='PASS_REPORT_FROM_COMPLETE_SAVED_CPU_TABLES',rows=3456,model_queries=0)))


if __name__=='__main__':
    main()
