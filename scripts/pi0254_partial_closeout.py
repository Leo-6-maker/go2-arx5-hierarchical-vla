"""Read-only original U endpoint rules with explicit missing paired R endpoint."""
from pathlib import Path
import json
import pi0254_compare as original


def main():
    root=original.ROOT
    endpoint,esha=original.read('U576_4608');baseline,bsha=original.read('G2000');interim,isha=original.read('U576_1152')
    assert len(endpoint)==len(interim)==384 and len(baseline)==256
    counters=original.inherited.read(root/'bindings/actual_training_counters_closeout.json')
    raw=original.inherited.read(root/'bindings/raw_training_counters_v2.json')
    assert raw['raw_U576_head_forwards']==41477 and raw['raw_successful_optimizer_updates']==4608
    r=original.inherited.read(root/'bindings/R576_terminal_partial.json')
    assert r['R_actual_optimizer_updates']==0 and r['submissions_consumed']==2 and not r['new_submission_authorized']
    queries=[json.loads(s) for s in (root/'query_attempts.jsonl').read_text().splitlines()]
    heads=[json.loads(s) for s in (root/'head_forwards.jsonl').read_text().splitlines()]
    assert len(queries)==1088 and len(heads)==4352
    assert sum(x['point']=='U576_4608' for x in queries)==384 and sum(x['point']=='U576_4608' for x in heads)==1536
    assert all(x['status']=='VALID' for x in endpoint)
    groups={group:{noise:original.stratify([x for x in endpoint if x['cohort']==group and x['noise_name']==noise]) for noise in ('B0','B1')} for group in ('O64','E64','P64')}
    decision=original.qualify(endpoint,baseline)
    report=dict(schema='PI0254_U_VALID_R_BLOCKED_PARTIAL_CLOSEOUT_V1',status='PARTIAL_R_INPUT_PAIRING_BLOCKED',
        U576_endpoint_status='VALID_COMPLETE_4608_AND_ORIGINAL384_CELLS_CPU_REPLAY',U576_original_rule=decision,U576_endpoint_groups=groups,
        old16={noise:original.summary([x for x in endpoint if x['cohort']=='O64' and x['old16'] and x['noise_name']==noise]) for noise in ('B0','B1')},
        U576_1152_diagnostic=original.stratify(interim),R576_status=r['status'],R576_endpoint=None,equal_budget_pair_complete=False,
        paired_rehearsal_causal_conclusion='UNAVAILABLE_NO_VALID_R_TRAINING',research_candidate='NONE_PAIRED_STUDY_INCOMPLETE',
        baseline_CPU_sha256=bsha,U576_final_CPU_sha256=esha,U576_interim_CPU_sha256=isha,
        query_attempts=len(queries),inference_head_forwards=len(heads),invalid_query_attempts=0,training_counters=counters,
        raw_training_counters_including_engineering=raw,
        remaining_scientific_queries='R milestone allocations unspent; not transferable to new conditions/candidates',
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE',new_model_queries=0,new_optimizer=0,
        no_deployment_sim_or_generalization_claim=True)
    original.inherited.write(root/'partial_closeout_with_raw_counters.json',report)
    print(json.dumps(dict(status=report['status'],U_rule=decision,queries=len(queries),heads=len(heads))),flush=True)


if __name__=='__main__':main()
