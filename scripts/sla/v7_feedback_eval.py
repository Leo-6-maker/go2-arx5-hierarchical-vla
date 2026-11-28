"""Evaluate each V7-A arm on the frozen 24-window Corelab feedback roster."""

import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import bind_imports, pin, read, require, sha
from scripts.pi0292.durable import write
from v7_core_endpoint import ARMS, PLAN, STEPS
from v7_core_provider import load_context_provider, provider_manifest


FEEDBACK_ROSTER = "a6e24bfd71bcd7034f18a9a5007cc0061f74190950ae1f4d080a4f466ddd698a"
ABI = ("foundation_sha256", "frozen_model", "action_sha256", "state_sha256", "tokenizer_sha256", "partition_sha256")


def check_pair(root, expected):
    audit = root/"CORE-PAIR-ACTUAL.json"
    pair = read(pin(audit,expected))
    require(pair["plan"] == PLAN and pair["step"] == STEPS and pair["formal_updates"] == 2*STEPS
            and pair["matched_cuda_steps"] == STEPS and pair["native_integrations"] == 0
            and set(pair["arms"]) == set(ARMS),"Require the independent full V7-A pair readback")
    for arm in ARMS:
        item = pair["arms"][arm]
        manifest,_ = provider_manifest(root/arm,item["provider_sha256"])
        require(item["complete_training_endpoint"] and item["rng_restored"]
                and set(item["adam_counters"].values()) == {STEPS}
                and manifest["v7_arm"] == arm and manifest["training_receipt_sha256"] == item["endpoint_sha256"]
                and manifest["training_contract_sha256"] == item["contract_sha256"],
                "Wrong full trainer state or native export")
    return audit,pair


@contextmanager
def original_feedback_entry(worker, package, expected_sha, arm):
    import v4_long_watch as feedback

    spec = read(worker)
    bind_imports(spec)
    from scripts.sla import hopper_reference

    original_candidate = feedback.candidate
    original_loader = getattr(hopper_reference, "provider_from_manifest", None)

    def validate(folder,parent,step,_audit):
        require(folder == package and step == STEPS,"Wrong V7 feedback role")
        manifest,_ = provider_manifest(folder,expected_sha)
        require(manifest["v7_arm"] == arm and all(manifest[key] == parent[key] for key in ABI)
                and manifest["frozen_wbc"] and manifest["original_l1"],"Changed original Go2 bridge")
        for old,new in zip(parent["arrays"],manifest["arrays"],strict=True):
            require(all(old[k] == new[k] for k in ("file","key","shape","dtype")),"Original Go2 array ABI changed")
            pin(folder/new["file"],new["sha256"])
        return manifest

    def load(folder,manifest,source,target,tokenizer):
        require(folder == package and manifest["v7_arm"] == arm,"Wrong sealed arm")
        provider,readback = load_context_provider(folder,expected_sha,source,target,tokenizer)
        require(manifest == readback,"Provider loader received a different manifest")
        return provider

    feedback.candidate = validate
    hopper_reference.provider_from_manifest = load
    try:
        yield feedback.watch
    finally:
        feedback.candidate = original_candidate
        if original_loader is None:
            del hopper_reference.provider_from_manifest
        else:
            hopper_reference.provider_from_manifest = original_loader


def run(site, worker, roster, providers, pair_sha, out, arm, *, execute=False):
    audit,pair = check_pair(providers,pair_sha)
    require(arm in ARMS,"Unknown evaluation arm")
    fixed = read(roster)
    require(len(fixed["rows"]) == 24 and fixed["original_worker"]["sha256"] == sha(worker),"Use the original feedback rows")
    with original_feedback_entry(worker,providers/arm,pair["arms"][arm]["provider_sha256"],arm) as watch:
        watch(site,worker,roster,providers/arm,STEPS,audit,out,execute=execute,retained_roster_sha=FEEDBACK_ROSTER)


def compare(control, treatment, output):
    def rows(file):
        report = read(file)
        require(report["step"] == STEPS and report["model_forwards"] == 24
                and report["training_updates"] == report["physics_ticks"] == 0 and not report["closed_loop"]
                and report["roster"]["sha256"] == FEEDBACK_ROSTER,"Wrong fixed V7 feedback scope")
        return report,report["rows"]

    c,first = rows(control)
    t,second = rows(treatment)
    require(c["checkpoint"] != t["checkpoint"],"The arms must use different endpoint identities")
    pairs = []
    for left,right in zip(first,second,strict=True):
        match = ("id","input","request","noise_id","noise_sha256","actual_sampler_sha256")
        require(all(left[k] == right[k] for k in match),"Mismatch in evaluation observation or actual sampler draws")
        row = {"id":left["id"]}
        for role,sample in (("control",left),("treatment",right)):
            metrics = sample["metrics"]
            row[role] = dict(seam_mm_s=1000*metrics["jaw"]["first_seam_speed_m_s"],
                             full_g7_violation=metrics["jaw"]["original_execution_speed_exceeded"],
                             original_H40_pass=metrics["original_envelope"]["original_full_H40_pass"])
        pairs.append(row)
    require(len(pairs) == 24,"Missing matched V7 feedback rows")
    summary = {}
    for role in ("control","treatment"):
        values = [row[role] for row in pairs]
        summary[role] = dict(seam_p95_mm_s=float(np.percentile([r["seam_mm_s"] for r in values],95)),
                             g7_count=sum(r["full_g7_violation"] for r in values),
                             h40_pass=sum(r["original_H40_pass"] for r in values))
    write(output,dict(rows=pairs,matched_windows=len(pairs),summary=summary,
                      model_forwards=48,training_updates=0,native_integrations=0,
                      authority="FIXED_OBSERVATION_MECHANISM_ONLY"))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    evaluate=commands.add_parser("run")
    for name in ("site","worker","roster","providers","out"):
        evaluate.add_argument(name,type=Path)
    evaluate.add_argument("pair_sha")
    evaluate.add_argument("arm",choices=ARMS)
    evaluate.add_argument("--execute",action="store_true")
    aggregate=commands.add_parser("compare")
    for name in ("control","treatment","output"):
        aggregate.add_argument(name,type=Path)
    args=vars(parser.parse_args())
    {"run":run,"compare":compare}[args.pop("command")](**args)
