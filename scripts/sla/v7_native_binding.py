"""Bind the V7 pair to the already qualified ten-case Corelab panels."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v7_core_endpoint import ARMS, PLAN, STEPS
from v7_core_provider import provider_manifest
from v7_feedback_eval import ABI, FEEDBACK_ROSTER, check_pair


SLA = Path("scripts/sla")
ROLES = dict(zip(ARMS,("V7CTRL0500","V7TREAT0500"),strict=True))
PHASES = ("HOLD","PLACE","INDEPENDENT_HOLD")
BOUND = "f22ac6c351de62a7802459c8a46730b97ed90935dae8a38534c609b09ee5babc"
EVIDENCE = {
    "HOLD": ("b5413e7526e58d62e26b148add647aba6f4dff59ff18532d66307e53ebed092c",
               "0d624d9ccd70bb0d2654630aac403ec2767557477f8d4a11f4bb77f83c6eeb14"),
    "PLACE": ("903ea6468d37a3db5faf41e2fb7e91d30da83ea064d80ade34ebe8fe21b434ba",
              "0d624d9ccd70bb0d2654630aac403ec2767557477f8d4a11f4bb77f83c6eeb14"),
    "INDEPENDENT_HOLD": ("e80c4956a74c9cd18f9a1bd483318ad43639d388bc654832c0b88ad01f29d2d4",
                         "fe7c1282baf0ac3174a82a176d213e1dc19d7356da4d9c5dd4bc01beed51fb1f"),
}
NATIVE_SOURCE = "66722811e969949f493a6a87242647d49ced4a538dfeb70f77d6d52f1aeb0697"


def ref(path):
    return dict(path=str(path),sha256=sha(path))


def validate_native(reference,parent,registration,arm):
    audit = registration["source_audit"]
    original = pin(audit["path"],audit["sha256"])
    root = Path(reference["path"]).resolve(strict=True)
    _,pair = check_pair(original.parent,audit["sha256"])
    proof = pair["arms"][arm]
    require(registration["v62_plan"] == PLAN and registration["source_step"] == STEPS
            and registration["v4_model_role"] == ROLES[arm] and root == original.parent/arm
            and reference["sha256"] == proof["provider_sha256"],"The native role does not match the full V7-A pair")
    model,_ = provider_manifest(root,reference["sha256"])
    require(model["v7_arm"] == arm and model["frozen_wbc"] and model["original_l1"]
            and all(parent[k] == model[k] for k in ABI),"The Go2 or WBC contract changed")
    for old,new in zip(parent["arrays"],model["arrays"],strict=True):
        require(all(old[k] == new[k] for k in ("file","key","shape","dtype")),"Native provider changed an original IO")
        pin(root/new["file"],new["sha256"])
    return model


def executor(path, phase):
    text = pin(path,EVIDENCE[phase][1]).read_text()
    original = """            self.model = provider_from_manifest(path,manifest,receiver["model"],
                                                data,receiver["tokenizer"]["path"])
"""
    modified = """            if manifest.get("v7_plan"):
                from v7_core_provider import load_context_provider

                self.model, checked = load_context_provider(path,self.candidate["sha256"],receiver["model"],
                                                            data,receiver["tokenizer"]["path"])
                require(checked == manifest, "The qualified V7 load differs from the chosen endpoint")
            else:
                self.model = provider_from_manifest(path,manifest,receiver["model"],
                                                    data,receiver["tokenizer"]["path"])
"""
    require(text.count(original) == 1,"The original inference seam changed")
    text = text.replace(original,modified)
    if phase == "INDEPENDENT_HOLD":
        old = 'binding["v62_plan"] == V65 and binding["role"] == source_role(binding["source_step"], V65)'
        require(text.count(old) == 1,"The pinned independent HOLD seam changed")
        text = text.replace(old,'binding["role"] == source_role(binding["source_step"], binding["v62_plan"])')
        old = 'additional.update(v61_roster=binding["v61_roster"], v62_plan=V65,'
        require(text.count(old) == 1,"Cannot bind the selected independent HOLD protocol")
        text = text.replace(old,'additional.update(v61_roster=binding["v61_roster"], v62_plan=binding["v62_plan"],')
    compile(text,str(path),"exec")
    return text


def stage(site, providers, pair_sha, root, phase, arm):
    audit,pair = check_pair(providers,pair_sha)
    standard = site/"RECOVERY3000-OVERLAY-01"
    prior = site/"RECOVERY3000-INDEPENDENT-OVERLAY-01" if phase == "INDEPENDENT_HOLD" else standard
    source = prior/f"RECOVERY_{phase}_PREFLIGHT.json"
    original = read(pin(source,EVIDENCE[phase][0]))
    code = executor(prior/SLA/"m38_a_native_v1.py",phase)
    source_module = pin(prior/SLA/"v62_source_native.py",NATIVE_SOURCE).read_text()
    require(source_module.count('if __name__ == "__main__":') <= 1,"Unexpected source runner")
    # Restrict the interpretive shim to the new protocol and a single arm.
    source_module += f'''

_original_source_role = source_role
_original_checked_manifest = checked_manifest

def source_role(step, plan=V62):
    if plan != {PLAN!r}:
        return _original_source_role(step,plan)
    require(step == {STEPS}, "Expected the fixed V7 endpoint")
    return {ROLES[arm]!r}

def checked_manifest(provider, parent, registration):
    if registration["v62_plan"] != {PLAN!r}:
        return _original_checked_manifest(provider,parent,registration)
    from v7_native_binding import validate_native
    return validate_native(provider,parent,registration,{arm!r})
'''
    compile(source_module,"v62_source_native.py","exec")
    shutil.copytree(prior,root,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    for name,data in (("m38_a_native_v1.py",code),("v62_source_native.py",source_module)):
        target = root/SLA/name
        require(not target.is_symlink(),"Require an isolated V7 overlay")
        target.write_text(data)
    for name in ("v7_native_binding.py","v7_core_endpoint.py","v7_core_provider.py","v7_full_model_paths.py",
                 "v7_previous_command.py","v7_feedback_eval.py"):
        shutil.copyfile(Path(__file__).with_name(name),root/SLA/name)
    provider = providers/arm
    model,_ = provider_manifest(provider,pair["arms"][arm]["provider_sha256"])
    role = ROLES[arm]
    transforms = {}
    for key in ("hold","place"):
        transforms[key] = [dict(item,id=item["id"].replace("RECOVERY3000",role),
                       **({"pair_id":item["pair_id"].replace("RECOVERY3000",role)} if "pair_id" in item else {}))
                       for item in original[key]]
    panel = root/f"V7_{phase}_PREFLIGHT.json"
    write(panel,dict(original,**transforms,role=role,v4_model_role=role,v62_plan=PLAN,source_step=STEPS,
          candidate=dict(path=str(provider),sha256=pair["arms"][arm]["provider_sha256"]),
          checkpoint_sha256=model["checkpoint_sha256"],source_audit=ref(audit)))
    return panel


def check_preflight(site, panel, result, phase, arm):
    historical = read(pin(site/"v65-3k-native-code-02/BOUND-NATIVE-ENTRY.json",BOUND))["groups"][phase]
    prior = read(pin(historical["preflight"]["path"],historical["preflight"]["sha256"]))
    new = read(result/"COMPARISON.json")
    model = read(panel)
    require(new["source_step"] == STEPS and new["v62_plan"] == PLAN and new["v4_model_role"] == ROLES[arm]
            and new["source_audit"] == model["source_audit"] and new["v4_panel"]["sha256"] == sha(panel)
            and new["gpu_noise_sha256"] == prior["gpu_noise_sha256"]
            and new["qualified_transport_spec"] == prior["qualified_transport_spec"],
            "Existing physical transport and noise must match")
    for old,row,key in zip(prior["schedule"],new["schedule"],historical["cases"],strict=True):
        fields = ("layout_slot","noise_label","policy_seed","max_ticks","policy_queries","task","link")
        require(all(old.get(k) == row.get(k) for k in fields)
                and row["id"] == old["id"].replace("RECOVERY3000",ROLES[arm])
                and row["checkpoint"] == model["checkpoint_sha256"],"An original native condition changed")
        selected = read(result/row["id"]/"EXECUTION_SPEC.json")
        require(selected["frozen_files"][selected["model_path"]] == key["geometry"]
                and selected["comparison_policy_seed"] == row["policy_seed"]
                and not selected["WBC_changes"] and not selected["safety_changes"]
                and not selected["hardware_connections_allowed"],"Wrong original world or safety binding")
    return ref(result/"COMPARISON.json")


def dispatch(base, site, original, overlay, panel, output, phase, arm, *, execute=False, offline=None, preflight=None):
    require(not output.exists(),"Reconcile existing native owner before any dispatch")
    if execute:
        require(offline is not None and preflight is not None,"Require fixed and verified evaluation prerequisites")
        check_preflight(site,panel,preflight,phase,arm)
        model = read(panel)
        feedback = read(offline/f"{arm}-FEEDBACK"/"RESULT.json")
        source = read(offline/f"{arm}-SOURCE"/"RESULT.json")
        require(feedback["roster"]["sha256"] == FEEDBACK_ROSTER and feedback["model_forwards"] == 24
                and source["plan"] == PLAN and source["arm"] == arm and source["model_forwards"] == 42
                and feedback["checkpoint"] == source["checkpoint"] == model["checkpoint_sha256"]
                and feedback["physics_ticks"] == source["physics_ticks"] == 0,
                "Source retention and Go2 feedback must bind the qualified endpoint")
    old_panel = site/"SOURCE0250-PLACE-01"
    old = read(old_panel/"SOURCE0250_P01_701/EXECUTION_SPEC.json")
    old_registration = read(old_panel/"COMPARISON.json")
    worker = read(base/"V2-M-PLACE-01/worker/EXECUTION_SPEC.json")
    env = dict(os.environ,**worker["native_environment"])
    env.update(CUDA_VISIBLE_DEVICES="",JAX_PLATFORMS="cpu",PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join(map(str,(overlay/SLA,overlay,original/"source",original/"source/src",env["PYTHONPATH"]))))
    binding = read(panel)
    reports = overlay/"reports"
    action = [worker["python"],"-B",str(overlay/SLA/"m38_a_native_v1.py"),
           str(output),str(original),str(base/"V2-M-PLACE-01/worker/cache"),
           str(reports/"SLA_WBC_INTERFACE_COMPARISON_V1"),
           binding["noise"]["place" if phase == "PLACE" else "hold"]["path"],
           old["terminal_step_library"],old_registration["qualified_transport_spec"]["path"],
           "--phase",binding["source_phase"],"--study-evidence",str(reports/"SLA_SUCCESS_EXECUTION_AUDIT_V2"),
           "--v4-registry",old_registration["v4_registry"]["path"],"--v4-panel",str(panel),
           "--v2-registry",str(overlay/"docs/protocols/SLA_SUCCESS_EXECUTION_AUDIT_V2_REGISTRY_20261126.json")]
    if execute:
        action.append("--execute")
    subprocess.run(action,env=env,check=True)
    evidence = check_preflight(site,panel,output,phase,arm)
    write(output/"V7-INVOCATION.json",dict(plan=PLAN,arm=arm,phase=phase,executed=execute,comparison=evidence))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    build=commands.add_parser("stage")
    for key in ("site","providers","root"):
        build.add_argument(key,type=Path)
    build.add_argument("pair_sha")
    run=commands.add_parser("dispatch")
    for key in ("base","site","original","overlay","panel","output"):
        run.add_argument(key,type=Path)
    run.add_argument("--execute",action="store_true")
    for key in ("offline","preflight"):
        run.add_argument("--"+key,type=Path)
    for item in (build,run):
        item.add_argument("phase",choices=PHASES)
        item.add_argument("arm",choices=ARMS)
    args=vars(parser.parse_args())
    selected=args.pop("command")
    print((stage if selected=="stage" else dispatch)(**args))
