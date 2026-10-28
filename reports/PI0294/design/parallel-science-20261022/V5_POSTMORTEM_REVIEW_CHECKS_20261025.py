"""CPU-only consistency checks; run from repository root or this directory."""
from pathlib import Path
import json, hashlib, collections, math
name = "V5_AB_POSTMORTEM_AND_GEN1_READINESS_REVIEWED_EVIDENCE_20261025.json"
candidates = [Path.cwd(), Path.cwd()/"reports/PI0294/design/parallel-science-20261022"]
folder = next(p for p in candidates if (p/name).is_file())
raw = (folder/name).read_bytes()
assert hashlib.sha256(raw).hexdigest() == "8c1325d978cb8947787c002b4639dc84d72310f5dff32be2da5f1e330c29a0f4"
d = json.loads(raw)
checks = []
def check(label, condition):
    assert condition, label
    checks.append(label)
rows = d["per_parent_exposure"]
check("265 unique bank/episode keys", len(rows)==265==len({(r["bank"],r["episode_id"]) for r in rows}))
check("finite geometry", all(math.isfinite(r[k]) for r in rows for k in ["distance_m","source_x_m","source_y_m","goal_x_m","goal_y_m","cube_yaw_rad"]))
check("actual PV1 TRAIN distance legal", all(r["distance_m"]>=.15 for r in rows))
for bank,task,n,a,b in [("old","T1",96,32768,16384),("old","T2",30,65536,32768),("new","T1",75,0,16384),("new","T2",64,0,32768)]:
    own=[r for r in rows if (r["bank"],r["task"])==(bank,task)]
    check(f"{bank}/{task} parent and visits", len(own)==n and sum(r["actual_visits"]["A_CONTROL"] for r in own)==a and sum(r["actual_visits"]["B_COVERAGE"] for r in own)==b)
    stored=d["data_distributions"][bank+"/"+task]
    check(f"{bank}/{task} family and direction counts", dict(collections.Counter(r["family"] for r in own))==stored["family_counts"] and dict(collections.Counter(r["direction"] for r in own))==stored["direction_counts"])
for task in ["T1","T2"]:
    own=[r for r in rows if r["bank"]=="old" and r["task"]==task]
    check(task+" all old parents visited", all(r["actual_visits"]["B_COVERAGE"]>0 for r in own))
    check(task+" B old anchors subset of A", all(r["actual_AB_shared_anchor_count"]==r["actual_unique_anchor_counts"]["B_COVERAGE"] for r in own) and sum(r["actual_unique_anchor_counts"]["B_COVERAGE"] for r in own)<sum(r["actual_unique_anchor_counts"]["A_CONTROL"] for r in own))
c=d["new_bank_codec_evidence"]
check("139 x 601 windows, not 139 model probes", c["all_H40_windows_checked"]==139*601 and c["actual_official_processor_anchor_checks"]==139*6)
nonzero=c["nonzero_records"]
check("codec nonzero counts retained", all(sum(r[k] for r in nonzero)==v for k,v in c["clip_counts"].items()))
check("codec affected episode counts retained", all(sum(r[k]>0 for r in nonzero)==v for k,v in c["nonzero_episode_counts"].items()))
check("physical decode gates", c["maximum_decode_error"]["XYZ_m"]<=1e-5 and c["maximum_decode_error"]["SO3_rad"]<=1e-4 and c["maximum_decode_error"]["gripper"]<=1e-5 and c["maximum_decode_error"]["relative_SE3_max_abs"]<=1e-8)
check("no zero-clip conflation", c["clip_counts"]["encode_clip_components"]==0 and c["clip_counts"]["native_inverse_clip_components_float32"]==7 and c["clip_counts"]["state_preclip_components"]==543)
atlas_raw=(folder/"V4A_A24_B24_COMPLETE_FAILURE_ATLAS_ACTUAL_20261025.json").read_bytes()
check("original published atlas hash", hashlib.sha256(atlas_raw).hexdigest()==d["input_sha256"]["V4A_A24_B24_COMPLETE_FAILURE_ATLAS_ACTUAL_20261025.json"])
atlas=json.loads(atlas_raw)
pairs=collections.defaultdict(dict)
for r in atlas["actual_condition_timelines"]: pairs[r["condition_id"]][r["candidate"]]=r
check("48 records / 24 complete pairs", len(pairs)==24 and all(set(v)=={"A_CONTROL","B_COVERAGE"} for v in pairs.values()))
counts=collections.Counter((p["A_CONTROL"]["status"]=="SUCCESS",p["B_COVERAGE"]["status"]=="SUCCESS") for p in pairs.values())
check("all pair outcome categories", counts=={(True,False):6,(False,True):1,(True,True):5,(False,False):12})
for key,predicate in [("discordant_conditions",lambda a,b:a!=b),("both_failed_conditions",lambda a,b:not a and not b)]:
    expected={cid for cid,p in pairs.items() if predicate(p["A_CONTROL"]["status"]=="SUCCESS",p["B_COVERAGE"]["status"]=="SUCCESS")}
    check(key+" complete exact IDs", expected=={p["condition_id"] for p in d[key]})
    check(key+" original score/ACK retained", all(p[arm]=={k:pairs[p["condition_id"]][candidate][k] for k in p[arm]} for p in d[key] for arm,candidate in [("A","A_CONTROL"),("B","B_COVERAGE")]))
g=d["GEN1_readiness"]
check("six old NEAR incompatible", len(g["V1_near_excluded_from_dynamic"])==6 and all(r["distance_m"]<.15 for r in g["V1_near_excluded_from_dynamic"]))
binding=g["revised_six_preview_binding"]
check("six preview geometries / three explicit replacements", len(binding)==6 and sum(r["geometry_changed"] for r in binding)==3 and all(r["distance_m"]>=.15 and r["split"]=="G_TRAIN" for r in binding))
check("twelve distinct visual design hashes", len({h for r in binding for h in r["design_hashes"]})==12)
check("24 previews NOT_RUN", g["expected_preview_images"]==24 and g["actual_images"]==g["new_constructors"]==g["new_integrations"]==0 and g["COMMON_HOME"]=="UNQUALIFIED")
check("no new native compute", all(d[k]==0 for k in ["new_model_queries","new_physics","new_training","new_native_writes"]))
for new_rate in [.5,.25]:
    c1=8192; ordinary_t1=8192; ordinary_t2=16384
    counts=[c1,int(ordinary_t1*(1-new_rate)),int(ordinary_t1*new_rate),int(ordinary_t2*(1-new_rate)),int(ordinary_t2*new_rate)]
    check(f"proposed {new_rate} matched cardinality", sum(counts)==1024*32 and sum(counts[:3])==sum(counts[3:])==16384)
print(json.dumps({"checks_passed":len(checks),"evidence_sha256":hashlib.sha256(raw).hexdigest(),"new_model_or_native_work":0,"limits":"Derived-data consistency and published atlas only; no new raw numeric qualification or model fit test.","checks":checks},ensure_ascii=False,indent=2))
