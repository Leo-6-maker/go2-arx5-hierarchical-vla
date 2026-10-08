"""Own Corelab evaluation seal; no healthy Hopper runtime is edited."""
import base64,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote

def stage():
    here=Path(__file__).resolve().parent
    names=('budget.py','authority.py','endpoint_worker.py','source_entry.py','evaluate_point.py','retention.py','retention_cpu_check.py','materialize_point.py','native_worker.py','prepare_core.py','diagnostic_runner.py','score.py','eval_cpu_check.py')
    files={n:base64.b64encode((here/n).read_bytes()).decode() for n in names}
    print(remote('core',f"""import base64,json,hashlib,shutil,subprocess
from pathlib import Path
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0284');old=r.parent/'PI0283'
assert not any((r/'source/episodes').glob('*/attempt-*/START.json')),'Do not restage active evaluation'
for n,b in {files!r}.items():(r/'code'/n).write_bytes(base64.b64decode(b))
for n in ('data','source/evidence','source/noise','source/runtime','models','transfers','audit'):(r/n).mkdir(parents=True,exist_ok=True)
for n in ('DATA_CONTRACT.json','EXPERT_CODEC_ADMISSION.json','processor_config.json','statistics.json','embodiment_id.json','DIAG48.jsonl'):shutil.copyfile(old/'data'/n,r/'data'/n)
for dest,source in [('data/episodes',old/'data/episodes'),('project-src',r.parent/'PI0276/project-src')]:
 p=r/dest
 if p.is_symlink() and p.resolve()!=source.resolve():
  assert not p.exists(),'Do not replace an active source reference'
  p.unlink()
 if not p.exists():p.symlink_to(source,target_is_directory=True)
 assert p.resolve()==source.resolve() and source.is_dir()
expert=old/'source/evidence/RESULT_EXPERT_TRAIN.json';shutil.copyfile(expert,r/'source/evidence/RESULT_EXPERT_TRAIN.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(r/n).write_text(json.dumps(v,indent=2)+'\\n')
card=json.loads((old/'design/SOURCE_CARD.json').read_text());pub=json.loads((r/'design/PUBLICATION_RECEIPT.json').read_text())
card.update(authority_release=pub['release'],canonical=pub['canonical'],points=['CTRL1024','CTRL2048','LR5X1024','LR5X2048'],development_points=['CTRL2048','LR5X2048'],panels=dict(TRAIN=list(range(10000,10004)),DEV=list(range(30000,30008)),CONFIRM=list(range(30100,30116))),technical_replacements={{}},code_sha256={{n:sha(r/'code'/n) for n in {names!r}}},policy_command=['/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python','-u',str(r/'code/endpoint_worker.py'),'--root',str(r)],native_pythonpath=':'.join([str(r/'code'),str(old/'code'),str(r.parent/'PI0280/official'),str(r.parent/'PI0276/ManiSkill'),str(r.parent/'PI0276/project-src'),str(r.parent/'PI0276/project-src/src'),str(r.parent/'PI0276/code')]),physical_caps=dict(starts=64,control=45000,physics=230000,rgb=45000,dispatches=90000,teacher=0),model_caps=dict(requests=6272,heads=100352),point_bindings={{}},conservative_episode_reserve_s=360)
card['predecessors'].update(native_source=dict(path=str(old/'code/source.py'),sha256=sha(old/'code/source.py')),native_source_guard_expert_result=dict(path=str(expert),sha256=sha(expert)))
write('design/SOURCE_CARD.json',card)
prior=json.loads((old/'design/IMPLEMENTATION.json').read_text())
dependencies={{str(p):sha(p) for p in (old/'code').glob('*.py')}}
implementation=dict(release=prior['release'],runtime_ABI_only=True,authority_release=pub['release'],canonical=pub['canonical'],published_commit='PENDING_EVALUATION_RUNTIME_PUSH',files={{n:sha(r/'code'/n) for n in {names!r}}},native_dependencies=dependencies,source_card_sha256=sha(r/'design/SOURCE_CARD.json'),codec_identity=prior['codec_identity'],expert_codec_replay_allowed=True,model_evaluation_allowed=True,expert_codec_result_sha256=sha(expert),training_allowed=False)
write('design/IMPLEMENTATION.json',implementation)
rc=json.loads((r/'design/RUNTIME_CARD.json').read_text());rc.update(dependencies={{**rc['dependencies'],**dependencies}},independent_CPU_audit=str(old/'runtime/closed_loop_audit.py'));write('design/RUNTIME_CARD.json',rc)
write('design/RUNTIME_AUTHORITY.json',dict(release=pub['release'],canonical=pub['canonical'],formal_cap=4096,all_updates_cap=4208,canary_cap=32,memory_cap=16,technical_cap=64,deadline_epoch=None,training_GPU_reservation_s=28800,training_allowed=False))
base=Path('/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10/assets/GR00T-N1.7-3B');native=json.loads((r.parent/'PI0280/design/NATIVE_CARD.json').read_text());assert len(native['rows'])==32
for row in native['rows']:
 assert sha(row['input_path'])==row['input_sha256']
 folder=r.parent/'PI0280/native/outputs'/row['id'];noise=folder/'actual_initial_noise.npy';stock=folder/'stock.npz'
 assert noise.exists() and stock.exists()
 row.update(noise_path=str(noise),noise_sha256=sha(noise),pristine_output=str(stock),pristine_output_sha256=sha(stock))
native.update(original_model_root=str(base),processor_identity={{n:sha(base/n) for n in ('processor_config.json','statistics.json','embodiment_id.json')}},selected='NONE',deployment='NONE',selection_metric=False);write('design/RETENTION32.json',native)
print(json.dumps(dict(status='CORE_EVALUATION_STAGED_NO_MODEL_PHYSICS_OR_TRAINING_CALL',code_sha256=implementation['files'],expert_replay_reused=True,DROID_tuples=len(native['rows']),TRAIN_panels=4,DEV_panels=2,CONFIRM_max=1,source_card_sha256=implementation['source_card_sha256'])))
"""))

if __name__=='__main__':stage()
