
import os,sys,json,time,resource,hashlib,traceback
from pathlib import Path
from contextlib import ExitStack
P=json.loads(sys.stdin.read());out=Path(P['out']);root=Path(P['root']);raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();sha=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:json.loads(Path(p).read_bytes());began=time.process_time()
def put(name,v):
 fd=os.open(out/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
 with os.fdopen(fd,'wb')as f:f.write(raw(v));f.flush();os.fsync(f.fileno())
try:
 assert os.getuid()==22182 and not os.environ.get('PBS_JOBID')and os.environ['CUDA_VISIBLE_DEVICES']=='';resource.setrlimit(resource.RLIMIT_CPU,(470,480));resource.setrlimit(resource.RLIMIT_CORE,(0,0));m=read(root/'RUN_MANIFEST.json');assert sha((root/'RUN_MANIFEST.json').read_bytes())==P['manifest_sha'];code=root/m['code']['capsule_relative'];sys.path[:0]=[str(code),str(code/'src')];assert m['code']['vla_commit']=='b0509a0bf296772e8e83856a2caf259b6ae0d908'
 binding=read(root/'OUTPUT_BINDING.private.json');assert read(binding['lease'])['status']=='CLOSED'and sha((root.parent.parent/'HOPPER_RESOURCE_LEDGER.json').read_bytes())==P['before'];assert sha((root/'control/resident-owner/TERMINAL_SETTLED.private.json').read_bytes())==P['settled_sha256']
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1);assert not torch.cuda.is_initialized()
 from scripts.pi0294.generalization import minimal_native_audit as audit,minimal_execution as e,minimal_inference_export as export
 from scripts.pi0294.generalization.minimal_train_session import dataset
 from scripts.pi0294.generalization.minimal_remaining_inputs import Inputs
 from scripts.pi0294.generalization.minimal_adapter import LEAVES
 from scripts.pi0294.dec1.readonly_checkpoint_hash import StreamHashes
 stream=StreamHashes();inputs=Inputs(root,m,guard=lambda:None);inputs.verify_current();data=dataset(root,out/'dataset-reader')
 request=read(Path(read(root/'control/resident-owner/SOURCE_REQUEST.private.json')['directory'])/'REQUEST.json');request=dict(request,job_id=P['job']);saved={};specs={s['id']:s for s in e.native_sessions()}
 for sid in('E2_SHARED_ACTION_HEAD_QUALIFY2','E1_ADAPTER_ONLY_FORMAL512'):
  cp=audit.checkpoint(root,specs[sid],request,data,stream)
  if sid=='E1_ADAPTER_ONLY_FORMAL512':assert cp['frozen_parameter_identity']==inputs.result['saved_E1_checkpoint']['frozen_parameter_identity']
  cp.pop('frozen_parameter_identity');saved[sid]=cp;put(sid+'_FIRST_STREAM.json',dict(status='FIRST_PASS_ONLY_FINAL_INDEPENDENT_REHASH_REQUIRED',checkpoint=cp,CPU_actual_s=time.process_time()-began))
 folder=root/'evidence/MINIMAL_THREE_REMAINING_SESSIONS';sf=folder/'E2_SHARED_ACTION_HEAD_QUALIFY2';rb=(sf/'SESSION_TERMINAL.json').read_bytes();receipt=json.loads(rb);assert stream(sf/'SESSION_TERMINAL.json')=='5bfab89a78799af9c82d8903bb56a0182a9ebb734b263b6b53968c61b15f5505';restore=receipt['actual_session_evidence']['restore'];assert restore['optimizer_scheduler_model_RNG_sampler_checked']is True and restore['step']==2 and restore['cursor']==64 and restore['new_optimizer_updates']==restore['new_queries']==0
 pair=export.compare_saved_draws(sf/'draws/step2-reference.npz',sf/'draws/step2-restored.npz');stream(sf/'draws/step2-reference.npz');stream(sf/'draws/step2-restored.npz')
 pp=root/'evidence/E1_ADAPTER_ONLY_FORMAL512-512/TRAIN_PROBES';probe=read(pp/'RESULT.json');declared=read(root/'plans/PV1_FIXED_TRAIN_PROBES.json');assert probe['complete']and probe['new_queries']==12 and len(probe['records'])==12 and probe['checkpoint_manifest_sha256']==saved['E1_ADAPTER_ONLY_FORMAL512']['checkpoint_manifest_sha256']and [x['id']for x in probe['records']]==[x['id']for x in declared['windows']];stream(pp/'RESULT.json')
 for row in probe['records']:assert row['split']=='TRAIN'and stream(pp/(row['id']+'.npz'))==row['arrays_sha256']
 inf=root/'inference/E1_ADAPTER_ONLY';complete=read(inf/'CHECKPOINT_COMPLETE.json');audit.own_members(inf,complete,stream);assert complete['wrapped_checkpoint_manifest_sha256']==saved['E1_ADAPTER_ONLY_FORMAL512']['checkpoint_manifest_sha256']and complete['global_step']==512 and complete['query_comparison_complete']is False
 # Compare serialized tensors directly, without constructing GR00T. All
 # non-category tensors must match bit-for-bit; each ordinary category table
 # must exactly substitute only the learned target10 slice.
 from safetensors import safe_open
 wrapped=root/'training/E1_ADAPTER_ONLY_FORMAL512/checkpoint-512'
 def mapping(folder,stack):
  readers={};mapping={}
  for p in sorted(folder.glob('*.safetensors')):
   reader=stack.enter_context(safe_open(str(p),framework='pt',device='cpu'));readers[p.name]=reader
   for key in reader.keys():assert key not in mapping;mapping[key]=reader
  return mapping
 with ExitStack()as stack:
  left=mapping(wrapped,stack);right=mapping(inf,stack);expected=set(left)
  for leaf in LEAVES:
   for name in('frozen_W','frozen_b','target_W','target_b'):expected.remove(leaf+'.'+name)
   expected.update((leaf+'.W',leaf+'.b'))
  assert set(right)==expected;checked=0
  for key,reader in right.items():
   actual=reader.get_tensor(key)
   leaf=next((name for name in LEAVES if key in(name+'.W',name+'.b')),None)
   if leaf:
    kind=key.rsplit('.',1)[1];frozen=left[leaf+'.frozen_'+kind].get_tensor(leaf+'.frozen_'+kind);target=left[leaf+'.target_'+kind].get_tensor(leaf+'.target_'+kind)
    assert actual.dtype==frozen.dtype==target.dtype and actual.shape==frozen.shape and torch.equal(actual[:10],frozen[:10])and torch.equal(actual[10],target)and torch.equal(actual[11:],frozen[11:])
   else:
    expected_tensor=left[key].get_tensor(key);assert actual.dtype==expected_tensor.dtype and actual.shape==expected_tensor.shape and torch.equal(actual,expected_tensor)
   checked+=1
 put('FINAL_REHASH_STARTED.json',dict(paths=len(stream.paths),unique_inodes=len(stream.inodes),first_pass_bytes_read=stream.bytes_read,ordinary_tensor_count=checked,ordinary_serialized_tensor_comparison_complete=True,CPU_actual_s=time.process_time()-began));witnesses=stream.witnesses();assert not torch.cuda.is_initialized()and witnesses['all_unique_inodes_independently_rehashed']
 put('RESULT.json',dict(status='PASS_ACTUAL_SAVED_E2_QUALIFICATION_E1_FULL512_AND_ORDINARY_SERIALIZED_TENSORS_INDEPENDENT_CPU_AUDIT',job_id=P['job'],native_exit_status=1,request_sha256=P['request_sha'],source_sha=m['code']['vla_commit'],original_financial_settlement_sha256=P['settled_sha256'],saved_states=saved,E2_full_restore=restore,E2_saved_output_pair=pair,E1_original_TRAIN_probes=12,E1_ordinary_manifest_sha256=sha((inf/'CHECKPOINT_COMPLETE.json').read_bytes()),ordinary_full_tensor_count=checked,ordinary_target10_table_substitution_and_all_other_tensors_exact=True,E1_actual_output_reconstruction_qualified=False,E1_bad_noise_language24_preserved_not_promoted=True,E2_formal_updates=0,behavior_success_not_measured=True,streamed_witnesses=witnesses,new_models_queries_updates_GPU_PBS_physics=0,CPU_actual_s=time.process_time()-began,observed_epoch=time.time()))
except BaseException as exc:
 put('FAILED_PRESERVED.json',dict(error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()[-5000:],CPU_actual_s=time.process_time()-began,no_model_query_or_optimization=True,no_automatic_retry=True));raise
