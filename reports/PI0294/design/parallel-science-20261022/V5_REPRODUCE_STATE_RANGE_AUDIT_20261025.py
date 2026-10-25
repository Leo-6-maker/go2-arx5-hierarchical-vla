"""Reproduce the finite saved-state/consumption audit without images or models.

Requires numpy/scipy and CPU torch; supply existing private captured receipts.
No download, SSH, registry, ledger or model.forward operation is performed.
"""
import argparse,ast,base64,collections,hashlib,io,json,sys,time,types,enum,csv
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
parser=argparse.ArgumentParser()
for name in ['numeric-receipt','source-materials','consumption','official-readback','official-directory','output','components-output']:parser.add_argument('--'+name,required=True)
args=parser.parse_args();R=Path(__file__).resolve().parents[4];S=Path(args.official_directory)
sys.path[:0]=[str(R),str(R/'src')]
from scripts.pi0294.pv1.state_range_audit import *
from scripts.pi0291_arx5_data.native_dataset import matrix_from10,pose10
torch.set_num_threads(1)
source=lambda path:(S/path.replace('/','_')).read_text()
start=time.process_time();read=lambda p:json.loads(p.read_bytes())
source=lambda p:(S/p.replace('/','_')).read_text()
receipt=Path(args.numeric_receipt);rr=read(receipt);assert rr['returncode']==0;data=rr['value'];assert len(data['numeric'])==265
materials=read(Path(args.source_materials))['files'];statistics_raw=base64.b64decode(materials['statistics']['base64']);assert sha(statistics_raw)=='fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7';stats=json.loads(statistics_raw)['new_embodiment']
actual_source=read(Path(args.official_readback));assert actual_source['returncode']==0
for path,record in actual_source['value']['official'].items():
 actual=source(path).encode();assert sha(actual)==record['sha256'] and actual==base64.b64decode(record['base64'])
assert statistics_raw==base64.b64decode(actual_source['value']['statistics_base64'])
apply,ns,source_proof=official_state_path(source('gr00t/data/utils.py'),source('gr00t/data/state_action/state_action_processor.py'),source('gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py'),stats,torch)
visits=read(Path(args.consumption))['value']['by_candidate_lane_episode']
manifest={bank:json.loads(base64.b64decode(x['base64'])) for bank,x in data['manifests'].items()};rows={bank:{r['id']:r for r in v['episodes'] if r['split']=='TRAIN'} for bank,v in manifest.items()}
loaded={};summaries=[];components=[]
for row in data['numeric']:
 bank,id=row['bank'],row['id'];raw=base64.b64decode(row['base64']);m=rows[bank][id];assert len(raw)==m['bytes'] and sha(raw)==m['sha256']==row['sha256'];assert m['task'] in ('T1','T2')
 with np.load(io.BytesIO(raw),allow_pickle=False) as z:packet={k:z[k].copy() for k in ['actual_state10','committed_absolute10','valid_window','label_representation']}
 assert packet['label_representation'].item()=='ABSOLUTE_COMMITTED_EEF10' and packet['valid_window'].shape in [(601,),(640,)] and np.array_equal(np.flatnonzero(packet['valid_window']),np.arange(601))
 loaded[bank,id]=packet
 lane=('original_' if bank=='old' else 'new_')+m['task'];v={a:visits[a].get(lane,{}).get(id,{}).get('anchor_counts',{}) for a in visits}
 summary,detail=state_records(packet['actual_state10'],stats,apply,episode_id=id,bank=bank,task=m['task'],visits=v)
 summary.update(npz_sha256=m['sha256'],parent=m.get('scene_group',m.get('group')),manifest_sha256=data['manifests'][bank]['sha256'])
 if bank=='new':assert summary['state_OOR_components_legal']==m['codec']['state_preclip_components'],(id,summary['state_OOR_components_legal'],m['codec']['state_preclip_components'])
 summaries.append(summary);components+=detail
print('State scan complete',len(summaries),len(components),flush=True)
# Execute exact official pose/chunking definitions with their original numpy/scipy imports.
ns.update(Enum=enum.Enum)
types_tree=ast.parse(source('gr00t/data/types.py'))
compile_nodes([n for n in types_tree.body if isinstance(n,ast.ClassDef) and n.name in ['ActionFormat','ActionType','ActionRepresentation']],ns)
for name in ['gr00t/data/state_action/pose.py','gr00t/data/state_action/action_chunking.py']:
 tree=ast.parse(source(name));compile_nodes([n for n in tree.body if not(isinstance(n,ast.ImportFrom) and (n.module or '').startswith(('gr00t','__future__')))],ns)
cls=next(n for n in ast.parse(source('gr00t/data/state_action/state_action_processor.py')).body if isinstance(n,ast.ClassDef) and n.name=='StateActionProcessor')
compile_nodes([n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['_convert_to_relative_action','_convert_to_absolute_action']],ns)
action_details=[];action_checks=[]
params={k:np.concatenate([np.asarray(stats['relative_action']['eef_9d'][k]),np.asarray(stats['action']['gripper_position'][k])]) for k in ['min','max']}
for id,m in rows['new'].items():
 if m['codec']['native_inverse_clip_components_float32']==0:continue
 packet=loaded['new',id];states=packet['actual_state10'];commands=packet['committed_absolute10'];count=0;pre=0;largest=dict(XYZ_m=0.,SO3_rad=0.,gripper=0.)
 for anchor in range(601):
  target=commands[anchor:anchor+40];ref=states[anchor,:9]
  relative=ns['_convert_to_relative_action'](None,target[:,:9],ref,ns['ActionType'].EEF,ns['ActionFormat'].XYZ_ROT6D)
  joined=np.concatenate([relative,target[:,9:]],axis=-1);z=ns['normalize_values_minmax'](joined,params);packed=z.astype(np.float32)
  pre+=int(np.count_nonzero(np.abs(z)>1));locations=np.argwhere(np.abs(packed)>1);count+=len(locations)
  if len(locations)==0:continue
  decoded=ns['unnormalize_values_minmax'](packed,params)
  absolute=ns['_convert_to_absolute_action'](None,decoded[:,:9],ref,ns['ActionType'].EEF,ns['ActionFormat'].XYZ_ROT6D)
  actual=np.concatenate([absolute,decoded[:,9:]],axis=-1);left,right=matrix_from10(target),matrix_from10(actual)
  errors=dict(XYZ_m=float(np.max(np.abs(actual[:,:3]-target[:,:3]))),SO3_rad=float(np.max(np.arccos(np.clip((np.trace(left[:,:3,:3]@np.swapaxes(right[:,:3,:3],-1,-2),axis1=-2,axis2=-1)-1)/2,-1,1)))),gripper=float(np.max(np.abs(actual[:,9]-target[:,9]))))
  largest={k:max(v,largest[k]) for k,v in errors.items()}
  for off,ch in locations:
   action_details.append(dict(episode_id=id,task=m['task'],anchor=anchor,offset=int(off),channel=int(ch),channel_name=CHANNELS[ch],command_index=anchor+int(off),command_time_s=(anchor+int(off))/20,within_Prefix8=bool(off<8),relative_raw=float(joined[off,ch]),R0_min=float(params['min'][ch]),R0_max=float(params['max'][ch]),normalized_before_float32=float(z[off,ch]),normalized_float32=float(packed[off,ch]),inverse_clipped_normalized=float(np.clip(packed[off,ch],-1,1)),actual_decode_error=errors))
 assert count==m['codec']['native_inverse_clip_components_float32'] and pre==m['codec']['action_preclip_components'],(id,count,pre,m['codec']['native_inverse_clip_components_float32'])
 action_checks.append(dict(episode_id=id,all601_H40_scanned=True,inverse_clip_components=count,preclip_components=pre,maximum_affected_decode_error=largest))
 print('Action scan',id,count,flush=True)
assert len(action_details)==7
new=[x for x in summaries if x['bank']=='new'];old=[x for x in summaries if x['bank']=='old'];assert sum(x['state_OOR_components_legal'] for x in new)==543
bounds=[]
for i,ch in enumerate(CHANNELS):
 group='eef_9d' if i<9 else 'gripper_position';j=i if i<9 else 0;lo=stats['state'][group]['min'][j];hi=stats['state'][group]['max'][j]
 stats_by_bank={}
 for bank in ['old','new']:
  own=[x for x in components if x['bank']==bank and x['channel']==i and x['legal_TRAIN_anchor']]
  excess=[x['excess_normalized'] for x in own]
  stats_by_bank[bank]=dict(OOR_components=len(own),parents=len({x['episode_id'] for x in own}),maximum_normalized_excess=max(excess,default=0.),p95_normalized_excess=float(np.quantile(excess,.95)) if excess else None,p99_normalized_excess=float(np.quantile(excess,.99)) if excess else None,maximum_raw_component_excess=max((x['physical_component_excess'] for x in own),default=0.),raw_global_min=min(x['raw_min'][i] for x in summaries if x['bank']==bank),raw_global_max=max(x['raw_max'][i] for x in summaries if x['bank']==bank))
 bounds.append(dict(channel=i,name=ch,min=lo,max=hi,span=hi-lo,near_constant=bool(np.isclose(lo,hi)),by_bank=stats_by_bank))
exposures=[]
for bank in ['new','old']:
 for task in ['T1','T2']:
  own=[x for x in summaries if x['bank']==bank and x['task']==task]
  lane=('new_' if bank=='new' else 'original_')+task
  for arm in ['A_CONTROL','B_COVERAGE']:
   n=sum(x['actual_OOR_visits'][arm] for x in own);den=sum(sum(r['anchor_counts'].values()) for r in visits[arm].get(lane,{}).values())
   exposures.append(dict(bank=bank,task=task,arm=arm,unique_OOR_anchors=sum(x['unique_OOR_anchors_legal'] for x in own),parents_with_OOR=sum(x['unique_OOR_anchors_legal']>0 for x in own),actual_OOR_visits=n,lane_visits=den,task_visits=65536,whole_branch_visits=131072,proportion_of_lane=n/den if den else None,proportion_of_task=n/65536,proportion_of_arm=n/131072))
summary=dict(schema='V5_ACTUAL_STATE_RANGE_CONSUMPTION_AUDIT_V1',review_source='e92aeecfa6b50353c32ef6cb811f50d5f89753d3',status='REPRESENTATION_VALID_FOR_MIX_STUDY',
 scope=dict(old_episodes=126,new_episodes=139,all_saved_states=139*641+126*601,legal_anchors=265*601,new_H40_windows=139*601,official_historical_six_probe_checks=834,new_state_components_denominator=139*601*10),
 numerical_receipt_sha256=sha(receipt.read_bytes()),source_readback_sha256=sha(Path(args.official_readback).read_bytes()),R0_sha256=sha(statistics_raw),source_proof=source_proof,channels=bounds,episodes=summaries,state_components=components,exposures=exposures,action_occurrences=action_details,action_episode_checks=action_checks,
 new_bank_legal_state_OOR_components=543,new_bank_all641_state_OOR_components=sum(x['state_OOR_components_all_saved'] for x in new),new_bank_OOR_legal_anchors=sum(x['unique_OOR_anchors_legal'] for x in new),
 no_active_near_constant_state_channel=all(not x['near_constant'] for x in bounds),state_train_float32_preserved=True,state_inference_float64_preserved=True,
 actual_native_models_loaded=0,new_queries=0,new_physics=0,new_native_writes=0,
 limits=['State-only exact official AST slices; image/tokenizer/VLM and model.forward not executed.','Train float32 and inference float64 state-only output differ by declared dtype; no clipping or zeroing observed.','Source timestamp is the validated 20Hz export grid, not a newly re-read raw clock.','Codec action rescan restricted to the five originally nonzero episodes; original zero-count episodes not re-decoded.','No claim normalized distribution has no effect on learning.'],CPU_s=time.process_time()-start)
# A boundary projection is diagnostic only and is never applied to inputs.
lo=np.array([x['min'] for x in bounds]);hi=np.array([x['max'] for x in bounds]);maxangle=0.;orth=0.
for (bank,identity),packet in loaded.items():
 x=packet['actual_state10'][:601];actual=matrix_from10(x)[:,:3,:3]
 orth=max(orth,float(np.max(abs(actual@np.swapaxes(actual,-1,-2)-np.eye(3)))))
 if bank=='new':
  boundary=x.copy();boundary[:,3:9]=np.clip(x[:,3:9],lo[3:9],hi[3:9]);other=matrix_from10(boundary)[:,:3,:3]
  angle=np.arccos(np.clip((np.trace(actual@np.swapaxes(other,-1,-2),axis1=-2,axis2=-1)-1)/2,-1,1));maxangle=max(maxangle,float(angle.max()))
summary['rotation_diagnostic']=dict(actual_reconstructed_R_orthogonality_max_abs=orth,max_SO3_to_R0_component_boundary_projection_rad=maxangle,projection_is_diagnostic_not_applied=True,component_delta_not_interpreted_as_rotation_angle=True)
summary['limits'].append('C1 and online policy states are not included in the old ordinary-bank zero-OOR claim.')
summary['audit_implementation_sha256']=sha((R/'scripts/pi0294/pv1/state_range_audit.py').read_bytes())
summary['reproducer_sha256']=sha(Path(__file__).read_bytes())
# Keep component detail in a separate finite CSV rather than duplicating it.
component_rows=[]
for record in summary.pop('state_components'):
 row={k:v for k,v in record.items() if k!='actual_visits'}
 row.update(A_actual_visits=record['actual_visits']['A_CONTROL'],B_actual_visits=record['actual_visits']['B_COVERAGE'])
 component_rows.append(row)
csv_path=Path(args.components_output);csv_path.parent.mkdir(parents=True,exist_ok=True)
with csv_path.open('w') as handle:
 writer=csv.DictWriter(handle,fieldnames=list(component_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(component_rows)
summary['state_components_detail']=dict(file=csv_path.name,sha256=sha(csv_path.read_bytes()),rows=len(component_rows),legal_TRAIN_rows=sum(x['legal_TRAIN_anchor'] for x in component_rows))
# Any observed alteration above raises; ordinary OOR does not create a new threshold.
out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(status=summary['status'],old_OOR=sum(x['state_OOR_components_legal'] for x in old),new_OOR=543,new_unique_anchors=summary['new_bank_OOR_legal_anchors'],new_all641_OOR=summary['new_bank_all641_state_OOR_components'],channels=bounds,exposures=exposures,action=action_details,CPU_s=summary['CPU_s'],sha256=sha(out.read_bytes()))))
