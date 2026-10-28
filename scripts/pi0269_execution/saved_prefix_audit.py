"""Bounded saved-state labels; no learned forward, rendering, or training data."""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5')
ROOT=BASE/'PI0269'
OLD=BASE/'PI0268/R1'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0268/R1')
sys.path[:0]=[str(BASE/'PI0267/code'),str(BASE/'PI0266/code'),str(BASE/'PI0250/runtime/reference-317a/src')]
from pi0267_expert import ArclengthCorrection
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    target=ROOT/'evidence/SAVED_STATE_PREFIX_AUDIT.json'
    assert not target.exists()
    card=read(BASE/'PI0267/design/SCENE_CARD.json');reg=card['registry'];anchor=np.array(card['policy_anchor_world_m'])
    families=[f for f in card['families'] if f['split']=='NEW_SIM_EVAL'][:2]
    rows=[];calls=0
    for family in families:
      for point in ('OLD2048','ROT2048'):
       for task in ('M1','M3'):
        for command in (0,1):
         teacher=ArclengthCorrection(reg,family,task,command)
         for backend in ('E0','E1'):
          folder=RAW/'virtual'/point/f"{family['family_id']}-{task}-c{command}-{backend}"
          result=read(folder/'result.json');assert result['validity']=='VALID'
          assert result['arrays_sha256']==sha(folder/'trajectory.npz')
          for planned in (0.,10.,35.):
           row=dict(family=family['family_id'],point=point,task=task,command=command,backend=backend,scheduled_s=planned,
                    source_result_sha256=sha(folder/'result.json'),source_arrays_sha256=result['arrays_sha256'])
           queries=[q for q in result['queries'] if abs(q.get('timestamp',q['replan']*.4)-planned)<1e-8]
           if not queries:
            row['status']='MISSING_AFTER_STOP' if result['actual_elapsed_s']<planned else 'NO_QUERY_AT_SCHEDULED_TIME'
            rows.append(row);continue
           assert len(queries)==1
           q=queries[0];obs=Path(q['query']['input']);pred=Path(q['query']['output'])
           assert sha(obs)==q['observation_sha256'] and sha(pred)==q['prediction_sha256']
           with np.load(obs) as z:
            p=z['position_world'].copy();r=z['rotation_world'].copy();now=float(z['timestamp'])
            assert now==planned and np.array_equal(z['state_v1'][:3],(p-anchor).astype(np.float32))
           with np.load(pred) as z:decoded=z['decoded'][0].copy()
           lp=decoded[:,:3].astype(float)+anchor;lr=rot6d_rows_to_matrix(decoded[:,3:9].astype(float))
           tp,tr,meta=teacher.label(p,r);calls+=1;assert calls<=96
           assert np.linalg.norm(tp[0]-p)<1e-12
           td=tp[8]-p;pd=lp[8]-p;norm=float(np.linalg.norm(td))
           row.update(status='VALID_LABEL' if meta['valid_label'] else 'AMBIGUOUS_LABEL',teacher=meta,actual_timestamp_s=now,
              observation_sha256=sha(obs),prediction_sha256=sha(pred),first_position_bias_m=float(np.linalg.norm(lp[0]-p)),
              teacher_displacement_m=td.tolist(),predicted_displacement_m=pd.tolist(),
              prefix_xyz_rmse_m=float(np.sqrt(np.mean(np.sum((lp[:9]-tp[:9])**2,axis=1)))),
              whole_xyz_rmse_m=float(np.sqrt(np.mean(np.sum((lp-tp)**2,axis=1)))),
              endpoint_error_m=float(np.linalg.norm(lp[8]-tp[8])),hold_case=norm<=1e-12,
              signed_alignment=None if norm<=1e-12 or not meta['valid_label'] else float(np.dot(pd,td)/norm),
              directional_cosine=None if norm<=1e-12 or np.linalg.norm(pd)<=1e-12 or not meta['valid_label'] else float(np.dot(pd,td)/(norm*np.linalg.norm(pd))),
              teacher_terminal_contraction=None if not meta['terminal_hold'] else dict(
                 current_error_m=float(np.linalg.norm(p-teacher.p[-1])),teacher_prefix_error_m=float(np.linalg.norm(tp[8]-teacher.p[-1])),
                 predicted_prefix_error_m=float(np.linalg.norm(lp[8]-teacher.p[-1]))))
           rows.append(row)
    assert len(rows)==96
    value=dict(status='PASS_BOUNDED_SAME_STATE_LABEL_DIAGNOSIS_NO_NEW_CORRECTNESS_DEFECT',proposals=96,label_calls=calls,rows=rows,
               source_teacher_sha256=sha(BASE/'PI0267/code/pi0267_expert.py'),source_scene_sha256=sha(BASE/'PI0267/design/SCENE_CARD.json'),
               scheduled_time_rule='Exact timestamps only; 35s is not a .4s query tick and is never substituted.',
               model_queries=0,render_calls=0,physics_calls=0,training_examples_added=0,source_script_sha256=sha(__file__))
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k!='rows'}))

if __name__=='__main__':main()
