"""Independent CPU-only readback of all native paired outputs; no neural import."""
import collections,json
from pathlib import Path
import numpy as np
from bind_pristine import ROOT,digest

def main():
    card=json.loads((ROOT/'design/NATIVE_CARD.json').read_text());result=json.loads((ROOT/'native/RESULT.json').read_text())
    rows=[];files=[]
    for row in card['rows']:
        directory=ROOT/'native/outputs'/row['id']
        with np.load(row['input_path']) as inputs:truth=inputs['descriptive_future_action']
        values=[]
        for lane in ('stock','thin_wrapper'):
            path=directory/(lane+'.npz')
            with np.load(path) as data:values.append({k:data[k].copy() for k in data.files})
            files.append(dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path)))
        a,b=values
        keys=['raw','noise_uint16','decoded_eef_9d','decoded_gripper_position','decoded_joint_position']
        assert all(np.array_equal(a[k],b[k]) for k in keys),row['id']
        assert a['raw'].shape==(1,40,132) and a['noise_uint16'].shape==(1,40,132)
        assert all(np.isfinite(a[k]).all() for k in keys if k!='noise_uint16')
        xyz=np.linalg.norm(a['decoded_eef_9d'][0,:,:3]-truth[:,:3],axis=-1)
        gripper=np.abs(a['decoded_gripper_position'][0,:,0]-truth[:,9])
        rows.append(dict(id=row['id'],episode=row['episode'],xyz_mean_m=float(xyz.mean()),xyz_p95_m=float(np.quantile(xyz,.95)),gripper_native_unit_MAE=float(gripper.mean()),latencies_s=[float(a['latency_s']),float(b['latency_s'])]))
    counter=collections.Counter(json.loads(x)['kind'] for x in (ROOT/'evidence/model_budget.jsonl').read_text().splitlines())
    assert len(rows)==32 and counter=={'policy_attempt':64,'action_head':256} and result['effective_weights_unchanged']
    receipt=dict(status='PASS_INDEPENDENT_CPU_NATIVE_READBACK',paired_outputs=32,counts=dict(counter),output_files=files,descriptive_rows=rows,median_policy_latency_s=float(np.median([v for r in rows for v in r['latencies_s']])),new_model_queries=0,new_training_updates=0,task_quality_gate='NONE_OFFICIAL_OPEN_LOOP_DESCRIPTIVE_ONLY')
    (ROOT/'evidence/NATIVE_CPU_READBACK.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('output_files','descriptive_rows')}))

if __name__=='__main__':main()
