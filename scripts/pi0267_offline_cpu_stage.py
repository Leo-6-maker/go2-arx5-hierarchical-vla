"""Close the fixed offline lane from saved outputs while virtual episodes continue."""
import sys
from pathlib import Path
import numpy as np
import torch

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0267';R1=ROOT/'R1'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1')
DATA=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260/R2')
sys.path[:0]=[str(ROOT/'code'),str(BASE/'PI0264/code'),str(BASE/'PI0263/code'),str(DATA/'R3'),str(DATA/'runtime')]
from pi0267_eval_report import read,sha,write,csv_write,independent_metrics
from pi0264_freeze_eval import source
import pi0260_r2_codec as codec


def main():
    torch.set_num_threads(1)
    prior_path=R1/'evidence/DONOR_N_OFFLINE_CPU_REPLAY.json';prior=read(prior_path)
    assert prior['status']=='PASS_FIRST_TWO_COMPLETE_OFFLINE_POINTS_INDEPENDENT_METRICS' and prior['rows']==1392
    cache={(r['point'],r['domain'],r['uid']):r for r in prior['checks']};assert len(cache)==1392
    processor=codec.processor(read(R1/'design/statistics.json'));checks=[];table=[];euler=[]
    for point in ('donor','N4096','R4096'):
        for domain,n in (('new',256),('legacy',440)):
            paths=sorted((RAW/'offline'/point/domain).glob('*/record.json'));assert len(paths)==n
            for path in paths:
                rec=read(path);row=rec['row'];assert rec['status']=='VALID'
                assert sha(rec['tensor'])==rec['tensor_sha256'];z=np.load(rec['tensor']);metrics=rec['metrics']
                key=point,domain,row['uid']
                if key in cache:
                    check=cache[key];assert check['record_sha256']==sha(path) and check['tensor_sha256']==rec['tensor_sha256']
                else:
                    expert=np.load(RAW/'offline/source'/row['uid']/'input.npz')['expert'] if domain=='new' else source(row)[1]
                    error=None
                    if metrics is not None:
                        computed=independent_metrics(z['decoded'][0].astype(float),expert.astype(float))
                        error=max(abs(computed[k]-metrics[k]) for k in computed);assert error<=1e-6
                    check=dict(point=point,domain=domain,uid=row['uid'],record_sha256=sha(path),tensor_sha256=rec['tensor_sha256'],metrics_max_abs=error)
                checks.append(check)
                if not rec['donor_cache_reused']:
                    assert np.array_equal(z['actual_initial_noise'],torch.from_numpy(z['noise']).to(torch.bfloat16).float().numpy())
                    tx,tv=z['trace_x'],z['trace_v'];assert tx.shape==tv.shape==(4,1,40,132)
                    for i in range(4):
                        expected=(torch.from_numpy(tx[i]).to(torch.bfloat16)+(.25*torch.from_numpy(tv[i]).to(torch.bfloat16))*torch.ones_like(torch.from_numpy(tx[i]).to(torch.bfloat16))).float().numpy()
                        assert np.array_equal(expected,tx[i+1] if i<3 else z['raw'],equal_nan=True),(point,row['uid'],i)
                    error=None
                    if np.isfinite(z['raw']).all() and np.isfinite(z['decoded']).all():
                        error=float(np.max(abs(codec.decode(processor,z['raw'][0],z['state']).astype(np.float32)-z['decoded'][0])))
                        assert error<=1e-7,(point,row['uid'],error)
                    euler.append(dict(point=point,domain=domain,uid=row['uid'],tensor_sha256=rec['tensor_sha256'],four_native_BF16_steps_exact=True,codec_max_abs=error))
                table.append(dict(point=point,domain=domain,uid=row['uid'],family=row.get('family'),task=row.get('task'),command=row.get('command'),
                    ell_fraction=row.get('ell_fraction'),variant=row.get('variant'),window=('recovery' if row['variant'] else 'nominal') if domain=='new' else 'legacy',
                    gate=rec['gate']['decision'],valid_numerical_failure=metrics is None,**(metrics or {})))
    assert len(checks)==2088 and len(euler)==1648
    summary=[]
    for point in ('donor','N4096','R4096'):
        for window in ('nominal','recovery','legacy'):
            own=[r for r in table if (r['point'],r['window'])==(point,window)];finite=[r for r in own if not r['valid_numerical_failure']]
            summary.append(dict(point=point,window=window,rows=len(own),finite=len(finite),REJECT=sum(r['gate']=='REJECT' for r in own),
                median_xyz_rmse_m=float(np.median([r['xyz_rmse_m'] for r in finite])) if finite else None,
                median_so3_rmse_rad=float(np.median([r['so3_rmse_rad'] for r in finite])) if finite else None))
    csv_write(R1/'evidence/OFFLINE2088_NUMERICAL_ROWS.csv',table)
    out=dict(status='PASS_FULL_OFFLINE2088_METRICS_AND_1648_NATIVE_S4_CODEC_REPLAY',rows=2088,reused_donor_rows=440,
        checks=checks,prediction_checks=euler,summary=summary,prior1392_receipt_sha256=sha(prior_path),source_sha256=sha(Path(__file__)),
        model_queries=0,gradient_queries=0,new_physics=0,closed_loop_qualification=None,full_primary_completion=False)
    write(R1/'evidence/COMPLETE_OFFLINE_CPU_REPLAY.json',out)
    print({k:v for k,v in out.items() if k not in ('checks','prediction_checks')},flush=True)


if __name__=='__main__':main()
