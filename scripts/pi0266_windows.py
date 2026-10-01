"""Read-only, fixed window admission and family-round-robin transfer selection."""
import hashlib
import itertools
import sys
from collections import Counter,defaultdict
import numpy as np
from scipy.spatial.transform import Rotation
from pi0266_common import ROOT,BASE,DATA,REF,read,write,sha
sys.path[:0]=[str(REF/'src'),str(ROOT/'code')]
from pi0258_geometry import full_fk,homogeneous
from go2arx5_vla.wbc.native.kinematics import NativeArmKinematics
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix


def round_robin(rows):
    groups=defaultdict(list)
    for row in sorted(rows,key=lambda w:(w['family'],w['timestamp'],w['id'])):groups[row['family']].append(row)
    return [item for group in itertools.zip_longest(*(groups[k] for k in sorted(groups))) for item in group if item is not None]


def audit_parents(parents):
    source=read(ROOT/'evidence/SOURCE_IDENTITY.json');urdf_path=next(x['path'] for x in source['files'] if x['name']=='native_urdf')
    import xml.etree.ElementTree as ET
    urdf=ET.parse(urdf_path).getroot();kin=NativeArmKinematics.from_urdf(urdf_path)
    anchor=np.asarray(read(ROOT/'design/SCENE_CARD.json')['policy_anchor_world_m']);records=[];input_groups=defaultdict(list)
    for parent in parents:
        out=__import__('pathlib').Path(parent['attempt']);result=parent['result'];windows=read(out/'windows.json') if (out/'windows.json').exists() else []
        native_rows=read(out/'rows.json');times=np.array([r['time'] for r in native_rows])
        for window in windows:
            record=dict(window,parent_attempt=str(out),parent_task_success=result['success'],parent_safety_failure=result['safety_failure'],parent_failure=result['failure'])
            if window['admitted']:
                assert sha(window['frame'])==window['frame_sha256'] and sha(window['proposal'])==window['proposal_sha256']
                f=dict(np.load(window['frame'],allow_pickle=False));p=dict(np.load(window['proposal'],allow_pickle=False));q=dict(np.load(__import__('pathlib').Path(window['proposal']).parent/'codec.npz',allow_pickle=False))
                assert float(f['time'])==float(p['time'])==window['timestamp']
                assert np.array_equal(f['state_v1'],p['state_v1'])
                assert all(f[k].shape==(96,96,3) and f[k].dtype==np.uint8 for k in ('body_rgb','wrist_rgb'))
                assert f['state_v1'].shape==(10,) and f['state_v1'].dtype==np.float32
                assert window['gate']['decision']!='REJECT' and window['checks']['complete_H40']
                row=native_rows[int(np.argmin(abs(times-window['timestamp'])))];assert abs(row['time']-window['timestamp'])<=1e-6
                assert not row.get('collision',{}).get('hits') and not row.get('hard_unsafe')
                root=np.asarray(row['root']);dof=np.asarray(row['DOF']).reshape(-1,2);joints=dict(zip(result['dof_names'],dof[:,0]))
                transform=homogeneous(Rotation.from_quat(root[3:7]).as_matrix(),root[:3]);fk=full_fk(urdf,joints,transform)['x5_link6']@homogeneous(np.eye(3),kin.tcp_offset_xyz)
                pe=float(np.linalg.norm(p['state_v1'][:3]+anchor-fk[:3,3]));re=float(Rotation.from_matrix(rot6d_rows_to_matrix(p['state_v1'][3:9])@fk[:3,:3].T).magnitude())
                assert pe<=1e-5 and re<=1e-4
                assert np.max(abs(p['raw_action'][0]-p['state_v1']))<=1e-5
                assert q['action'].shape==q['action_mask'].shape==(40,132) and q['state'].shape==(1,132)
                assert np.all(q['action_mask'][:,:10]==1) and np.all(q['action_mask'][:,10:]==0)
                assert np.isfinite(q['action']).all() and np.isfinite(q['state']).all() and np.isfinite(q['decoded_action']).all()
                codec_pe=float(np.linalg.norm(q['decoded_action'][:,:3]-p['raw_action'][:,:3],axis=-1).max())
                codec_re=float(Rotation.from_matrix(rot6d_rows_to_matrix(q['decoded_action'][:,3:9])@rot6d_rows_to_matrix(p['raw_action'][:,3:9]).transpose(0,2,1)).magnitude().max())
                assert codec_pe<=1e-5 and codec_re<=1e-4
                assert np.max(abs(q['decoded_action'][:,9]-p['raw_action'][:,9]))<=1e-5
                record.update(independent_full_URDF_FK_xyz_m=pe,independent_full_URDF_FK_so3_rad=re,
                    independent_codec_xyz_m=codec_pe,independent_codec_so3_rad=codec_re,
                    codec_sha256=sha(__import__('pathlib').Path(window['proposal']).parent/'codec.npz'),
                    bucket=4 if window['metadata']['terminal_hold'] else window['metadata']['quartile'])
                digest=hashlib.sha256()
                for key in ('body_rgb','wrist_rgb','state_v1'):digest.update(f[key].tobytes())
                digest.update((window['task']+':'+str(window['command'])).encode())
                key=digest.hexdigest();record['observable_content_sha256']=key;input_groups[key].append((record,p['raw_action']))
            records.append(record)
    # An exact visible/state/command alias with conflicting futures is excluded, never resolved by timestamp/family ID.
    for items in input_groups.values():
        if len(items)<2:continue
        first=items[0][1];conflict=any(np.linalg.norm(a[:,:3]-first[:,:3],axis=-1).max()>1e-5 or
            Rotation.from_matrix(rot6d_rows_to_matrix(a[:,3:9])@rot6d_rows_to_matrix(first[:,3:9]).transpose(0,2,1)).magnitude().max()>1e-4 for _,a in items[1:])
        if conflict:
            for record,_ in items:record.update(admitted=False,label_class='NO_LABEL_OBSERVABLE_ALIAS')
    return records


def main():
    collection=read(ROOT/'evidence/COLLECTION_VERDICT.json');assert len(collection['rows'])==96 and collection['all_registered_cases_valid']
    records=audit_parents(collection['rows'])
    accepted=[w for w in records if w['admitted']];counts=Counter((w['task'],w['command'],w['bucket']) for w in accepted)
    missing=[]
    if len({w['id'] for w in accepted})<512:missing.append(dict(rule='distinct_windows',actual=len({w['id'] for w in accepted}),required=512))
    if len({w['family'] for w in accepted})<8:missing.append(dict(rule='families',actual=len({w['family'] for w in accepted}),required=8))
    for task,command in itertools.product(('M1','M3'),(0,1)):
        total=sum(counts[task,command,k] for k in range(5))
        if total<64:missing.append(dict(rule='task_command_windows',task=task,command=command,actual=total,required=64))
        for k in range(4):
            if counts[task,command,k]<8:missing.append(dict(rule='active_quartile',task=task,command=command,quartile=k,actual=counts[task,command,k],required=8))
        if not counts[task,command,4]:missing.append(dict(rule='terminal_hold_bank_bucket',task=task,command=command,actual=0,required='nonempty bucket to fill8units'))
    transfer=[]
    if not missing:
        for task in ('M1','M3'):
            for bucket,n in ((0,30),(1,30),(2,30),(3,30),(4,8)):
                pools=[round_robin([w for w in accepted if (w['task'],w['command'],w['bucket'])==(task,c,bucket)]) for c in (0,1)]
                for i in range(n):transfer.append(dict(task=task,bucket=bucket,selected=[pools[c][i%len(pools[c])] for c in (0,1)],cycled=[i>=len(pools[c]) for c in (0,1)]))
        assert len(transfer)==256 and Counter(w['task'] for w in transfer)=={'M1':128,'M3':128}
    receipt=dict(schema='PI0266_ACTUAL_FEEDBACK_WINDOW_ADMISSION_V1',status='PASS_ACTUAL_WINDOW_ADMISSION_CONTINUE_MATCHED_TRAINING' if not missing else 'PI0266_FEEDBACK_WINDOW_SHORTFALL',
        overall='IN_PROGRESS' if not missing else 'PARTIAL',parent_episodes=96,proposed=len(records),accepted=len(accepted),
        unique_accepted_source_windows=len({w['id'] for w in accepted}),unique_observable_contents=len({w['observable_content_sha256'] for w in accepted}),
        eligible_families=sorted({w['family'] for w in accepted}),shortages=missing,counts=[dict(task=t,command=c,bucket=k,n=n) for (t,c,k),n in sorted(counts.items())],
        rejection_counts=dict(Counter(w['label_class'] for w in records if not w['admitted'])),records=records,transfer_bank=transfer,
        independent_CPU_path='same-owner full URDF tree traversal; distinct from the arm-chain producer implementation',model_queries=0,optimizer_updates=0,
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    write(ROOT/'evidence/WINDOW_ADMISSION.json',receipt)
    print(__import__('json').dumps({k:receipt[k] for k in ('status','accepted','unique_accepted_source_windows','eligible_families','shortages')}),flush=True)


def self_check(attempt):
    assert [w['id'] for w in round_robin([dict(family=f,timestamp=t,id=f+str(t)) for f,t in [('b',2),('a',1),('a',0)]])]==['a0','b2','a1']
    result=read(attempt/'result.json');assert result['validity']=='VALID' and result['case']['stage']=='DIAGNOSTIC'
    records=audit_parents([dict(attempt=str(attempt),result=result)])
    assert sum(w['admitted'] for w in records)==result['valid_windows']
    report=dict(status='PASS_ACTUAL_DIAGNOSTIC_WINDOW_CPU_BOUNDARY_CHECK',attempt=str(attempt),windows=len(records),
        maximum_fk_xyz_m=max(w['independent_full_URDF_FK_xyz_m'] for w in records if w['admitted']),
        maximum_fk_so3_rad=max(w['independent_full_URDF_FK_so3_rad'] for w in records if w['admitted']),
        maximum_codec_xyz_m=max(w['independent_codec_xyz_m'] for w in records if w['admitted']),
        maximum_codec_so3_rad=max(w['independent_codec_so3_rad'] for w in records if w['admitted']),
        GR00T_queries=0,optimizer_updates=0,training_admission=False,source_sha256=sha(__file__))
    write(ROOT/'evidence/WINDOW_CPU_BOUNDARY_CHECK.json',report);print(__import__('json').dumps(report),flush=True)


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--self-check':self_check(__import__('pathlib').Path(sys.argv[2]))
    else:main()
