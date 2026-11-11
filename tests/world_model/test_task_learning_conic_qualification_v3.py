import json
import numpy as np
import pytest
from go2arx5_vla.world_model.task_learning_conic_qualification_v3 import compare,identity


def prepare(tmp_path):
    x=np.zeros((40,10));state=np.zeros(10);row=dict(method='A',index=0,split='TRAIN',phase='PREGRASP',noise_group=0,
        episode='synthetic',anchor=0,seed=0,source_prediction_sha256='a'*64,raw_identity=identity(x),decoded_identity=identity(x),
        valid=True,output_file='sample.npz')
    for name in('one','two'):
        d=tmp_path/name;d.mkdir();np.savez(d/'sample.npz',raw=x,decoded=x,state=state)
        result=dict(completed=540,valid=540,model_identities={'A':'synthetic'},source_sha='synthetic',solver_version='0.11.1',rows=[row],CPU_s=0.)
        (d/'RESULT.private.json').write_text(json.dumps(result))
    return x,state,row


def test_full_denominator_cannot_be_replaced_by_summary(tmp_path):
    prepare(tmp_path)
    # A dishonest summary count cannot substitute for540 actual saved rows.
    with pytest.raises(ValueError):compare(tmp_path/'one',tmp_path/'two')


def test_saved_arrays_are_checked_in_addition_to_matching_declared_hashes(tmp_path):
    x,state,row=prepare(tmp_path)
    for name in('one','two'):
        p=tmp_path/name/'RESULT.private.json';result=json.loads(p.read_bytes())
        result['rows']=[dict(row,method=m,index=i)for m in('A','B','C')for i in range(180)]
        p.write_text(json.dumps(result))
    assert compare(tmp_path/'one',tmp_path/'two')['matched']==540
    np.savez(tmp_path/'two/sample.npz',raw=x,decoded=x+.001,state=state)
    with pytest.raises(ValueError,match='saved Core arrays'):compare(tmp_path/'one',tmp_path/'two')
