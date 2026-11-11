"""Private comparison integrity tests; not native or GPU qualification."""
import json

import numpy as np
import pytest

from go2arx5_vla.world_model import task_learning_backbone_compare_v1 as comp
from go2arx5_vla.world_model.storage import digest


def fixture(folder, changed=False):
    folder.mkdir();rows=[]
    for name,stage,value in [('constants','load',[1.,2.]),('backbone','input',[3.,4.]),
                             ('visual.patch_embed.proj','output',[5.1 if changed else 5.,6.])]:
        file=folder/(name+'-'+stage+'.npz');array=np.asarray(value,dtype=np.float32)
        np.savez(file,tensor0000=array)
        rows.append(dict(query_index=0,module=name,stage=stage,call=0,file=file.name,
                         sha256=digest(file),tensor_bytes=array.nbytes,active_allow_tf32=False,
                         active_bf16_reduced_precision_reduction=True,
                         structure=[dict(key='tensor0000',path=['value'],dtype='torch.float32',shape=[2])]))
    file=folder/'manifest.json';file.write_text(json.dumps(dict(mode=comp.MODE,diagnostic_only=True,
        complete=True,failure=None,original_query_count=40,traced_query_count=1,
        numeric_flags_changed=False,tensors_mutated=False,package_qualified=False,
        tensor_bytes=sum(r['tensor_bytes']for r in rows),records=rows,
        candidate_package_sha256='package',runtime=dict(HF_Qwen_source_sha256='implementation'))))
    return dict(path=str(file),sha256=digest(file))


def test_first_operator_difference_is_separate_from_identical_constants_and_inputs(tmp_path):
    a=fixture(tmp_path/'a');b=fixture(tmp_path/'b',changed=True)
    x=comp.compare(a,b)
    assert x['compared_boundaries']==3 and x['bit_exact_boundaries']==2
    assert x['first_non_bit_exact_boundary']['module']=='visual.patch_embed.proj'
    assert x['first_non_bit_exact_boundary']['tensors'][0]['changed_elements']==1
    assert not x['package_qualified'] and x['model_loads']==0


def test_same_trace_has_no_first_difference(tmp_path):
    a=fixture(tmp_path/'a');x=comp.compare(a,a)
    assert x['bit_exact_boundaries']==3 and x['first_non_bit_exact_boundary']is None


def test_changed_private_member_cannot_hide_behind_a_valid_manifest_pin(tmp_path):
    a=fixture(tmp_path/'a');b=fixture(tmp_path/'b')
    (tmp_path/'b/constants-load.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='pinned regular backbone member'):comp.compare(a,b)


def test_partial_original_population_is_not_complete_first_query_evidence(tmp_path):
    from pathlib import Path
    a=fixture(tmp_path/'a');b=fixture(tmp_path/'b');p=Path(b['path']);j=json.loads(p.read_text())
    j['original_query_count']=1;p.write_text(json.dumps(j));b['sha256']=digest(p)
    with pytest.raises(ValueError,match='complete first-query'):comp.compare(a,b)
