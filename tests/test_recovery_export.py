import hashlib,json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from scripts.pi0294.multitask.recovery_export import expert_suffix_packet,fit_statistics
from scripts.pi0294.native_dataset import NativeDataset
from scripts.pi0292.train_adapter import DOMAIN
from scripts.pi0294.test_pv1_export import FakeProcessor
from scripts.pi0294.pv1.export import codec_packet


def full_packet():
    state=np.tile([0.,0.,0.,1.,0.,0.,0.,1.,0.,.5],(641,1))
    commands=state[:-1].copy();commands[:,0]=np.arange(640)/10000
    commands[:32,0]=999.  # Executed student actions, forbidden as expert labels.
    return dict(actual_state10=state,committed_absolute10=commands,
        valid_window=np.arange(640)<601,rgb_paths=np.tile(['body.png','wrist.png'],(641,1)),
        rgb_sha256=np.full((641,2),'0'*64),anchor_time_s=np.arange(601)/20,
        offset_s=np.arange(40)/20,instruction=np.asarray('Put book away'),
        label_representation=np.asarray('ABSOLUTE_COMMITTED_EEF10'))


def test_first_student_actions_never_enter_suffix_or_statistics():
    full=full_packet();p=expert_suffix_packet(full)
    assert p['actual_state10'].shape==(609,10)and p['committed_absolute10'].shape==(608,10)
    np.testing.assert_array_equal(p['committed_absolute10'],full['committed_absolute10'][32:])
    np.testing.assert_array_equal(p['source_observation_index'],np.arange(32,641))
    assert int(p['valid_window'].sum())==569 and p['anchor_time_s'][0]==1.6
    originals=[]
    for _ in range(5):
        q=full_packet();q['committed_absolute10'][:32,0]=0;originals.append(q)
    stats=fit_statistics(originals+[p])['new_embodiment']
    assert stats['action']['eef_9d']['max'][0]==.0639
    assert not np.any(p['committed_absolute10']==999)
    with pytest.raises(ValueError):fit_statistics(originals)


def test_original_reader_accepts_exact_variable_suffix_and_last_full_window(tmp_path):
    p=expert_suffix_packet(full_packet())
    for n,color in [('body.png','red'),('wrist.png','blue')]:
        Image.new('RGB',(256,256),color).save(tmp_path/n)
    for i,n in enumerate(('body.png','wrist.png')):
        p['rgb_sha256'][:,i]=hashlib.sha256((tmp_path/n).read_bytes()).hexdigest()
    np.savez_compressed(tmp_path/'suffix.npz',**p)
    (tmp_path/'statistics.json').write_text('{}\n')
    m=dict(domain=DOMAIN,training_admitted=False,
        statistics_sha256=hashlib.sha256((tmp_path/'statistics.json').read_bytes()).hexdigest(),
        episodes=[dict(id='EXPERT_SUFFIX',split='TRAIN',windows=569,file='suffix.npz',
            sha256=hashlib.sha256((tmp_path/'suffix.npz').read_bytes()).hexdigest(),raw_root=str(tmp_path))])
    (tmp_path/'DATASET_MANIFEST.json').write_text(json.dumps(m))
    d=NativeDataset(tmp_path,'unused','unused');assert len(d)==569
    state,commands,images,text=d.raw(568)
    np.testing.assert_array_equal(commands,full_packet()['committed_absolute10'][600:640])
    assert len(images)==2 and text=='Put book away'
    with pytest.raises(PermissionError):NativeDataset(tmp_path,'unused','unused',for_optimization=True)


def test_existing_codec_checks_all_569_suffix_windows_and_rejects_padding_or_duplicates():
    arrays=expert_suffix_packet(full_packet());calls=[]
    enums=SimpleNamespace(ActionType=SimpleNamespace(EEF=1),ActionFormat=SimpleNamespace(XYZ_ROT6D=1))
    tag=SimpleNamespace(EmbodimentTag=SimpleNamespace(NEW_EMBODIMENT='new_embodiment'))
    with patch.dict('sys.modules',{'gr00t.data.types':enums,'gr00t.data.embodiment_tags':tag}):
        result=codec_packet(arrays,FakeProcessor(),lambda:calls.append(1),anchors=range(569))
        assert len(calls)==result['H40_windows_actually_checked']==569
        assert result['committed_label_elements_checked']==569*40*10
        with pytest.raises(ValueError):codec_packet(arrays,FakeProcessor(),anchors=[568,569])
        with pytest.raises(ValueError):codec_packet(arrays,FakeProcessor(),anchors=[0,0])
