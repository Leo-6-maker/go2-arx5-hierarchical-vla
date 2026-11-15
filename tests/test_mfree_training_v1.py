import json
from pathlib import Path
import numpy as np
import pytest
from go2arx5_vla.sla_adapter.mfree_training_v1 import MobileDataset,learning_rates
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.gripper_v2 import VERSION,EPSILON


def fixture(root):
    episodes=[];windows=0
    for e,n in enumerate((5,6,7,8)):
        eid=f'F1-T{e}-r1';folder=root/eid;folder.mkdir();windows+=n
        state=np.tile([.2,0,.7,1,0,0,0,1,0,.7],(n,1));steps=5703+np.arange(n)*250
        phase=np.array((['PREGRASP','DESCEND','CLOSE','LIFT','HOLD']+['HOLD']*n)[:n])
        labels=folder/'LABELS.npz';np.savez(labels,state7=np.zeros((n,7)),latent7=np.zeros((n,40,7)),
            state_eef10=state,nominal_policy_H40=np.repeat(state[:,None],40,axis=1),
            supervision_mask=np.ones((n,40),bool),consumed_mask=np.tile([True]*10+[False]*30,(n,1)),
            actual_consumed_ticks=np.full(n,250),native_step=steps,frame_index=np.arange(n),phase=phase)
        frames=folder/'frames.npz';rgb=np.full((n,2,256,256,3),e,dtype=np.uint8)
        np.savez(frames,rgb=rgb,native_step=steps,state_eef10=state)
        episodes.append(dict(id=eid,split='TRAIN',task='PICK_HOLD',windows=n,language='Pick up the cube.',
            labels=dict(file=str(labels.relative_to(root)),sha256=sha(labels)),
            frame_shards=[dict(file=str(frames.relative_to(root)),sha256=sha(frames),first_frame=0,frames=n)],
            expert_stage_audit_only=phase.tolist(),mount=dict(T_policy_from_mount=np.eye(4).tolist())))
    stats=dict(codec_version=VERSION,epsilon=EPSILON,split='TRAIN',state_gripper_transformed=False,
        fit_episodes=4,fit_unique_windows=windows,state=dict(q01=[0]*7,q99=[1]*7),actions=dict(q01=[0]*7,q99=[1]*7))
    (root/'STATISTICS.json').write_text(json.dumps(stats))
    m=dict(schema='MFREE_LIVE_FINAL_PLAN_TRAIN_DATASET_V1',state='committed',codec=VERSION,split='TRAIN',
        phase_input_to_student=False,teacher_memory_input=False,BASE_head=False,unconsumed_suffix_is_executed=False,
        plan_supervision_suffix=True,episodes=episodes,windows=windows,
        statistics=dict(file='STATISTICS.json',sha256=sha(root/'STATISTICS.json')))
    (root/'MANIFEST.private.json').write_text(json.dumps(m));return m


def test_variable_length_live_frames_keep_actual_query_clock(tmp_path):
    fixture(tmp_path);d=MobileDataset(tmp_path,sha(tmp_path/'MANIFEST.private.json'),resize_with_pad=None)
    assert len(d)==26
    state,_,rgb,text=d.raw(11)
    assert d.identity(11)==dict(episode_id='F1-T2-r1',anchor=0,native_step=5703)
    assert np.all(rgb[0]==2) and np.all(rgb[1]==2) and text=='Pick up the cube.'
    assert state.shape==(10,) and len(d.catalog_rows())==26


def test_future_suffix_cannot_be_relabelled_consumed_even_with_new_hash(tmp_path):
    m=fixture(tmp_path);ep=m['episodes'][0];p=tmp_path/ep['labels']['file']
    with np.load(p,allow_pickle=False)as z:data={k:z[k].copy()for k in z.files}
    data['consumed_mask'][0,10]=True;np.savez(p,**data);ep['labels']['sha256']=sha(p)
    (tmp_path/'MANIFEST.private.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='LIVE_H40_P10'):
        MobileDataset(tmp_path,sha(tmp_path/'MANIFEST.private.json'),resize_with_pad=None)


def test_fixed1000_lr_clock():
    rates=learning_rates()
    assert len(rates)==1000 and rates[49]==5e-5 and rates[-1]==1e-5
    assert all(b>a for a,b in zip(rates[:49],rates[1:50]))
    assert all(b<a for a,b in zip(rates[49:-1],rates[50:]))
