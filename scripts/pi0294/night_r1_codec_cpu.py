"""Official codec fixtures; never fit dataset statistics or certify a bank."""
import numpy as np
from scripts.pi0292.train_adapter import official_import,TAG
from .night_r1_consumer import modality,states,actions


def run(source):
    official_import(source)
    from gr00t.data.state_action.state_action_processor import StateActionProcessor
    from gr00t.data.types import ActionType,ActionFormat
    from scripts.pi0292.eval_contract import matrix10
    from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows,geodesic_angle
    from .base_intent import rotation
    def stats(n,constant=False):
        return {k:([0.]*n if constant else ([-2.]*n if k in ('min','q01') else [2.]*n if k in ('max','q99') else [1.]*n if k=='std' else [0.]*n)) for k in ('min','max','mean','std','q01','q99')}
    results=[]
    for constant in (False,True):
        statistics={TAG:dict(state=dict(eef_9d=stats(9),gripper_position=stats(1),base_observed9=stats(9)),
            action=dict(eef_9d=stats(9),gripper_position=stats(1),base_soft_velocity3=stats(3,constant)),relative_action=dict(eef_9d=stats(9)))}
        processor=StateActionProcessor(modality(),statistics=statistics,use_percentiles=False,clip_outliers=False,use_relative_action=True);processor.eval()
        for yaw in (0.,.7):
            anchor=np.eye(4);anchor[:3,:3]=rotation(yaw);anchor[:3,3]=[.2,-.1,.5]
            eef=np.r_[anchor[:3,3],rotation_matrix_to_rot6d_rows(anchor[:3,:3])]
            state=np.r_[eef,.5,.01,-.02,.03,.04,-.05,.06,.3,.02,-.01]
            commands=np.tile(np.r_[eef,.5,0.,0.,0.],(40,1));commands[:,:3]+=np.arange(40)[:,None]*np.array([.0001,-.0002,.0003])
            commands[:,9]=np.linspace(.2,.8,40)
            if not constant:commands[:,10:13]=np.linspace([.01,-.01,.02],[.02,-.02,.03],40)
            sd,ad=states(state),actions(commands)
            encoded=processor.apply_action(ad,TAG,state=sd)
            decoded=processor.unapply_action(encoded,TAG,state=sd)
            recovered=np.concatenate([decoded[k] for k in ('eef_9d','gripper_position','base_soft_velocity3')],axis=-1)
            expected=np.linalg.inv(anchor)@matrix10(commands[:,:10])
            expected_rel=np.concatenate([expected[:,:3,3],rotation_matrix_to_rot6d_rows(expected[:,:3,:3])],axis=-1)
            official_rel=processor._convert_to_relative_action(commands[:,:9],state[:9],ActionType.EEF,ActionFormat.XYZ_ROT6D)
            se3_error=float(np.max(np.abs(official_rel-expected_rel)));xyz=float(np.max(np.linalg.norm(recovered[:,:3]-commands[:,:3],axis=-1)))
            angle=float(np.max(geodesic_angle(matrix10(recovered[:,:10])[:,:3,:3],matrix10(commands[:,:10])[:,:3,:3])))
            base_error=float(np.max(np.abs(recovered[:,10:13]-commands[:,10:13])));grip_error=float(np.max(np.abs(recovered[:,9]-commands[:,9])))
            normalized_state=processor.apply_state(sd,TAG)
            if not all(np.isfinite(x).all() for x in [*encoded.values(),*normalized_state.values(),recovered]):raise ValueError('FINITE_OFFICIAL_CODEC')
            assert se3_error<1e-8 and xyz<1e-5 and angle<1e-4 and base_error<1e-5 and grip_error<1e-5
            assert sum(x.shape[-1] for x in normalized_state.values())==19 and sum(x.shape[-1] for x in encoded.values())==13
            results.append(dict(constant_zero_BASE=constant,anchor_yaw=yaw,independent_SE3_error=se3_error,XYZ_m=xyz,SO3_rad=angle,BASE_error=base_error,gripper_error=grip_error))
    return dict(status='PASS_OFFICIAL_EEF13_STATE19_CODEC_FIXTURES_ONLY',official_source='51d4c89f72fda44cbf77285c6a8114b52676b8a1',
        cases=results,actual_bank=False,dataset_statistics_fit=0,synthetic_fixture_statistics_only=True,relative_EEF_count=1,BASE_relative_conversion=False,
        model_queries=0,physics_steps=0,actual_RGB_processor_collator='NOT_RUN_NO_ADMITTED_BANK')
