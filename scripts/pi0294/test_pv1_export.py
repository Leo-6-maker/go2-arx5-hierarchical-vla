"""Synthetic contract fixtures, never evidence of actual official codec passage."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.pi0291_arx5_data.native_dataset import matrix_from10,pose10
from scripts.pi0294.pv1.capture import capture_cards
from scripts.pi0294.pv1.plan import original_group
from scripts.pi0294.pv1.export import packet,codec_packet,physical_errors
from scripts.pi0294.pv1.scope import registry_role,EXPORT_ROLE


def fixture():
    card=capture_cards([original_group('TRAIN','F0',0,0)],'CLEARANCE_PAIR')[0]
    samples=[]
    for i in range(641):
        T=np.eye(4);T[:3,3]=[.45,i*1e-5,.5]
        obs=dict(instruction=card['instruction'],actual_tcp=T.tolist(),actual_gripper=1.,
            actual_gripper_speed=0.,timestamp_s=i/20,valid=True)
        for camera in ('body_rgb','wrist_rgb'):
            obs[camera]=dict(valid=True,frame_index=i,snapshot_id=str(i),timestamp_s=i/20,
                path=camera+'/'+str(i).zfill(4)+'.png',sha256='a'*64)
        command=T.copy();command[2,3]+=.002
        samples.append(dict(policy_obs=obs,snapshot_id=str(i),
            causal_history=dict(episode_id=card['id'],segment_id='SEG0',previous_samples=i,
                previous_committed=i,reset=False,abort=False,takeover=False),
            supervision=None if i==640 else dict(actually_committed=True,valid=True,sequence=i,
                command_time_s=i/20,committed=command.tolist(),committed_gripper=0.)))
    return card,samples


class FakeSAP:
    clip_outliers=False
    use_relative_action=True
    use_percentiles=False
    def _convert_to_relative_action(self,targets,state,*args):
        state=np.r_[state,0.];targets=np.c_[targets,np.zeros(40)]
        return pose10(np.linalg.inv(matrix_from10(state))@matrix_from10(targets),np.zeros(40))[:,:9]
    def apply_action(self,action,tag,state):
        return dict(eef_9d=self._convert_to_relative_action(action['eef_9d'],state['eef_9d'][0]),
            gripper_position=action['gripper_position'])
    def apply_state(self,state,tag):return state


class FakeProcessor:
    state_action_processor=FakeSAP()
    def decode_action(self,packed,tag,state):
        relative=packed[0]
        reference=np.r_[state['eef_9d'][0,0],0.]
        absolute=pose10(matrix_from10(reference)@matrix_from10(relative[:,:10]),relative[:,9])
        return dict(eef_9d=absolute[None,:,:9],gripper_position=absolute[None,:,9:])


class ExportContractTests(unittest.TestCase):
    def test_only_executed_640_commands_and_601_whole_horizons(self):
        card,samples=fixture();p=packet(samples,card)
        self.assertEqual(p['actual_state10'].shape,(641,10))
        self.assertEqual(p['committed_absolute10'].shape,(640,10))
        self.assertEqual(int(p['valid_window'].sum()),601)
        self.assertAlmostEqual(p['committed_absolute10'][0,2]-p['actual_state10'][0,2],.002)
        for mutate in ('uncommitted','clock','privilege','DEV'):
            c,s=copy.deepcopy(card),copy.deepcopy(samples)
            if mutate=='uncommitted':s[13]['supervision']['actually_committed']=False
            if mutate=='clock':s[13]['policy_obs']['timestamp_s']+=.05
            if mutate=='privilege':s[0]['policy_obs']['success']=True
            if mutate=='DEV':c['split']='DEV'
            with self.assertRaises((PermissionError,ValueError)):packet(s,c)

    def test_all_601_windows_not_six_anchor_smoke(self):
        card,samples=fixture();arrays=packet(samples,card);guards=[]
        enums=SimpleNamespace(ActionType=SimpleNamespace(EEF=1),ActionFormat=SimpleNamespace(XYZ_ROT6D=1))
        tag=SimpleNamespace(EmbodimentTag=SimpleNamespace(NEW_EMBODIMENT='new_embodiment'))
        with patch.dict('sys.modules',{'gr00t.data.types':enums,'gr00t.data.embodiment_tags':tag}):
            receipt=codec_packet(arrays,FakeProcessor(),lambda:guards.append(1))
        self.assertEqual(receipt['status'],'PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP')
        self.assertEqual(len(guards),601)
        self.assertEqual(receipt['committed_label_elements_checked'],240400)

    def test_actual_decoder_error_is_not_hidden_by_relabeling_or_refit(self):
        expected=np.tile(np.r_[[.45,0.,.5],[1.,0.,0.,0.,1.,0.],1.],(40,1))
        changed=expected.copy();changed[30,2]+=.002
        self.assertAlmostEqual(physical_errors(expected,changed)['XYZ_m'],.002)
        changed[0,0]=np.nan
        with self.assertRaises(ValueError):physical_errors(expected,changed)
        self.assertNotEqual(registry_role(EXPORT_ROLE,'export-01'),registry_role(EXPORT_ROLE,'export-02'))
