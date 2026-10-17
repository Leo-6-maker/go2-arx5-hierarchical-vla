import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.pi0292.relay_protocol import sealed,unseal
from scripts.pi0294.native_dataset import sha
from .mp1.student_label_audit import join
from .test_mp1_student_probe_audit import fixture as student_fixture


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    actions=root/'actions';actions.mkdir()
    plan,originals=student_fixture(actions)
    old=root/'old';pool=[];history=[]
    for state in plan['states']:
        identifier=state['id'];seq=state['sequence'];tick=state['tick'];attempt='ORIGINAL-'+state['original_episode_uuid']
        request=unseal(originals[identifier]['request'])
        packet=sealed({k:request[k] for k in ['protocol','run_uuid','episode_uuid','sequence','tick','payload']}|dict(kind='observation'))
        put(old/'evidence/TRAIN_STATES'/identifier/'observation.json',packet)
        pool.append({k:state[k] for k in ['id','original_episode_uuid','sequence','tick','task','observation_sha256']}|
            dict(original_attempt_id=attempt,split='TRAIN',physically_safety_valid=True,snapshot_relative=identifier))
        if seq==0:continue
        history.append(dict(attempt_id=attempt,sequence=seq,tick=tick,full_state_identical=True))
        folder=old/'forks'/(identifier+'-SHORT');folder.mkdir(parents=True)
        target=np.tile(state['payload']['state_eef10'],(40,1));target[:,0]+=.002
        actual=np.tile(state['payload']['state_eef10'],(40,1));valid=np.arange(40)==0
        for camera in ['body_rgb','wrist_rgb']:
            (folder/camera).mkdir();Image.new('RGB',(256,256),(120,30,90)).save(folder/camera/'0000.png')
        paths=np.asarray([['body_rgb/0000.png','wrist_rgb/0000.png']]*40)
        hashes=np.asarray([[sha(folder/'body_rgb/0000.png'),sha(folder/'wrist_rgb/0000.png')]]*40)
        np.savez(folder/'expert_labels.npz',committed_absolute10=target,actual_state10=actual,
            valid_window=valid,rgb_paths=paths,rgb_sha256=hashes,label_representation=np.asarray('ABSOLUTE_COMMITTED_EEF10'))
        acks=[]
        for i in range(5):
            p=folder/f'prefix-{i:03d}.npz';np.savez(p,tick_before=tick+i*200,tick_after=tick+(i+1)*200,expert_commands=target[i*8:(i+1)*8])
            acks.append(dict(tick_before=tick+i*200,tick_after=tick+(i+1)*200,integrations=200,commands=8,physical_receipt_sha256=sha(p)))
        put(folder/'PREFIX_ACKS.json',acks)
        result=dict(source_student_episode=state['original_episode_uuid'],anchor_tick=tick,anchor_sequence=seq,
            status='SAFE_TWO_SECOND_CONTINUATION',live_native_failures=0,labels_ACK_certified=True,
            actual_expert_commands=40,actual_integrations=1000,engine_history_copied=True,
            maximum_live_shadow_difference=0.,labels_sha256=sha(folder/'expert_labels.npz'))
        put(folder/'RESULT.json',result)
        put(old/'evidence/fork-audits'/folder.name/'AUDIT.private.json',dict(status='PASS_INDEPENDENT_SAVED_FORK_AUDIT',
            restoration='EXACT_NONINITIAL_HISTORY',task_qualification_evidence_eligible=True,checks=dict(actual=True),result=result))
    put(old/'plans/TRAIN_STATE_POOL.json',dict(states=pool))
    put(old/'evidence/INITIAL_STATIC_HISTORY_AUDIT.json',dict(status='PASS_ALL_USED_NONINITIAL_CORRECTION_STATES_EXACT_STATIC_HISTORY',records=history))
    return old,actions,plan


class StudentLabelTest(unittest.TestCase):
    def test_exact_left_join_excludes_four_initial_states_and_groups_true_parents(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,actions,plan=fixture(Path(tmp));v=join(old,actions,plan,'b'*64)
            self.assertEqual((v['fixed_states'],v['qualified_error_states'],v['excluded_initial_states']),(12,8,4))
            self.assertEqual(v['independent_historical_student_parents'],4)
            for task in ['T1','T2']:
                s=v['task_summaries'][task]
                self.assertEqual((s['queries'],len(s['episodes'])),(4,2))
                self.assertAlmostEqual(s['episode_macro']['prefix8']['XYZ_euclidean_m']['mean'],.002)
            self.assertTrue(all(r['original_S1_for_this_S_BC_probe']=='NOT_RUN' for r in v['rows']))
            self.assertFalse(v['qualified_H40_is_complete_32s_recovery_success'])

    def test_unknown_history_or_duplicate_source_parent_tick_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,actions,plan=fixture(Path(tmp));p=old/'evidence/INITIAL_STATIC_HISTORY_AUDIT.json';v=json.loads(p.read_text())
            for change in ['not_identical','duplicate']:
                bad=copy.deepcopy(v)
                if change=='not_identical':bad['records'][0]['full_state_identical']=False
                else:bad['records'].append(bad['records'][0])
                put(p,bad)
                with self.subTest(change=change),self.assertRaisesRegex(ValueError,'CONTROL_HISTORY'):
                    join(old,actions,plan,'b'*64)

    def test_actual_ACK_hash_or_commands_changed_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,actions,plan=fixture(Path(tmp));state=next(s for s in plan['states'] if s['sequence']>0)
            p=old/'forks'/(state['id']+'-SHORT')/'prefix-000.npz';p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'ACK_HASH'):join(old,actions,plan,'b'*64)

    def test_source_camera_pixels_or_expert_state_must_match_actual_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,actions,plan=fixture(Path(tmp));state=next(s for s in plan['states'] if s['sequence']>0)
            folder=old/'forks'/(state['id']+'-SHORT');p=folder/'expert_labels.npz'
            with np.load(p,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
            arrays['actual_state10'][0,0]+=.1;np.savez(p,**arrays)
            result=json.loads((folder/'RESULT.json').read_text());result['labels_sha256']=sha(p);put(folder/'RESULT.json',result)
            audit=old/'evidence/fork-audits'/folder.name/'AUDIT.private.json';v=json.loads(audit.read_text());v['result']=result;put(audit,v)
            with self.assertRaisesRegex(ValueError,'ORACLE_STATE'):join(old,actions,plan,'b'*64)

    def test_no_dropped_rows_or_checkpoint_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,actions,plan=fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError,'WEIGHT_IDENTITY'):join(old,actions,plan,'c'*64)
            bad=copy.deepcopy(plan);bad['states'].pop()
            with self.assertRaisesRegex(ValueError,'NO_ROW_LOSS'):join(old,actions,bad,'b'*64)


if __name__=='__main__':unittest.main()
