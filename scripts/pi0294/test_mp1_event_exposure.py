import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from go2arx5_vla.task_data.codec import pose10
from scripts.pi0294.native_dataset import sha
from .mp1.event_exposure import episode_events, window_events, summarize_visits, probe_strata
from .test_mp1_probe_compare import fixture as probe_fixture


def fixture(root):
    root=Path(root);raw=root/'raw';raw.mkdir();numeric=root/'numeric.npz'
    n=64;states=np.tile(pose10(np.eye(4),1.),(n+1,1));commands=states[:n].copy()
    commands[7:40,9]=0;stages=['approach']*30+['detach']*10+['withdraw']*8+['hold']*17
    samples=[]
    for i in range(n+1):
        command=None if i==n else dict(sequence=i,valid=True,actually_committed=True,
            command_time_s=i/20,committed=np.eye(4).tolist(),installed=np.eye(4).tolist(),
            committed_gripper=float(commands[i,9]),desired_gripper=float(commands[i,9]))
        samples.append(dict(policy_obs=dict(timestamp_s=i/20,actual_tcp=np.eye(4).tolist(),actual_gripper=1.),
            supervision=command,privileged=dict(teacher_stage=stages[i])))
    events=[dict(stage='detach',previous='approach',time_s=30/20),
            dict(stage='withdraw',previous='detach',time_s=40/20)]
    (raw/'samples.json').write_text(json.dumps(samples));(raw/'TEACHER_EVENTS.json').write_text(json.dumps(events))
    (raw/'FILE_MANIFEST.json').write_text(json.dumps([dict(path=n,sha256=sha(raw/n)) for n in ['samples.json','TEACHER_EVENTS.json']]))
    np.savez(numeric,committed_absolute10=commands,actual_state10=states[:n-39],
        valid_window=np.ones(n-39,dtype=bool),label_representation='ABSOLUTE_COMMITTED_EEF10')
    row=dict(id='A',task='T2',origin='new',split='TRAIN',file=numeric.name,sha256=sha(numeric),
        raw_FILE_MANIFEST_sha256=sha(raw/'FILE_MANIFEST.json'),samples_sha256=sha(raw/'samples.json'),windows=n-39)
    return row,raw


class EventExposureTest(unittest.TestCase):
    def test_prefix_and_late_boundaries(self):
        events=dict(close=[7,8,39,40],open=[],detach=[],withdraw=[])
        v=window_events(events,0)['close']
        self.assertEqual(v,dict(prefix8=[7],late_H40=[8,39]))
        self.assertEqual(window_events(events,1)['close'],dict(prefix8=[6,7],late_H40=[38,39]))

    def test_actual_commands_and_stage_events_not_coarse_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            row,raw=fixture(tmp);v=episode_events(tmp,raw,row)
            self.assertEqual(v['events'],dict(close=[7],open=[40],detach=[30],withdraw=[40]))
            self.assertEqual(v['raw_stages'][30],'detach')

    def test_changed_numeric_installed_command_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            row,raw=fixture(tmp);p=Path(tmp)/row['file']
            with np.load(p,allow_pickle=False) as z:packet={k:z[k].copy() for k in z.files}
            packet['committed_absolute10'][0,0]=.01;np.savez(p,**packet);row['sha256']=sha(p)
            with self.assertRaisesRegex(ValueError,'EQUAL_ACTUAL_ORIGINAL'):
                episode_events(tmp,raw,row)

    def test_tampered_teacher_event_bytes_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            row,raw=fixture(tmp);(raw/'TEACHER_EVENTS.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'TEACHER_EVENT_BYTES_CHANGED'):
                episode_events(tmp,raw,row)

    def test_numeric_state_table_is_window_anchors_not_padded_future_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            row,raw=fixture(tmp);p=Path(tmp)/row['file']
            with np.load(p,allow_pickle=False) as z:packet={k:z[k].copy() for k in z.files}
            packet['actual_state10']=np.tile(packet['actual_state10'][0],(65,1))
            np.savez(p,**packet);row['sha256']=sha(p)
            with self.assertRaisesRegex(ValueError,'COMPLETE_ACTUAL_COMMAND_STATE'):
                episode_events(tmp,raw,row)

    def test_wrong_raw_event_transition_is_not_imputed_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            row,raw=fixture(tmp);p=raw/'TEACHER_EVENTS.json';events=json.loads(p.read_text());events[0]['time_s']=1.;p.write_text(json.dumps(events))
            manifest=raw/'FILE_MANIFEST.json';members=json.loads(manifest.read_text());members[1]['sha256']=sha(p);manifest.write_text(json.dumps(members));row['raw_FILE_MANIFEST_sha256']=sha(manifest)
            with self.assertRaisesRegex(ValueError,'RECORDED_STATE_STAGE'):
                episode_events(tmp,raw,row)

    def test_dev_rejected_before_reading_any_files(self):
        with self.assertRaisesRegex(PermissionError,'TRAIN_ONLY'):
            episode_events('/missing','/missing',dict(split='DEV'))

    def test_repeated_visits_and_event_instances_have_distinct_denominators(self):
        inv=dict(task='T1',origin='new',windows=25,raw_stages=['hold']*65,
            events=dict(close=[7],open=[40],detach=[],withdraw=[]))
        v=summarize_visits({'A':inv},{'A':{0:3,1:2}})
        a=v['episodes']['A'];self.assertEqual(a['sample_visits'],5)
        self.assertEqual(a['events']['close']['event_occurrences'],1)
        self.assertEqual(a['events']['close']['prefix8_visit_hits'],5)
        self.assertEqual(a['events']['open']['late_H40_visit_hits'],2)
        self.assertEqual(a['events']['open']['uncovered_in_sampled_prefix8'],1)
        self.assertEqual(v['task_origin_groups']['T1:new']['episodes'],1)
        with self.assertRaisesRegex(ValueError,'JOIN_LOSS'):
            summarize_visits({'A':inv},{})
        with self.assertRaisesRegex(ValueError,'VALID_COMMITTED_WINDOW'):
            summarize_visits({'A':inv},{'A':{25:1}})

    def test_probe_strata_keeps_96_rejects_in_one_true_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'probes';manifest,_=probe_fixture(folder,offset=1.)
            p=folder/'ON_EXPERT_RESULTS.json';saved=json.loads(p.read_text());rows=[];windows=[]
            for i in range(96):
                old=saved['records'][i%2];identifier='P'+str(i)
                (folder/(identifier+'.npz')).write_bytes((folder/(old['id']+'.npz')).read_bytes())
                rows.append(dict(old,id=identifier));windows.append(dict(manifest['windows'][i%2],id=identifier))
            p.write_text(json.dumps(dict(saved,records=rows)));manifest['windows']=windows
            proof=dict(status='PASS_SAVED_CANDIDATE_TRAIN_PROBE_METRICS_INPUTS_NOISE_AND_WEIGHT_IDENTITY',candidate_result_sha256=sha(p))
            inventories={'TRAIN-T1':dict(task='T1',windows=25,raw_stages=['hold']*65,
                events=dict(close=[7],open=[],detach=[],withdraw=[]))}
            v=probe_strata(folder,inventories,manifest,proof)
            self.assertEqual(v['distinct_episodes'],1)
            s=v['raw_stage_and_event_groups']['T1:prefix8:close']
            self.assertEqual(s['queries'],96);self.assertEqual(s['independent_episodes'],1)
            self.assertEqual(s['trajectory_gate_only'],dict(REJECT=96))
            self.assertEqual(s['signed_XYZ_prefix8_episode_macro_mean_m'],[1.,0.,0.])
            paired=dict(proof,status='PASS_PAIRED_SAVED_TRAIN_PROBES_IDENTICAL_INPUTS_NOISE_AND_RECOMPUTED_METRICS',
                windows=96,actual_raw_TRAIN_numeric_state_H40_and_dual_camera_inputs_checked=True,new_model_queries=0,new_physics=0)
            self.assertEqual(probe_strata(folder,inventories,manifest,paired)['draws'],96)
            with self.assertRaises(PermissionError):
                probe_strata(folder,inventories,manifest,dict(paired,actual_raw_TRAIN_numeric_state_H40_and_dual_camera_inputs_checked=False))
            proof['candidate_result_sha256']='changed'
            with self.assertRaisesRegex(PermissionError,'PRIOR_INDEPENDENTLY_AUDITED'):
                probe_strata(folder,inventories,manifest,proof)


if __name__=='__main__':unittest.main()
