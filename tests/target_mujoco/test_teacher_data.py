"""Narrow CPU contracts. All fixtures explicitly synthetic, never gate evidence."""
import copy
import numpy as np
import pytest

from go2arx5_vla.target_mujoco.dataset import (check_official, enforce_roundtrip, pose10,
    training_population, validate_manifest, windows)
from go2arx5_vla.target_mujoco.recorder import IDENTITIES, LABEL_SOURCE, Recorder, validate_sample
from go2arx5_vla.target_mujoco.scorer import BOOL_FIELDS, physical_grasp, score_episode
from go2arx5_vla.target_mujoco.teacher import Teacher
from go2arx5_vla.trajectory.so3 import quat_to_matrix, rotation_matrix_to_rot6d, rotation_matrix_to_rot6d_rows


def pose(xyz=(0,0,0)):
    result = np.eye(4); result[:3,3] = xyz
    return result


def feedback():
    row = dict(tcp_world=pose((-.16,0,.48)), object_pose=pose((-.16,0,.42)), gripper_fraction=1.,
        object_linear_velocity=np.zeros(3), object_angular_velocity=np.zeros(3))
    row.update({key: False for key in BOOL_FIELDS})
    row.update(object_present=True,contact_valid=True,gripper_state_valid=True,safety_valid=True)
    return row


def spec():
    from go2arx5_vla.target_mujoco.teacher import SPEC_FIELDS
    return dict({name:"CPU_SYNTHETIC_NOT_DEPLOYMENT" for name in SPEC_FIELDS}, status="PROVISIONAL_GRIPPER")


def record_rows(count=641):
    rows = []
    for i in range(count):
        t = i / 20
        target = dict(tcp_xyz=[t*.01,.1,.5],tcp_quaternion_xyzw=[.1,.2,.3,np.sqrt(.86)],gripper=.5)
        obs = dict(target,instruction="Place the red cube in the blue area.")
        for camera in ("body_rgb","wrist_rgb"):
            obs[camera] = dict(path=f"{camera}{i}.png",timestamp_s=t,valid=True,source="mujoco_render",frame_index=i,snapshot_id=f"s{i}")
        sup = dict(requested=target,accepted=target,installed=target,committed=target,
            command_time_s=t,issue_time_s=t,sequence=i,ref_id="r",valid=True,actually_sent=True,action_label_source=LABEL_SOURCE)
        rows.append(dict(episode_id="CPU-1",layout_id=288100,split="TRAIN",task_id="GO2_MJ_PLACE_V1",
            evidence_kind="CPU_SYNTHETIC_NOT_PHYSICS",identity={key:key for key in IDENTITIES},
            observation_time_s=t,segment_id=0,policy_observation=obs,supervision=sup,
            execution=dict(ref_id="r",sequence=i,valid_mask={"base":False},base=None,snapshot_id=f"s{i}"),
            privileged={"teacher_stage":"carry"},times={"simulation":t,"ros":t,"wall_monotonic":t}))
    return rows


def physical_fixture():
    times = np.arange(641)/20
    objects = np.tile(pose((-.16,0,.42)),(641,1,1))
    grasp = (times>=1)&(times<=4)
    objects[(times>=1.5)&(times<=4),2,3] += .08
    objects[times>=2.5,0,3] = .16
    objects[times>4,2,3] = .42
    data = dict(time_s=times,object_pose=objects,tcp_world=np.tile(pose((.16,0,.55)),(641,1,1)),
        object_linear_velocity=np.zeros((641,3)),object_angular_velocity=np.zeros((641,3)))
    data.update({name:np.zeros(641,dtype=bool) for name in BOOL_FIELDS})
    for name in ("object_present","contact_valid","gripper_state_valid","safety_valid"): data[name][:] = True
    for name in ("native_grasp","left_object_contact","right_object_contact","opposing_pad_contact","relative_motion_valid"): data[name] = grasp.copy()
    return data


def test_exact_success_and_physical_grasp():
    data = physical_fixture()
    def score(value):
        return score_episode(value,goal_center=[.16,0,.42],initial_object_pose=pose((-.16,0,.42)),
            table_height_m=.4,evidence_source="CPU_SYNTHETIC_NOT_PHYSICS")
    result = score(data)
    assert result["persistent_success"] and result["hold_pass"] == result["hold_observed"] == 41
    for key in ("opposing_pad_contact","relative_motion_valid"):
        bad = copy.deepcopy(data); bad[key][:] = False
        assert not physical_grasp(bad).any() and not score(bad)["persistent_success"]
    for key in ("numerical_bound_warning","illegal_contact","native_abort"):
        bad = copy.deepcopy(data); bad[key][640] = True
        assert not score(bad)["persistent_success"]
    for i in (560,579,600):
        bad = copy.deepcopy(data); bad["object_pose"][i,0,3] = -.16
        assert score(bad)["hold_pass"] == 40 and not score(bad)["persistent_success"]
    partial = {key:value[:-1] for key,value in data.items()}
    assert not score(partial)["persistent_success"]


def test_h40_602_relative_labels_and_boundaries():
    rows = record_rows()
    samples = windows(rows)
    assert len(samples) == 602 and samples[0]["action"].shape == (40,10)
    assert np.array_equal(samples[0]["action"][0],pose10(rows[0]["supervision"]["committed"]))
    assert np.allclose(samples[0]["relative_action"][0,:3],0)
    assert samples[0]["point_offsets_s"][-1] == 1.95
    assert len(windows(rows[:40])) == 1 and windows(rows[:39]) == []
    for flag in ("reset","native_abort","takeover","invalid_reference"):
        bad = copy.deepcopy(rows[:40]); bad[20]["privileged"][flag] = True
        assert not windows(bad)
    bad = copy.deepcopy(rows[:40]); bad[20]["segment_id"] = 1
    assert not windows(bad)
    bad = copy.deepcopy(rows[:40]); bad[20]["supervision"]["valid"] = False
    assert not windows(bad)
    bad = copy.deepcopy(rows[:40]); bad[20]["execution"]["achieved_tcp"] = [999,999,999]
    assert np.array_equal(windows(bad)[0]["action"],samples[0]["action"])


def test_namespace_clock_and_no_synthetic_admission(tmp_path):
    rows = record_rows(2)
    validate_sample(rows[0])
    for key in ("object_pose","teacher_stage","BASE","success"):
        bad = copy.deepcopy(rows[0]); bad["policy_observation"][key] = True
        with pytest.raises(ValueError): validate_sample(bad)
    bad = copy.deepcopy(rows[0]); bad["policy_observation"]["wrist_rgb"]["timestamp_s"] = .01
    with pytest.raises(ValueError): validate_sample(bad)
    bad = copy.deepcopy(rows[1]); bad["policy_observation"]["wrist_rgb"]["timestamp_s"] = .02
    with pytest.raises(ValueError): validate_sample(bad)
    report, samples = validate_manifest(tmp_path)
    assert report["status"] == "DATA_NOT_ADMITTED" and not samples
    with pytest.raises(ValueError): check_official([dict(evidence_kind="CPU_SYNTHETIC_NOT_PHYSICS")],tmp_path,tmp_path,tmp_path)
    train, dev = dict(split="TRAIN"),dict(split="DEV")
    assert training_population([dev,train]) == [train]
    recorder = Recorder("CPU-1",288100,"CPU",rows[0]["identity"],evidence_kind="CPU_SYNTHETIC_NOT_PHYSICS")
    for row in rows:
        recorder.append(**{key:row[key] for key in ("observation_time_s","policy_observation","supervision","execution","privileged","times")})
    recorder.boundary("reset",.05)
    with pytest.raises(ValueError):
        recorder.append(**{key:rows[1][key] for key in ("observation_time_s","policy_observation","supervision","execution","privileged","times")})


def test_nonidentity_rotation_layout_and_teacher_limits():
    native = dict(tcp_xyz=[.1,.2,.3],tcp_quaternion_xyzw=[.1,.2,.3,np.sqrt(.86)],gripper=.5)
    encoded = pose10(native)
    matrix = quat_to_matrix(native["tcp_quaternion_xyzw"])
    assert np.allclose(encoded[3:9],rotation_matrix_to_rot6d_rows(matrix))
    assert not np.allclose(encoded[3:9],rotation_matrix_to_rot6d(matrix))
    target = np.tile(encoded,(40,1))
    assert enforce_roundtrip(target,target)["SO3_rad"] < 1e-7
    for bad_rotation in ([0]*6,[1,0,0,2,0,0]):
        bad = target.copy(); bad[0,3:9] = bad_rotation
        with pytest.raises(ValueError): enforce_roundtrip(target,bad)
    rng = np.random.default_rng(288)
    for _ in range(8):
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        native_random = dict(native,tcp_quaternion_xyzw=q)
        correct = np.tile(pose10(native_random),(40,1))
        assert enforce_roundtrip(correct,correct)["SO3_rad"] < 1e-7
        columns = correct.copy(); columns[:,3:9] = rotation_matrix_to_rot6d(quat_to_matrix(q))
        with pytest.raises(ValueError): enforce_roundtrip(correct,columns)
    args = (pose((-.16,0,.48)),pose((-.16,0,.42)),[.16,0,.42],pose((0,0,.005)),[-1,-1,0],[1,1,1])
    with pytest.raises(ValueError,match="BLOCKED_GRIPPER_SPEC"): Teacher(*args,gripper_spec=spec())
    with pytest.raises(ValueError): Teacher(*args,gripper_spec=spec(),provisional=True,linear_speed_m_s=.15)
    with pytest.raises(ValueError): Teacher(*args,gripper_spec=spec(),provisional=True,declared_versions=("1","2","3"),version="1")
    teacher = Teacher(*args,gripper_spec=spec(),provisional=True)
    teacher.query(0,feedback())
    plan = teacher.plan(0,feedback())
    assert plan.tcp_world.shape == (40,4,4) and plan.point_times_s[-1] == 1.95
    assert np.max(np.linalg.norm(np.diff(plan.tcp_world[:,:3,3],axis=0),axis=1)) <= .05*.05+1e-10
    with pytest.raises(ValueError): teacher.plan(0,feedback())
    with pytest.raises(ValueError): teacher.query(.1,feedback())
    for i in range(1,9):
        command = teacher.query(i/20,feedback())
        assert np.allclose(command.tcp_world,plan.tcp_world[i])
    replacement = teacher.plan(.4,feedback())
    assert np.allclose(replacement.tcp_world[0],plan.tcp_world[8])
