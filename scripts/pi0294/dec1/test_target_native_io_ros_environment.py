"""CPU-only original ROS environment and one-exec failure boundaries."""
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

from . import target_native_io_ros_environment as unit


@pytest.fixture
def binding(tmp_path,monkeypatch):
    native=tmp_path/'original-native';ros=Path('/opt/ros/jazzy');install=native/'install'
    roots=(install/'lib/python3.12/site-packages',ros/'lib/python3.12/site-packages')
    libs=(native/'src/ctrl/rl_sar/library/inference_runtime/libtorch/lib',
        native/'src/ctrl/rl_sar/library/mujoco/lib',install/'lib',
        ros/'opt/rviz_ogre_vendor/lib',ros/'lib/x86_64-linux-gnu',ros/'opt/gz_math_vendor/lib',
        ros/'opt/gz_utils_vendor/lib',ros/'opt/gz_cmake_vendor/lib',ros/'lib')
    env=dict(PYTHONPATH=':'.join(map(str,roots)),LD_LIBRARY_PATH=':'.join(map(str,libs)),
        AMENT_PREFIX_PATH=str(install)+':'+str(ros),COLCON_PREFIX_PATH=str(install),
        ROS_DISTRO='jazzy',ROS_VERSION='2',RMW_IMPLEMENTATION='')
    origins={name:str((roots[0] if name=='ocs2_msgs' else roots[1])/name/'__init__.py') for name in unit.MODULES}
    extension=str(roots[1]/'rclpy'/('_rclpy_pybind11'+unit.SUFFIX))
    pins={p:'a'*64 for p in (*origins.values(),extension)}
    value=dict(schema=unit.SCHEMA,environment=env,python='/original-venv/bin/python',
        extension_suffix=unit.SUFFIX,module_origins=origins,rclpy_extension=extension)
    inputs=dict(native_root=str(native),runtime_python=value['python'],original_readonly_files=deepcopy(pins),
        original_ROS_environment=value)
    # These represent external readonly installations, never native imports.
    monkeypatch.setattr(unit,'filehash',lambda p:pins[str(p)])
    monkeypatch.setattr(Path,'is_dir',lambda p:True)
    return inputs


def test_exact_original_environment_only_no_native_import(binding):
    original=deepcopy(binding)
    assert unit.validate_binding(binding)==binding['original_ROS_environment']
    env=unit.desired_environment('/sealed/code',binding)
    assert env['PYTHONPATH'].startswith('/sealed/code:/sealed/code/src:')
    assert 'RMW_IMPLEMENTATION' not in env and 'LD_PRELOAD' not in env
    assert binding==original


@pytest.mark.parametrize('fault',('extra_variable','wrong_abi','different_python','changed_library_order',
    'wrong_module_origin','missing_module','changed_extension','changed_module','missing_directory'))
def test_rejects_unbound_original_runtime_before_exec(binding,monkeypatch,fault):
    v=binding['original_ROS_environment']
    if fault=='extra_variable':v['environment']['LD_PRELOAD']='/new/controller.so'
    elif fault=='wrong_abi':v['extension_suffix']='.cpython-311-x86_64-linux-gnu.so'
    elif fault=='different_python':v['python']='/different/python'
    elif fault=='changed_library_order':v['environment']['LD_LIBRARY_PATH']=':'.join(reversed(v['environment']['LD_LIBRARY_PATH'].split(':')))
    elif fault=='wrong_module_origin':v['module_origins']['rclpy']='/shadow/rclpy/__init__.py'
    elif fault=='missing_module':v['module_origins'].pop('ocs2_msgs')
    elif fault=='changed_extension':binding['original_readonly_files'][v['rclpy_extension']]='b'*64
    elif fault=='changed_module':binding['original_readonly_files'][v['module_origins']['rclpy']]='b'*64
    elif fault=='missing_directory':monkeypatch.setattr(Path,'is_dir',lambda p:False)
    with pytest.raises(PermissionError):unit.validate_binding(binding)


def test_same_interpreter_exec_once_after_guard_with_full_original_environment(binding,tmp_path,monkeypatch):
    root=tmp_path/'child';(root/'evidence').mkdir(parents=True);code=root/'code'
    monkeypatch.setattr(unit,'initial_environment',lambda:{})
    called=[];guards=[]
    class Replaced(BaseException):pass
    def execute(path,argv,env):
        called.append((path,argv,env));raise Replaced()
    monkeypatch.setattr(unit.os,'execve',execute)
    with pytest.raises(Replaced):unit.ensure_startup(root,code,binding,lambda:guards.append(True))
    path,argv,env=called[0]
    assert path==argv[0]==binding['runtime_python'] and argv[1]=='-c'
    assert 'target_native_io_entry import run' in argv[2] and str(root) in argv[2]
    assert env['LD_LIBRARY_PATH']==binding['original_ROS_environment']['environment']['LD_LIBRARY_PATH']
    assert len(guards)==2 and (root/'evidence/ROS_ENVIRONMENT_REEXEC_INTENT.json').exists()
    assert not(root/'evidence/NATIVE_IO_OWNER_INTENT.json').exists()
    with pytest.raises(PermissionError,match='ALREADY_ATTEMPTED'):unit.ensure_startup(root,code,binding,lambda:None)
    assert len(called)==1


@pytest.mark.parametrize('shadow',(False,True))
def test_startup_loader_environment_and_metadata_origin_required(binding,tmp_path,monkeypatch,shadow):
    root=tmp_path/'child';(root/'evidence').mkdir(parents=True);code=root/'code'
    env=unit.desired_environment(code,binding)
    for k in (*unit.KEYS,'PYTHONDONTWRITEBYTECODE'):
        if k in env:monkeypatch.setenv(k,env[k])
        else:monkeypatch.delenv(k,raising=False)
    monkeypatch.setattr(unit,'initial_environment',lambda:env.copy())
    monkeypatch.setattr(unit.sys,'version_info',(3,12,3))
    monkeypatch.setattr(unit.sysconfig,'get_config_var',lambda name:unit.SUFFIX)
    seen=[]
    def spec(name):
        seen.append(name);p=binding['original_ROS_environment']['module_origins'][name]
        return SimpleNamespace(origin='/shadow/rclpy.py' if shadow and name=='rclpy' else p)
    monkeypatch.setattr(unit.importlib.util,'find_spec',spec)
    monkeypatch.setattr(unit.os,'execve',lambda *a:pytest.fail('matching startup must not exec'))
    if shadow:
        with pytest.raises(PermissionError,match='SHADOW'):unit.ensure_startup(root,code,binding,lambda:None)
    else:
        unit.ensure_startup(root,code,binding,lambda:None)
        assert seen==list(unit.MODULES)
    assert not list((root/'evidence').iterdir())
