"""Bind the existing ROS installation after full current-child admission.

No setup shell is executed. The sealed original environment is supplied to a
single exec of the same interpreter, so its dynamic loader sees the original
library paths at process startup. This neither admits a child nor imports ROS.
"""
import importlib.util
import os
from pathlib import Path
import sys
import sysconfig

from .target_zero_counter import filehash, put
from .target_native_io_budget import need

SCHEMA='DEC1_ORIGINAL_ROS_JAZZY_ENVIRONMENT_BINDING_V1'
MODULES=('rclpy','rosidl_runtime_py','ocs2_msgs','std_msgs','nav_msgs','sensor_msgs')
KEYS=('PYTHONPATH','LD_LIBRARY_PATH','AMENT_PREFIX_PATH','COLCON_PREFIX_PATH',
      'ROS_DISTRO','ROS_VERSION','RMW_IMPLEMENTATION')
SUFFIX='.cpython-312-x86_64-linux-gnu.so'


def validate_binding(inputs):
    value=inputs['original_ROS_environment'];env=value['environment']
    native=Path(inputs['native_root']);ros=Path('/opt/ros/jazzy')
    install=native/'install'
    need(value['schema']==SCHEMA and set(env)==set(KEYS)
         and value['python']==inputs['runtime_python'] and value['extension_suffix']==SUFFIX,
         'IO_SAME_ORIGINAL_PYTHON_AND_ROS_ABI')
    python_roots=(install/'lib/python3.12/site-packages',ros/'lib/python3.12/site-packages')
    libraries=(native/'src/ctrl/rl_sar/library/inference_runtime/libtorch/lib',
        native/'src/ctrl/rl_sar/library/mujoco/lib',install/'lib',
        ros/'opt/rviz_ogre_vendor/lib',ros/'lib/x86_64-linux-gnu',
        ros/'opt/gz_math_vendor/lib',ros/'opt/gz_utils_vendor/lib',
        ros/'opt/gz_cmake_vendor/lib',ros/'lib')
    need(env==dict(PYTHONPATH=':'.join(map(str,python_roots)),
        LD_LIBRARY_PATH=':'.join(map(str,libraries)),
        AMENT_PREFIX_PATH=str(install)+':'+str(ros),COLCON_PREFIX_PATH=str(install),
        ROS_DISTRO='jazzy',ROS_VERSION='2',RMW_IMPLEMENTATION=''),
        'IO_ONLY_ORIGINAL_DELIVERED_ROS_ENVIRONMENT')
    origins=value['module_origins'];pins=inputs['original_readonly_files']
    need(set(origins)==set(MODULES),'IO_COMPLETE_ORIGINAL_ROS_MODULE_BINDING')
    for name,path in origins.items():
        prefix=python_roots[0] if name=='ocs2_msgs' else python_roots[1]
        need(path==str(prefix/name/'__init__.py') and path in pins
             and filehash(path)==pins[path],'IO_ROS_MODULE_ORIGIN_CHANGED')
    extension=str(python_roots[1]/'rclpy'/('_rclpy_pybind11'+SUFFIX))
    need(value['rclpy_extension']==extension and extension in pins
         and filehash(extension)==pins[extension],'IO_ORIGINAL_ROS_COMPILED_EXTENSION')
    need(all(path.is_dir() for path in (*python_roots,*libraries)),
         'IO_ORIGINAL_ROS_SEARCH_DIRECTORY_MISSING')
    return value


def loader_environment(inputs,value):
    """Bind the already-installed ELF closure separately from ROS setup.

    ROS env.sh does not cover all delivered GLFW/Pinocchio/OctoMap libraries.
    A future cold request may carry the exact read-only ldd closure; every
    resolved file is sealed and rehashed before the same-interpreter exec.
    This does not build, replace or execute a WBC binary.
    """
    base=value['environment']['LD_LIBRARY_PATH']
    closure=inputs.get('original_ELF_linker_closure')
    if closure is None:return base  # Historical binding remains interpretable.
    native=Path(inputs['native_root']);dep=native.parent/'dependency-root'
    extra=('/usr/lib/gcc/x86_64-linux-gnu/13','/usr/lib/x86_64-linux-gnu',
        str(dep/'opt/ros/jazzy/lib/x86_64-linux-gnu'),str(dep/'usr/lib/x86_64-linux-gnu'))
    expected=base+':'+':'.join(extra)
    need(closure['schema']=='DEC1_ORIGINAL_FROZEN_ELF_LINKER_CLOSURE_V1'
         and closure['library_path']==expected,'IO_EXACT_ORIGINAL_ELF_SEARCH_ORDER')
    files=closure['resolved_files'];checks=closure['programs'];pins=inputs['original_readonly_files']
    need(set(checks)=={'sim_binary','solver_binary','controller_binary'} and files,
         'IO_COMPLETE_ORIGINAL_ELF_PROGRAM_SET')
    covered=set()
    for key,check in checks.items():
        need(check['returncode']==0 and check['missing']==[] and check['resolved']
             and check['binary_sha256']==pins[inputs[key]]
             and filehash(inputs[key])==check['binary_sha256'],'IO_ORIGINAL_ELF_BINARY_OR_DEPENDENCY')
        covered.update(check['resolved'])
    need(covered==set(files),'IO_EXACT_TRANSITIVE_ELF_CLOSURE')
    for name,pin in files.items():
        need(Path(name).is_absolute() and pins.get(name)==pin and filehash(name)==pin,
             'IO_ORIGINAL_ELF_DEPENDENCY_CHANGED')
    return expected


def desired_environment(code,inputs):
    value=validate_binding(inputs)
    result=dict(os.environ);result.update(value['environment'])
    result['LD_LIBRARY_PATH']=loader_environment(inputs,value)
    result['PYTHONPATH']=str(code)+':'+str(Path(code)/'src')+':'+value['environment']['PYTHONPATH']
    # Do not let an unrelated inherited RMW selection change the frozen graph.
    result.pop('RMW_IMPLEMENTATION',None)
    result.pop('LD_PRELOAD',None);result.pop('LD_AUDIT',None)
    result['PYTHONDONTWRITEBYTECODE']='1'
    return result


def initial_environment():
    return dict(os.fsdecode(item).split('=',1) for item in Path('/proc/self/environ').read_bytes().split(b'\0')
                if b'=' in item)


def ensure_startup(root,code,inputs,guard):
    """One exec before owner/native intent; repeated incomplete startup rejects."""
    guard();expected=desired_environment(code,inputs);startup=initial_environment()
    keys=(*KEYS,'PYTHONDONTWRITEBYTECODE')
    ready=all(startup.get(k,'')==expected.get(k,'')==os.environ.get(k,'') for k in keys)
    if ready:
        need(sys.version_info[:2]==(3,12) and sysconfig.get_config_var('EXT_SUFFIX')==SUFFIX,
             'IO_RUNNING_INTERPRETER_ABI_CHANGED')
        for name,path in inputs['original_ROS_environment']['module_origins'].items():
            spec=importlib.util.find_spec(name)
            need(spec is not None and spec.origin==path,'IO_ROS_IMPORT_SHADOW_OR_MISSING_'+name)
        return
    intent=Path(root)/'evidence/ROS_ENVIRONMENT_REEXEC_INTENT.json'
    need(not intent.exists(),'IO_ROS_ENVIRONMENT_EXEC_ALREADY_ATTEMPTED_NO_LOOP')
    put(intent,dict(status='SAME_ADMITTED_CHILD_ORIGINAL_ROS_ENVIRONMENT_EXEC_INTENT',
        python=inputs['runtime_python'],once=True,new_registry_or_resource_reservation=False,
        native_imports_or_model_constructors=0))
    guard()
    bootstrap=('import sys;sys.path[:0]='+repr([str(code),str(Path(code)/'src')])+';'
        'from scripts.pi0294.dec1.target_native_io_entry import run;'
        'v=run('+repr(str(root))+');'
        'raise SystemExit(0 if v["status"]=="ACTUAL_ZERO_IO_FEEDBACK_AND_STOP" else 1)')
    os.execve(inputs['runtime_python'],[inputs['runtime_python'],'-c',bootstrap],expected)
    raise RuntimeError('IO_EXEC_RETURNED_WITHOUT_PROCESS_REPLACEMENT')
