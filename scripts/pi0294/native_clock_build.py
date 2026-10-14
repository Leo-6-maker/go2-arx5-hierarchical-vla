"""Incremental simulation-clock overlay; original controller and gain files untouched."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from scripts.pi0292.durable import write,read
from .r4a import validate_scope,cpu_stage


def build(root,native,startup):
    root,native,startup=map(Path,(root,native,startup));validate_scope(root)
    with cpu_stage(root,'FULLBODY_CLOCK_BUILD_01',cores=4,cpu_s_cap=3600):
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('NATIVE_CPU_BUILD_ONLY')
        out=root/'fullbody/clock-v1';out.mkdir(exist_ok=False)
        cfg=read(startup/'startup-v4/config.json')
        for name,h in cfg['core_hashes'].items():
            if hashlib.sha256((native/name).read_bytes()).hexdigest()!=h:raise ValueError('FROZEN_NATIVE_MEMBER_CHANGED:'+name)
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        source=startup/'startup-v2/private-source';before=(source/'rl_sim_mujoco.cpp').read_text()
        if sha(source/'rl_sim_mujoco.cpp')!='7bdacaecdf3ad66dd62fe0ed5046a1fb1c6611e185673dcb18c9cb94de61229c':raise ValueError('QUALIFIED_STARTUP_SOURCE_CHANGED')
        after=before.replace('std::chrono::steady_clock::now()','PI0294SimNow()')
        header='''#include <atomic>
#include <rosgraph_msgs/msg/clock.hpp>
static std::atomic<int64_t> PI0294SimNs{0};
static rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr PI0294ClockPub;
static std::chrono::steady_clock::time_point PI0294SimNow() {
  return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(PI0294SimNs.load()));
}
'''
        after=after.replace('#include <functional>','#include <functional>\n'+header)
        needle='    const std::lock_guard<std::recursive_mutex> lock(sim->mtx);'
        assert after.count(needle)==1
        after=after.replace(needle,needle+'\n    PI0294SimNs.store(static_cast<int64_t>(std::llround(this->mj_data->time * 1e9)));')
        needle='    ros_node_ = std::make_shared<rclcpp::Node>("go2_x5_mujoco_ros");'
        assert after.count(needle)==1
        after=after.replace(needle,needle+'\n    PI0294ClockPub = ros_node_->create_publisher<rosgraph_msgs::msg::Clock>("/clock", 10);')
        after=after.replace('msg.header.stamp = ros_node_->now();','msg.header.stamp = rclcpp::Time(PI0294SimNs.load());')
        needle='    if (!ros_odom_pub_ || !this->params.Get<bool>("use_base_state_sensor", false)) return;'
        assert after.count(needle)==1
        after=after.replace(needle,'    if (PI0294ClockPub) { rosgraph_msgs::msg::Clock clock; clock.clock = rclcpp::Time(PI0294SimNs.load()); PI0294ClockPub->publish(clock); }\n'+needle)
        (out/'rl_sim_mujoco.cpp').write_text(after)
        for name in ('rl_sim_mujoco.hpp','mujoco_utils.hpp','native_home.hpp','native_startup.hpp'):
            p=source/name if (source/name).is_file() else startup/'startup-v2'/name
            (out/name).write_bytes(p.read_bytes())
        node=native/'src/ctrl/go2_x5_ocs2/src/go2_x5_ocs2_node.cpp';node_before=node.read_text()
        node_after=node_before.replace('  void tick() {\n    std::lock_guard<std::mutex> lock(mutex_);',
            '  void tick() {\n    std::lock_guard<std::mutex> lock(mutex_);\n    const auto source_tick = now().nanoseconds();\n    if (source_tick == pi0294_last_tick_) return;\n    pi0294_last_tick_ = source_tick;')
        assert node_after!=node_before
        node_after=node_after.replace('  rclcpp::Time start_time_;','  rclcpp::Time start_time_;\n  int64_t pi0294_last_tick_{-1};')
        (out/'go2_x5_ocs2_node.cpp').write_text(node_after)
        (out/'PRIVATE_CLOCK.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True)))+'\n'+''.join(difflib.unified_diff(node_before.splitlines(True),node_after.splitlines(True))))
        commands=[]
        def execute(argv,label,cwd=None):
            p=subprocess.run(argv,cwd=cwd,capture_output=True,text=True,timeout=600)
            (out/(label+'.log')).write_text(p.stdout+p.stderr)
            commands.append(dict(label=label,argv=argv,exit=p.returncode))
            write(out/'BUILD_COMMANDS.private.json',commands);p.check_returncode()
        for part,filename,object_name,binary in (
                ('mujoco','rl_sim_mujoco.cpp','src/rl_sim_mujoco.cpp.o','rl_sim_mujoco'),
                ('go2_x5_ocs2','go2_x5_ocs2_node.cpp','src/go2_x5_ocs2_node.cpp.o','go2_x5_ocs2_node')):
            directory=native/'build'/part;cmake=directory/'CMakeFiles'/(binary+'.dir');flags={}
            for line in (cmake/'flags.make').read_text().splitlines():
                if ' = ' in line:
                    key,v=line.split(' = ',1);flags[key]=shlex.split(v)
            args=['/usr/bin/c++','-I'+str(out),'-isystem','/opt/ros/jazzy/include/rosgraph_msgs']+flags['CXX_DEFINES']+flags['CXX_INCLUDES']+flags['CXX_FLAGS']
            execute(args+['-o',str(out/(filename+'.o')),'-c',str(out/filename)],'compile-'+binary)
            link=shlex.split((cmake/'link.txt').read_text())
            expected='CMakeFiles/'+binary+'.dir/'+object_name;assert expected in link
            link[link.index(expected)]=str(out/(filename+'.o'))
            if part=='mujoco':
                expected='CMakeFiles/rl_sim_mujoco.dir/library/thirdparty/mujoco_simulate/simulate.cc.o';assert expected in link
                link[link.index(expected)]=str(startup/'startup-v2/simulate.cc.o')
                link.append('/opt/ros/jazzy/lib/librosgraph_msgs__rosidl_typesupport_cpp.so')
            output=link.index('-o')+1;link[output]=str(out/binary)
            execute(link,'link-'+binary,directory)
        write(out/'CLOCK_BUILD_RECEIPT.json',dict(status='CLOCK_OVERLAY_COMPILED_NOT_PHYSICS_QUALIFIED',
            source_pins=cfg['core_hashes'],sim_sha256=sha(out/'rl_sim_mujoco'),node_sha256=sha(out/'go2_x5_ocs2_node'),
            private_diff_sha256=sha(out/'PRIVATE_CLOCK.diff'),controller_costs_gains_limits_unchanged=True,
            source_clock='ACTUAL_MJDATA_TIME_WITH_ATOMIC_NANOSECONDS',watchdogs='SAME_INTERVAL_IN_SIM_TIME',
            clock_configuration_required='use_sim_time=true_FOR_OWN_NATIVE_ROS_GRAPH',
            new_physics=0,leg_RL_forwards=0,GR00T_queries=0,plant_overlay_ready=False))


def relink(root,native,startup):
    root,native,startup=map(Path,(root,native,startup));validate_scope(root)
    with cpu_stage(root,'FULLBODY_CLOCK_RELINK_02',cores=1,cpu_s_cap=900):
        out=root/'fullbody/clock-v1';commands=read(out/'BUILD_COMMANDS.private.json')
        if (len(commands)!=4 or [c['exit'] for c in commands]!=[0,0,0,1]
                or commands[-1]['label']!='link-go2_x5_ocs2_node'):
            raise PermissionError('EXACT_COMPILED_OBJECTS_TRANSITIVE_LINK_FAILURE_ONLY')
        cfg=read(startup/'startup-v4/config.json');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        for name,h in cfg['core_hashes'].items():
            if sha(native/name)!=h:raise ValueError('FROZEN_NATIVE_MEMBER_CHANGED:'+name)
        dependency=native.parent/'dependency-root/usr/lib/x86_64-linux-gnu'
        libraries=[dependency/'liboctomap.so.1.9',dependency/'liboctomath.so.1.9']
        pins={p.name:sha(p) for p in libraries};objects={p.name:sha(p) for p in out.glob('*.cpp.o')}
        argv=commands[-1]['argv']+['-Wl,-rpath-link,'+str(dependency),'-Wl,-rpath,'+str(dependency)]
        result=subprocess.run(argv,cwd=native/'build/go2_x5_ocs2',capture_output=True,text=True,timeout=300)
        (out/'link-go2_x5_ocs2_node-02.log').write_text(result.stdout+result.stderr)
        write(out/'RELINK_02.private.json',dict(argv=argv,exit=result.returncode,reused_objects=objects,library_sha256=pins))
        result.check_returncode()
        env=dict(os.environ,LD_LIBRARY_PATH=str(dependency)+':'+str(native.parent/'dependency-root/opt/ros/jazzy/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH',''))
        loaded=subprocess.run(['ldd',str(out/'go2_x5_ocs2_node')],env=env,capture_output=True,text=True)
        (out/'ldd-node-02.log').write_text(loaded.stdout+loaded.stderr)
        if loaded.returncode or 'not found' in loaded.stdout:raise RuntimeError('NATIVE_RUNTIME_TRANSITIVE_LIBRARY_STILL_UNRESOLVED')
        assert objects=={p.name:sha(p) for p in out.glob('*.cpp.o')}
        write(out/'CLOCK_BUILD_RECEIPT.json',dict(status='CLOCK_OVERLAY_COMPILED_NOT_PHYSICS_QUALIFIED',
            source_pins=cfg['core_hashes'],sim_sha256=sha(out/'rl_sim_mujoco'),node_sha256=sha(out/'go2_x5_ocs2_node'),
            private_diff_sha256=sha(out/'PRIVATE_CLOCK.diff'),transitive_library_sha256=pins,
            reused_objects_unchanged=True,controller_costs_gains_limits_unchanged=True,
            source_clock='ACTUAL_MJDATA_TIME_WITH_ATOMIC_NANOSECONDS',watchdogs='SAME_INTERVAL_IN_SIM_TIME',
            clock_configuration_required='use_sim_time=true_FOR_OWN_NATIVE_ROS_GRAPH',
            new_physics=0,leg_RL_forwards=0,GR00T_queries=0,plant_overlay_ready=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','native','startup'):p.add_argument('--'+name,required=True)
    a=p.parse_args();build(a.root,a.native,a.startup)
