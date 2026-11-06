"""Fresh registered trace-only capacity successor; never launch the simulator.

Keep the frozen executable, observer and actuation DSO. Compile the same trace
sources at an explicit 131072 or 262144 capacity and check the actual DSO
resolution and serialized-key events in a small loader process.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import resource
import signal
import struct
import subprocess
import time

CPP = 'src/go2arx5_vla/target_mujoco/init_trace.cpp'
HPP = 'src/go2arx5_vla/target_mujoco/init_trace.hpp'
STATUS = 'TRACE_ONLY_131072_CAPACITY_COMPILED_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED'
ORIGINAL_CAPACITY_ASSERT = (b'static_assert(WM02_INIT_TRACE_CAPACITY==65536 || WM02_INIT_TRACE_CAPACITY==131072,\n'
    b'              "Only original and finite NEXT2 twenty-second capacities are supported");')
EXTENDED_CAPACITY_ASSERT = (b'static_assert(WM02_INIT_TRACE_CAPACITY==65536 || WM02_INIT_TRACE_CAPACITY==131072 || WM02_INIT_TRACE_CAPACITY==262144,\n'
    b'              "Only explicit original, NEXT2 and FUSION retimed capacities are supported");')
DOCTOR = r'''#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
int main(int argc,char**argv){
  using E=void(*)(uint32_t,double,uint64_t,const float*,uint32_t);
  auto emit=(E)dlsym(RTLD_DEFAULT,"WM02InitEmit");
  if(!emit)return 71;
  Dl_info info{};if(!dladdr((void*)emit,&info)||!info.dli_fname)return 72;
  printf("TRACE=%s\n",info.dli_fname);
  const char*names[]={"WM02OuterNext","WM02ObserverNext"};
  for(auto n:names){auto f=(const char*(*)())dlsym(RTLD_DEFAULT,n);
    if(!f)return 73;printf("%s=%s\n",n,f());}
  int count=argc==2?std::atoi(argv[1]):0;
  for(int i=0;i<count;++i)emit(i%2?18:17,0.,i,nullptr,0);
}
'''


def need(value, reason):
    if not value: raise ValueError('FUSION_TRACE_CAPACITY_'+reason)


def build(spec, output):
    from go2arx5_vla.world_model.storage import digest, write_json
    from scripts.pi0292.durable import write as replace_json
    out, source = Path(output), Path(spec['source_root'])
    start, cpu = time.monotonic(), time.process_time()
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    capacity = spec.get('trace_capacity', 131072)
    need(type(capacity) is int and capacity in (131072, 262144),
         'EXPLICIT_SUPPORTED_INTEGER_CAPACITY')
    synthetic_records = 70000 * (capacity // 131072)
    status = f'TRACE_ONLY_{capacity}_CAPACITY_COMPILED_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED'
    need(spec.get('profile') == 'core-wm02-init-build'
         and spec.get('domain') == 'TARGET_INIT_BUILD_ONLY_V1'
         and spec.get('physics') is False
         and spec.get('purpose') == 'FUSION_TRACE_CAPACITY_ONLY'
         and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'ORIGINAL_CPU_BUILD_PROFILE')
    need(time.time() < spec['deadline_epoch'], 'FINITE_DEADLINE')
    original = {key: digest(spec[key]) for key in (
        'native_binary', 'observer_library', 'outer_library', 'base_trace_library')}
    need(original == spec['unchanged_inputs_sha256'], 'EXACT_FROZEN_EXECUTOR_AND_DSO_INPUTS')
    need(digest(spec['base_candidate_result']) == spec['base_candidate_result_sha256'],
         'ORIGINAL_C10_BUILD_RECEIPT')
    base_candidate=json.loads(Path(spec['base_candidate_result']).read_text())
    need(base_candidate['binary_sha256'] == original['native_binary'], 'SAME_FROZEN_C10_BINARY')
    source_compatibility = dict(changed_members=[], extension='NONE_EXACT_ORIGINAL_SOURCE')
    for name in (CPP, HPP):
        actual = digest(source/name)
        original_pin = base_candidate['candidate_sources'][Path(name).name]
        need(actual == spec['source_files'][name], 'CAPSULE_TRACE_SOURCE_BYTES')
        if actual != original_pin:
            import hashlib
            raw = (source/name).read_bytes()
            need(name == CPP and capacity == 262144 and raw.count(EXTENDED_CAPACITY_ASSERT) == 1,
                 'ONLY_NEW_262144_ASSERT_EXTENSION_ALLOWED')
            restored = raw.replace(EXTENDED_CAPACITY_ASSERT, ORIGINAL_CAPACITY_ASSERT)
            need(hashlib.sha256(restored).hexdigest() == original_pin,
                 'ALL_EXECUTABLE_TRACE_SOURCE_BYTES_EXCEPT_CAPACITY_ASSERT_MUST_MATCH_ORIGINAL')
            source_compatibility = dict(changed_members=[CPP],
                extension='CAPACITY_STATIC_ASSERT_AND_MESSAGE_ONLY_DEFAULT_65536_UNCHANGED',
                original_CPP_sha256=original_pin, new_CPP_sha256=actual,
                restored_original_CPP_sha256=hashlib.sha256(restored).hexdigest())
    commands = []

    def execute(argv, name, env=None):
        need(time.time() < spec['deadline_epoch'], 'DEADLINE_BEFORE_'+name)
        tick = time.monotonic()
        result = subprocess.run(argv, cwd=out, env=env, capture_output=True,
                                text=True, timeout=45)
        (out/(name+'.stdout')).write_text(result.stdout)
        (out/(name+'.stderr')).write_text(result.stderr)
        commands.append(dict(argv=list(map(str, argv)), name=name,
                             exit_code=result.returncode, wall_s=time.monotonic()-tick))
        replace_json(out/'COMMANDS.private.json', commands)
        need(result.returncode == 0, name+'_EXIT_'+str(result.returncode))
        return result.stdout

    needed = set()
    for key in ('native_binary', 'outer_library', 'observer_library'):
        text = execute(['readelf', '-d', spec[key]], key+'_needed')
        needed.update(s for s in re.findall(r'\(NEEDED\).*?\[(.*?)\]', text)
                      if Path(s).name == 'libwm02_init_trace.so')
    need(len(needed) == 1, 'SINGLE_ACTUAL_TRACE_NEEDED_IDENTITY')
    soname = next(iter(needed))
    need(soname in (spec['base_trace_library'], Path(spec['base_trace_library']).name),
         'BASE_TRACE_DSO_NEEDED_IDENTITY')
    trace = out/'libwm02_init_trace.so'
    execute(['g++', '-std=c++17', '-O2', '-shared', '-fPIC', '-pthread',
        '-DWM02_INIT_TRACE_CAPACITY='+str(capacity), '-Wl,-soname,'+soname,
        str(source/CPP), '-o', str(trace)], 'trace_compile')
    doctor = out/'loader_doctor.cpp'; doctor.write_text(DOCTOR)
    executable = out/'loader_doctor'
    execute(['g++', '-std=c++17', '-O2', str(doctor), '-ldl', '-o', str(executable)],
            'loader_compile')
    env = dict(os.environ, **spec['native_environment'])
    for name in list(env):
        if (name.startswith(('GR00T_TASK_', 'PI0288_', 'PI0289_'))
                or name in ('LD_PRELOAD', 'WM02_INIT_TRACE', 'WM02_INIT_SEAL_REQUEST')):
            env.pop(name)
    original_trace, new_trace = out/'original-loader.trace', out/'capacity-loader.trace'
    old_env = dict(env, LD_PRELOAD=':'.join(spec[k] for k in (
        'outer_library','observer_library','base_trace_library')), WM02_INIT_TRACE=str(original_trace))
    new_env = dict(env, LD_PRELOAD=':'.join([str(trace), spec['outer_library'],
                   spec['observer_library']]), WM02_INIT_TRACE=str(new_trace))
    old = dict(line.split('=',1) for line in execute([str(executable),'0'],
               'original_load_order', old_env).splitlines() if '=' in line)
    new = dict(line.split('=',1) for line in execute([str(executable),str(synthetic_records)],
               'new_trace_first_load_order', new_env).splitlines() if '=' in line)
    need(Path(new['TRACE']).resolve() == trace.resolve(), 'ACTUAL_NEW_TRACE_DLSYM_OWNER')
    need(Path(old['TRACE']).resolve() == Path(spec['base_trace_library']).resolve(),
         'ACTUAL_ORIGINAL_TRACE_DLSYM_OWNER')
    need(all(old[k] == new[k] for k in ('WM02OuterNext','WM02ObserverNext')),
         'UNCHANGED_OUTER_OBSERVER_NEXT_RESOLUTION')
    with original_trace.open('rb') as stream:
        header = stream.read(16)
    need(header[:8] == b'WM02IT01' and struct.unpack('<II', header[8:]) == (840,65536),
         'ACTUAL_ORIGINAL_65536_HEADER')
    kinds, records = Counter(), 0
    with new_trace.open('rb') as stream:
        header = stream.read(16)
        need(header[:8] == b'WM02IT01' and struct.unpack('<II',header[8:]) == (840,capacity),
             'ACTUAL_NEW_DECLARED_CAPACITY_HEADER')
        while raw := stream.read(840):
            need(len(raw) == 840 and struct.unpack_from('<Q',raw)[0] == records,
                 'COMPLETE_ORDERED_RECORDS')
            kinds[struct.unpack_from('<I',raw,24)[0]] += 1; records += 1
    summary = json.loads(Path(str(new_trace)+'.summary').read_text())
    need(records == synthetic_records
         and kinds == {17:synthetic_records//2,18:synthetic_records//2}
         and summary['events'] == summary['written'] == synthetic_records
         and summary['dropped'] == 0,
         'SERIALIZED_KEY_EVENTS_COMPLETE_BEYOND_PREVIOUS_CAPACITY')
    need(original == {key:digest(spec[key]) for key in original}, 'FROZEN_INPUTS_UNCHANGED')
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = dict(status=status, source_sha=spec['source_sha'], trace_capacity=capacity,
        trace_library_sha256=digest(trace), trace_library=str(trace),
        source_files={name:spec['source_files'][name] for name in (CPP,HPP)},
        trace_source_compatibility=source_compatibility,
        native_binary_sha256=original['native_binary'], observer_sha256=original['observer_library'],
        outer_library_sha256=original['outer_library'], original_trace_library_sha256=original['base_trace_library'],
        original_candidate_result_sha256=spec['base_candidate_result_sha256'],
        native_binary_changed=False, WBC_changes=False, new_target_integrations=0,
        leg_policy_queries=0, WM_updates=0, Actor_updates=0, GR00T_updates=0,
        original_capacity=65536, actual_trace_headers_verified=True,
        new_trace_needed_soname=soname, actual_preload_order=['trace_library','outer_library','observer_library'],
        original_loader=old, new_loader=new, original_next_resolution_unchanged=True,
        loader_only_checks=2, synthetic_trace_records=records, key_kind_counts=dict(kinds),
        trace_summary=summary, native_dynamic_qualification=False,
        CPU_s=time.process_time()-cpu, wall_s=time.monotonic()-start,
        child_CPU_s=after.ru_utime+after.ru_stime-children.ru_utime-children.ru_stime,
        commands=commands)
    write_json(out/'RESULT.private.json', result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    args=parser.parse_args(argv)
    from scripts.pi0292.durable import write
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model.runtime import admitted_run
    spec=json.loads(args.spec.read_text());root=Path(spec['registered_root']);lease_path=Path(spec['lease'])
    lease=json.loads(lease_path.read_text())
    need(lease.get('status')=='ACTIVE' and lease.get('run_uuid')==root.name
         and lease.get('spec_sha256')==digest(args.spec), 'FRESH_REGISTERED_WORKER')
    need(not(root/'SUPERVISOR.private.json').exists(), 'NEVER_RESTART_EXISTING_BUILD_WORKER')
    write(root/'SUPERVISOR.private.json',dict(pid=os.getpid(),source_sha=spec['source_sha'],started_epoch=time.time()))
    def stop(signum, frame): raise KeyboardInterrupt('OWNED_BUILD_SIGNAL_'+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    start=time.monotonic();code=1;failure=None
    try:
        with admitted_run(args.spec,'build-01','core-wm02-init-build',root/'build-01') as admitted:
            build(admitted,root/'build-01')
        code=0
    except BaseException as error:
        failure=type(error).__name__+': '+str(error)
    finally:
        write(root/'WORKER_EXIT.private.json',dict(exit_code=code,failure=failure,wall_s=time.monotonic()-start))
        current=json.loads(lease_path.read_text())
        need(current.get('status')=='ACTIVE' and current.get('spec_sha256')==digest(args.spec),
             'ORIGINAL_WORKER_LEASE_UNCHANGED')
        current.update(status='CLOSED',closed_epoch=time.time(),worker_exit_code=code)
        write(lease_path,current)
    return code


if __name__=='__main__': raise SystemExit(main())
