"""Build-only target candidate: existing R1 synchronization, latched observer and bounded trace."""
from pathlib import Path
import difflib,hashlib,json,os,re,shlex,shutil,subprocess,time
from .storage import digest,write_json
from scripts.pi0292.durable import write as replace_json
from scripts.pi0294.task_pause_r1_build import patch as r1_patch,HARNESS

START='// WM02_TRACE_BEGIN\n';END='// WM02_TRACE_END\n'
def marked(s):return START+s+'\n'+END
def remove_trace(s):
    s=re.sub(re.escape(START)+r'.*?'+re.escape(END),'',s,flags=re.S)
    s=s.replace('wm02_init::pop(rl.output_dof_pos_queue, _output_dof_pos, 0)','rl.output_dof_pos_queue.try_pop(_output_dof_pos)')
    return s.replace('wm02_init::pop(rl.output_dof_vel_queue, _output_dof_vel, 1)','rl.output_dof_vel_queue.try_pop(_output_dof_vel)')

def once(text,needle,addition,*,after=True):
    if text.count(needle)!=1:raise ValueError('Frozen candidate patch anchor differs:'+needle[:60])
    return text.replace(needle,needle+marked(addition)if after else marked(addition)+needle)

def end_function(text,signature,addition):
    # Mask literals/comments before counting braces; do not edit vendor syntax.
    mask=re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',lambda m:' '*len(m[0]),text,flags=re.S)
    start=mask.index('{',mask.index(signature));level=1;i=start+1
    while level:
        level+=(mask[i]=='{')-(mask[i]=='}');i+=1
    return text[:i-1]+marked(addition)+text[i-1:]


def instrument(cpp,sdk):
    original_cpp,original_sdk=cpp,sdk;cpp=marked('#include "init_trace.hpp"')+cpp;sdk=marked('#include "init_trace.hpp"')+sdk
    cpp=once(cpp,'    if (this->rl_init_done && simulation_running)\n    {\n','    const auto wm02_version=WM02InitBegin(PI0294SimNs.load()*1e-9);')
    cpp=once(cpp,'        this->obs.actions = this->Forward();\n','        wm02_init::emit(wm02_init::INFERENCE_DONE,PI0294SimNs.load()*1e-9,wm02_version,this->obs.actions);')
    for key,channel in [('pos',0),('vel',1)]:
        cpp=once(cpp,f'            output_dof_{key}_queue.push(this->output_dof_{key});',f'            WM02InitPublish({channel},wm02_version);',after=False)
    cpp=once(cpp,'    std::vector<float> clamped_obs = this->ComputeObservation();\n','    wm02_init::emit(wm02_init::INPUT,PI0294SimNs.load()*1e-9,WM02InitCurrent(),clamped_obs);')
    cpp=once(cpp,'        return this->obs.actions;\n','        wm02_init::emit(wm02_init::INFERENCE_SKIPPED,PI0294SimNs.load()*1e-9,WM02InitCurrent());',after=False)
    cpp=once(cpp,'        actions = this->model->forward({this->history_obs});','        wm02_init::emit(wm02_init::HISTORY,PI0294SimNs.load()*1e-9,WM02InitCurrent(),this->history_obs);',after=False)
    cpp=end_function(cpp,'void RL_Sim::GetState(','''    if(mj_data)wm02_init::vectors(wm02_init::OBS,mj_data->time,0,{&state->motor_state.q,&state->motor_state.dq});''')
    cpp=end_function(cpp,'void RL_Sim::SetCommand(','''    if(mj_data){
     wm02_init::vectors(wm02_init::PD_TARGET,mj_data->time,0,{&command->motor_command.q,&command->motor_command.dq,&command->motor_command.kp,&command->motor_command.kd,&command->motor_command.tau});
     wm02_init::numbers(wm02_init::PD_STATE_CTRL,mj_data->time,0,{{mj_data->qpos,mj_model->nq},{mj_data->qvel,mj_model->nv},{mj_data->ctrl,mj_model->nu}});
    }''')
    cpp=once(cpp,'PI0289ControlTrace(this->mj_data,', '''double wm02_flags[]={double(this->rl_init_done),double(simulation_running),double(task_clock_paused_),double(ros_arm_mode_=="HOLD"),double(ros_arm_mode_=="WBC"),double(ros_arm_mode_=="OCS2"),double(ros_arm_mode_=="DAMPING")};
    WM02InitNumeric(wm02_init::MODE,mj_data->time,0,wm02_flags,7);
    ''',after=False)
    a='rl.output_dof_pos_queue.try_pop(_output_dof_pos)';b='rl.output_dof_vel_queue.try_pop(_output_dof_vel)'
    if sdk.count(a)!=1 or sdk.count(b)!=1:raise ValueError('Actual SDK queue consumer differs')
    sdk=sdk.replace(a,'wm02_init::pop(rl.output_dof_pos_queue, _output_dof_pos, 0)').replace(b,'wm02_init::pop(rl.output_dof_vel_queue, _output_dof_vel, 1)')
    sdk=once(sdk,'if (wm02_init::pop(rl.output_dof_pos_queue, _output_dof_pos, 0) && wm02_init::pop(rl.output_dof_vel_queue, _output_dof_vel, 1))\n    {\n','        WM02InitApplied();')
    if remove_trace(cpp)!=original_cpp or remove_trace(sdk)!=original_sdk:raise ValueError('Instrumentation changed non-trace controller text')
    return cpp,sdk


def flags(path):
    result={}
    for line in Path(path).read_text().splitlines():
        if' = 'in line:k,v=line.split(' = ',1);result[k]=shlex.split(v)
    return result


def build(spec,output):
    output=Path(output);start=time.monotonic();cpu=time.process_time();native=Path(spec['native_root']);clock=Path(spec['clock_root']);startup=Path(spec['startup_root'])
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU build only')
    for p,h in spec['frozen_files'].items():
        if digest(p)!=h:raise ValueError('Frozen build input changed:'+p)
    lease=Path(spec['target_lease']);lease_before=digest(lease)
    if lease_before!=spec['target_lease_sha256']or json.loads(lease.read_text())['status']!='CLOSED':raise PermissionError('Historical target lease must stay CLOSED')
    code=Path(spec['source_root']);names=('rl_sim_mujoco.cpp','rl_sim_mujoco.hpp','mujoco_utils.hpp');before=[(clock/n).read_text()for n in names];patched=r1_patch(*before)
    sdk=Path(spec['sdk_source']).read_text();cpp,sdk_after=instrument(patched[0],sdk)
    for n,s in zip(names,(cpp,patched[1],patched[2])):(output/n).write_text(s)
    (output/'rl_sdk.cpp').write_text(sdk_after)
    for name in('native_home.hpp','native_startup.hpp'):shutil.copyfile(clock/name,output/name)
    for name in('init_trace.hpp','init_trace.cpp'):shutil.copyfile(code/'src/go2arx5_vla/target_mujoco'/name,output/name)
    commands=[]
    # Match the existing user-space ROS/native dependency roots. This is a
    # per-process loader environment, with no host installation or path changes.
    library_roots=[Path('/opt/ros/jazzy/lib'),native.parent/'dependency-root/usr/lib/x86_64-linux-gnu']
    if not all(p.is_dir()for p in library_roots):raise FileNotFoundError('Original native library roots unavailable')
    build_env=dict(os.environ,LD_LIBRARY_PATH=':'.join(map(str,library_roots))+(':'+os.environ['LD_LIBRARY_PATH']if os.environ.get('LD_LIBRARY_PATH')else''))
    def execute(argv,label,cwd=None,env=None):
        q=subprocess.run(argv,cwd=cwd,env=build_env if env is None else env,capture_output=True,text=True,timeout=600)
        (output/(label+'.log')).write_text(q.stdout+q.stderr);commands.append(dict(label=label,argv=[str(v)for v in argv],exit_code=q.returncode));replace_json(output/'COMMANDS.private.json',commands);q.check_returncode();return q
    trace=output/'libwm02_init_trace.so';execute(['g++','-std=c++17','-O2','-shared','-fPIC','-pthread',str(output/'init_trace.cpp'),'-o',str(trace)],'trace_compile')
    for key,source in [('main',output/'rl_sim_mujoco.cpp'),('sdk',output/'rl_sdk.cpp')]:
        f=flags(spec[key+'_flags']);obj=output/('rl_sim_mujoco.cpp.o'if key=='main'else'rl_sdk.cpp.o')
        execute(['/usr/bin/c++','-I'+str(output),'-isystem','/opt/ros/jazzy/include/rosgraph_msgs']+f['CXX_DEFINES']+f['CXX_INCLUDES']+f['CXX_FLAGS']+['-o',str(obj),'-c',str(source)],key+'_compile')
    archive=output/'librl_sdk.a';shutil.copyfile(spec['sdk_archive'],archive)
    original_members=subprocess.check_output(['ar','t',str(archive)],text=True).splitlines()
    if original_members.count('rl_sdk.cpp.o')!=1:raise ValueError('Unexpected original SDK archive')
    preserved_members={n:hashlib.sha256(subprocess.check_output(['ar','p',str(archive),n])).hexdigest()for n in original_members if n!='rl_sdk.cpp.o'}
    execute(['ar','r',str(archive),str(output/'rl_sdk.cpp.o')],'sdk_archive')
    if subprocess.check_output(['ar','t',str(archive)],text=True).splitlines()!=original_members:raise ValueError('SDK archive member set changed')
    for n,h in preserved_members.items():
        if hashlib.sha256(subprocess.check_output(['ar','p',str(archive),n])).hexdigest()!=h:raise ValueError('Unrelated SDK archive member changed')
    directory=native/'build/mujoco';link=shlex.split(Path(spec['link_flags']).read_text());link[link.index('CMakeFiles/rl_sim_mujoco.dir/src/rl_sim_mujoco.cpp.o')]=str(output/'rl_sim_mujoco.cpp.o')
    link[link.index('CMakeFiles/rl_sim_mujoco.dir/library/thirdparty/mujoco_simulate/simulate.cc.o')]=str(startup/'startup-v2/simulate.cc.o');link[link.index('librl_sdk.a')]=str(archive)
    link[link.index('-o')+1]=str(output/'rl_sim_mujoco');link+=['/opt/ros/jazzy/lib/librosgraph_msgs__rosidl_typesupport_cpp.so',str(trace),'-Wl,-rpath,'+str(output)]
    execute(link,'native_link',directory);execute(['ldd',str(output/'rl_sim_mujoco')],'native_ldd')
    if'not found'in(output/'native_ldd.log').read_text():raise RuntimeError('Candidate native dependencies unresolved')
    # Compile both alternatives. Exactly one outer wrapper belongs in a chain.
    dep=Path(spec['mujoco_library']);include=Path(spec['mujoco_include']);common=['g++','-std=c++17','-O2','-shared','-fPIC','-I'+str(include),'-I'+str(code/'scripts/pi0294'),'-I'+str(output),'-ldl']
    for label in('task_actuation','task_actuation_r1','observer'):
        src=(code/'src/go2arx5_vla/target_mujoco/observer.cpp'if label=='observer'else code/'scripts/pi0294'/(label+'.cpp')).read_text();src='#include "init_trace.hpp"\n'+src
        function='WM02ObserverNext'if label=='observer'else'WM02OuterNext'
        src+='\nextern "C" const char* '+function+'(){Dl_info i{};return dladdr(dlsym(RTLD_NEXT,"mj_step"),&i)?i.dli_fname:nullptr;}\n'
        if label=='observer':
            src=src.replace('  real(m,d);','  wm02_init::numbers(wm02_init::STEP_BEFORE,d->time,count+1,{{d->qpos,m->nq},{d->qvel,m->nv},{d->ctrl,m->nu}});\n  real(m,d);\n  wm02_init::numbers(wm02_init::STEP_AFTER,d->time,count+1,{{d->qpos,m->nq},{d->qvel,m->nv},{d->ctrl,m->nu}});')
            src=src.replace('if(safety_stopped) {raise(SIGINT);return;}','if(safety_stopped) {wm02_init::emit(wm02_init::REFUSED,d->time,count);raise(SIGINT);return;}')
            src=src.replace('safety_stopped=true;','safety_stopped=true;wm02_init::emit(wm02_init::STOP,d->time,count);')
        p=output/(label+'.cpp');p.write_text(src);execute(common+[str(p),str(trace),str(dep),'-Wl,-rpath,'+str(output),'-Wl,-rpath,'+str(dep.parent),'-o',str(output/(label+'.so'))],label+'_compile')
    doctor=output/'loader_doctor.cpp';doctor.write_text('#include <dlfcn.h>\n#include <cstdio>\nint main(){using F=const char*(*)();for(auto n:{"WM02OuterNext","WM02ObserverNext"}){auto f=(F)dlsym(RTLD_DEFAULT,n);if(!f)return 71;printf("%s=%s\\n",n,f());}return 0;}\n')
    execute(['g++','-std=c++17','-include','initializer_list',str(doctor),'-ldl','-o',str(output/'loader_doctor')],'loader_doctor_compile')
    chains={}
    for label in('task_actuation','task_actuation_r1'):
        paths=[output/(label+'.so'),output/'observer.so',trace];env=dict(build_env,LD_PRELOAD=':'.join(map(str,paths)),WM02_INIT_TRACE=str(output/(label+'-doctor.trace')))
        q=execute([str(output/'loader_doctor')],label+'_load_order',env=env);actual=dict(line.split('=',1)for line in q.stdout.splitlines())
        if Path(actual['WM02OuterNext']).resolve()!=paths[1].resolve()or Path(actual['WM02ObserverNext']).resolve()!=dep.resolve():raise RuntimeError('Actual dlsym step chain differs')
        chains[label]=[dict(path=str(p),sha256=digest(p))for p in paths]+[dict(path=str(dep),sha256=digest(dep))]
    (output/'pause_harness.cpp').write_text(HARNESS);execute(['g++','-std=c++17','-pthread',str(output/'pause_harness.cpp'),'-o',str(output/'pause_harness')],'pause_harness_compile');execute([str(output/'pause_harness')],'pause_harness')
    if digest(lease)!=lease_before:raise RuntimeError('Historical target lease changed')
    result=dict(status='INIT_CANDIDATE_COMPILED_AND_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED',source_sha=spec['source_sha'],
        binary_sha256=digest(output/'rl_sim_mujoco'),candidate_sources={n:digest(output/n)for n in(*names,'rl_sdk.cpp','init_trace.hpp','init_trace.cpp')},
        original_SDK_sha256=digest(spec['sdk_source']),controller_nontrace_text_identical=True,SDK_archive_member_names=original_members,preserved_SDK_members=preserved_members,original_R1_math_assertions_pass=True,
        actual_preload_chains=chains,runtime_environment={'LD_LIBRARY_PATH':build_env['LD_LIBRARY_PATH']},target_lease_sha256=lease_before,target_lease_status='CLOSED',native_dynamics_permission='NOT_GRANTED',
        actual_target_integrations=0,leg_policy_queries=0,WM_updates=0,Actor_updates=0,loader_only_checks=2,
        trace_contract='65536 bounded records; async writer10ms; first inputs plus recent28 payload slots; exact pos/vel consumed versions; no emit-path IO/wait or command math change',
        CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    write_json(output/'CANDIDATE.private.json',result);return result
