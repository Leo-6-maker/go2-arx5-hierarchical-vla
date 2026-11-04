"""Actual C++ wrappers over a fake mj_step; zero native/fixture physics integration."""
from pathlib import Path
import json,os,shutil,subprocess,tempfile,unittest
try:import mujoco
except ModuleNotFoundError:raise unittest.SkipTest('MuJoCo headers/library needed, no plant is run')
from go2arx5_vla.world_model.initialization_audit import load_packets

ROOT=Path(__file__).resolve().parents[2]
FAKE=r'''
#include <mujoco/mujoco.h>
#include <fstream>
#include <cstdlib>
extern "C" void mj_step(const mjModel*,mjData* d) {
 std::ofstream f(std::getenv("FAKE_CALLS"),std::ios::app);f<<"FAKE_NOT_PHYSICS\n";
 d->time+=.002;d->qpos[7]=.2;
}
'''
DRIVER=r'''
#include <mujoco/mujoco.h>
#include <signal.h>
#include <cstdio>
#include <dlfcn.h>
void delayed_exit(int) {}
int main(int argc,char**argv) {
 signal(SIGINT,SIG_IGN);
 auto m=mj_loadModel(argv[1],nullptr);if(!m)return 81;
 auto d=mj_makeData(m);for(auto name:{"task_left_slide","task_right_slide"}) {
  int j=mj_name2id(m,mjOBJ_JOINT,name);d->qpos[m->jnt_qposadr[j]]=.042;
 }
 // Ignore the exit request deliberately, as an owner still unwinding would.
 for(int i=0;i<6;i++)mj_step(m,d);
 mj_deleteData(d);mj_deleteModel(m);return 0;
}
'''


def exercise(folder,observer,actuation):
    folder=Path(folder);folder.mkdir();dep=Path(mujoco.__file__).parent;include=str(dep/'include')
    library=next(dep.glob('libmujoco.so.*'));compiler=shutil.which('g++')
    bodies=['<body name="base"><freejoint/><inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/><site name="x5_ee"/>']
    motors=[]
    for i in range(18):
        bodies.append(f'<body><inertial pos="0 0 0" mass=".1" diaginertia=".01 .01 .01"/><joint name="j{i}" range="-.1 .1"/></body>')
        motors.append(f'<motor joint="j{i}"/>')
    for side in ('left','right'):
        bodies.append(f'<body><inertial pos="0 0 0" mass=".1" diaginertia=".01 .01 .01"/><joint name="task_{side}_slide" type="slide" range="0 .044"/></body>')
        motors.append(f'<motor name="task_{side}_motor" joint="task_{side}_slide"/>')
    bodies.append('</body><body><freejoint/><inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/></body>')
    (folder/'fixture.xml').write_text('<mujoco><compiler angle="radian"/><option timestep=".002"/><worldbody>'+''.join(bodies)+'</worldbody><actuator>'+''.join(motors)+'</actuator></mujoco>')
    model=mujoco.MjModel.from_xml_path(str(folder/'fixture.xml'))
    mujoco.mj_saveModel(model,str(folder/'fixture.mjb'),None)
    for name,text in [('observer',observer),('actuation',actuation),('fake',FAKE),('driver',DRIVER)]:
        p=folder/(name+'.cpp');p.write_text(text)
        args=[compiler,'-std=c++17','-O1','-I'+include,'-I'+str(ROOT/'scripts/pi0294'),str(p)]
        if name!='driver':args+=['-shared','-fPIC','-ldl','-o',str(folder/(name+'.so'))]
        else:args+=[str(library),'-ldl','-Wl,-rpath,'+str(dep),'-o',str(folder/'driver')]
        subprocess.run(args,check=True,capture_output=True,text=True)
    (folder/'quota').write_text('10\n')
    (folder/'binding').write_text('GRTQUOTA2\nfixture\nfixture-card\nfixture-owner\n10\n1\n0\n')
    env=dict(os.environ,LD_PRELOAD=':'.join(str(folder/(n+'.so'))for n in ('actuation','observer','fake')),
        PI0288_PHYSICS_TRACE=str(folder/'trace.jsonl'),PI0288_PHYSICS_CAP='10',
        GR00T_TASK_GRIPPER_COMMAND=str(folder/'absent-command'),GR00T_TASK_STATE=str(folder/'state.bin'),
        GR00T_TASK_STEP_QUOTA=str(folder/'quota'),FAKE_CALLS=str(folder/'fake-calls'),
        GR00T_TASK_QUOTA_BINDING=str(folder/'binding'),GR00T_TASK_RUN_UUID='fixture',
        GR00T_TASK_CARD_SHA256='fixture-card',GR00T_TASK_RESERVATION_ID='fixture-owner')
    q=subprocess.run([str(folder/'driver'),str(folder/'fixture.mjb')],env=env,capture_output=True,text=True,timeout=20)
    if q.returncode:raise RuntimeError('C ABI fixture failed exit '+str(q.returncode)+': '+q.stderr)
    return dict(fake_calls=len((folder/'fake-calls').read_text().splitlines()),
        journal_packets=len(load_packets(folder/'state.bin.journal')),
        stop_steps=[x['steps']for x in map(json.loads,(folder/'trace.jsonl').read_text().splitlines())if x['kind']=='stop'])


@unittest.skipUnless(shutil.which('g++'),'C++ compiler required')
class SafetyStopLatch(unittest.TestCase):
    def test_delayed_signal_handler_cannot_add_steps_or_duplicate_transitions(self):
        with tempfile.TemporaryDirectory()as d:
            for name in ('task_actuation.cpp','task_actuation_r1.cpp'):
                with self.subTest(wrapper=name):
                    result=exercise(Path(d)/name,(ROOT/'src/go2arx5_vla/target_mujoco/observer.cpp').read_text(),
                        (ROOT/'scripts/pi0294'/name).read_text())
                    self.assertEqual(result,dict(fake_calls=1,journal_packets=1,stop_steps=[1]))


if __name__=='__main__':unittest.main()
