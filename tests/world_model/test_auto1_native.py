import unittest,tempfile,subprocess,shutil,itertools,sys,json
from pathlib import Path
from go2arx5_vla.world_model.auto1_native import phase_limit,task_observer_source,validate_parent_loader
from scripts.pi0294.task_contacts import allowed
from go2arx5_vla.world_model.target_sequential import native_display_command,TargetSequentialAdapter

class Auto1Native(unittest.TestCase):
    @unittest.skipUnless(shutil.which('g++'),'C++ compiler required')
    def test_original_quota_validator_allows_three_init_ticks_but_never_four(self):
        import os
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp);(p/'q.cpp').write_text('#include "task_quota_r1.hpp"\nint main(){uint64_t bound=0;return gr00t_quota_r1::read_bound(bound)?0:1;}\n')
            subprocess.run(['g++','-std=c++17','-I'+str(root/'scripts/pi0294'),str(p/'q.cpp'),'-o',str(p/'q')],check=True,capture_output=True)
            env=dict(os.environ,GR00T_TASK_QUOTA_BINDING=str(p/'binding'),GR00T_TASK_RUN_UUID='fixture',GR00T_TASK_CARD_SHA256='card',GR00T_TASK_RESERVATION_ID='owner')
            for bound,warmup,expected in((2503,5,0),(2504,5,1),(2503,6,1)):
                (p/'binding').write_text(f'GRTQUOTA2\nfixture\ncard\nowner\n{bound}\n{warmup}\n0\n')
                self.assertEqual(subprocess.run([str(p/'q')],env=env).returncode,expected)
    def test_shutdown_keeps_display_alive_until_our_simulator_child_exits(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);flag=root/'display-survived'
            code="import subprocess,sys;from pathlib import Path;display=subprocess.Popen(['/usr/bin/tail','-f','/dev/null']);sim=subprocess.Popen(['/usr/bin/sleep','100'],start_new_session=True);print('READY',flush=True);sim.wait();Path(sys.argv[1]).write_text(str(display.poll()is None));display.terminate();display.wait()"
            p=subprocess.Popen([sys.executable,'-c',code,str(flag)],stdout=subprocess.PIPE,text=True,start_new_session=True)
            try:
                self.assertEqual(p.stdout.readline().strip(),'READY');adapter=TargetSequentialAdapter(dict(native_binary='/usr/bin/sleep'),root);adapter.processes=[('sim',p)];adapter.close()
                self.assertEqual(flag.read_text(),'True');self.assertEqual(p.returncode,0);self.assertEqual(adapter.exit_records[0]['signals'],['SIGINT']);self.assertIsNone(adapter.exit_records[0]['display_child_lookup_error'])
            finally:
                if p.poll()is None:p.kill();p.wait()
                p.stdout.close()
    def test_preload_reaches_only_simulator_not_display_shell_and_helpers(self):
        s=dict(xvfb_run='/private/xvfb-run',outer_library='/outer',observer_library='/observer',trace_library='/trace',native_binary='/sim')
        argv,env=native_display_command(s,'/run',dict(LD_PRELOAD='bad',WM02_INIT_TRACE='bad',OTHER='kept'))
        self.assertNotIn('LD_PRELOAD',env);self.assertNotIn('WM02_INIT_TRACE',env);self.assertEqual(env['OTHER'],'kept')
        self.assertEqual(argv,['/private/xvfb-run','-a','/usr/bin/env','LD_PRELOAD=/outer:/observer:/trace','WM02_INIT_TRACE=/run/init.trace','WM02_INIT_SEAL_REQUEST=/run/trace_seal_request','/sim','go2_x5','scene'])
    def test_ROS_parent_loader_must_be_bound_at_exec_not_only_for_future_children(self):
        validate_parent_loader('/existing/ros:/existing/native','/existing/ros:/existing/native')
        for value in(None,'','/other'):
            with self.assertRaises(PermissionError):validate_parent_loader('/existing/ros:/existing/native',value)
    def test_import_and_help_do_not_construct_or_import_native_models(self):
        code="import sys;from scripts.wm02.commission_target import main;from go2arx5_vla.world_model.target_sequential import TargetSequentialAdapter;assert not {'rclpy','mujoco','torch'}&sys.modules.keys()"
        subprocess.run([sys.executable,'-c',code],check=True)
        subprocess.run([sys.executable,'-m','scripts.wm02.commission_target','--help'],check=True,capture_output=True)
    def test_phases_never_skip_failed_or_missing_native_stage(self):
        self.assertEqual(phase_limit('cold-1',None),50)
        with self.assertRaises(PermissionError):phase_limit('g2-reference-qualification',None)
        self.assertEqual(phase_limit('g2-reference-qualification',dict(phase='hold-5s-2',status='NATIVE_STAGE_PASSED',shutdown_clean=True,owned_process_exits=[dict(exit_code=0)])),2503)
        with self.assertRaises(PermissionError):phase_limit('g2-reference-qualification',dict(phase='hold-5s-2',status='NATIVE_STAGE_PASSED',shutdown_clean=False))
        self.assertEqual(phase_limit('cold-2',dict(phase='cold-1',status='NATIVE_STAGE_PASSED')),50)
        with self.assertRaises(PermissionError):phase_limit('cold-2',dict(phase='cold-1',status='NATIVE_STAGE_PASSED',owned_process_exits=[dict(exit_code=139)]))
        for previous in(None,dict(phase='cold-1',status='FAIL'),dict(phase='cold-2',status='NATIVE_STAGE_PASSED')):
            with self.assertRaises(PermissionError):phase_limit('hold-2s-2',previous)
        with self.assertRaises(PermissionError):phase_limit('unknown',None)
    @unittest.skipUnless(shutil.which('g++'),'C++ compiler required')
    def test_restored_cpp_rule_matches_original_task_pairs_and_penetration_in_both_orders(self):
        snippet=task_observer_source('    illegal_contact|=!allowed;')
        source='''#include <string>
#include <cstdio>
int main(int argc,char**argv){const char*a=argv[1],*b=argv[2];
bool allowed=(std::string(a)=="floor"&&(std::string(b)=="FL"||std::string(b)=="FR"||std::string(b)=="RL"||std::string(b)=="RR"))||(std::string(b)=="floor"&&(std::string(a)=="FL"||std::string(a)=="FR"||std::string(a)=="RL"||std::string(a)=="RR"));
struct C{double dist;};struct D{C contact[1];};D data{{{std::stod(argv[3])}}};D*d=&data;int i=0;bool illegal_contact=false;
'''+snippet+'\nreturn illegal_contact?1:0;}\n'
        names=['floor','FL','FR','RL','RR','task_table','task_cube_geom','task_left_pad','task_right_pad','x5_arm']
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp);(p/'rule.cpp').write_text(source);subprocess.run(['g++','-std=c++17',str(p/'rule.cpp'),'-o',str(p/'rule')],check=True,capture_output=True)
            for a,b in itertools.product(names,repeat=2):
                q=subprocess.run([str(p/'rule'),a,b,'0.0'],check=False)
                self.assertEqual(q.returncode,0 if allowed(a,b)else 1,(a,b))
            for a,b in [('task_table','task_cube_geom'),('task_cube_geom','task_left_pad')]:
                self.assertEqual(subprocess.run([str(p/'rule'),a,b,'-0.005']).returncode,1)

if __name__=='__main__':unittest.main()
