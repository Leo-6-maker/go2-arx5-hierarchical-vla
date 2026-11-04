import unittest,tempfile,subprocess,shutil,itertools,sys,json
from pathlib import Path
from go2arx5_vla.world_model.auto1_native import phase_limit,task_observer_source,validate_parent_loader
from scripts.pi0294.task_contacts import allowed

class Auto1Native(unittest.TestCase):
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
        self.assertEqual(phase_limit('cold-2',dict(phase='cold-1',status='NATIVE_STAGE_PASSED')),50)
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
