import unittest,subprocess,tempfile,shutil
from pathlib import Path
from go2arx5_vla.world_model.native_control_clock import patch,BEGIN


class NativePhysicalClock(unittest.TestCase):
    def test_patch_preserves_original_callback_and_integrator_text(self):
        cpp=BEGIN+'POLICY_MATH\n        // this->TorqueProtect(this->output_dof_tau);\n}\n    this->SetCommand(&this->robot_command);\n            wm02_cold_history_ready_ = false;'
        utils='            // run single step, let next iteration deal with timing\n            mj_step(m, d);\n              // call mj_step\n              mj_step(m, d);'
        a,b=patch(cpp,utils);self.assertIn('POLICY_MATH',a);self.assertEqual(b.count('mj_step(m, d);'),2)
        with self.assertRaises(ValueError):patch(cpp+BEGIN,utils)

    @unittest.skipUnless(shutil.which('g++'),'compiler required')
    def test_frozen_5ms_PD_20ms_policy_2ms_physics_no_duplicate_history(self):
        header=Path(__file__).resolve().parents[2]/'src/go2arx5_vla/target_mujoco/control_clock.hpp'
        source=r'''
#include "control_clock.hpp"
int main(){wm02_control_clock::Clock c;const long pd=5000000,policy=20000000;
 if(c.ready(0))return 1;c.control(0,pd);
 if(!c.policy_due(0,policy)||c.ready(0))return 2;
 for(int i=0;i<1000;++i)if(c.policy_due(0,policy))return 3;
 c.policy_complete(0,policy);
 if(!c.ready(0)||!c.ready(.002)||!c.ready(.004)||c.ready(.006))return 4;
 c.control(.006,pd);if(!c.ready(.006)||!c.ready(.008)||c.ready(.010))return 5;
 c.control(.020,pd);if(c.ready(.020)||!c.policy_due(.020,policy))return 6;
 // PD can continue recording real feedback during in-flight inference.
 c.control(.020,pd);if(c.ready(.020))return 7;
 c.policy_complete(.020,policy);if(!c.ready(.020)||c.policy_due(.020,policy))return 8;
 c.reset();if(c.ready(0)||!c.policy_due(0,policy))return 9;
 return 0;}
'''
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp);(p/'t.cpp').write_text(source)
            subprocess.run(['g++','-std=c++17','-pthread','-I'+str(header.parent),str(p/'t.cpp'),'-o',str(p/'test')],check=True,capture_output=True)
            subprocess.run([str(p/'test')],check=True,timeout=5)
