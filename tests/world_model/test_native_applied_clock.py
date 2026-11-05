"""Application ordering regressions using the actual C++ clock headers."""
import shutil,subprocess,tempfile,unittest
from pathlib import Path
from go2arx5_vla.world_model.native_applied_clock import patch,INCLUDE
from go2arx5_vla.world_model.native_control_clock import BEGIN


class AppliedClock(unittest.TestCase):
    def test_patch_preserves_PD_policy_and_physics_formulas(self):
        cpp=(BEGIN+'POLICY_MATH\n        wm02_command_pair::publish([&]() {\nPAIR_PUSHES\n        });\n'
             '        // this->TorqueProtect(this->output_dof_tau);\n}\n'
             '    this->SetCommand(&this->robot_command);\n            wm02_cold_history_ready_ = false;')
        sdk='        WM02InitApplied();\nORIGINAL_TARGET_ASSIGNMENTS\n'
        utils=('            // run single step, let next iteration deal with timing\n            mj_step(m, d);\n'
               '              // call mj_step\n              mj_step(m, d);')
        a,s,b=patch(cpp,sdk,utils)
        self.assertIn('POLICY_MATH',a);self.assertIn('PAIR_PUSHES',a)
        self.assertIn('ORIGINAL_TARGET_ASSIGNMENTS',s)
        self.assertEqual(b.count('mj_step(m, d);'),2)
        self.assertTrue(a.startswith(INCLUDE));self.assertTrue(s.startswith(INCLUDE))
        with self.assertRaises(ValueError):patch(cpp,sdk+sdk,utils)
        with self.assertRaises(ValueError):patch(cpp.replace('PAIR_PUSHES','PAIR_PUSHES\n        wm02_command_pair::publish([&]() {\n'),sdk,utils)

    @unittest.skipUnless(shutil.which('g++'),'compiler required')
    def test_slow_inference_and_delayed_consumption_cannot_advance_physics(self):
        headers=Path(__file__).resolve().parents[2]/'src/go2arx5_vla/target_mujoco'
        source=r'''
#include "applied_control_clock.hpp"
#include <future>
#include <atomic>
int main(){
 const long pd=5000000,pp=20000000;
 // Reproduce the recorded V1 ordering defect without a simulator.
 wm02_control_clock::Clock old;old.control(0,pd);old.policy_due(0,pp);old.policy_complete(0,pp);
 if(!old.ready(0))return 1;
 wm02_applied_control_clock::Clock c;c.control(0,pd);
 if(!c.policy_due(0,pp)||c.ready(0))return 2;
 std::promise<void> release;auto wait=release.get_future();
 auto infer=std::async(std::launch::async,[&]{wait.wait();c.published(0,pp);c.policy_complete(0,pp);});
 // PD is independent and remains callable throughout an unfinished inference.
 for(int i=0;i<10000;++i){c.control(0,pd);if(c.ready(0))return 3;}
 release.set_value();infer.get();
 if(c.ready(0))return 4; // Done + published is still not applied.
 c.applied();c.control(0,pd);
 if(!c.ready(0)||!c.ready(.002)||!c.ready(.004)||c.ready(.006))return 5;
 c.control(.006,pd);if(!c.ready(.006)||c.ready(.010))return 6;
 c.control(.020,pd);if(c.ready(.020)||!c.policy_due(.020,pp))return 7;
 c.published(.020,pp);c.policy_complete(.020,pp);
 for(int i=0;i<10000;++i){c.control(.020,pd);if(c.ready(.020))return 8;}
 c.applied();c.control(.020,pd);if(!c.ready(.020)||c.policy_due(.020,pp))return 9;
 // The first consumer may run before the inference completion mark.
 c.control(.040,pd);c.policy_due(.040,pp);c.published(.040,pp);c.applied();
 if(c.ready(.040))return 10;
 c.policy_complete(.040,pp);if(!c.ready(.040))return 11;
 c.reset();if(c.ready(0)||!c.policy_due(0,pp))return 12;
 c.control(0,pd);c.published(0,pp);c.policy_complete(0,pp);
 if(c.ready(0))return 13;c.applied();if(!c.ready(0))return 14;
 return 0;
}
'''
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);(root/'t.cpp').write_text(source)
            subprocess.run(['g++','-std=c++17','-pthread','-O2','-I'+str(headers),str(root/'t.cpp'),'-o',str(root/'test')],check=True,capture_output=True)
            subprocess.run([str(root/'test')],check=True,timeout=8)

if __name__=='__main__':unittest.main()
