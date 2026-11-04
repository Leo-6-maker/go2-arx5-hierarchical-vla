import unittest,tempfile,subprocess,shutil
from pathlib import Path
from go2arx5_vla.world_model.native_command_pair import patch,producer_block,CONSUMER


class NativeCommandPair(unittest.TestCase):
    def test_exact_seam_does_not_edit_target_or_PD_math(self):
        cpp='POLICY_MATH\n'+producer_block()+'TAU_MATH\n';sdk='if ('+CONSUMER+') { ORIGINAL_PD_MATH; }'
        a,b=patch(cpp,sdk)
        self.assertIn('POLICY_MATH',a);self.assertIn('TAU_MATH',a);self.assertIn('ORIGINAL_PD_MATH',b)
        with self.assertRaises(ValueError):patch(cpp+producer_block(),sdk)

    @unittest.skipUnless(shutil.which('g++'),'compiler required')
    def test_adversarial_split_publication_and_two_translation_units(self):
        header=Path(__file__).resolve().parents[2]/'src/go2arx5_vla/target_mujoco/command_pair.hpp'
        # A producer is deliberately held between its position and velocity
        # queue writes. PD must return without taking just the first channel.
        source=r'''
#include "command_pair.hpp"
#include <atomic>
#include <thread>
#include <queue>
extern std::mutex* other_mutex();
int main(){
 if(other_mutex()!=&wm02_command_pair::mutex())return 10;
 std::queue<int> q,v;std::atomic<bool> first{false},release{false};
 std::thread producer([&]{wm02_command_pair::publish([&]{q.push(7);first=true;while(!release.load())std::this_thread::yield();v.push(7);});});
 while(!first.load())std::this_thread::yield();
 bool called=false;
 bool accepted=wm02_command_pair::consume([&]{called=true;q.pop();return true;});
 if(accepted||called)return 1;
 release=true;producer.join();int a=0,b=0;
 if(!wm02_command_pair::consume([&]{a=q.front();q.pop();b=v.front();v.pop();return true;}))return 2;
 if(a!=7||a!=b)return 3;
 for(int i=0;i<10000;++i){wm02_command_pair::publish([&]{q.push(i);v.push(i);});
  if(!wm02_command_pair::consume([&]{a=q.front();q.pop();b=v.front();v.pop();return true;})||a!=b)return 4;}
 return 0;
}
'''
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp);(p/'main.cpp').write_text(source);(p/'other.cpp').write_text('#include "command_pair.hpp"\nstd::mutex* other_mutex(){return &wm02_command_pair::mutex();}\n')
            subprocess.run(['g++','-std=c++17','-pthread','-I'+str(header.parent),str(p/'main.cpp'),str(p/'other.cpp'),'-o',str(p/'test')],check=True,capture_output=True)
            subprocess.run([str(p/'test')],check=True,timeout=5)
