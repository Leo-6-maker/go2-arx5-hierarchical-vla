import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from go2arx5_vla.world_model.native_clock_key import patch, CALLBACK, SERIALIZED, TOGGLE, APPLIED


class NativeClockKey(unittest.TestCase):
    def test_only_delivery_and_observation_change_not_controller_math(self):
        old = ('const std::lock_guard<std::recursive_mutex> lock(sim->mtx);\n'
            'FROZEN_CONTROL_MATH();\n'+TOGGLE+'\n    this->control.ClearInput();\n'+CALLBACK)
        new = patch(old)
        self.assertEqual(new.replace(SERIALIZED, CALLBACK).replace(TOGGLE+APPLIED, TOGGLE), old)
        with self.assertRaises(ValueError): patch(new)
        with self.assertRaises(ValueError): patch(old+CALLBACK)

    @unittest.skipUnless(shutil.which('g++'), 'compiler required')
    def test_event_between_key_test_and_clear_is_lost_without_serialization(self):
        # Deliberately schedule the old interleaving without a C++ data race.
        # The real Control::ClearInput restores last_keyboard, so a newly
        # received key can vanish after the consumer already tested it.
        code = r'''
#include <mutex>
#include <thread>
#include <atomic>
struct Control {
 int current=0,last=0;
 void SetKeyboard(int key){if(current!=key){last=current;current=key;}}
 void ClearInput(){current=last;}
};
int main(){Control old;bool consumed=old.current==1;
 old.SetKeyboard(1);old.ClearInput();
 if(consumed||old.current!=0)return 1;
 Control fixed;std::recursive_mutex m;std::atomic<bool> attempted{false};
 std::unique_lock<std::recursive_mutex> pd(m);
 std::thread callback([&]{attempted=true;std::lock_guard<std::recursive_mutex> key(m);fixed.SetKeyboard(1);});
 while(!attempted.load())std::this_thread::yield();
 if(fixed.current!=0)return 2;
 fixed.ClearInput();pd.unlock();callback.join();
 std::lock_guard<std::recursive_mutex> next_pd(m);
 if(fixed.current!=1)return 3;
 fixed.ClearInput();if(fixed.current!=0)return 4;
 return 0;}
'''
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'test.cpp').write_text(code)
            subprocess.run(['g++', '-std=c++17', '-pthread', str(root/'test.cpp'), '-o', str(root/'test')], capture_output=True, check=True)
            subprocess.run([str(root/'test')], capture_output=True, check=True, timeout=5)


if __name__ == '__main__': unittest.main()
