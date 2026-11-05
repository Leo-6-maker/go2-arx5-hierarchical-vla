import unittest,tempfile,subprocess,shutil
from pathlib import Path
from go2arx5_vla.world_model.native_cold_history import patch

class NativeColdHistory(unittest.TestCase):
    @unittest.skipUnless(shutil.which('g++'),'compiler required')
    def test_first_real_observation_reset_once_and_reset_boundary_rearms(self):
        insert='        this->history_obs_buf.insert(clamped_obs);';reset='            PI0289ResetStartup(this->mj_model, this->mj_data);';member='    std::recursive_mutex task_model_pause_mutex_;'
        cpp='void first(std::vector<float> clamped_obs){\n'+insert+'\n}\nvoid reset(){\n'+reset+'\n}'
        hpp=member;changed,header,utils=patch(cpp,hpp,'untouched utils');self.assertEqual(utils,'untouched utils')
        source='''#include <vector>
#include <mutex>
struct Buffer {int resets=0,inserts=0;std::vector<float> last;void reset(std::vector<int>,const std::vector<float>&v){++resets;last=v;}void insert(const std::vector<float>&v){++inserts;last=v;}};
void PI0289ResetStartup(void*,void*){}
struct X {Buffer history_obs_buf;void*mj_model=nullptr,*mj_data=nullptr;
'''+header+'\n'+changed+'''};
int main(){X x;x.first({1,2});x.first({3,4});if(x.history_obs_buf.resets!=1||x.history_obs_buf.inserts!=2)return 1;x.reset();x.first({5,6});return x.history_obs_buf.resets==2&&x.history_obs_buf.last[0]==5?0:2;}
'''
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp);(p/'test.cpp').write_text(source);subprocess.run(['g++','-std=c++17',str(p/'test.cpp'),'-o',str(p/'test')],check=True,capture_output=True);subprocess.run([str(p/'test')],check=True)
        with self.assertRaises(ValueError):patch(cpp+insert,hpp,utils)
if __name__=='__main__':unittest.main()
