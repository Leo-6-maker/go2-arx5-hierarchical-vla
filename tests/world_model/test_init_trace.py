import json,os,struct,subprocess,tempfile,unittest
from pathlib import Path
from go2arx5_vla.world_model.init_trace_reader import read_trace,summarize
from go2arx5_vla.world_model.init_candidate import marked,remove_trace,end_function

ROOT=Path(__file__).resolve().parents[2]
HARNESS=r'''
#include "init_trace.hpp"
#include <future>
#include <deque>
#include <cassert>
struct Q{std::deque<std::vector<float>> data;bool try_pop(std::vector<float>&v){if(data.empty())return false;v=data.front();data.pop_front();return true;}};
int main(){
 std::promise<void> begin,release;auto r=release.get_future().share();
 auto infer=std::async(std::launch::async,[&]{WM02InitBegin(0);begin.set_value();r.wait();std::vector<float>v(1024,3.25f);wm02_init::emit(wm02_init::HISTORY,0,WM02InitCurrent(),v);wm02_init::emit(wm02_init::INFERENCE_DONE,0,WM02InitCurrent());});
 begin.get_future().wait();double x[3]={1.,2.,3.};WM02InitNumeric(wm02_init::PD_STATE_CTRL,.002,0,x,3);release.set_value();infer.get();
 Q pos,vel;std::vector<float>v;WM02InitPublish(0,1);pos.data.push_back({1});
 bool applied=wm02_init::pop(pos,v,0)&&wm02_init::pop(vel,v,1);assert(!applied);
 WM02InitPublish(0,2);pos.data.push_back({2});WM02InitPublish(1,1);vel.data.push_back({1});
 if(wm02_init::pop(pos,v,0)&&wm02_init::pop(vel,v,1))WM02InitApplied();else return 7;
 WM02InitNumeric(wm02_init::STEP_AFTER,.004,1,x,3);wm02_init::emit(wm02_init::STOP,.004,1);wm02_init::emit(wm02_init::REFUSED,.004,1);return 0;
}
'''
class InitTrace(unittest.TestCase):
    def test_bounded_trace_keeps_PD_during_inference_and_actual_queue_versions(self):
        with tempfile.TemporaryDirectory()as tmp:
            d=Path(tmp);h=d/'harness.cpp';h.write_text(HARNESS);binary=d/'harness';trace=d/'trace'
            subprocess.run(['g++','-std=c++17','-O2','-pthread','-I'+str(ROOT/'src/go2arx5_vla/target_mujoco'),str(h),str(ROOT/'src/go2arx5_vla/target_mujoco/init_trace.cpp'),'-o',str(binary)],check=True,capture_output=True)
            subprocess.run([str(binary)],env=dict(os.environ,WM02_INIT_TRACE=str(trace)),check=True,timeout=10)
            rows=read_trace(trace);kinds=[r['kind']for r in rows];self.assertLess(kinds.index(2),kinds.index(10));self.assertLess(kinds.index(10),kinds.index(5))
            applied=next(r for r in rows if r['kind']==8);self.assertEqual((applied['applied_position_version'],applied['applied_velocity_version']),(2,1))
            s=summarize(trace);self.assertEqual(s['integrations_after_first_STOP'],0);self.assertEqual(s['writer']['dropped'],0)
            snapshot=Path(str(trace)+'.snapshots').read_bytes();g,slot,token,kind,n=struct.unpack('<5Q',snapshot[:40]);self.assertEqual((kind,n),(4,1024));self.assertEqual(struct.unpack('<1024f',snapshot[40:]),(3.25,)*1024)
    def test_patch_stripping_preserves_controller_text_and_reader_rejects_truncation(self):
        source='void X() { const char* s="}"; /* { */\n}\n';patched=end_function(source,'void X()', 'int trace_only=1;');self.assertEqual(remove_trace(patched),source)
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'trace';p.write_bytes(b'WM02IT01'+struct.pack('<II',840,65536)+b'partial')
            with self.assertRaises(ValueError):read_trace(p)
if __name__=='__main__':unittest.main()
