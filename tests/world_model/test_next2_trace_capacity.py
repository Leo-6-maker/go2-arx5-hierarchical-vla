"""Exercise the finite trace boundary which failed the actual20s candidate."""
import unittest,tempfile,subprocess,os,json
from pathlib import Path
from go2arx5_vla.world_model.init_trace_reader import read_trace
ROOT=Path(__file__).resolve().parents[2]
class ExtendedTrace(unittest.TestCase):
    def test_70000_events_are_complete_in_explicit_next2_capacity(self):
        with tempfile.TemporaryDirectory()as tmp:
            r=Path(tmp);source=r/'h.cpp';source.write_text('#include "init_trace.hpp"\nint main(){for(int i=0;i<70000;++i)WM02InitEmit(16,0,i,nullptr,0);}\n');exe=r/'h';trace=r/'trace'
            subprocess.run(['g++','-std=c++17','-O2','-pthread','-DWM02_INIT_TRACE_CAPACITY=131072','-I'+str(ROOT/'src/go2arx5_vla/target_mujoco'),str(source),str(ROOT/'src/go2arx5_vla/target_mujoco/init_trace.cpp'),'-o',str(exe)],check=True,capture_output=True)
            subprocess.run([str(exe)],env=dict(os.environ,WM02_INIT_TRACE=str(trace)),check=True,timeout=10);s=json.loads(Path(str(trace)+'.summary').read_text());self.assertEqual(s['events'],70000);self.assertEqual(s['written'],70000);self.assertEqual(s['dropped'],0);self.assertEqual(len(read_trace(trace)),70000)
if __name__=='__main__':unittest.main()
