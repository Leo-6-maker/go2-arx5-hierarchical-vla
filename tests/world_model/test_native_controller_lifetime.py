import unittest,tempfile,subprocess,shutil,signal
from pathlib import Path
from go2arx5_vla.world_model.native_controller_lifetime import patch,OLD

class ControllerLifetime(unittest.TestCase):
    @unittest.skipUnless(shutil.which('g++'),'compiler required')
    def test_signal_joins_owned_worker_before_invalidating_context(self):
        header='''#include <atomic>
#include <thread>
#include <chrono>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <exception>
static std::atomic<bool> context{true},running{false};
class Go2X5Ocs2Node{std::thread worker;public:void LaunchMrt(){running=true;worker=std::thread([]{while(running){if(!context)std::abort();std::this_thread::yield();}});puts("READY");fflush(stdout);}~Go2X5Ocs2Node(){if(!context)std::abort();running=false;if(worker.joinable())worker.join();}};
namespace rclcpp{struct InitOptions{};enum class SignalHandlerOptions{None};void init(int,char**,InitOptions,SignalHandlerOptions){}bool ok(){return context;}void shutdown(){context=false;}namespace executors{struct SingleThreadedExecutor{void add_node(std::shared_ptr<Go2X5Ocs2Node>){}void remove_node(std::shared_ptr<Go2X5Ocs2Node>){}void spin_once(std::chrono::milliseconds t){std::this_thread::sleep_for(t);}};}}
'''
        changed=patch(header+OLD);self.assertEqual(changed[:len(header)],header)
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp);(p/'test.cpp').write_text(changed);subprocess.run(['g++','-std=c++17','-pthread',str(p/'test.cpp'),'-o',str(p/'test')],check=True,capture_output=True)
            for sig in(signal.SIGINT,signal.SIGTERM):
                q=subprocess.Popen([str(p/'test')],stdout=subprocess.PIPE,text=True)
                try:self.assertEqual(q.stdout.readline().strip(),'READY');q.send_signal(sig);self.assertEqual(q.wait(timeout=3),0)
                finally:
                    if q.poll()is None:q.kill();q.wait()
                    q.stdout.close()
        with self.assertRaises(ValueError):patch(header+OLD+OLD)
if __name__=='__main__':unittest.main()
