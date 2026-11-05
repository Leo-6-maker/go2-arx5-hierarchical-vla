import unittest,tempfile,subprocess,shutil,signal
from pathlib import Path
from go2arx5_vla.world_model.native_controller_lifetime import patch,unpatch,OLD,OLD_CLASS

class ControllerLifetime(unittest.TestCase):
    @unittest.skipUnless(shutil.which('g++'),'compiler required')
    def test_shared_node_cycle_is_broken_and_worker_joined_before_context_shutdown(self):
        # Reproduce MRT's actual SharedPtr back to the Node. A Node destructor
        # alone cannot join a worker while this ownership cycle remains live.
        header='''#include <atomic>
#include <thread>
#include <chrono>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <exception>
static std::atomic<bool> context{true},running{false};
class Go2X5Ocs2Node;
struct MRT {std::shared_ptr<Go2X5Ocs2Node> node_;std::thread worker;void start(){running=true;worker=std::thread([]{while(running){if(!context)std::abort();std::this_thread::yield();}});}void shutdownNodes(){if(!context)std::abort();running=false;if(worker.joinable())worker.join();}~MRT(){shutdownNodes();}};
struct NodeBase{virtual ~NodeBase()=default;};
class Go2X5Ocs2Node:public NodeBase,public std::enable_shared_from_this<Go2X5Ocs2Node>{
public:std::unique_ptr<MRT>mrt_=std::make_unique<MRT>();std::unique_ptr<int>visualizer_,ee_observation_,core_;bool launched_=false;
void LaunchMrt(){mrt_->node_=shared_from_this();mrt_->start();launched_=true;puts("READY");fflush(stdout);}
'''+OLD_CLASS+'''
};
namespace rclcpp{struct InitOptions{};enum class SignalHandlerOptions{None};void init(int,char**,InitOptions,SignalHandlerOptions){}bool ok(){return context;}void shutdown(){if(running)std::abort();context=false;}namespace executors{struct SingleThreadedExecutor{void add_node(std::shared_ptr<Go2X5Ocs2Node>){}void remove_node(std::shared_ptr<Go2X5Ocs2Node>){}void spin_once(std::chrono::milliseconds t){std::this_thread::sleep_for(t);}};}}
'''
        original=header+OLD;changed=patch(original);self.assertEqual(unpatch(changed),original)
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
