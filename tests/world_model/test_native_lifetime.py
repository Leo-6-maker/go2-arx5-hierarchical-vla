import unittest,subprocess,tempfile
from pathlib import Path
from go2arx5_vla.world_model.native_lifetime import patch_shutdown

class NativeLifetime(unittest.TestCase):
    def test_patch_moves_cleanup_after_all_readers_and_keeps_control_body(self):
        cpp='''void control(){ ORIGINAL_CONTROLLER(); }
if (!rclcpp::ok()) rclcpp::init(argc, argv);
    ros_thread_ = std::thread([this]() { rclcpp::spin(ros_node_); });
    std::thread physicsthreadhandle(&PhysicsThread, sim.get(), filename.c_str());
    physicsthreadhandle.detach();
RL_Sim::~RL_Sim()
{
    loop_control->shutdown();
    loop_rl->shutdown();
    if (ros_node_) {
        if (rclcpp::ok()) rclcpp::shutdown();
        if (ros_thread_.joinable()) ros_thread_.join();
    }
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}
'''
        hpp='    std::thread ros_thread_;';utils='void PhysicsThread(){ PhysicsLoop(*sim);\n  // delete everything we allocated\n  mj_deleteData(d);\n  mj_deleteModel(m);\n}'
        c,h,u=patch_shutdown(cpp,hpp,utils)
        self.assertIn(cpp.splitlines()[0],c);self.assertNotIn('detach()',c);self.assertIn('std::thread physics_thread_;',h);self.assertIn('rclcpp::SignalHandlerOptions::None',c)
        self.assertLess(c.index('physics_thread_.join()'),c.index('mj_deleteData(d)'));self.assertNotIn('mj_deleteData',u);self.assertIn('PhysicsLoop(*sim)',u)
        for earlier,later in(('loop_rl->shutdown()','ros_executor_->cancel()'),('ros_executor_->cancel()','ros_thread_.join()'),('ros_thread_.join()','PI0294ClockPub.reset()'),('ros_node_.reset()','rclcpp::shutdown()')):self.assertLess(c.index(earlier),c.index(later))
        with self.assertRaises(ValueError):patch_shutdown(c,h,u)

if __name__=='__main__':unittest.main()
