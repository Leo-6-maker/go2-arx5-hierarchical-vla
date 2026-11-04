"""Ordered ROS/MRT teardown only; controller class and all control math stay byte-identical."""
OLD='''int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<Go2X5Ocs2Node>();
    node->LaunchMrt();
    rclcpp::spin(node);
  } catch (const std::exception &e) {
    fprintf(stderr, "go2_x5_ocs2: %s\\n", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
'''
NEW='''#include <csignal>
static volatile std::sig_atomic_t wm02_controller_stop = 0;
static void wm02_controller_signal(int) { wm02_controller_stop = 1; }
int main(int argc, char **argv) {
  std::signal(SIGINT, wm02_controller_signal);
  std::signal(SIGTERM, wm02_controller_signal);
  rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  int result = 0;
  try {
    auto node = std::make_shared<Go2X5Ocs2Node>();
    node->LaunchMrt();
    {
      rclcpp::executors::SingleThreadedExecutor executor;
      executor.add_node(node);
      while (rclcpp::ok() && !wm02_controller_stop)
        executor.spin_once(std::chrono::milliseconds(20));
      executor.remove_node(node);
    }
    // MRT stores this Node's SharedPtr. Explicitly join/release it before
    // dropping our owner; waiting for the Node destructor leaves a cycle.
    node->ShutdownMrt();
    node.reset();
  } catch (const std::exception &e) {
    fprintf(stderr, "go2_x5_ocs2: %s\\n", e.what());
    result = 1;
  }
  rclcpp::shutdown();
  return result;
}
'''

OLD_CLASS='  ~Go2X5Ocs2Node() override { if (mrt_ && launched_) mrt_->shutdownNodes(); }'
NEW_CLASS='''  void ShutdownMrt() {
    if (mrt_ && launched_) mrt_->shutdownNodes();
    launched_ = false;
    // Preserve original reverse member lifetime order, with the context live.
    visualizer_.reset();
    ee_observation_.reset();
    core_.reset();
    mrt_.reset();  // Releases MRT's shared Node owner after its workers join.
  }

  ~Go2X5Ocs2Node() override { ShutdownMrt(); }'''

def unpatch(text):
    return text.replace(NEW,OLD).replace(NEW_CLASS,OLD_CLASS)

def patch(text):
    if text.count(OLD)!=1 or text.count(OLD_CLASS)!=1:raise ValueError('EXACT_FROZEN_CONTROLLER_LIFETIME_ANCHORS_REQUIRED')
    changed=text.replace(OLD,NEW).replace(OLD_CLASS,NEW_CLASS)
    if unpatch(changed)!=text:raise ValueError('CONTROLLER_CLASS_OR_MATH_CHANGED')
    return changed
