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
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    while (rclcpp::ok() && !wm02_controller_stop)
      executor.spin_once(std::chrono::milliseconds(20));
    executor.remove_node(node);
    node.reset();  // Join MRT workers while the ROS context is still valid.
  } catch (const std::exception &e) {
    fprintf(stderr, "go2_x5_ocs2: %s\\n", e.what());
    result = 1;
  }
  rclcpp::shutdown();
  return result;
}
'''

def patch(text):
    if text.count(OLD)!=1:raise ValueError('EXACT_FROZEN_CONTROLLER_MAIN_REQUIRED')
    changed=text.replace(OLD,NEW)
    if changed.replace(NEW,OLD)!=text:raise ValueError('CONTROLLER_CLASS_OR_MATH_CHANGED')
    return changed
