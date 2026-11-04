"""Keep existing native mjData alive until every owned reader has stopped.

Changes shutdown ownership only. Normal controller, solver and physics-loop
code is left byte-for-byte unchanged by this patch.
"""

def patch_shutdown(cpp,hpp,utils):
    start='    std::thread physicsthreadhandle(&PhysicsThread, sim.get(), filename.c_str());\n    physicsthreadhandle.detach();'
    member='    std::thread ros_thread_;'
    destroy='    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;'
    cleanup='  // delete everything we allocated\n  mj_deleteData(d);\n  mj_deleteModel(m);'
    for text,needle in((cpp,start),(hpp,member),(cpp,destroy),(utils,cleanup)):
        if text.count(needle)!=1:raise ValueError('Exact frozen native lifetime anchor required:'+needle[:60])
    cpp=cpp.replace(start,'    physics_thread_ = std::thread(&PhysicsThread, sim.get(), filename.c_str());')
    hpp=hpp.replace(member,member+'\n    std::thread physics_thread_;\n    std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> ros_executor_;')
    # exitrequest is already set on normal UI exit; repeat it explicitly at the
    # destructor boundary before stopping PD, policy and ROS readers.
    anchor='RL_Sim::~RL_Sim()\n{'
    if cpp.count(anchor)!=1:raise ValueError('Exact native destructor required')
    cpp=cpp.replace(anchor,anchor+'\n    sim->exitrequest.store(1);')
    init='if (!rclcpp::ok()) rclcpp::init(argc, argv);'
    if cpp.count(init)!=1:raise ValueError('Exact native ROS initialization required')
    # Keep the existing native SIGINT exitrequest handler. The default ROS
    # handler otherwise shuts its context down concurrently with live PD
    # publishers, before this object's ordered reader shutdown can run.
    cpp=cpp.replace(init,'if (!rclcpp::ok()) rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);')
    spin='    ros_thread_ = std::thread([this]() { rclcpp::spin(ros_node_); });'
    shutdown='''    if (ros_node_) {
        if (rclcpp::ok()) rclcpp::shutdown();
        if (ros_thread_.joinable()) ros_thread_.join();
    }'''
    if cpp.count(spin)!=1 or cpp.count(shutdown)!=1:raise ValueError('Exact native ROS executor lifecycle required')
    cpp=cpp.replace(spin,'''    ros_executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    ros_executor_->add_node(ros_node_);
    ros_thread_ = std::thread([this]() { ros_executor_->spin(); });''')
    # The global /clock publisher otherwise survives until process-static
    # destruction, after its context/participant. Dispose every native ROS
    # handle with its context still alive, after stopping callback readers.
    cpp=cpp.replace(shutdown,'''    if (physics_thread_.joinable()) physics_thread_.join();
    if (ros_node_) {
        ros_executor_->cancel();
        if (ros_thread_.joinable()) ros_thread_.join();
        ros_executor_->remove_node(ros_node_);
        ros_executor_.reset();
        PI0294ClockPub.reset();
        ros_arm_state_pub_.reset(); ros_arm_mode_pub_.reset();
        ros_arm_target_pub_.reset(); ros_odom_pub_.reset();
        ros_odom_mocap_pub_.reset(); ros_gait_phase_pub_.reset();
        ros_arm_command_sub_.reset(); ros_arm_mode_sub_.reset();
        ros_fsm_key_sub_.reset(); ros_base_command_sub_.reset();
        ros_node_.reset();
        if (rclcpp::ok()) rclcpp::shutdown();
    }''')
    cpp=cpp.replace(destroy,'''    if (physics_thread_.joinable()) physics_thread_.join();
    mj_deleteData(d); d = nullptr;
    mj_deleteModel(m); m = nullptr;
'''+destroy)
    utils=utils.replace(cleanup,'  // RL_Sim owns cleanup after joining physics, PD, policy and ROS readers.')
    return cpp,hpp,utils
