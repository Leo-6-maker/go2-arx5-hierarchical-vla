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
    hpp=hpp.replace(member,member+'\n    std::thread physics_thread_;')
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
    cpp=cpp.replace(destroy,'''    if (physics_thread_.joinable()) physics_thread_.join();
    mj_deleteData(d); d = nullptr;
    mj_deleteModel(m); m = nullptr;
'''+destroy)
    utils=utils.replace(cleanup,'  // RL_Sim owns cleanup after joining physics, PD, policy and ROS readers.')
    return cpp,hpp,utils
