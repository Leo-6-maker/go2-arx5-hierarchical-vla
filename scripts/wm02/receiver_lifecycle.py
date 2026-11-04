"""Stop the owned receiver executor before destroying its ROS context."""
import signal


def spin_until_stopped(executor, ok, stopped):
    # Callback errors remain fatal. Only an explicitly delivered owned signal
    # ends this loop; a RuntimeError is never converted into a clean shutdown.
    while ok() and not stopped():
        executor.spin_once(timeout_sec=.05)


def run(rclpy, factory):
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    previous = {sig: signal.signal(sig, request_stop)
                for sig in (signal.SIGINT, signal.SIGTERM)}
    node = executor = None
    try:
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        node = factory()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin_until_stopped(executor, rclpy.ok, lambda: stop)
    finally:
        try:
            if executor is not None:
                executor.shutdown()
                if node is not None:
                    executor.remove_node(node)
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
