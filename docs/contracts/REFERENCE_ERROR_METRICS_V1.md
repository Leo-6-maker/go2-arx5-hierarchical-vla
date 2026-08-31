# Reference and tracking error metrics V1

The evaluation API keeps three named measurements separate:

`REFERENCE_RECONSTRUCTION_ERROR` compares the original continuous desired EEF
trajectory with the trajectory reconstructed from an N-point chunk or WBC
preview. It measures interface/discretization loss before policy execution.

`CONTROLLER_TRACKING_ERROR` compares the exact reference consumed by WBC with
the achieved EEF pose. It measures controller/simulator behavior after the
reference has entered the runtime.

`END_TO_END_TASK_SPACE_ERROR` compares the original continuous desired EEF
trajectory with the achieved EEF pose. It is useful for system reporting but
must not replace the first two measurements.
