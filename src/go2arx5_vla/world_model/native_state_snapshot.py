"""Own native sensor values while preserving observation/control math.

GetState moves a new base.lin_vel vector into shared RobotState. RunModel used
to copy it concurrently on the non-native_mrt path. Publish fixed atomic fields
after the sensor read, and copy one coherent current frame before claiming a
policy clock bucket. No lock or inference is added to the PD path.
"""
VERSION='LOCKFREE_ROBOT_STATE_VECTOR_SNAPSHOT_V1'
INCLUDE='// WM02_ROBOT_STATE_HANDOFF_INCLUDE\n#include "state_snapshot.hpp"\n'
END_GET='// WM02_TRACE_END\n}\n\nvoid RL_Sim::SetCommand('
POLICY_TIME='        const double wm02_policy_time_s=PI0294SimNs.load()*1e-9;\n'
PUBLISH='''    // WM02_ROBOT_STATE_SNAPSHOT_PUBLISH_BEGIN
    if(mj_data) {
        wm02_robot_state_snapshot::Frame frame;
        for(int i=0;i<18;++i){frame.values[i]=state->motor_state.q[i];frame.values[18+i]=state->motor_state.dq[i];}
        for(int i=0;i<3;++i){frame.values[36+i]=state->imu.gyroscope[i];frame.values[43+i]=state->base.lin_vel[i];frame.values[46+i]=state->base.position[i];}
        for(int i=0;i<4;++i)frame.values[39+i]=state->imu.quaternion[i];
        frame.sample_ns=std::llround(mj_data->time*1e9);
        wm02_robot_state_snapshot::snapshot().publish(frame);
    }
    // WM02_ROBOT_STATE_SNAPSHOT_PUBLISH_END
'''
READ='''        // WM02_ROBOT_STATE_SNAPSHOT_READ_BEGIN
        wm02_robot_state_snapshot::Frame wm02_sensor_frame;
        if(!wm02_robot_state_snapshot::snapshot().try_read(wm02_sensor_frame,std::llround(wm02_policy_time_s*1e9)))return;
        // WM02_ROBOT_STATE_SNAPSHOT_READ_END
'''
COPIES={
    '        this->obs.ang_vel = this->robot_state.imu.gyroscope;\n':
    '        this->obs.ang_vel.assign(wm02_sensor_frame.values.begin()+36,wm02_sensor_frame.values.begin()+39);\n',
    '        this->obs.base_quat = this->robot_state.imu.quaternion;\n':
    '        this->obs.base_quat.assign(wm02_sensor_frame.values.begin()+39,wm02_sensor_frame.values.begin()+43);\n',
    '        this->obs.dof_pos = this->robot_state.motor_state.q;\n':
    '        this->obs.dof_pos.assign(wm02_sensor_frame.values.begin(),wm02_sensor_frame.values.begin()+18);\n',
    '        this->obs.dof_vel = this->robot_state.motor_state.dq;\n':
    '        this->obs.dof_vel.assign(wm02_sensor_frame.values.begin()+18,wm02_sensor_frame.values.begin()+36);\n',
    '        this->obs.lin_vel = this->robot_state.base.lin_vel;\n':
    '        this->obs.lin_vel.assign(wm02_sensor_frame.values.begin()+43,wm02_sensor_frame.values.begin()+46);\n',
    '        this->obs.base_height = {this->robot_state.base.position[2]};\n':
    '        this->obs.base_height = {wm02_sensor_frame.values[48]};\n'}


def remove_handoff(text):
    for value in(INCLUDE,PUBLISH,READ):
        if text.count(value)!=1:raise ValueError('EXACT_SENSOR_HANDOFF_REMOVAL_REQUIRED')
        text=text.replace(value,'',1)
    for old,new in COPIES.items():
        if text.count(new)!=1:raise ValueError('EXACT_SENSOR_COPY_REMOVAL_REQUIRED')
        text=text.replace(new,old,1)
    return text


def patch(cpp):
    if any(v in cpp for v in(INCLUDE,PUBLISH,READ)):
        raise ValueError('SENSOR_HANDOFF_ALREADY_PRESENT_NEVER_PATCH_TWICE')
    for value in(END_GET,POLICY_TIME,*COPIES):
        if cpp.count(value)!=1:raise ValueError('ORIGINAL_SINGLE_SENSOR_PRODUCER_AND_COPY_ANCHORS_REQUIRED')
    if cpp.count('this->GetState(&this->robot_state);')!=1:
        raise ValueError('ONE_ORIGINAL_CONTROL_SENSOR_PRODUCER_REQUIRED')
    start=cpp.index('void RL_Sim::RunModel()');begin=cpp.index(next(iter(COPIES)))
    end=cpp.index(list(COPIES)[-1]);forward=cpp.index('        this->obs.actions = this->Forward();',end)
    due=cpp.index('        if(!wm02_applied_control_clock::clock().policy_due(')
    if not start<cpp.index(POLICY_TIME)<due<begin<end<forward:
        raise ValueError('CURRENT_SENSOR_FRAME_BEFORE_CLAIMING_POLICY_BUCKET_AND_INFERENCE')
    result=INCLUDE+cpp.replace(END_GET,'// WM02_TRACE_END\n'+PUBLISH+'}\n\nvoid RL_Sim::SetCommand(',1)
    result=result.replace(POLICY_TIME,POLICY_TIME+READ,1)
    for old,new in COPIES.items():result=result.replace(old,new,1)
    if remove_handoff(result)!=cpp:raise ValueError('CONTROLLER_OR_OBSERVATION_MATH_CHANGED')
    return result
