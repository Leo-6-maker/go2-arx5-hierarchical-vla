"""Coherent q/dq queue handoff; original policy and PD formula text is retained."""
from .init_candidate import START, END

INCLUDE='#include "command_pair.hpp"\n#include <stdexcept>\n'
CONSUMER='wm02_init::pop(rl.output_dof_pos_queue, _output_dof_pos, 0) && wm02_init::pop(rl.output_dof_vel_queue, _output_dof_vel, 1)'


def producer_block():
    return ''.join('''        if (!this->output_dof_'''+key+'''.empty())
        {
'''+START+f'            WM02InitPublish({channel},wm02_version);\n'+END+f'            output_dof_{key}_queue.push(this->output_dof_{key});\n'+'''        }
''' for key,channel in [('pos',0),('vel',1)])


def patch(cpp,sdk):
    block=producer_block()
    if cpp.count(block)!=1 or sdk.count(CONSUMER)!=1:
        raise ValueError('EXACT_INSTRUMENTED_Q_DQ_HANDOFF_REQUIRED')
    wrapped='''        // WM02 coherent pair: no inference or PD math under this lock.
        if (this->output_dof_pos.size()!=this->output_dof_vel.size())
            throw std::logic_error("WM02_POSITION_VELOCITY_LAYOUT_MISMATCH");
        wm02_command_pair::publish([&]() {
'''+block+'''        });
'''
    new_cpp=INCLUDE+cpp.replace(block,wrapped)
    replacement='wm02_command_pair::consume([&]() { return '+CONSUMER+'; })'
    new_sdk=INCLUDE+sdk.replace(CONSUMER,replacement)
    # Exact inverse is the math-preservation check, including every policy,
    # gain, target, torque formula and unmodified third tau queue operation.
    if (new_cpp[len(INCLUDE):].replace(wrapped,block)!=cpp or
            new_sdk[len(INCLUDE):].replace(replacement,CONSUMER)!=sdk):
        raise ValueError('COMMAND_PAIR_PATCH_CHANGED_UNRELATED_NATIVE_TEXT')
    return new_cpp,new_sdk
