"""V2 successor: gate integration on the actual q/dq application, not inference.

This changes the handoff clock only. Legacy V1 builds remain reproducible.
The existing sim mutex spans SDK consumption, target assignment, SetCommand and
the PD completion mark, so the integrator cannot observe a half-applied command.
"""
from .native_control_clock import patch as legacy_patch

VERSION='FROZEN_PERIOD_PD_POLICY_APPLIED_CLOCK_V2'
INCLUDE='#include "applied_control_clock.hpp"\n'
NAMESPACE='wm02_applied_control_clock::clock()'


def patch(cpp,sdk,utils):
    publish='        wm02_command_pair::publish([&]() {\n'
    applied='        WM02InitApplied();'
    if cpp.count(publish)!=1 or sdk.count(applied)!=1:
        raise ValueError('EXACT_COHERENT_PAIR_APPLICATION_ANCHORS_REQUIRED')
    a,b=legacy_patch(cpp,utils)
    a=a.replace('#include "control_clock.hpp"\n',INCLUDE,1)
    b=b.replace('#include "control_clock.hpp"\n',INCLUDE,1)
    a=a.replace('wm02_control_clock::clock()',NAMESPACE)
    b=b.replace('wm02_control_clock::clock()',NAMESPACE)
    publication='        '+NAMESPACE+'.published(wm02_policy_time_s,wm02_policy_period_ns);\n'
    acknowledgement='\n        '+NAMESPACE+'.applied();'
    a=a.replace(publish,publish+publication)
    s=INCLUDE+sdk.replace(applied,applied+acknowledgement)
    # All original control, policy, torque and integration text is recoverable.
    inverse_a=a.replace(publication,'').replace(INCLUDE,'#include "control_clock.hpp"\n',1).replace(NAMESPACE,'wm02_control_clock::clock()')
    inverse_b=b.replace(INCLUDE,'#include "control_clock.hpp"\n',1).replace(NAMESPACE,'wm02_control_clock::clock()')
    if (inverse_a,inverse_b)!=legacy_patch(cpp,utils) or s[len(INCLUDE):].replace(acknowledgement,'')!=sdk:
        raise ValueError('APPLIED_CLOCK_PATCH_CHANGED_UNRELATED_NATIVE_TEXT')
    return a,s,b
