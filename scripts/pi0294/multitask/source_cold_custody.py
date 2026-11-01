"""Recognize only evidenced, closed source roots that never launched."""


def is_cold_terminal(terminal, *, launch_intent_exists, detached_owner_exists, stop=None):
    rows=terminal['results']
    legacy=terminal.get('cold_never_started')is True
    explicit=(terminal.get('status')=='FINITE_SOURCE_CHAIN_TERMINAL'
        and terminal.get('physics_integrations')==0
        and terminal.get('new_model_queries_updates_PBS')==0
        and len(rows)==1
        and rows[0].get('OS_exit_status')=='NOT_STARTED_READBACK_SIGKILL_AT_HARD_CPU15'
        and rows[0].get('raw_terminal_exists')is False
        and rows[0].get('qualified_source_pilot')is False
        and isinstance(stop,dict)and stop.get('new_physics')==0
        and stop.get('model_queries')==0 and stop.get('source_episode_attempts')==0)
    if not(legacy or explicit):return False
    assert not((legacy and rows!=[])or launch_intent_exists or detached_owner_exists), 'COLD_CUSTODY_REQUIRES_PROVEN_ZERO_LAUNCH'
    return True
