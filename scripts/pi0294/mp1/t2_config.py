"""Fixed saved-M1 T2 shadow diagnosis, not a new policy evaluation."""
SCHEMA='MP1_SAVED_M1_T2_SHADOW_DIAGNOSIS_V1'
SCOPE='MP1_SAVED_M1_T2_SHADOW_DIAGNOSIS'
CAPS=dict(main_starts=0,recovery_forks=0,live_and_recovery_integrations=0,
    shadow_integrations=21000,generation_queries=0,optimizer_updates=0)
CPU_CAPS=dict(core=3600,hopper=0)
ATTEMPTS=tuple('PI0294-E1-M1-1024-CANONICAL-'+f+'-000-T2' for f in ['F0','F1'])


def is_diagnostic(stage):return stage.get('schema')==SCHEMA
