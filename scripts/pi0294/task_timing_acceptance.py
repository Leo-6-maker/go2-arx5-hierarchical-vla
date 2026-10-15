"""Versioned CPU timing receipt; preserves the earlier plant/gripper receipt."""
import json
from pathlib import Path
import unittest
from .task_capture_runtime import cpu_stage,read,sha


def run(root):
    from scripts.pi0292.durable import write
    root=Path(root)
    with cpu_stage(root,'TASK_TIMING_ACCEPTANCE_CPU',cores=1,cpu_s_cap=120):
        proof=root/'evidence/TASK_TIMING_ENGINEERING_ACCEPTANCE.json'
        if proof.exists():raise FileExistsError('TIMING_RECEIPT_ALREADY_SEALED')
        base=root/'evidence/TASK_V2_ENGINEERING_ACCEPTANCE.json';old=read(base)
        # Plant, working stroke and numerical gripper contract reuse their prior receipts.
        for n in ('task_plant.py','task_gripper_contract.py'):
            if sha(Path(__file__).with_name(n))!=old['source_files'][n]:raise ValueError('PLANT_OR_GRIPPER_CONTRACT_CHANGED')
        cb=root/'runtime/TASK_PAUSE_CLOCK_01/CLOCK_BUILD_RECEIPT.json';clock=read(cb)
        tb=root/'runtime/TASK_ADAPTER_BUILD_03/BUILD_RECEIPT.json';build=read(tb)
        previous=read(root/'runtime/TASK_ADAPTER_BUILD_02/BUILD_RECEIPT.json')
        if (build['binaries']['task_observer']!=previous['binaries']['task_observer']
                or build['original_observer_sha256']!=previous['original_observer_sha256']
                or build['sources']['task_actuation']!=sha(Path(__file__).with_name('task_actuation.cpp'))
                or build['native_WBC_rebuild'] or build['native_threshold_changes']
                or clock['controller_math_changes'] or clock['gain_changes'] or clock['safety_threshold_changes']):
            raise ValueError('FROZEN_CORE_AND_SAFETY_REQUIRED')
        for n,digest in clock['binary_sha256'].items():
            if sha(cb.parent/n)!=digest:raise ValueError('TIMING_BINARY_READBACK')
        for n,digest in build['binaries'].items():
            if sha(tb.parent/(n+'.so'))!=digest:raise ValueError('TASK_BINARY_READBACK')
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('scripts.pi0294.'+n) for n in
            ('test_task_clock','test_task_gripper_contract','test_task_state','test_task_contacts','test_task_capture_scope'))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        cp=root/'evidence/TASK_TIMING_CONTRACT_CPU.json'
        write(cp,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,new_physics=0,new_model_queries=0))
        if not result.wasSuccessful():raise ValueError('TIMING_CPU_CONTRACT_FAIL')
        member=lambda path:dict(relative=str(path.relative_to(root)),sha256=sha(path))
        sources={n:sha(Path(__file__).with_name(n)) for n in ('task_actuation.cpp','task_native_worker.py','task_clock.py','task_pause_build.py')}
        value=dict(status='PASS_CPU_EXACT_STEP_PAUSE_ONLY',physical_qualification=False,
            base_acceptance_sha256=sha(base),clock_build_receipt=member(cb),task_build_receipt=member(tb),CPU_receipt=member(cp),
            source_files=sources,source_age_limit_s=.025,controller_math_changes=0,safety_threshold_changes=0,
            paused_state='ACTUAL_MJDATA_FSM_LEG_HISTORY_AND_OWNED_SOLVER_PROCESS_WAIT',
            non_realtime=True,raw_attempts_preserved=True,no_quality_retry=True)
        write(proof,value);print(json.dumps(value))
