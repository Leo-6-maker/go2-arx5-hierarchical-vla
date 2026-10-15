"""CPU-only admission of one versioned task-gripper reset/working stroke fix."""
import argparse
import json
from pathlib import Path
import unittest
from .task_capture_runtime import cpu_stage,read,sha
from .task_gripper_contract import VERSION,MARGIN,MECHANICAL_TRAVEL,WORKING_OPEN


def run(root):
    from scripts.pi0292.durable import write
    root=Path(root)
    with cpu_stage(root,'TASK_V2_ACCEPTANCE_CPU',cores=1,cpu_s_cap=120):
        proof=root/'evidence/TASK_V2_ENGINEERING_ACCEPTANCE.json'
        if proof.exists():raise FileExistsError('V2_RECEIPT_ALREADY_SEALED')
        legacy=root/'NATIVE_SAFETY_FAIL.json'
        if sha(legacy)!='175ae9cdab3d108e4826968c5ffb3724a34ec069a3ed1f909cf82fafb6a39a71':
            raise ValueError('ORIGINAL_FAILURE_CHANGED')
        plant_path=root/'runtime/TASK_PLANT_CPU_03/CPU_RECEIPT.json';plant=read(plant_path)
        if (plant['version']!=VERSION or not plant['initial_contacts_valid'] or plant['physics_steps']
                or plant['frozen_robot']['status']!='COMPILED_NATIVE_PROPERTIES_EXACTLY_PRESERVED'
                or plant['parameters']['working_margin_m']!=MARGIN):raise ValueError('V2_COMPILED_PLANT_CONTRACT')
        for name in ('task_left_slide','task_right_slide'):
            j=next(x for x in plant['joint_margins'] if x['name']==name)
            if j['q']!=WORKING_OPEN or j['upper_margin']<MARGIN-1e-12:raise ValueError('ACTUAL_RESET_MARGIN')
        build_path=root/'runtime/TASK_ADAPTER_BUILD_02/BUILD_RECEIPT.json';build=read(build_path)
        if build['native_WBC_rebuild'] or build['native_threshold_changes']:raise ValueError('NATIVE_CORE_OR_GATE_CHANGED')
        if build['sources']['task_actuation']!=sha(Path(__file__).with_name('task_actuation.cpp')):
            raise ValueError('ACTUAL_NEW_GRIPPER_LIBRARY_SOURCE')
        for name,digest in build['binaries'].items():
            if sha(build_path.parent/(name+'.so'))!=digest:raise ValueError('TASK_V2_BINARY_READBACK')
        suite=unittest.defaultTestLoader.loadTestsFromName('scripts.pi0294.test_task_gripper_contract')
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        receipt=root/'evidence/TASK_V2_WORKING_CONTRACT_CPU.json'
        write(receipt,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,new_physics=0,new_model_queries=0))
        if not result.wasSuccessful():raise ValueError('V2_CPU_CONTRACT_TEST')
        source={name:sha(Path(__file__).with_name(name)) for name in
            ('task_gripper_contract.py','task_plant.py','task_actuation.cpp','task_native_worker.py')}
        value=dict(status='PASS_CPU_RESET_COMMAND_AND_FROZEN_CORE_CONTRACT_ONLY',version=VERSION,
            original_failure_sha256=sha(legacy),original_failure_preserved=True,
            mechanical_travel_m=MECHANICAL_TRAVEL,working_margin_m=MARGIN,
            derivation='DECLARED_0_04_M_S_TIMES_0_05_S_ONE_COMMAND_INTERVAL',
            WBC_changes=0,safety_threshold_changes=0,gripper_gain_mass_force_speed_changes=0,
            source_files=source,physical_qualification=False,
            original_V1_pilot_retried=False,remaining_fixed_pilots_only=True,
            plant_cpu_receipt=dict(relative=str(plant_path.relative_to(root)),sha256=sha(plant_path)),
            build_receipt=dict(relative=str(build_path.relative_to(root)),sha256=sha(build_path)),
            contract_cpu_receipt=dict(relative=str(receipt.relative_to(root)),sha256=sha(receipt)))
        write(proof,value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
