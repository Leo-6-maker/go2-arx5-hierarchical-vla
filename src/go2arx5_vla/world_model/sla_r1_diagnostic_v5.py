"""R1's registered eight TRAIN conditions through the original endpoint consumer.

The fixed R0/R1 population is distinct from original S1 TRAIN4 and DEV16.
No teacher fallback, inference clip or retry is introduced.
"""
import ast
import hashlib
import json
from pathlib import Path

from . import task_learning_grasp_v1 as task
from .stable_task_scene_v1 import pinned
from .storage import digest

SCHEMA = 'SLA_R0_R1_FIXED_TRAIN8_PANEL_V5'
ADMISSION = 'SLA_R1_LOGIT500_NATIVE_DIAGNOSTIC_ADMISSION_V5'
METHODS = ('R0_ORIGINAL_S1_500', 'R1_LOGIT_GRIP500')
DECLARATION = 'experiments/sla_go2_posttrain/r1_train8_inputs_v2.json'


def declaration():
    p = Path(__file__).resolve().parents[3]/DECLARATION
    return json.loads(p.read_bytes()), digest(p)


def qualification(q, *, verify_package=True):
    from .sla_native_v1 import closed
    from ..sla_adapter.gripper_latent_v2 import VERSION, EPSILON
    task.need(q.get('schema') == ADMISSION and q.get('method') == METHODS[1]
              and q.get('old_DEV_TEST_admitted') is False, 'DISTINCT_R1_LOGIT500_IDENTITY')
    rgb = closed(q['ordinary_RGB_reconstruction'])
    root = Path(q['ordinary_RGB_reconstruction']['path']).parent
    spec = json.loads((root/'EXECUTION_SPEC.json').read_bytes())
    task.need(rgb.get('schema') == 'SLA_R1_CORE_SITE_RGB32_ACTUAL_V3' and rgb['passes']
              and rgb['ordinary_RGB_generations'] == 32 and rgb['actual_parent_loads'] == 2
              and rgb['all_fresh_process_bit_exact'] and rgb['step'] == 500 and rgb['cursor'] == 16000
              and rgb['package'] == q['package'] and rgb['optimizer_updates'] == rgb['physical_requests'] == 0,
              'ACTUAL_R1_CORE_FRESH32_RGB_NOT_A_SOFTWARE_CHECK')
    transfer = pinned(q['transfer_manifest'])
    delivery = Path(q['transfer_manifest']['path']).parent
    pins = {r['file']: r for r in transfer['files']}
    def delivered(name):
        p = delivery/name
        task.need(p.resolve().is_relative_to(delivery.resolve()) and digest(p) == pins[name]['sha256'],
                  'ACTUAL_BYTE_VERIFIED_H200_R1_TRANSFER')
        return json.loads(p.read_bytes())
    run, lease, worker, candidate = [delivered(n) for n in
        ('RESULT.private.json','control/LEASE.json','WORKER_EXIT.private.json','candidate500/RESULT.private.json')]
    task.need(lease['status'] == 'CLOSED' and lease['worker_exit_code'] == worker['exit_code'] == run['exit_code'] == 0
              and run['optimizer_updates'] == 500 and run['optimization_visits'] == 16000
              and candidate['actual']['optimizer_updates'] == 500
              and candidate['actual']['completed_training_visits'] == 16000
              and candidate['step'] == 500 and candidate['cursor'] == 16000,
              'ACTUAL_R1_FIXED500_AND_16000_VISITS')
    m = pinned(dict(path=str(Path(q['package']['path'])/'MANIFEST.private.json'), sha256=q['package']['sha256']))
    # The native supervisor checks metadata without importing the model's
    # SciPy/JAX environment. The GPU provider repeats its serving contract.
    task.need(m.get('schema') == 'SLA_GRIP_LOGIT_ORDINARY_RGB_INFERENCE_PACKAGE_V3'
              and m.get('codec') == m.get('bindings',{}).get('codec_version') == VERSION
              and m.get('label_epsilon') == EPSILON
              and m.get('decode_order') == 'INVERSE_LATENT_NORMALIZATION_THEN_SIGMOID_ONCE_THEN_ORIGINAL_NATIVE_NOMINAL_MAPPING'
              and all(m.get(k) is False for k in ('state_gripper_transformed','teacher_required',
                                                'training_cache_required','optimizer_required')),
              'R1_DISTINCT_SINGLE_SIGMOID_TEACHER_FREE_PACKAGE')
    task.need(m['step'] == 500 and m['cursor'] == 16000 and not m['qualification_only']
              and m['original_complete_sha256'] == candidate['last_checkpoint']['sha256']
              and m['bindings']['recipe'] == 'SLA_EFFECTIVE32_DEVICE_REDUCTION_V3'
              and m['parent_manifest']['sha256'] == q['parent_manifest']['sha256']
              and digest(q['parent_manifest']['path']) == m['parent_manifest']['sha256'],
              'R1_SAME_ACTUAL_TRAINED_PACKAGE')
    task.need(spec['GPU_uuid'] == q['GPU_uuid'] and spec['parent_params'] == q['parent_params']
              and spec['python'] == q['runtime']['python'] and spec['upstream_root'] == q['runtime']['upstream_root']
              and spec['JAX_memory_fraction'] == q['runtime']['JAX_memory_fraction']
              and q['allocator_limit_bytes'] == spec['allocator_limit_bytes']
              and max(rgb['GPU_memory'].values()) <= q['allocator_limit_bytes'],
              'R1_SAME_QUALIFIED_CORE_RUNTIME_AND_CAPACITY')
    if verify_package:
        for row in m['adapter_arrays']:
            p = Path(q['package']['path'])/row['file']
            task.need(p.resolve().is_relative_to(Path(q['package']['path']).resolve()) and digest(p) == row['sha256'],
                      'R1_ALL_DEPLOYED_ADAPTER_BYTES')
        for name, key in (('STATISTICS.json','statistics_sha256'),('paligemma_tokenizer.model','tokenizer_sha256')):
            task.need(digest(Path(q['package']['path'])/name) == m[key], 'R1_EXACT_STATISTICS_TOKENIZER')
    return q


def inference_identity(source, qualified_source, *, logit=True):
    """Export-only refactors do not require repeating already qualified RGB.

    Compare every inference dependency in full, except provider_v3 where only
    the actual Provider class and its contract are executed at serving time.
    """
    source, qualified_source = Path(source), Path(qualified_source)
    prefix = 'src/go2arx5_vla/sla_adapter/'
    names = ('model_v1.py','config_v1.py','codec_v1.py','gripper_v2.py','dataset_v1.py','provider_v1.py','checkpoint_v1.py')
    for name in names:
        task.need(digest(source/(prefix+name)) == digest(qualified_source/(prefix+name)),
                  'R1_INFERENCE_DEPENDENCY_CHANGED_'+name)
    if not logit:
        return digest(source/(prefix+'provider_v1.py'))
    def subset(path):
        tree = ast.parse(path.read_text())
        return ast.dump(ast.Module(body=[n for n in tree.body if
            isinstance(n,(ast.Import,ast.ImportFrom)) or
            isinstance(n,ast.Assign) or
            isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Provider','contract')],type_ignores=[]),
            include_attributes=False)
    a, b = [subset(p/(prefix+'provider_v3.py')) for p in (source, qualified_source)]
    task.need(a == b, 'R1_QUALIFIED_PROVIDER_CLASS_AND_CONTRACT_CHANGED')
    return hashlib.sha256(a.encode()).hexdigest()


def validate_panel(panel):
    d, pin = declaration()
    task.need(panel.get('schema') == SCHEMA and panel.get('task_learning_role') == task.SLA_ROLE
              and panel['preregistration']['sha256'] == pin
              and pinned(panel['preregistration']) == d and panel['B1_global_step'] == 500
              and panel['population'] == 'TRAIN8' and tuple(panel['policies']) == METHODS
              and panel['protocol'] == task.protocol(task.SLA_ROLE), 'FIXED_R0_R1_TRAIN8_IDENTITY')
    expected = [(m,r) for m in METHODS for r in d['population']]
    rows = panel['assignments']
    task.need(len(rows) == 16 and len({r['native_run'] for r in rows}) == 16, 'FULL_R0_R1_16_CONDITIONS')
    for row,(method, original) in zip(rows,expected,strict=True):
        task.need(row['method'] == method and all(row[k] == original[k] for k in
            ('layout','parent','split','restart','query_seeds')) and row['purpose'] == SCHEMA
            and row['command'] == task.COMMAND and Path(row['native_run']).name == row['native_run'],
            'R0_R1_NO_LAYOUT_SEED_RESTART_OR_METHOD_ORDER_CHANGE')
    return rows


def summarize(panel, outcomes):
    rows = validate_panel(panel)
    task.need(len(outcomes) <= len(rows) and all(all(a[k] == b[k] for k in
        ('native_run','method','layout','restart','query_seeds')) for a,b in zip(outcomes,rows)),
        'R0_R1_EVERY_FIXED_REQUEST_ONCE')
    methods = {}
    for method in METHODS:
        values = [v for v in outcomes if v['method'] == method]
        methods[method] = dict(requested=8, completed=len(values), attempted=sum(v['attempted'] for v in values),
            task_successes=sum(v['technical_valid'] and v['task_success'] for v in values),
            valid_task_failures=sum(v['technical_valid'] and not v['task_success'] for v in values),
            technical_failures=sum(not v['technical_valid'] for v in values),
            hard_safety_stop_conditions=sum(v['hard_safety_stop'] for v in values),
            actual_SLA_queries=sum(v.get('actual_SLA_queries',0) for v in values),
            actual_native_integrations=sum(v.get('actual_native_integrations',0) for v in values))
    return dict(schema=SCHEMA+'_ACTUAL',requested=16,completed=len(outcomes),complete=len(outcomes)==16,
                methods=methods,original_DEV_conditions=0,TEST_conditions=0,model_updates=0,
                qualified_training_demonstrations=0,WBC_changes=False,generalization_claim=False)
