import io
import json

import pytest

from go2arx5_vla.sla_adapter import adapt5k_checkpoint, adapt5k_contract, adapt5k_package
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.world_model import adapt5k_native
from scripts.sla import adapt5k_endpoint_transport as transport
from scripts.sla import adapt5k_export_endpoint as export
from scripts.sla import goal_v2_adapt_trained_replay_watch as replay_watch
from scripts.wm02.adapt5k_core_native_launch import deployment_replay


def test_native_launcher_uses_each_endpoint_own_replay_shape(tmp_path):
    replay = tmp_path/'RGB16.json'
    replay.write_text(json.dumps(dict(environment={'GPU_uuid':'GPU-test'},rows=16)))
    expected = json.loads(replay.read_bytes())
    direct = export.pin(replay)
    wrapped = tmp_path/'deployment.json'
    wrapped.write_text(json.dumps(dict(proof={'restore':direct})))
    assert deployment_replay(dict(arm='ADAPT0',deployment=export.pin(wrapped))) == expected
    for arm in ('ADAPT2K','ADAPT5K'):
        assert deployment_replay(dict(arm=arm,deployment=direct)) == expected


def test_go2_export_is_atomic_and_keeps_the_exact_endpoint_identity(tmp_path, monkeypatch):
    source = tmp_path/'source'
    source.mkdir()

    def asset(name, data):
        path = source/name
        path.write_bytes(data)
        return export.pin(path)

    weights = []
    for i in range(29):
        row = asset(f'{i:06d}.npy',str(i).encode())
        domain = 'shared' if i < 14 else 'target'
        weights.append(dict(key=f"['parameters']['{domain}']['w{i}']",file=f'{i:06d}.npy',
                            sha256=row['sha256'],dtype='float32',shape=[1]))
    weights.append(dict(key="['parameters']['libero']['w']",file='000029.npy',sha256='1'*64,
                        dtype='float32',shape=[1]))
    c20 = asset('STATISTICS.json',b'c20')
    monkeypatch.setattr(export,'C20_STATISTICS',c20['sha256'])
    monkeypatch.setattr(adapt5k_package,'C20_STATISTICS',c20['sha256'])
    monkeypatch.setattr(adapt5k_checkpoint,'read_complete',lambda *_: (source,dict(phase_step=2000,arrays=weights)))
    checkpoint = dict(path=str(source),sha256='3'*64)
    panel = asset('panel',json.dumps(dict(old_TEST_access=False,restore_ordinals=export.RESTORE,
                                         bindings={'target':'fixed'})).encode())
    outputs = {str(i):asset(f'{i:03}.npz',f'archived-RGB-{i:02}'.encode()) for i in export.RESTORE}
    rows = [dict(ordinal=i,index=100+i,identity=f'frame-{i}',noise_id=i,normalized_sha256='f'*64,
                 decoded_sha256='d'*64,output=outputs.get(str(i))) for i in range(96)]
    formal = dict(phase=2000,checkpoint=checkpoint,panel=panel,model_generations=376,
        status='ADAPT_OFFLINE_RESULTS_SEALED',role='formal',exit_code=0,new_updates=0,native_requests=0,
        old_TEST_access=False,optimization_visits=0,rows={'target':rows})
    restore = dict(phase=2000,checkpoint=checkpoint,panel=panel,ordinary_predictions=288,
                   independent_restore_exact=48,phase_train_predictions=88)
    parent_manifest = asset('parent_manifest',b'{}')
    sealed = dict(checkpoint=checkpoint,binding={},
        model=dict(binding=dict(partition_sha256='4'*64,initial_state_sha256='5'*64),frozen_fingerprint='6'*64),
        base=dict(dataset=str(source),parent_step=500,parent_checkpoint={'sha256':'7'*64},
                  parent_manifest=parent_manifest,D1_freeze_sha256='9'*64,tokenizer=asset('tokenizer',b't')),
        target=asset('state',b's'),formal=asset('formal',json.dumps(formal).encode()),
        restore=asset('restore',json.dumps(restore).encode()),panel=panel,outputs=outputs,
        outcome=asset('eval-scope',b'e'))
    root = tmp_path/'job'
    root.mkdir()
    corrupt = source/'000010.npy'
    corrupt.write_bytes(b'changed')
    with pytest.raises(ValueError,match='MEMBER_BYTES_CHANGED'):
        export.archive(root,2000,sealed,lambda:None)
    assert not list(root.iterdir())
    corrupt.write_bytes(b'10')
    card = export.archive(root,2000,sealed,lambda:None)
    custody = export.read_pin(card)
    package = root/'sealed/package'
    manifest = adapt5k_package.Provider.read_manifest(package,custody['package_sha256'])
    assert manifest['plan'] == adapt5k_contract.PLAN
    assert manifest['checkpoint_sha256'] == sealed['checkpoint']['sha256']
    assert manifest['phase'] == 2000
    assert manifest['native_admission'] is False
    assert len(manifest['arrays']) == 29
    assert len(custody['members']) == 52
    assert not (package/'parameters/000029.npy').exists()
    assert all(sha(root/'sealed'/name)==row['sha256'] for name,row in custody['members'].items())
    with pytest.raises(FileExistsError):
        export.archive(root,2000,sealed,lambda:None)
    assert sha(package/'MANIFEST.private.json') == custody['package_sha256']

    wire = io.BytesIO()
    transport.send(root/'sealed',wire,2000,card['sha256'])
    bytes_on_wire = wire.getvalue()
    core = tmp_path/'core'
    with pytest.raises(ValueError,match='UNEXPECTED_ARCHIVE_MEMBERS_OR_PHASE'):
        transport.receive(core,io.BytesIO(bytes_on_wire),5000,card['sha256'])
    with pytest.raises(ValueError,match='MEMBER_HASH_CHANGED'):
        transport.receive(core,io.BytesIO(bytes_on_wire.replace(b'archived-RGB-00',b'corrupted-RGB00')),
                          2000,card['sha256'])
    assert not core.exists()
    receipt = transport.receive(core,io.BytesIO(bytes_on_wire),2000,card['sha256'])
    local_package = dict(path=str(core/'package'),sha256=custody['package_sha256'])
    evaluation = transport.pin(core/'archive/FORMAL.private.json')
    state_inputs = dict(D1_freeze_sha256='9'*64,target_D1={'statistics':sealed['target']})
    legacy = dict(parent_params='frozen_sla',GPU_uuid='GPU-test',neural_owner_lock='lock',dataset='frozen_data',
                  eval_input=asset('d1_inputs',json.dumps(state_inputs).encode()))
    gpu = dict(legacy,parent_manifest=parent_manifest)
    monkeypatch.setattr(replay_watch,'PANEL',panel['sha256'])
    monkeypatch.setattr(replay_watch,'bindings',lambda data: {'target':'fixed'})
    fields, trained_checkpoint = replay_watch.replay_inputs(2000,receipt,legacy,gpu)
    assert trained_checkpoint==checkpoint['sha256'] and fields['evaluation']==evaluation
    with pytest.raises(ValueError,match='CORE_INPUT_CONTRACT'):
        replay_watch.replay_inputs(2000,receipt,legacy,dict(gpu,GPU_uuid='GPU-other'))
    with pytest.raises(ValueError,match='EXACT_FORMAL_EVAL'):
        replay_watch.replay_inputs(5000,receipt,legacy,gpu)
    replay_rows = [dict(rows[i],normalized_exact=True,physical_exact=True,guard_exact=True) for i in export.RESTORE]
    replay = dict(schema=transport.RGB_SCHEMA,core_custody=receipt,status='CORE_ADAPT_TRAINED_RGB16_EXACT',
        phase=2000,passed=True,exit_code=0,package=local_package['sha256'],checkpoint_sha256=checkpoint['sha256'],
        evaluation=evaluation,panel=panel,rows=replay_rows,site=dict(host='corelab-4080s',GPU_uuid='GPU-test'),
        generation_attempts=16,model_generations=16,physics=False,hardware_allowed=False,
        optimizer_updates=0,optimization_visits=0,native_requests=0,native_ticks=0)
    deploy = asset('strict-pass',json.dumps(replay).encode())
    assert adapt5k_native.endpoint('ADAPT2K',local_package,evaluation,deploy,gpu_uuid='GPU-test')[1] == formal
    with pytest.raises(PermissionError,match='OWN_EXACT_TRAINED_RGB16_DEPLOYMENT'):
        adapt5k_native.endpoint('ADAPT2K',local_package,evaluation,deploy,gpu_uuid='GPU-other')
    with pytest.raises(PermissionError,match='PHASE_CHANGED'):
        adapt5k_native.endpoint('ADAPT5K',local_package,evaluation,deploy,gpu_uuid='GPU-test')
    replay['rows'][0]['physical_exact'] = False
    failed = asset('failed',json.dumps(replay).encode())
    with pytest.raises(PermissionError,match='EFFECT_CHANGED'):
        adapt5k_native.endpoint('ADAPT2K',local_package,evaluation,failed,gpu_uuid='GPU-test')

    manifest_path = package/'MANIFEST.private.json'
    manifest_path.write_text(json.dumps(dict(manifest,phase=500)))
    with pytest.raises(ValueError,match='SEALED_NONPHYSICAL_IDENTITY'):
        adapt5k_package.Provider.read_manifest(package,sha(manifest_path))
