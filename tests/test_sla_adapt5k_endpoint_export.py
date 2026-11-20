import json

import pytest

from go2arx5_vla.sla_adapter import adapt5k_checkpoint, adapt5k_contract, adapt5k_package
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from scripts.sla import adapt5k_export_endpoint as export


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
    sealed = dict(checkpoint=dict(path=str(source),sha256='3'*64),binding={},
        model=dict(binding=dict(partition_sha256='4'*64,initial_state_sha256='5'*64),frozen_fingerprint='6'*64),
        base=dict(dataset=str(source),parent_step=500,parent_checkpoint={'sha256':'7'*64},
                  parent_manifest={'sha256':'8'*64},D1_freeze_sha256='9'*64,tokenizer=asset('tokenizer',b't')),
        target=asset('state',b's'),formal=asset('formal',b'f'),restore=asset('restore',b'r'),panel=asset('panel',b'p'),
        outputs={str(i):asset(f'{i:03}.npz',str(i).encode()) for i in export.RESTORE},
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

    manifest_path = package/'MANIFEST.private.json'
    manifest_path.write_text(json.dumps(dict(manifest,phase=500)))
    with pytest.raises(ValueError,match='SEALED_NONPHYSICAL_IDENTITY'):
        adapt5k_package.Provider.read_manifest(package,sha(manifest_path))
