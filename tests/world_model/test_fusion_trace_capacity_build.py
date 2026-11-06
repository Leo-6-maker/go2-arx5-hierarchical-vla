"""Compile real trace DSOs against tiny loader fixtures, never a simulator."""
from pathlib import Path
import json
import subprocess
import time
import pytest

from scripts.wm02.build_fusion_trace_capacity import (
    build, CPP, HPP, STATUS, ORIGINAL_CAPACITY_ASSERT, EXTENDED_CAPACITY_ASSERT)
from go2arx5_vla.world_model.storage import digest

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('capacity', [131072, 262144])
def test_real_dso_overrides_absolute_old_dependency_and_keeps_next_resolution(tmp_path,monkeypatch,capacity):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    base=tmp_path/'base';base.mkdir()
    old=base/'libwm02_init_trace.so'
    old_cpp=ROOT/CPP
    if capacity==262144:
        old_cpp=base/'init_trace.cpp'
        old_cpp.write_bytes((ROOT/CPP).read_bytes().replace(EXTENDED_CAPACITY_ASSERT,ORIGINAL_CAPACITY_ASSERT))
        (base/'init_trace.hpp').write_bytes((ROOT/HPP).read_bytes())
    subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-pthread',
        str(old_cpp),'-o',str(old)],check=True,capture_output=True)
    fixtures={
        'outer':('extern "C" unsigned long WM02InitCurrent();\n'
            'extern "C" const char*WM02OuterNext(){return "FIXTURE_OBSERVER_NEXT";}\n'
            'extern "C" unsigned long fixture_outer(){return WM02InitCurrent();}\n'),
        'observer':('extern "C" unsigned long WM02InitCurrent();\n'
            'extern "C" const char*WM02ObserverNext(){return "FIXTURE_MUJOCO_NEXT";}\n'
            'extern "C" unsigned long fixture_observer(){return WM02InitCurrent();}\n'),
        'native':('extern "C" unsigned long WM02InitCurrent();\n'
            'int main(){return int(WM02InitCurrent());}\n')}
    paths={}
    for name,text in fixtures.items():
        source=base/(name+'.cpp');source.write_text(text)
        target=base/(name+'.so' if name!='native' else 'fixture-not-a-simulator')
        command=['g++','-std=c++17','-O2']
        if name!='native':command+=['-shared','-fPIC']
        subprocess.run(command+[str(source),str(old),'-o',str(target)],check=True,capture_output=True)
        paths[name]=str(target)
    spec=dict(profile='core-wm02-init-build',domain='TARGET_INIT_BUILD_ONLY_V1',physics=False,
        purpose='FUSION_TRACE_CAPACITY_ONLY',source_root=str(ROOT),source_sha='OFFLINE_FIXTURE',
        source_files={p:digest(ROOT/p)for p in (CPP,HPP)},deadline_epoch=time.time()+90,
        native_binary=paths['native'],outer_library=paths['outer'],observer_library=paths['observer'],
        base_trace_library=str(old),native_environment={})
    if capacity==262144:spec['trace_capacity']=capacity
    spec['unchanged_inputs_sha256']={k:digest(spec[k])for k in (
        'native_binary','outer_library','observer_library','base_trace_library')}
    parent=base/'candidate.json'
    parent.write_text(json.dumps(dict(binary_sha256=digest(paths['native']),
        candidate_sources={Path(p).name:digest(old_cpp if p==CPP else ROOT/p)for p in (CPP,HPP)})))
    spec.update(base_candidate_result=str(parent),base_candidate_result_sha256=digest(parent))
    out=tmp_path/'new';out.mkdir()
    result=build(spec,out)
    assert result['status']==f'TRACE_ONLY_{capacity}_CAPACITY_COMPILED_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED'
    assert result['trace_capacity']==capacity
    records=70000*(capacity//131072)
    assert result['synthetic_trace_records']==records
    assert result['key_kind_counts']=={17:records//2,18:records//2}
    assert result['trace_source_compatibility']['changed_members']==([CPP]if capacity==262144 else [])
    assert result['trace_summary']['dropped']==0
    assert result['original_next_resolution_unchanged']
    assert result['new_trace_needed_soname']==str(old)
    assert result['new_target_integrations']==result['leg_policy_queries']==0
    assert result['native_dynamic_qualification']is False
    assert spec['unchanged_inputs_sha256']=={k:digest(spec[k])for k in spec['unchanged_inputs_sha256']}
    assert all(row['exit_code']==0 for row in result['commands'])


@pytest.mark.parametrize('bad', [True, 262144., 0, 524288])
def test_unknown_or_noninteger_capacity_rejected_before_build(tmp_path,monkeypatch,bad):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    with pytest.raises(ValueError,match='SUPPORTED_INTEGER_CAPACITY'):
        build(dict(source_root=str(ROOT),trace_capacity=bad),tmp_path)
