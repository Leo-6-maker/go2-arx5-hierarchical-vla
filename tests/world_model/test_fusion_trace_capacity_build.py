"""Compile real trace DSOs against tiny loader fixtures, never a simulator."""
from pathlib import Path
import json
import subprocess
import time

from scripts.wm02.build_fusion_trace_capacity import build, CPP, HPP, STATUS
from go2arx5_vla.world_model.storage import digest

ROOT=Path(__file__).resolve().parents[2]


def test_real_131072_dso_overrides_absolute_old_dependency_and_keeps_next_resolution(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    base=tmp_path/'base';base.mkdir()
    old=base/'libwm02_init_trace.so'
    subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-pthread',
        str(ROOT/CPP),'-o',str(old)],check=True,capture_output=True)
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
    spec['unchanged_inputs_sha256']={k:digest(spec[k])for k in (
        'native_binary','outer_library','observer_library','base_trace_library')}
    parent=base/'candidate.json'
    parent.write_text(json.dumps(dict(binary_sha256=digest(paths['native']),
        candidate_sources={Path(p).name:digest(ROOT/p)for p in (CPP,HPP)})))
    spec.update(base_candidate_result=str(parent),base_candidate_result_sha256=digest(parent))
    out=tmp_path/'new';out.mkdir()
    result=build(spec,out)
    assert result['status']==STATUS and result['trace_capacity']==131072
    assert result['synthetic_trace_records']==70000
    assert result['key_kind_counts']=={17:35000,18:35000}
    assert result['trace_summary']['dropped']==0
    assert result['original_next_resolution_unchanged']
    assert result['new_trace_needed_soname']==str(old)
    assert result['new_target_integrations']==result['leg_policy_queries']==0
    assert result['native_dynamic_qualification']is False
    assert spec['unchanged_inputs_sha256']=={k:digest(spec[k])for k in spec['unchanged_inputs_sha256']}
    assert all(row['exit_code']==0 for row in result['commands'])
