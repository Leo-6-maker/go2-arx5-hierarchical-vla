"""Pure planner fixtures. Actual captured records are checked privately too."""
import hashlib
import pytest
from . import target_sdk_build_plan as unit


@pytest.fixture
def records(monkeypatch):
    flags=b'CXX_DEFINES = -DCPU_FIXTURE\nCXX_INCLUDES = -I"/sdk/include with space"\nCXX_FLAGS = -O3 -std=gnu++17\n'
    link=('/usr/bin/c++ -O3 '+unit.NODE_OBJECT+' -o mobile_manipulator_mpc_node '
          'libocs2_mobile_manipulator_ros.a ../ocs2_core/libocs2_core.a /usr/lib/libpthread.so -Wl,-rpath,/sdk/install/lib\n').encode()
    monkeypatch.setattr(unit,'FLAGS_SHA256',hashlib.sha256(flags).hexdigest())
    monkeypatch.setattr(unit,'LINK_SHA256',hashlib.sha256(link).hexdigest())
    return flags,link


def plan(records,**kw):
    return unit.plan(*records,sdk_root='/sdk',sealed_client_directory='/registered/capsule/dec1',
                     registered_build_directory=kw.get('out','/registered/output/sdk-build'))


def test_only_own_companion_object_replaces_original_node(records):
    value=plan(records)
    assert value['cwd']=='/registered/output/sdk-build'
    assert '-I/sdk/include with space' in value['compile_argv']
    assert value['link_argv'].count('/registered/output/sdk-build/target_sdk_client.o')==1
    assert unit.NODE_OBJECT not in value['link_argv']
    assert '/sdk/build/ocs2_core/libocs2_core.a' in value['existing_link_inputs']
    assert value['vendor_sources_compiled']==[] and value['native_action'] is False
    assert value['model_constructors_during_build']==0


@pytest.mark.parametrize('out',['/sdk','/sdk/build/new','/registered/../escape','relative'])
def test_no_output_in_vendor_or_nonnormalized_path(records,out):
    with pytest.raises(PermissionError):plan(records,out=out)


def test_changed_build_record_fails_before_any_dispatch(records):
    a,b=records
    with pytest.raises(PermissionError):plan((a+b'CHANGED',b))


def test_unknown_original_link_shape_cannot_be_repaired_implicitly(records,monkeypatch):
    a,b=records;b=b.replace(unit.NODE_OBJECT.encode(),b'unknown_source.o')
    monkeypatch.setattr(unit,'LINK_SHA256',hashlib.sha256(b).hexdigest())
    with pytest.raises(PermissionError):plan((a,b))


def test_delivered_transitive_dependencies_are_explicit_pinned_inputs_and_runtime_dependencies(records):
    value=plan(records);argv=value['link_argv'];deps=value['dependent_shared_library_sha256']
    assert set(deps)=={'/dependency-root/usr/lib/x86_64-linux-gnu/'+x for x in unit.DEPENDENCY_SHARED_SHA256}
    start=argv.index('-Wl,--push-state,--no-as-needed');stop=argv.index('-Wl,--pop-state')
    assert argv[start+1:stop]==list(deps)
    assert set(deps)<=set(value['existing_link_inputs'])
    assert '-Wl,-rpath-link,/dependency-root/usr/lib/x86_64-linux-gnu' in argv
    assert '-Wl,-rpath,/dependency-root/usr/lib/x86_64-linux-gnu' in argv
    assert value['vendor_sources_compiled']==[] and value['vendor_binaries_modified'] is False
    assert '-Wl,--disable-new-dtags' in argv
    assert set(value['runtime_library_directories'])=={'/dependency-root/usr/lib/x86_64-linux-gnu','/usr/lib'}


@pytest.mark.parametrize('sdk',['/sdk,inject/sdk','/sdk:inject/sdk','/sdk\ninject/sdk'])
def test_search_directory_cannot_inject_extra_linker_paths(records,sdk):
    with pytest.raises(PermissionError):
        unit.plan(*records,sdk_root=sdk,sealed_client_directory='/registered/code',registered_build_directory='/registered/out')
