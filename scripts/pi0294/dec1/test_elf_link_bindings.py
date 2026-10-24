"""Real compiler/linker CPU fixture, no controller or model execution."""
from pathlib import Path
import shutil
import subprocess
import pytest
from . import elf_link_bindings as unit
from .sdk_elf_identity import ELF,compare

@pytest.fixture
def linked(tmp_path):
    cc=shutil.which('cc');ar=shutil.which('ar')
    if not cc or not ar:pytest.skip('system C compiler/ar absent')
    source=tmp_path/'fixture.c'
    source.write_text('int original_constant=37; int original_bss;\nint binding_callee(int x){return x+1;}\nint binding_other(int x){return x+2;}\nint binding_candidate(int x){return binding_callee(x)+original_constant+original_bss;}\nint main(void){return binding_candidate(0);}\n')
    obj=tmp_path/'fixture.o';binary=tmp_path/'fixture';archive=tmp_path/'fixture.a'
    subprocess.run([cc,'-O0','-fno-pie','-c',str(source),'-o',str(obj)],check=True,capture_output=True)
    subprocess.run([cc,'-no-pie',str(obj),'-o',str(binary)],check=True,capture_output=True)
    subprocess.run([ar,'rcs',str(archive),str(obj)],check=True,capture_output=True)
    return binary.read_bytes(),{'fixture.a':archive.read_bytes()}

def test_exact_named_calls_and_complete_constant_bytes(linked):
    raw,archives=linked;v=unit.inspect(raw,archives,contains=('binding_candidate',))
    assert v['all_relocations_resolved'] and v['counts']['EXACT_NAMED_BINDING_MATCH']>=2
    data=[r for r in v['relocations'] if r['object_target']['name']=='original_constant'][0]
    assert data['named_object_bytes']==4 and data['named_object_bytes_equal']
    assert v['code_executed'] is v['native_prefix_safety_qualified'] is v['header_object_state_ABI_qualified'] is False

def test_wrong_call_can_match_masked_body_but_never_binding(linked):
    raw,archives=linked;elf=ELF(raw);report=unit.inspect(raw,archives,contains=('binding_candidate',))
    call=next(r for r in report['relocations'] if r['object_target']['name']=='binding_callee')
    function=elf.symbols()['binding_candidate'];section=elf.sections[function['section']]
    offset=section[4]+function['begin']+call['offset'];other=elf.symbols()['binding_other']['value']
    displacement=(other+call['addend']-call['linked_place'])%(1<<32);changed=bytearray(raw);changed[offset:offset+4]=displacement.to_bytes(4,'little')
    neutral=compare(bytes(changed),archives,contains=('binding_candidate',))
    assert neutral['compared_functions'][0]['status']=='ADDRESS_NEUTRAL_BODY_MATCH'
    exact=unit.inspect(bytes(changed),archives,contains=('binding_candidate',))
    assert exact['counts']['NAMED_BINDING_MISMATCH']==1 and not exact['all_relocations_resolved']

def test_changed_constant_contents_are_not_hidden_by_correct_address(linked):
    raw,archives=linked;elf=ELF(raw);value=next(iter(unit.definitions(elf)['original_constant']))
    address,section,size=value;offset=elf.sections[section][4]+address-elf.sections[section][3]
    changed=bytearray(raw);changed[offset:offset+size]=(38).to_bytes(size,'little')
    result=unit.inspect(bytes(changed),archives,contains=('binding_candidate',))
    assert result['counts']['NAMED_CONSTANT_BYTES_MISMATCH']==1 and not result['all_relocations_resolved']


def test_bss_address_is_bound_without_inventing_file_contents(linked):
    raw,archives=linked;value=unit.inspect(raw,archives,contains=('binding_candidate',))
    row=next(r for r in value['relocations'] if r['object_target']['name']=='original_bss')
    assert row['status']=='EXACT_NAMED_BINDING_MATCH'
    assert row['named_object_storage']=='NOBITS_NOT_FILE_BYTES'
    assert 'named_object_bytes_equal' not in row
