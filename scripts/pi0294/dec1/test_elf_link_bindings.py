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


@pytest.fixture
def dynamic_linked(tmp_path):
    cc=shutil.which('cc');ar=shutil.which('ar')
    if not cc or not ar:pytest.skip('system C compiler/ar absent')
    source=tmp_path/'dynamic.c';assembly=tmp_path/'local.s'
    source.write_text('#include <stdio.h>\nconst char *binding_literal(void){return "exact frozen literal";}\nint binding_external(void){return puts(binding_literal());}\nvoid *binding_got(void){return (void*)&puts;}\ndouble binding_number(double x){return x*1.25;}\nint main(void){return 0;}\n')
    assembly.write_text('.section .text.rare,"ax",@progbits\n.type local_frozen,@function\nlocal_frozen:\nlea 1(%rdi),%eax\nret\n.size local_frozen,.-local_frozen\n.text\n.globl binding_local\n.type binding_local,@function\nbinding_local:\ncall local_frozen\nret\n.size binding_local,.-binding_local\n.section .note.GNU-stack,"",@progbits\n')
    obj=tmp_path/'dynamic.o';local=tmp_path/'local.o';binary=tmp_path/'dynamic';archive=tmp_path/'dynamic.a'
    subprocess.run([cc,'-O2','-fPIC','-ffunction-sections','-c',str(source),'-o',str(obj)],check=True,capture_output=True)
    subprocess.run([cc,'-c',str(assembly),'-o',str(local)],check=True,capture_output=True)
    subprocess.run([cc,'-no-pie',str(obj),str(local),'-o',str(binary)],check=True,capture_output=True)
    subprocess.run([ar,'rcs',str(archive),str(obj),str(local)],check=True,capture_output=True)
    return binary.read_bytes(),{'dynamic.a':archive.read_bytes()}


def test_external_plt_and_got_name_not_runtime_implementation(dynamic_linked):
    raw,archives=dynamic_linked
    report=unit.inspect(raw,archives,contains=('binding_external','binding_got'))
    statuses=set(report['counts'])
    assert 'EXACT_PLT_NAME_BINDING_RUNTIME_UNQUALIFIED' in statuses
    assert 'EXACT_GOT_NAME_BINDING_RUNTIME_UNQUALIFIED' in statuses
    assert report['dynamic_DSO_implementations_qualified'] is False
    assert report['all_relocations_resolved'] is False


def test_local_section_call_resolves_to_exact_local_function(dynamic_linked):
    raw,archives=dynamic_linked;v=unit.inspect(raw,archives,contains=('binding_local',))
    row=next(r for r in v['relocations'] if r['status']=='EXACT_LOCAL_CODE_BINDING_MATCH')
    assert row['target_function']=='local_frozen' and row['target_function_byte_offset']==0
    assert v['all_relocations_resolved']


def test_merge_literal_bytes_and_substitution_detection(dynamic_linked):
    raw,archives=dynamic_linked;v=unit.inspect(raw,archives,contains=('binding_literal',))
    assert v['counts']=={'EXACT_LOCAL_CONSTANT_BYTES_MATCH':1}
    old=b'exact frozen literal\0';assert raw.count(old)==1
    changed=raw.replace(old,b'wrong frozen literal\0')
    assert len(changed)==len(raw)
    v=unit.inspect(changed,archives,contains=('binding_literal',))
    assert v['counts']=={'LOCAL_CONSTANT_BYTES_MISMATCH':1}


def test_changed_plt_destination_fails_even_if_stub_bytes_exist(dynamic_linked):
    raw,archives=dynamic_linked;elf=ELF(raw);v=unit.inspect(raw,archives,contains=('binding_external',))
    call=next(r for r in v['relocations'] if r['status']=='EXACT_PLT_NAME_BINDING_RUNTIME_UNQUALIFIED')
    record=unit.function_map(elf)[call['function']];offset=elf.sections[record['section']][4]+record['begin']+call['offset']
    changed=bytearray(raw);changed[offset:offset+4]=(call['actual_field']+16).to_bytes(4,'little')
    v=unit.inspect(bytes(changed),archives,contains=('binding_external',))
    assert v['counts']['PLT_NAME_BINDING_MISMATCH']==1 and not v['all_link_fields_bound']


def test_numeric_merge_constant_bytes_are_checked(dynamic_linked):
    import struct
    raw,archives=dynamic_linked;v=unit.inspect(raw,archives,contains=('binding_number',))
    assert v['counts']=={'EXACT_LOCAL_CONSTANT_BYTES_MATCH':1}
    row=v['relocations'][0];assert row['constant_semantic']=='COMPLETE_MERGE_CONSTANT'
    old=struct.pack('<d',1.25);new=struct.pack('<d',1.5);assert raw.count(old)==1
    v=unit.inspect(raw.replace(old,new),archives,contains=('binding_number',))
    assert v['counts']=={'LOCAL_CONSTANT_BYTES_MISMATCH':1}


def test_another_objects_local_label_cannot_supply_a_binding(dynamic_linked,monkeypatch):
    raw,archives=dynamic_linked;v=unit.inspect(raw,archives,contains=('binding_literal',));target=v['relocations'][0]['object_target']
    assert target['binding']==0
    original=unit.definitions
    def unrelated(elf):
        values=original(elf);values[target['name']]={(1,1,1)};return values
    monkeypatch.setattr(unit,'definitions',unrelated)
    v=unit.inspect(raw,archives,contains=('binding_literal',))
    assert v['counts']=={'EXACT_LOCAL_CONSTANT_BYTES_MATCH':1}
