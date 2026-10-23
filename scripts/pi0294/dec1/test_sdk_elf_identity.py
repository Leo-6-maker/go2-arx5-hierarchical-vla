"""Generated CPU ELF fixtures only; no vendor assets or executable invocation."""
import shutil
import struct
import subprocess
import pytest
from . import sdk_elf_identity as unit


@pytest.fixture
def fixture(tmp_path):
    if not shutil.which('g++') or not shutil.which('ar'):
        pytest.skip('CPU C++/ar fixture tools unavailable; no native qualification claim')
    header = 'namespace ocs2 { struct MobileManipulatorInterface { MobileManipulatorInterface(); virtual ~MobileManipulatorInterface(); virtual int value(); int state; }; }'
    (tmp_path/'part.cpp').write_text(header + ' namespace ocs2 { MobileManipulatorInterface::MobileManipulatorInterface():state(7){} MobileManipulatorInterface::~MobileManipulatorInterface(){} int MobileManipulatorInterface::value(){return state+5;} }')
    (tmp_path/'main.cpp').write_text(header + ' int main(){ocs2::MobileManipulatorInterface x; return x.value();}')
    for argv in (
        ['g++','-O2','-fPIC','-c',str(tmp_path/'part.cpp'),'-o',str(tmp_path/'part.o')],
        ['ar','rcs',str(tmp_path/'fixture.a'),str(tmp_path/'part.o')],
        ['g++','-O2','-fPIE','-pie','-Wl,--export-dynamic',str(tmp_path/'main.cpp'),str(tmp_path/'fixture.a'),'-o',str(tmp_path/'fixture')],
    ):
        subprocess.run(argv, check=True, capture_output=True, timeout=20)
    return (tmp_path/'fixture').read_bytes(), (tmp_path/'fixture.a').read_bytes()


def test_real_ELF_ar_and_only_recognized_linker_relaxation(fixture):
    binary, archive = fixture
    result = unit.compare(binary, {'CPU_FIXTURE':archive})
    constructors = [x for x in result['compared_functions'] if 'C1E' in x['symbol'] or 'C2E' in x['symbol']]
    assert len(constructors) == 2
    assert all(x['status'] == 'ADDRESS_NEUTRAL_BODY_MATCH' for x in constructors)
    assert any(x['recognized_linker_relaxations'] for x in constructors)
    assert result['code_executed'] is False and result['ABI_qualified'] is False
    assert result['native_prefix_safety_qualified'] is False
    assert result['complete_build_input_provenance_qualified'] is False


def test_changed_nonrelocation_math_is_not_erased(fixture):
    binary, archive = fixture
    _, obj = next(unit.members(archive))
    name = next(n for n in obj.symbols() if '5valueEv' in n)
    s = obj.symbols()[name]
    where = archive.index(obj.raw) + obj.sections[s['section']][4] + s['begin']
    changed = bytearray(archive); changed[where] ^= 1
    row = next(x for x in unit.compare(binary, {'CPU_CHANGED':bytes(changed)})['compared_functions'] if x['symbol'] == name)
    assert row['status'] == 'BODY_DIFFERENT_NOT_MATCHED'


def test_unsupported_relocation_never_silently_passes(fixture):
    binary, archive = fixture
    _, obj = next(unit.members(archive))
    name = next(n for n in obj.symbols() if 'C1E' in n)
    symbol = obj.symbols()[name]; changed = bytearray(archive); patched = False
    for i, section in enumerate(obj.sections):
        if section[1] != 4 or section[7] != symbol['section']: continue
        for at in range(0, section[5], 24):
            offset, info, _ = struct.unpack_from('<QQq', obj.section(i), at)
            if symbol['begin'] <= offset < symbol['begin'] + symbol['size']:
                where = archive.index(obj.raw) + section[4] + at + 8
                struct.pack_into('<Q', changed, where, (info & ~0xffffffff) | 255)
                patched = True; break
        if patched: break
    assert patched
    row = next(x for x in unit.compare(binary, {'CPU_BAD_RELOCATION':bytes(changed)})['compared_functions'] if x['symbol'] == name)
    assert row['status'] == 'RELOCATION_UNSUPPORTED_NOT_MATCHED'


@pytest.mark.parametrize('raw', [b'', b'!<arch>\ninvalid', b'BAD_ELF' * 20])
def test_malformed_archive_rejected(raw):
    with pytest.raises(ValueError): list(unit.members(raw))


def test_truncated_archive_and_non_x86_ELF_rejected(fixture):
    binary, archive = fixture
    with pytest.raises(ValueError): list(unit.members(archive[:-10]))
    changed = bytearray(binary); struct.pack_into('<H', changed, 18, 183)
    with pytest.raises(ValueError, match='X86_64'): unit.ELF(bytes(changed))
