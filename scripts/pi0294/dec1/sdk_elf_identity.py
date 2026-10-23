"""Bounded ELF64/ar byte comparison only; never loads or executes native code.

Address-neutral body equality is LIMITED evidence: masked relocation targets,
headers, ABI, callers and the complete linked program remain unqualified.
"""
import hashlib
import struct


def check(value, why):
    if not value: raise ValueError('ELF_READONLY_' + why)


def digest(raw): return hashlib.sha256(raw).hexdigest()


class ELF:
    def __init__(self, raw):
        check(type(raw) is bytes and 64 <= len(raw) <= 64 * 1024**2, 'BOUNDED_FILE')
        check(raw[:7] == b'\x7fELF\x02\x01\x01', 'ELF64_LITTLE_ENDIAN_V1')
        h = struct.unpack_from('<16sHHIQQQIHHHHHH', raw)
        check(h[1] in (1, 2, 3) and h[2] == 62 and h[11] == 64, 'X86_64_REL_EXEC_OR_DYN')
        check(0 < h[12] < 65535 and h[6] + h[12] * 64 <= len(raw), 'SECTION_TABLE')
        self.raw, self.kind = raw, h[1]
        self.sections = [struct.unpack_from('<IIQQQQIIQQ', raw, h[6] + 64*i) for i in range(h[12])]
        self._symbols = None

    def section(self, index):
        check(0 <= index < len(self.sections), 'SECTION_INDEX')
        s = self.sections[index]
        check(s[4] + s[5] <= len(self.raw), 'SECTION_RANGE')
        return self.raw[s[4]:s[4]+s[5]]

    def symbols(self):
        if self._symbols is not None: return self._symbols
        found = {}
        for i, section in enumerate(self.sections):
            if section[1] not in (2, 11): continue
            check(section[9] == 24 and section[5] % 24 == 0, 'SYMBOL_ENTRY_SIZE')
            strings = self.section(section[6]); rows = self.section(i)
            for offset in range(0, len(rows), 24):
                name, info, _, idx, value, size = struct.unpack_from('<IBBHQQ', rows, offset)
                if not (info & 15 == 2 and info >> 4 in (1, 2) and 0 < idx < len(self.sections) and size): continue
                check(name < len(strings), 'SYMBOL_NAME_OFFSET')
                end = strings.find(b'\0', name); check(end >= name, 'SYMBOL_NAME_TERMINATION')
                text = strings[name:end].decode('utf-8')
                section = self.sections[idx]
                begin = value if self.kind == 1 else value-section[3]
                check(0 <= begin and begin+size <= section[5], 'FUNCTION_SECTION_RANGE')
                record = dict(name=text, section=idx, begin=begin, size=size, value=value)
                if text in found:
                    check(found[text] == record, 'AMBIGUOUS_DEFINED_FUNCTION')
                found[text] = record
        self._symbols = found
        return found

    def code(self, symbol):
        s = self.symbols()[symbol]
        return self.section(s['section'])[s['begin']:s['begin']+s['size']]

    def relocations(self, symbol):
        s = self.symbols()[symbol]; found = []
        for i, section in enumerate(self.sections):
            if section[1] != 4 or section[7] != s['section']: continue
            check(section[9] == 24 and section[5] % 24 == 0, 'RELA_ENTRY_SIZE')
            rows = self.section(i)
            for at in range(0, len(rows), 24):
                offset, info, addend = struct.unpack_from('<QQq', rows, at)
                if s['begin'] <= offset < s['begin']+s['size']:
                    kind = info & 0xffffffff
                    widths = {1:8, 2:4, 3:4, 4:4, 9:4, 10:4, 11:4, 41:4, 42:4}
                    check(kind in widths, 'UNSUPPORTED_X86_64_RELOCATION_' + str(kind))
                    start = offset-s['begin']; width = widths[kind]
                    check(start+width <= s['size'], 'RELOCATION_OUTSIDE_FUNCTION')
                    found.append(dict(offset=start, width=width, kind=kind, addend=addend))
        return found


def members(raw):
    check(type(raw) is bytes and raw.startswith(b'!<arch>\n') and len(raw) <= 64*1024**2, 'BOUNDED_AR_ARCHIVE')
    pos = 8; names = b''
    while pos < len(raw):
        check(pos+60 <= len(raw), 'AR_HEADER')
        h = raw[pos:pos+60]; check(h[-2:] == b'`\n', 'AR_HEADER_MAGIC')
        count = int(h[48:58]); check(count >= 0 and pos+60+count <= len(raw), 'AR_MEMBER_RANGE')
        name = h[:16].decode().strip(); value = raw[pos+60:pos+60+count]
        pos += 60+count; pos += pos % 2
        if name == '//': names = value; continue
        if name in ('/', '/SYM64/'): continue
        if name.startswith('#1/'):
            n = int(name[3:]); check(0 < n <= len(value), 'BSD_AR_NAME')
            name, value = value[:n].decode().rstrip('\0'), value[n:]
        elif name.startswith('/'):
            n = int(name[1:]); check(0 <= n < len(names), 'GNU_AR_NAME_OFFSET')
            end = names.find(b'/\n', n); check(end >= n, 'GNU_AR_LONG_NAME')
            name = names[n:end].decode()
        else: name = name.rstrip('/')
        if value.startswith(b'\x7fELF'): yield name, ELF(value)


def compare(binary, archives, contains=('MobileManipulatorInterface', 'StateSoftConstraint', 'StateInputSoftConstraint', 'MRT_BASE')):
    """No PASS for source/build provenance or ABI follows from this comparison."""
    exe = ELF(binary); symbols = exe.symbols(); rows = []
    for archive, raw in archives.items():
        for member, obj in members(raw):
            for name in obj.symbols():
                if name not in symbols or not any(word in name for word in contains): continue
                original = exe.code(name); candidate = obj.code(name)
                row = dict(symbol=name, archive=archive, member=member,
                    linked_bytes=len(original), object_bytes=len(candidate))
                if len(original) != len(candidate):
                    row.update(status='SIZE_DIFFERENT_NOT_MATCHED'); rows.append(row); continue
                try: relocations = obj.relocations(name)
                except ValueError as error:
                    row.update(status='RELOCATION_UNSUPPORTED_NOT_MATCHED',error=str(error)); rows.append(row); continue
                a, b = bytearray(original), bytearray(candidate)
                used = set()
                relaxations = []
                for r in relocations:
                    start = r['offset']
                    # GNU x86-64 GOTPCRELX may resolve a local symbol directly:
                    # MOV [GOT] -> LEA [symbol], or FF15 indirect call -> 67E8.
                    # Recognize only these exact opcode forms AT the relocation;
                    # never erase arbitrary differing instruction bytes.
                    if r['kind'] in (41, 42) and start >= 3:
                        rex_ok = (r['kind'] != 42 or
                                  0x40 <= b[start-3] <= 0x4f and a[start-3] == b[start-3])
                        if (rex_ok and b[start-2] == 0x8b and a[start-2] == 0x8d
                                and b[start-1] == a[start-1] and b[start-1] & 0xc7 == 5):
                            a[start-2] = b[start-2]
                            relaxations.append(dict(offset=start-2,kind='GOTPCRELX_MOV_TO_LEA'))
                        elif (r['kind'] == 41 and b[start-2:start] == b'\xff\x15'
                              and a[start-2:start] == b'\x67\xe8'):
                            a[start-2:start] = b[start-2:start]
                            relaxations.append(dict(offset=start-2,kind='GOTPCRELX_INDIRECT_TO_DIRECT_CALL'))
                    indexes = set(range(r['offset'],r['offset']+r['width']))
                    check(not used & indexes, 'OVERLAPPING_RELOCATIONS'); used |= indexes
                    a[r['offset']:r['offset']+r['width']] = bytes(r['width'])
                    b[r['offset']:r['offset']+r['width']] = bytes(r['width'])
                row.update(status='ADDRESS_NEUTRAL_BODY_MATCH' if a==b else 'BODY_DIFFERENT_NOT_MATCHED',
                    masked_bytes=len(used),relocations=relocations,recognized_linker_relaxations=relaxations,
                    linked_neutral_sha256=digest(a),object_neutral_sha256=digest(b),
                    relocation_target_identity_qualified=False)
                rows.append(row)
    return dict(schema='READONLY_ELF_STATIC_ARCHIVE_LIMITED_BODY_COMPARISON_V1',
        binary_sha256=digest(binary),archive_sha256={k:digest(v) for k,v in archives.items()},
        compared_functions=rows,code_executed=False,ABI_qualified=False,
        complete_build_input_provenance_qualified=False,native_prefix_safety_qualified=False)
