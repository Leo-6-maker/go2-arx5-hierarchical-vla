"""Read-only x86-64 static-link relocation evidence; never native execution.

Retain unsupported, unresolved and mismatched bindings explicitly. A matching
subset cannot certify header/class layout, full executable or prefix safety.
"""
import struct
from .sdk_elf_identity import ELF, check, digest, members


def symbol_rows(elf,table):
    section=elf.sections[table]
    check(section[1] in (2,11) and section[9]==24 and section[5]%24==0,'BIND_SYMBOL_TABLE')
    strings=elf.section(section[6]);rows=elf.section(table);result=[]
    for at in range(0,len(rows),24):
        name,info,other,index,value,size=struct.unpack_from('<IBBHQQ',rows,at)
        check(name<len(strings),'BIND_SYMBOL_NAME');end=strings.find(b'\0',name);check(end>=name,'BIND_SYMBOL_TERMINATION')
        result.append(dict(name=strings[name:end].decode(),kind=info&15,binding=info>>4,
            visibility=other&3,section=index,value=value,size=size))
    return result


def definitions(elf):
    found={}
    for i,s in enumerate(elf.sections):
        if s[1] not in (2,11):continue
        for row in symbol_rows(elf,i):
            if row['name'] and 0<row['section']<len(elf.sections):
                found.setdefault(row['name'],set()).add((row['value'],row['section'],row['size']))
    return found


def function_relocations(obj,name,function=None):
    function=function or obj.symbols()[name]
    for i,section in enumerate(obj.sections):
        if section[1]!=4 or section[7]!=function['section']:continue
        check(section[9]==24 and section[5]%24==0,'BIND_RELA_SIZE')
        symbols=symbol_rows(obj,section[6]);raw=obj.section(i)
        for at in range(0,len(raw),24):
            offset,info,addend=struct.unpack_from('<QQq',raw,at)
            if not function['begin']<=offset<function['begin']+function['size']:continue
            index=info>>32;check(index<len(symbols),'BIND_RELOCATION_SYMBOL_INDEX')
            yield dict(offset=offset-function['begin'],kind=info&0xffffffff,addend=addend,target=symbols[index])


def function_map(elf):
    """Include local cold functions, not just dynamic/global symbol exports."""
    found={};ambiguous=set()
    for table,section in enumerate(elf.sections):
        if section[1] not in (2,11):continue
        for row in symbol_rows(elf,table):
            if row['kind']!=2 or not row['size'] or not 0<row['section']<len(elf.sections):continue
            sec=elf.sections[row['section']];begin=row['value']-(sec[3] if elf.kind!=1 else 0)
            check(0<=begin and begin+row['size']<=sec[5],'BIND_ALL_FUNCTION_RANGE')
            record=dict(name=row['name'],section=row['section'],begin=begin,size=row['size'],value=row['value'])
            if row['name'] in found and found[row['name']]!=record:ambiguous.add(row['name'])
            found[row['name']]=record
    return {name:row for name,row in found.items() if name not in ambiguous}


def function_code(elf,record):
    return elf.section(record['section'])[record['begin']:record['begin']+record['size']]


def section_names(elf):
    index=struct.unpack_from('<16sHHIQQQIHHHHHH',elf.raw)[13]
    strings=elf.section(index);result={}
    for i,sec in enumerate(elf.sections):
        at=sec[0];check(at<len(strings),'BIND_SECTION_NAME');end=strings.find(b'\0',at)
        check(end>=at,'BIND_SECTION_NAME_END');result[i]=strings[at:end].decode()
    return result


def dynamic_slots(elf):
    """ELF relocation symbol names, not resolved runtime DSO implementations."""
    found={}
    for i,sec in enumerate(elf.sections):
        if sec[1]!=4:continue
        check(sec[9]==24 and sec[5]%24==0,'BIND_DYNAMIC_RELA_SIZE')
        names=symbol_rows(elf,sec[6]);raw=elf.section(i)
        for at in range(0,len(raw),24):
            address,info,_=struct.unpack_from('<QQq',raw,at);kind=info&0xffffffff
            if kind not in (6,7):continue
            index=info>>32;check(index<len(names),'BIND_DYNAMIC_SYMBOL_INDEX');name=names[index]['name']
            if name:found.setdefault(name,set()).add(address)
    return found


def plt_stubs(elf,slots):
    result={};reverse={address:name for name,addresses in slots.items() for address in addresses}
    for index,name in section_names(elf).items():
        if not name.startswith('.plt'):continue
        section=elf.sections[index];stride=section[9] or (8 if name=='.plt.got' else 16)
        if stride not in (8,16):continue
        raw=elf.section(index)
        for start in range(0,len(raw),stride):
            code=raw[start:start+stride]
            for offset in (0,1,4,5):
                if offset+6>len(code) or code[offset:offset+2]!=b'\xff\x25':continue
                if offset==1 and code[:1]!=b'\xf2':continue
                if offset==4 and code[:4]!=b'\xf3\x0f\x1e\xfa':continue
                if offset==5 and code[:5]!=b'\xf3\x0f\x1e\xfa\xf2':continue
                displacement=int.from_bytes(code[offset+2:offset+6],'little',signed=True)
                got=section[3]+start+offset+6+displacement
                if got in reverse:result.setdefault(reverse[got],set()).add(section[3]+start)
    return result


def va_bytes(elf,address,length):
    for index,section in enumerate(elf.sections):
        if section[1]==1 and section[2]&2 and not section[2]&1 and section[3]<=address and address+length<=section[3]+section[5]:
            begin=address-section[3];return elf.section(index)[begin:begin+length]
    return None


def local_binding(obj,linked,target,relocation,place,actual,linked_functions):
    kind=relocation['kind'];width=8 if kind==1 else 4
    if kind not in (1,2,4,10,11) or not 0<target['section']<len(obj.sections):return None
    adjust=4 if kind in (2,4) else 0
    position=target['value']+relocation['addend']+adjust
    section=obj.sections[target['section']]
    if not 0<=position<section[5]:return None
    if section[2]&4:
        candidates=[r for r in function_map(obj).values() if r['section']==target['section'] and r['begin']<=position<r['begin']+r['size']
                    and r['name'] in linked_functions and r['size']==linked_functions[r['name']]['size']]
        if len(candidates)!=1:return None
        fn=candidates[0];address=linked_functions[fn['name']]['value']+position-fn['begin']
        expected=(address-adjust-(place if adjust else 0))%(1<<(8*width))
        return dict(status='EXACT_LOCAL_CODE_BINDING_MATCH' if actual==expected else 'LOCAL_CODE_BINDING_MISMATCH',
            target_function=fn['name'],target_function_byte_offset=position-fn['begin'],actual_field=actual,expected_field=expected)
    # Only compiler-declared merge constants have a proven byte boundary.
    # Local names such as .LC0 are not assumed unique across object files.
    if section[1]!=1 or section[2]&1 or not section[2]&16:return None
    raw=obj.section(target['section']);stride=section[9]
    if section[2]&32:
        if stride!=1:return None
        end=raw.find(b'\0',position)
        if end<position or end-position>16384:return None
        begin=position;length=end-position+1;semantic='REFERENCED_STRING_SUFFIX'
    else:
        if not 0<stride<=64:return None
        begin=position//stride*stride;length=stride;semantic='COMPLETE_MERGE_CONSTANT'
    original=raw[begin:begin+length]
    if len(original)!=length:return None
    signed=actual if actual<1<<(width*8-1) else actual-(1<<(width*8))
    pointer=place+adjust+signed if adjust else actual
    candidate=va_bytes(linked,pointer-(position-begin),length)
    if candidate is None:return None
    return dict(status='EXACT_LOCAL_CONSTANT_BYTES_MATCH' if original==candidate else 'LOCAL_CONSTANT_BYTES_MISMATCH',
        constant_semantic=semantic,constant_bytes=length,original_constant_sha256=digest(original),linked_constant_sha256=digest(candidate))


def inspect(binary,archives,contains=('RL','ObservationBuffer','InferenceRuntime')):
    linked=ELF(binary);check(linked.kind in (2,3),'BIND_LINKED_EXECUTABLE')
    names=definitions(linked);functions=function_map(linked);slots=dynamic_slots(linked);stubs=plt_stubs(linked,slots);rows=[]
    for archive,raw in archives.items():
        for member,obj in members(raw):
            for name,function in function_map(obj).items():
                if name not in functions or not any(word in name for word in contains):continue
                if len(function_code(obj,function))!=len(function_code(linked,functions[name])):
                    rows.append(dict(archive=archive,member=member,function=name,status='FUNCTION_SIZE_MISMATCH'));continue
                for relocation in function_relocations(obj,name,function):
                    target=relocation['target'];kind=relocation['kind'];offset=relocation['offset']
                    row=dict(archive=archive,member=member,function=name,offset=offset,relocation_kind=kind,
                        addend=relocation['addend'],object_target=target)
                    code=function_code(linked,functions[name]);width=8 if kind==1 else 4
                    check(offset+width<=len(code),'BIND_FIELD_OUTSIDE_FUNCTION')
                    actual=int.from_bytes(code[offset:offset+width],'little');place=functions[name]['value']+offset
                    dynamic=None
                    if kind==4 and target['name'] in stubs:
                        candidates=stubs[target['name']]
                        expected={(address+relocation['addend']-place)%(1<<32) for address in candidates}
                        dynamic=dict(status='EXACT_PLT_NAME_BINDING_RUNTIME_UNQUALIFIED' if actual in expected else 'PLT_NAME_BINDING_MISMATCH',
                            runtime_DSO_implementation_qualified=False,verified_stub_addresses=sorted(candidates),actual_field=actual)
                    elif kind in (9,41,42) and target['name'] in slots:
                        expected={(address+relocation['addend']-place)%(1<<32) for address in slots[target['name']]}
                        dynamic=dict(status='EXACT_GOT_NAME_BINDING_RUNTIME_UNQUALIFIED' if actual in expected else 'GOT_NAME_BINDING_MISMATCH',
                            runtime_DSO_implementation_qualified=False,actual_field=actual,verified_dynamic_slots=sorted(slots[target['name']]))
                    if dynamic is not None:
                        row.update(dynamic);rows.append(row);continue
                    # Named defined functions/objects have exact link addresses.
                    # Never guess stripped/local/PLT/GOT targets from body match.
                    addresses=names.get(target['name'],set()) if target['name'] and target['binding'] in (1,2) else set()
                    values={a[0] for a in addresses}
                    if kind not in (1,2,4,10,11):row['status']='UNSUPPORTED_BINDING_KIND'
                    elif len(values)!=1:
                        local=local_binding(obj,linked,target,relocation,place,actual,functions)
                        row.update(local or dict(status='UNRESOLVED_OR_AMBIGUOUS_TARGET'))
                    else:
                        width=8 if kind==1 else 4;start=offset;end=start+width
                        check(end<=len(code),'BIND_FIELD_OUTSIDE_FUNCTION')
                        symbol=next(iter(values));place=functions[name]['value']+offset
                        expected=symbol+relocation['addend']-(place if kind in (2,4) else 0)
                        actual=int.from_bytes(code[start:end],'little')
                        expected%=1<<(8*width)
                        row.update(status='EXACT_NAMED_BINDING_MATCH' if actual==expected else 'NAMED_BINDING_MISMATCH',
                            linked_symbol_address=symbol,linked_place=place,actual_field=actual,expected_field=expected)
                        # Bind the complete named object's bytes as well when
                        # its symbol has a known finite size, without assuming
                        # that arbitrary rodata/string section offsets survived.
                        if target['kind']==1 and target['size'] and 0<target['section']<len(obj.sections):
                            candidates=[a for a in addresses if a[0]==symbol and a[2]==target['size']]
                            if len(candidates)==1:
                                _,section,size=candidates[0];begin=symbol-linked.sections[section][3]
                                if obj.sections[target['section']][1]==8 or linked.sections[section][1]==8:
                                    row['named_object_storage']='NOBITS_NOT_FILE_BYTES'
                                    rows.append(row)
                                    continue
                                a=obj.section(target['section'])[target['value']:target['value']+size]
                                b=linked.section(section)[begin:begin+size]
                                check(len(a)==len(b)==size,'BIND_NAMED_OBJECT_RANGE')
                                # Objects with their own relocations are not raw
                                # constants; retain this limitation explicitly.
                                has_reloc=any(s[1] in (4,9) and s[7]==target['section'] for s in obj.sections)
                                if not has_reloc:
                                    row.update(named_object_bytes=size,named_object_original_sha256=digest(a),
                                        named_object_linked_sha256=digest(b),named_object_bytes_equal=a==b)
                                    if a!=b:row['status']='NAMED_CONSTANT_BYTES_MISMATCH'
                    rows.append(row)
    counts={status:sum(row['status']==status for row in rows) for status in sorted({r['status'] for r in rows})}
    return dict(schema='READONLY_X86_64_EXPLICIT_LINK_BINDING_EVIDENCE_V1',binary_sha256=digest(binary),
        archive_sha256={name:digest(raw) for name,raw in archives.items()},relocations=rows,counts=counts,
        all_relocations_resolved=bool(rows) and all(row['status'] in ('EXACT_NAMED_BINDING_MATCH','EXACT_LOCAL_CODE_BINDING_MATCH','EXACT_LOCAL_CONSTANT_BYTES_MATCH') for row in rows),
        all_link_fields_bound=bool(rows) and all(row['status'].startswith('EXACT_') for row in rows),
        dynamic_DSO_implementations_qualified=False,selected_function_scope=list(contains),header_object_state_ABI_qualified=False,native_prefix_safety_qualified=False,code_executed=False)
