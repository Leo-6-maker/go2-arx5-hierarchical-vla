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


def function_relocations(obj,name):
    function=obj.symbols()[name]
    for i,section in enumerate(obj.sections):
        if section[1]!=4 or section[7]!=function['section']:continue
        check(section[9]==24 and section[5]%24==0,'BIND_RELA_SIZE')
        symbols=symbol_rows(obj,section[6]);raw=obj.section(i)
        for at in range(0,len(raw),24):
            offset,info,addend=struct.unpack_from('<QQq',raw,at)
            if not function['begin']<=offset<function['begin']+function['size']:continue
            index=info>>32;check(index<len(symbols),'BIND_RELOCATION_SYMBOL_INDEX')
            yield dict(offset=offset-function['begin'],kind=info&0xffffffff,addend=addend,target=symbols[index])


def inspect(binary,archives,contains=('RL','ObservationBuffer','InferenceRuntime')):
    linked=ELF(binary);check(linked.kind in (2,3),'BIND_LINKED_EXECUTABLE')
    names=definitions(linked);functions=linked.symbols();rows=[]
    for archive,raw in archives.items():
        for member,obj in members(raw):
            for name,function in obj.symbols().items():
                if name not in functions or not any(word in name for word in contains):continue
                if len(obj.code(name))!=len(linked.code(name)):
                    rows.append(dict(archive=archive,member=member,function=name,status='FUNCTION_SIZE_MISMATCH'));continue
                for relocation in function_relocations(obj,name):
                    target=relocation['target'];kind=relocation['kind'];offset=relocation['offset']
                    row=dict(archive=archive,member=member,function=name,offset=offset,relocation_kind=kind,
                        addend=relocation['addend'],object_target=target)
                    # Named defined functions/objects have exact link addresses.
                    # Never guess stripped/local/PLT/GOT targets from body match.
                    addresses=names.get(target['name'],set()) if target['name'] else set()
                    values={a[0] for a in addresses}
                    if kind not in (1,2,4,10,11):row['status']='UNSUPPORTED_BINDING_KIND'
                    elif len(values)!=1:row['status']='UNRESOLVED_OR_AMBIGUOUS_TARGET'
                    else:
                        width=8 if kind==1 else 4;start=offset;end=start+width
                        check(end<=len(linked.code(name)),'BIND_FIELD_OUTSIDE_FUNCTION')
                        symbol=next(iter(values));place=functions[name]['value']+offset
                        expected=symbol+relocation['addend']-(place if kind in (2,4) else 0)
                        actual=int.from_bytes(linked.code(name)[start:end],'little')
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
        all_relocations_resolved=bool(rows) and all(row['status']=='EXACT_NAMED_BINDING_MATCH' for row in rows),
        header_object_state_ABI_qualified=False,native_prefix_safety_qualified=False,code_executed=False)
