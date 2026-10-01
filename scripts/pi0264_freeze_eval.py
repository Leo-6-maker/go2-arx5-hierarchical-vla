"""Seal old regressions and actual new text/noise inputs before any PI0264 canary."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0264')
OLD=ROOT.with_name('PI0263')
sys.path.insert(0,str(OLD/'code'))
import pi0263_eval as previous
from pi0264_plan import read,sha,write,TEMPLATES,CHALLENGE,AUTHORITY
from pi0260_r3_inputs import batch


def new_packet(source):
    packet,expert=previous.crossed.packet(source['base'])
    packet['language']=source['text']
    return packet,expert


def noise(row):
    if row.get('noise_file'):
        p=ROOT/row['noise_file']
    else:p=previous.r3.R3/'noise'/(row['noise']+'.npz')
    with np.load(p,allow_pickle=False) as z:return z['float32'].copy()


def source(row):
    return previous.source(row) if row['kind'] in ('legacy','crossed') else new_packet(row['source_row'])


def main():
    import torch
    old=read(OLD/'UNION_EVAL_CARD.json')
    binding=read(OLD/'model_bindings/X2000.json')
    assert binding['checkpoint_tree_sha256']=='4ec3b3a44ad40a057cfdcd7ad992a86b68889b4efba98be70ed109e571497694'
    assert '@a2f9f767a11e4af269377603f3da432f4931d623/' in binding['source_revision']
    rows=[]
    for oldrow in old['rows']:
        cell=OLD/'eval/X2000/cells'/oldrow['uid'];record=read(cell/'record.json')
        assert record['status']=='VALID' and record['model_binding']==binding
        assert record['row']==oldrow and record['eval_card_sha256']==sha(OLD/'UNION_EVAL_CARD.json')
        assert sha(cell/'tensors.npz')==record['tensor_sha256']
        rows.append(dict(oldrow,donor_cell=str(cell),donor_record_sha256=sha(cell/'record.json'),
                         donor_tensor_sha256=record['tensor_sha256'],panel='A'))
    assert len(rows)==1152
    original_cross={}
    for a in old['aliases']:
        if a['panel']=='crossed':
            r=a['source_row']
            if r['template']=='T0' and r['noise']=='B0':original_cross[r['split'],r['family'],r['command']]=r
    train=sorted({f for s,f,c in original_cross if s=='TRAIN'})
    val=sorted({f for s,f,c in original_cross if s=='KIN-VAL'})
    assert len(train)==30 and len(val)==10
    # Identical mature torch CPU FP32 normal generator and BF16 bit cast; new deterministic seeds only.
    fresh=[];digests={}
    for family in val:
        for name in ('N0','N1'):
            key=f'PI0264:fresh-eval:26441:{family}:{name}'
            seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')
            f32=torch.randn((1,40,132),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32)
            bits=f32.to(torch.bfloat16).view(torch.uint16).numpy()
            p=ROOT/'noise'/family/(name+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():
                with np.load(p,allow_pickle=False) as z:
                    assert np.array_equal(z['float32'],f32.numpy()) and np.array_equal(z['bfloat16_bits'],bits)
            else:np.savez_compressed(p,float32=f32.numpy(),bfloat16_bits=bits)
            digest=previous.digest(f32.numpy())
            coincidences=[label for label,d in digests.items() if d==digest]
            for oldnoise in ('B0','B1'):
                with np.load(previous.r3.R3/'noise'/(oldnoise+'.npz'),allow_pickle=False) as z:
                    if np.array_equal(z['float32'],f32.numpy()):coincidences.append(oldnoise)
            digests[family+'/'+name]=digest
            fresh.append(dict(family=family,name=name,key=key,seed=seed,shape=[1,40,132],
                seed_conversion='SHA256 UTF8 key; first8 digest bytes interpreted unsigned big-endian uint64 for torch.manual_seed.',
                generator='torch CPU float32 standard normal',float32_sha256=digest,
                bfloat16_bits_sha256=previous.digest(bits),file=p.relative_to(ROOT).as_posix(),file_sha256=sha(p),
                accidental_byte_coincidences=coincidences,shared_across='both commands,all six formulations,all three points'))
    write(ROOT/'FRESH_NOISE_MANIFEST.json',dict(status='PI0264_ACTUAL_N0_N1_BEFORE_MODEL_FROZEN',rows=fresh,
        generator_source_sha256=sha(previous.r3.R3/'pi0260_r3_noise.py'),model_queries=0,quality_resampling=False))
    proc=previous.crossed.processor();text_checks=[]
    for panel,split,families,templates,packets in (
        ('B','TRAIN',train,{k:TEMPLATES[k] for k in ('T2','T3','T4','T5','T6','T7')},('B0','B1')),
        ('C','KIN-VAL',val,{**CHALLENGE,**{k:TEMPLATES[k] for k in ('T0','T1')}},('N0','N1'))):
        for family in families:
            for command in (0,1):
                base=original_cross[split,family,command]
                for template,pattern in templates.items():
                    text=pattern.format(color=base['color'])
                    spec=dict(base=base,split=split,family=family,command=command,color=base['color'],template=template,text=text)
                    packet,expert=new_packet(spec)
                    policy=batch(proc,packet)['inputs']
                    import re
                    normalized=re.sub(r'[^\w\s]','',text.lower()) if proc.formalize_language else text
                    visible=policy['input_ids'][policy['attention_mask'].bool()].tolist()
                    decoded=proc.collator.processor.tokenizer.decode(visible,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    assert normalized in decoded,('TARGET_ORIENTATION_CLAUSE_TRUNCATED',text)
                    text_checks.append(dict(panel=panel,family=family,command=command,template=template,text=text,
                        language_sha256=hashlib.sha256(text.encode()).hexdigest(),token_count=len(visible),
                        inherited_formalize_language=proc.formalize_language,decoded_contains_whole_language=True))
                    for name in packets:
                        row=dict(kind='new_text',source_row=spec,uid=f'u-{len(rows):04d}',noise=name,panel=panel)
                        if panel=='C':row['noise_file']=f'noise/{family}/{name}.npz'
                        row['identities']=previous.tuple_identity(policy,packet,expert,noise(row))
                        rows.append(row)
    assert len(rows)==2112 and {p:sum(r['panel']==p for r in rows) for p in ('A','B','C')}=={'A':1152,'B':720,'C':240}
    # Logical rows retain distinct labels. Accidental exact forward identity collisions are documented, never resampled.
    write(ROOT/'EVAL_TEXT_INPUT_CHECKS.json',dict(status='PASS_ACTUAL_NEW_EVAL_LANGUAGE_AND_INPUT_SEAL',rows=text_checks,model_queries=0))
    write(ROOT/'UNION_EVAL_CARD.json',dict(schema='PI0264_FIXED_THREE_PANEL_6336_V1',authority_commit=AUTHORITY,
        source_card_sha256=sha(OLD/'UNION_EVAL_CARD.json'),statistics_sha256=sha(previous.r3.ROOT/'dataset/statistics.json'),
        legacy_noise_manifest_sha256=sha(previous.r3.R3/'noise_manifest.json'),
        fresh_noise_manifest_sha256=sha(ROOT/'FRESH_NOISE_MANIFEST.json'),donor_binding=binding,
        rows=rows,legacy_aliases=old['aliases'],points=['X2000','B2048','L2048'],
        rows_per_point=2112,logical_rows=6336,donor_old_reuse=1152,planned_new_queries=5184,
        new_attempt_cap=5440,inference_head_cap=21760,invalid_only_reserve_cap=256,
        model_queries_before_freeze=0,quality_release_condition='Both B/L complete final2048 immutable model bindings and main integrity receipt.',
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print(json.dumps(dict(status='PASS_PI0264_PRE_CANARY_ALL_PANELS_AND_ACTUAL_NOISE_SEALED',rows=2112,model_queries=0,
        card_sha256=sha(ROOT/'UNION_EVAL_CARD.json'))))


if __name__=='__main__':main()
