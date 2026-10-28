"""Compare saved HF header/config metadata, with no tensor or model loading."""
from pathlib import Path
import argparse
import hashlib
import json
from stage25_o1_parameter_delta_audit import SERIALIZED_TIED_ALIAS_RULES, SERIALIZED_DTYPE_CAST_RULES


def compare(root):
    data=json.loads((root/'header_comparison.json').read_text());b=data['Base']['keys'];g=data['G']['keys']
    alias=SERIALIZED_TIED_ALIAS_RULES[0]
    assert b.keys()-g.keys()=={alias['base_only_key']} and not g.keys()-b.keys()
    assert b[alias['base_only_key']]['shape']==b[alias['paired_key']]['shape']==g[alias['paired_key']]['shape']
    assert len(b)==1031 and len(g)==1030
    changed=[]
    for key in g:
        assert b[key]['shape']==g[key]['shape'],key
        if b[key]['dtype']!=g[key]['dtype']:
            assert (b[key]['dtype'],g[key]['dtype'])==('BF16','F32') and any(key.startswith(p) for rule in SERIALIZED_DTYPE_CAST_RULES for p in rule['key_prefixes']),key
            changed.append(key)
    assert len(changed)==537
    base=json.loads((root/'base_config.json').read_text());adapted=json.loads((root/'G_config.json').read_text())
    differences={k:[base.get(k),adapted.get(k)] for k in base.keys()|adapted.keys() if base.get(k)!=adapted.get(k)}
    assert set(differences)=={'backbone_trainable_params_fp32','state_dropout_prob','transformers_version','tune_llm','tune_visual','tune_vlln'}
    assert base['architectures']==adapted['architectures']==['Gr00tN1d7']
    assert base['action_horizon']==adapted['action_horizon']==40 and base['max_action_dim']==adapted['max_action_dim']==132
    assert base['max_num_embodiments']==adapted['max_num_embodiments']==32
    assert hashlib.sha256((root/'base_model.safetensors.index.json').read_bytes()).hexdigest()=='407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59'
    result=dict(status='PASS_HEADER_AND_CONFIG_LINEAGE_COMPATIBILITY_NOT_MODEL_LOAD',common_keys=len(g),dtype_pairs=len(changed),
        config_differences=differences,tied_alias=alias,public_header_only_transport_bytes=sum(data[t]['transport_bytes'] for t in ('Base','G')),
        model_tensor_data_bytes=0,GR00T_loads=0,queries=0,backward=0,optimizer=0,
        Base={k:v for k,v in data['Base'].items() if k!='keys'},G={k:v for k,v in data['G'].items() if k!='keys'},
        common_proposed_overrides=dict(tune_llm=False,tune_visual=False,tune_vlln=False,state_dropout_prob=0.,backbone_trainable_params_fp32=False,
            action_head_storage='F32 including frozen vlln; frozen backbone BF16',trainable_scopes=['projector','diffusion']),
        unverified=['actual NEW_EMBODIMENT mapping/processor migration','new TRAIN-derived C_MT','eligible multitask data and executable matched training binding'])
    (root/'initializer_metadata_receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('status','common_keys','dtype_pairs','model_tensor_data_bytes','GR00T_loads')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();compare(a.root)
