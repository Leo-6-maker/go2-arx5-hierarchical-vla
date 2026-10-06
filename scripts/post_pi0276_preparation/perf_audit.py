"""Read-only performance inventory from saved configuration and source, never a model."""
import argparse,ast,hashlib,json,re
from pathlib import Path

def audit(config,trainer_state,source,args_opcodes=None,trainer_log=None):
    text=Path(config).read_text();state=json.loads(Path(trainer_state).read_text());code=Path(source).read_text()
    def setting(key):
        values=re.findall(r'^\s*'+re.escape(key)+r':\s*([^\n]+)',text,re.M)
        return values[-1].strip() if values else 'NOT_OBSERVED'
    history=state.get('log_history',[]);end=[v for v in history if 'train_runtime' in v]
    sites=[dict(line=i,text=line.strip()) for i,line in enumerate(code.splitlines(),1) if any(s in line for s in ('float(','.item(','.cpu(','get_train_dataloader','sample_audit','torch.save'))]
    result=dict(status='SAVED_CONFIG_AND_SOURCE_AUDIT',sources={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (config,trainer_state,source)},
        resolved_loader_settings={k:setting(k) for k in ('dataloader_num_workers','dataloader_pin_memory','per_device_train_batch_size','gradient_accumulation_steps','bf16')},
        final_training_summary=end,global_step=state['global_step'],candidate_sync_and_IO_sites=sites,
        unavailable={k:'NOT_OBSERVED' for k in ('GPU_utilization','data_dequeue_wait','H2D_timeline','GPU_compute_timeline','checkpoint_IO_duration','network_archive_duration','canary_setup_duration')},
        profiler_insertion_map=['data dequeue','CPU processor','input/label hash','H2D','native forward','S4 generation and backward','detached metric transfers and journal writes','optimizer','checkpoint'],
        future_GPU_timing='CUDA events/profiler or synchronized interval endpoints; no per-operator synchronization',
        claim='Saved train_runtime is the Trainer interval including its bookkeeping, not pure GPU compute.',new_model_calls=0)
    if args_opcodes:
        lines=Path(args_opcodes).read_text().splitlines()
        for key in ('dataloader_pin_memory','per_device_train_batch_size','dataloader_num_workers'):
            matches=[i for i,line in enumerate(lines) if "'"+key+"'" in line]
            if len(matches)!=1:raise ValueError('Resolved argument opcode is missing or ambiguous: '+key)
            values='\n'.join(lines[matches[0]+1:matches[0]+4])
            match=re.search(r'BININT1\s+(\d+)',values)
            if key=='dataloader_pin_memory':
                if 'NEWTRUE' not in values and 'NEWFALSE' not in values:raise ValueError('Unknown saved Boolean opcode')
                value='NEWTRUE' in values
            else:
                if not match:raise ValueError('Unknown saved integer opcode')
                value=int(match.group(1))
            result['resolved_loader_settings'][key]=value
        result['sources'][str(args_opcodes)]=hashlib.sha256(Path(args_opcodes).read_bytes()).hexdigest()
        result['resolved_args_evidence']='Saved pickle opcode inspection without unpickling objects'
    if trainer_log:
        result['final_training_log_lines']=[line for line in Path(trainer_log).read_text().splitlines() if 'train_runtime' in line]
        result['sources'][str(trainer_log)]=hashlib.sha256(Path(trainer_log).read_bytes()).hexdigest()
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--trainer-state',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--args-opcodes',type=Path);p.add_argument('--trainer-log',type=Path);a=p.parse_args()
    value=audit(a.config,a.trainer_state,a.source,a.args_opcodes,a.trainer_log);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value['resolved_loader_settings']))
