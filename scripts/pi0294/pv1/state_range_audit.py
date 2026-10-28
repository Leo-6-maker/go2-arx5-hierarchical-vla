"""Offline saved-state audit. No model, image reader, native writes or transport.

Official numeric functions and state-only tensor slices are compiled unchanged
from caller-supplied, hash-bound source. This is not full VLM preprocessing.
"""
import ast
from copy import deepcopy
import hashlib
from types import SimpleNamespace
import numpy as np

CHANNELS=('x_m','y_m','z_m','R00','R01','R02','R10','R11','R12','gripper')


def sha(raw):return hashlib.sha256(raw).hexdigest()


def compile_nodes(nodes,namespace):
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+deepcopy(nodes),type_ignores=[])
    ast.fix_missing_locations(tree)
    exec(compile(tree,'<hash-bound-official-state-only>','exec'),namespace)


def official_state_path(utils_source,processor_source,tensor_source,statistics,torch):
    """Use exact official apply_state and exact train/eval tensor statements."""
    ns=dict(np=np,deepcopy=deepcopy,torch=torch)
    utils=ast.parse(utils_source)
    compile_nodes([n for n in utils.body if isinstance(n,ast.FunctionDef) and n.name in ('normalize_values_minmax','unnormalize_values_minmax')],ns)
    tree=ast.parse(processor_source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StateActionProcessor')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='apply_state')
    compile_nodes([fn],ns)
    groups=['eef_9d','gripper_position']
    params={g:{k:np.asarray(v) for k,v in statistics['state'][g].items()} for g in groups}
    config=SimpleNamespace(modality_keys=groups)
    sap=SimpleNamespace(apply_sincos_state_encoding=False,clip_outliers=False,
        modality_configs={'new_embodiment':{'state':config}},norm_params={'new_embodiment':{'state':params}})
    sap.apply_state=lambda state,embodiment_tag:ns['apply_state'](sap,state,embodiment_tag)
    owner=SimpleNamespace(state_action_processor=sap,modality_configs=sap.modality_configs,
        exclude_state=False,state_dropout_prob=0.,training=True,max_state_dim=132)
    cls=next(n for n in ast.parse(tensor_source).body if isinstance(n,ast.ClassDef) and n.name=='Gr00tN1d7Processor')
    methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    def target(n,name):return isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
    train=methods['__call__'].body
    begin=next(i for i,n in enumerate(train) if target(n,'state_keys'))
    end=next(i for i in range(begin,len(train)) if target(train[i],'image_transform') or isinstance(train[i],ast.If) and any(target(x,'image_transform') for x in train[i].body))
    nodes=train[begin:end]
    # The original final state conversion, not a handwritten float transform.
    final=next(n for n in train if target(n,'transformed_inputs'))
    state_expr=next(v for k,v in zip(final.value.keys,final.value.values) if isinstance(k,ast.Constant) and k.value=='state')
    eval_nodes=methods['process_observation'].body
    eval_end=next(i for i,n in enumerate(eval_nodes) if target(n,'image_keys'))
    eval_nodes=eval_nodes[1:eval_end]  # skip docstring, stop before image reading
    def make(name,args,body,ret):
        f=ast.FunctionDef(name=name,args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=a) for a in args],kwonlyargs=[],kw_defaults=[],defaults=[]),body=deepcopy(body)+[ast.Return(value=ret)],decorator_list=[])
        compile_nodes([f],ns)
    make('training_state',['self','state_data','norm_state_dict','embodiment_tag'],nodes,state_expr)
    make('inference_state',['self','observation','embodiment_tag'],eval_nodes,ast.Subscript(value=ast.Name(id='transformed_observation',ctx=ast.Load()),slice=ast.Constant(value='state'),ctx=ast.Load()))
    collator_class=next(n for n in ast.parse(tensor_source).body if isinstance(n,ast.ClassDef) and n.name=='Gr00tN1d7DataCollator')
    collator_method=deepcopy(next(n for n in collator_class.body if isinstance(n,ast.FunctionDef) and n.name=='__call__'))
    # Run all original numeric statements, omitting only BatchFeature container
    # construction; no VLM content is provided or tokenized in this audit.
    collator_method.name='collate_state_only'
    collator_method.body[-1]=ast.Return(value=ast.Name(id='batch',ctx=ast.Load()))
    compile_nodes([collator_method],ns)
    tag=SimpleNamespace(value='new_embodiment')
    def apply(states):
        data={'eef_9d':states[:,:9],'gripper_position':states[:,9:]}
        normalized=sap.apply_state(data,'new_embodiment')
        train_tensor=ns['training_state'](owner,data,normalized,tag)
        # Actual inference has B,T,D; run the same values with a singleton T.
        obs={'state.'+k:v[:,None,:] for k,v in data.items()}
        inference_tensor=ns['inference_state'](owner,obs,tag)[:,0]
        collated=ns['collate_state_only'](None,[{'state':train_tensor[i:i+1]} for i in range(len(states))])['state'][:,0]
        if not torch.equal(collated,train_tensor):raise ValueError('FINAL_COLLATOR_CHANGED_STATE')
        return np.concatenate([normalized[g] for g in groups],axis=-1),collated,inference_tensor
    return apply,ns,dict(apply_state_AST_sha256=sha(ast.dump(fn,include_attributes=False).encode()),
        training_state_AST_sha256=sha(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()),
        inference_state_AST_sha256=sha(ast.dump(ast.Module(body=eval_nodes,type_ignores=[]),include_attributes=False).encode()),
        source_sha256=dict(utils=sha(utils_source.encode()),state_action_processor=sha(processor_source.encode()),processor=sha(tensor_source.encode())),
        collator_numeric_AST_sha256=sha(ast.dump(collator_method,include_attributes=False).encode()),
        exact_official_state_only_slices=True,full_processor_images_or_model_executed=False)


def contiguous_runs(anchors):
    result=[]
    for a in sorted(set(int(x) for x in anchors)):
        if result and a==result[-1][1]+1:result[-1][1]=a
        else:result.append([a,a])
    return result


def state_records(states,statistics,apply,*,episode_id,bank,task,visits):
    if states.shape not in ((601,10),(641,10)) or not np.isfinite(states).all():raise ValueError('EXACT_FINITE_SAVED_STATE10')
    normalized,train,inference=apply(states)
    if train.shape!=(len(states),132) or inference.shape!=(len(states),132):raise ValueError('STATE_TENSOR_SHAPE')
    if (not np.array_equal(train[:,:10].numpy(),normalized.astype(np.float32))
        or not np.array_equal(inference[:,:10].numpy(),normalized)
        or np.count_nonzero(train[:,10:].numpy()) or np.count_nonzero(inference[:,10:].numpy())):
        raise ValueError('STATE_TENSOR_ALTERED_BEYOND_DECLARED_FLOAT32_AND_PADDING')
    bounds={k:np.concatenate([np.asarray(statistics['state'][g][k]) for g in ['eef_9d','gripper_position']]) for k in ['min','max']}
    lo,hi=bounds['min'],bounds['max'];mask=np.isclose(hi,lo)
    components=[]
    for anchor,ch in zip(*np.where(np.abs(normalized)>1)):
        raw=float(states[anchor,ch]);z=float(normalized[anchor,ch]);outside=max(lo[ch]-raw,raw-hi[ch],0.)
        components.append(dict(bank=bank,episode_id=episode_id,task=task,anchor=int(anchor),source_relative_s=int(anchor)/20,
            timestamp_semantics='SOURCE_EXPORT_VALIDATED_20HZ_GRID',channel=int(ch),channel_name=CHANNELS[ch],raw_value=raw,
            R0_min=float(lo[ch]),R0_max=float(hi[ch]),R0_span=float(hi[ch]-lo[ch]),near_constant=bool(mask[ch]),
            normalized_float64=z,excess_normalized=max(abs(z)-1,0),physical_component_excess=float(outside),
            physical_excess_over_R0_span=float(outside/(hi[ch]-lo[ch])),training_tensor_float32=float(train[anchor,ch]),
            inference_tensor_float64=float(inference[anchor,ch]),legal_TRAIN_anchor=bool(anchor<601),
            actual_visits={a:int(visits.get(a,{}).get(str(anchor),0)) for a in ['A_CONTROL','B_COVERAGE']}))
    legal={r['anchor'] for r in components if r['legal_TRAIN_anchor']}
    summary=dict(bank=bank,episode_id=episode_id,task=task,saved_states=len(states),legal_TRAIN_anchors=601,
        state_OOR_components_legal=sum(r['legal_TRAIN_anchor'] for r in components),
        state_OOR_components_all_saved=len(components),unique_OOR_anchors_legal=len(legal),continuous_runs_legal=contiguous_runs(legal),
        actual_OOR_visits={a:sum(int(v) for k,v in visits.get(a,{}).items() if int(k) in legal) for a in ['A_CONTROL','B_COVERAGE']},
        raw_min=states.min(axis=0).tolist(),raw_max=states.max(axis=0).tolist(),
        normalized_min=normalized.min(axis=0).tolist(),normalized_max=normalized.max(axis=0).tolist(),
        near_constant_active_channels=np.where(mask)[0].tolist(),state_path_exact=True,finite_tensor=True,final_collator_state_preserved=True,
        declared_train_cast='float64 -> torch default float32; zero padding 10 -> 132',inference_cast='preserved float64 by state-only observation path')
    return summary,components
