"""Two finite CPU scene compilations, zero integration and zero admission."""
import json,time,traceback
from pathlib import Path
import numpy as np
from scripts.pi0292.durable import write
from go2arx5_vla.world_model.storage import digest


def asset_binding(template):
    """Resolve only the two already-frozen XML members, without a path scan."""
    members={}
    for name in('go2_x5.xml','scene.xml'):
        matches=[Path(p)for p in template['frozen_files']if Path(p).name==name]
        if len(matches)!=1:raise ValueError('UNIQUE_FROZEN_NATIVE_ASSET_XML_REQUIRED')
        path=matches[0]
        if digest(path)!=template['frozen_files'][str(path)]:raise ValueError('FROZEN_NATIVE_ASSET_XML_CHANGED')
        members[name]=dict(path=str(path),sha256=digest(path))
    roots={str(Path(p['path']).parent)for p in members.values()}
    if len(roots)!=1:raise ValueError('SAME_FROZEN_NATIVE_ASSET_DIRECTORY_REQUIRED')
    root=roots.pop()
    if template.get('asset_root',root)!=root:raise ValueError('EXPLICIT_ASSET_ROOT_DIFFERS_FROM_FROZEN_XML')
    return root,members


def run(spec):
    import mujoco
    from go2arx5_vla.world_model.multitask_scene_v6 import overlay_xml,robot_contract
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from scripts.pi0294.task_contacts import allowed
    root=Path(spec['registered_root']);start=time.monotonic();cpu=time.process_time();records=[]
    if(spec['physics']or spec['new_physical_requests_budget']or spec['new_model_generations_budget']or spec['optimizer_updates']
       or spec['plan_commit']!='ae3d3c0dceade437b0104a150cd0402fef650052' or len(spec['templates'])!=2 or mujoco.get_mjcb_control()is not None):
        raise PermissionError('FINITE_V6_CPU_SCENE_AUDIT_ONLY_NO_CALLBACK_OR_NATIVE')
    for family,pin in spec['templates'].items():
        if digest(pin['path'])!=pin['sha256']:raise ValueError('OLD_TRAIN_SUPPORT_TEMPLATE_CHANGED')
        template=json.loads(Path(pin['path']).read_bytes());asset_root,assets=asset_binding(template);template=dict(template,asset_root=asset_root);path=template['model_path']
        if digest(path)!=template['frozen_files'][path]:raise ValueError('ORIGINAL_FROZEN_MODEL_BYTES_CHANGED')
        reuse=spec.get('reuse_static_family',{}).get(family)
        if reuse:
            if digest(reuse['result']['path'])!=reuse['result']['sha256']or digest(reuse['spec']['path'])!=reuse['spec']['sha256']:raise ValueError('SEALED_STATIC_CANDIDATE_REUSE_CHANGED')
            prior=json.loads(Path(reuse['result']['path']).read_bytes());old_spec=json.loads(Path(reuse['spec']['path']).read_bytes())
            relative='src/go2arx5_vla/world_model/multitask_scene_v6.py'
            if old_spec['source_files'][relative]!=spec['source_files'][relative]:raise ValueError('STATIC_SCENE_BUILDER_CHANGED_BEFORE_REUSE')
            row=next(r for r in prior['records']if r['family']==family)
            if not row['valid_static_candidate']or row['template']!=pin or row['original_scene_sha256']!=digest(path)or digest(row['model']['path'])!=row['model']['sha256']:
                raise ValueError('EXACT_SUCCESSFUL_STATIC_CANDIDATE_ONLY_NO_RECOMPILE')
            records.append(dict(row,reused_static_result=reuse['result'],new_compilations=0));continue
        original=mujoco.MjModel.from_binary_path(path);cube=original.body('task_cube').id;address=int(original.jnt_qposadr[original.body_jntadr[cube]])
        origin=original.key_qpos[0,address:address+3].copy();axis=template['mobile_condition']['axis']
        offset=np.asarray(spec.get('prototype_origin_offsets',{}).get(family,[0.,0.,0.]),float)
        if offset.shape!=(3,)or not np.isfinite(offset).all()or offset[2]!=0:raise ValueError('PREDECLARED_STATIC_XY_LAYOUT_OFFSET_REQUIRED')
        origin+=offset
        direction={'+x':[1.,0.],'-x':[-1.,0.],'+y':[0.,1.],'-y':[0.,-1.]}[axis];u=np.asarray(direction);v=np.asarray([-u[1],u[0]])
        card=dict(object_positions_world={c:(origin+np.r_[sign*.045*v,0.]).tolist()for c,sign in(('red',-1),('blue',1))},
            zones={z:dict(center_world_XY=(origin[:2]+.09*u+sign*.06*v).tolist(),half_size_XY=[.035,.035])for z,sign in(('left',1),('right',-1))},direction_world_XY=direction,prototype_origin_offset_world_m=offset.tolist())
        folder=root/family;folder.mkdir();record=dict(family=family,template=pin,asset_XML_binding=assets,original_scene_sha256=digest(path),card=card,valid_static_candidate=False)
        try:
            xml,binding=overlay_xml(template,original,card);xml_path=folder/'SCENE.private.xml';xml_path.write_text(xml)
            with forbid_integration(mujoco)as integrations:
                candidate=mujoco.MjModel.from_xml_string(xml);contract=robot_contract(original,candidate);data=mujoco.MjData(candidate);mujoco.mj_resetDataKeyframe(candidate,data,0);mujoco.mj_forward(candidate,data)
                if integrations:raise RuntimeError('SCENE_AUDIT_ATTEMPTED_PHYSICS')
            model=folder/'SCENE.private.mjb';mujoco.mj_saveModel(candidate,str(model),None)
            record.update(valid_static_candidate=True,binding=binding,robot_contract=contract,model=dict(path=str(model),sha256=digest(model)),
                compiled_ABI=dict(nq=candidate.nq,nv=candidate.nv,nu=candidate.nu),old_reader_accepts_ABI=False,
                existing_contact_predicates=dict(red_support=allowed('task_cube_geom','task_table'),blue_support=allowed('task_blue_cube_geom','task_table'),
                    pad_table=allowed('task_left_pad','task_table'),wrong_object_pad=allowed('task_left_pad','task_blue_cube_geom')))
        except (ValueError,AssertionError)as error:record.update(reason=str(error),traceback=traceback.format_exc())
        records.append(record);write(root/'PROGRESS.private.json',dict(phase='ZERO_INTEGRATION_MULTITASK_SCENE_AUDIT',completed=len(records),new_native_requests=0,new_integrations=0,new_updates=0))
    result=dict(exit_code=0,status='STATIC_SCENE_CANDIDATES_ONLY_NATIVE_BINDING_UNQUALIFIED',source_sha=spec['source_sha'],records=records,
        valid_candidates=sum(r['valid_static_candidate']for r in records),new_native_requests=0,new_integrations=0,new_updates=0,new_visits=0,new_model_generations=0,
        native_dispatch_qualified=False,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu)
    return result


def main(spec):
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    try:result=run(spec)
    except BaseException:result=dict(exit_code=1,status='PRESERVED_STATIC_MULTITASK_TECHNICAL_FAILURE',failure=traceback.format_exc(),new_native_requests=0,new_integrations=0,new_updates=0,new_model_generations=0)
    write(Path(spec['registered_root'])/'RESULT.private.json',result);close(spec,result['exit_code']);return result['exit_code']
