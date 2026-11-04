"""Build a private controller lifecycle successor; no simulator or solver execution."""
def main(argv=None):
    import argparse,json,shlex,subprocess,time
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    for k in('spec','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    from go2arx5_vla.world_model.storage import digest,write_json
    from go2arx5_vla.world_model.native_controller_lifetime import patch
    with admitted_run(a.spec,a.run_id,'core-wm02-init-build',a.output)as s:
        if s.get('physics')is not False or s.get('domain')!='TARGET_INIT_BUILD_ONLY_V1':raise PermissionError('BUILD_ONLY_IDENTITY_REQUIRED')
        for k in('controller_source','controller_binary','controller_flags','controller_link_receipt'):
            if digest(s[k])!=s[k+'_sha256']:raise ValueError('FROZEN_CONTROLLER_BUILD_INPUT_CHANGED:'+k)
        start=time.monotonic();out=a.output;before=Path(s['controller_source']).read_text();after=patch(before);cpp=out/'go2_x5_ocs2_node.cpp';cpp.write_text(after);flags={}
        for line in Path(s['controller_flags']).read_text().splitlines():
            if ' = 'in line:
                key,value=line.split(' = ',1);flags[key]=shlex.split(value)
        obj=out/'go2_x5_ocs2_node.cpp.o';compile_cmd=['/usr/bin/c++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-o',str(obj),'-c',str(cpp)]
        link=json.loads(Path(s['controller_link_receipt']).read_text())['argv'];old_obj=str(Path(s['controller_source']).with_suffix('.cpp.o'))
        if link.count(old_obj)!=1 or link[link.index('-o')+1]!=s['controller_binary']:raise ValueError('EXACT_EXISTING_CONTROLLER_LINK_REQUIRED')
        link[link.index(old_obj)]=str(obj);binary=out/'go2_x5_ocs2_node';link[link.index('-o')+1]=str(binary);commands=[]
        for name,cmd in(('compile',compile_cmd),('link',link),('loader',['ldd',str(binary)])):
            q=subprocess.run(cmd,cwd=s['controller_build_directory'],capture_output=True,text=True,timeout=180);commands.append(dict(stage=name,argv=cmd,exit_code=q.returncode));(out/(name+'.log')).write_text(q.stdout+q.stderr);write_json(out/'BUILD_COMMANDS.private.json',commands);q.check_returncode()
            if name=='loader'and'not found'in q.stdout:raise RuntimeError('CONTROLLER_LOADER_UNRESOLVED')
        result=dict(status='CONTROLLER_LIFETIME_COMPILED_NOT_NATIVE_QUALIFIED',source_sha=s['source_sha'],ancestor_binary_sha256=s['controller_binary_sha256'],binary_sha256=digest(binary),original_source_sha256=digest(s['controller_source']),patched_source_sha256=digest(cpp),controller_class_and_math_byte_identical=after.replace(__import__('go2arx5_vla.world_model.native_controller_lifetime',fromlist=['NEW']).NEW,__import__('go2arx5_vla.world_model.native_controller_lifetime',fromlist=['OLD']).OLD)==before,change='ORDERED_MRT_JOIN_BEFORE_ROS_CONTEXT_SHUTDOWN',physics_integrations=0,WM_updates=0,Actor_updates=0,wall_s=time.monotonic()-start)
        write_json(out/'RESULT.json',result);print(json.dumps(result));return 0
if __name__=='__main__':raise SystemExit(main())
