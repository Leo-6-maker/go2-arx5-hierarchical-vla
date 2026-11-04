"""PC full Go2+ARX5 saved-mesh replay, no target executor or integration."""
import argparse,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in('spec','config','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--mode',choices=('render-check','passive'),default='render-check');p.add_argument('--frames',type=int,default=120);p.add_argument('--max-wall',type=float,default=30);p.add_argument('--rate',type=float,default=1.)
    a=p.parse_args(argv)
    if not 1<=a.frames<=900 or not 1<=a.max_wall<=600 or not .01<=a.rate<=4:p.error('Finite replay endpoint/rate required')
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,'local-PC-offline',a.output)as spec:
        if spec.get('physics')is not False or spec.get('domain')!='PC_NATIVE_REPLAY_ONLY_V1':raise PermissionError('Only registered passive PC replay')
        config=json.loads(a.config.read_text());config['source_sha']=spec['source_sha']
        from go2arx5_vla.world_model.native_replay import render_check,passive
        result=render_check(config,a.output,a.frames)if a.mode=='render-check'else passive(config,a.output,a.max_wall,a.rate);print(json.dumps(result))
    return 0
if __name__=='__main__':raise SystemExit(main())
