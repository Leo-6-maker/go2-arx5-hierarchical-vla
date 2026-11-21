"""Read an archived prediction through the original verifier and retain every native margin."""
import json
import types

import numpy as np


def per_point(raw, state, policy_from_world, reference, previous, expected):
    from go2arx5_vla.sla_adapter.dense_g7_v6 import label_margin_report
    from go2arx5_vla.world_model import mfree_query_diagnostics_v2 as native
    from scripts.sla.replay_scale20k_core import guard_signature

    captured = []
    original = native.analyze
    def series(path, **kwargs):
        result = native.path_series(path, **kwargs)
        captured.append({name:dict(magnitudes=v['magnitudes'],limit=v['limit'])
                         for name,v in result.items() if isinstance(v,dict) and v.get('limit') is not None})
        return result
    # An isolated function namespace intercepts the two unchanged path_series calls.
    private = dict(original.__globals__, path_series=series)
    observed = types.FunctionType(original.__code__,private,original.__name__,original.__defaults__,original.__closure__)
    envelope = observed(raw,state,policy_from_world,reference)
    seam = label_margin_report(raw,previous)
    if (envelope!=expected['envelope'] or seam!=expected['seam'] or
            json.loads(json.dumps(guard_signature(envelope,seam)))!=expected['guard'] or len(captured)!=2):
        raise ValueError('Full replay of the original envelope differs from the sealed values')
    return dict(zip(('nominal','reference'),captured,strict=True))


def margin_comparison(lhs,rhs,seam_left,seam_right):
    result={}
    for path in ('nominal','reference'):
        if set(lhs[path])!=set(rhs[path]):
            raise ValueError('Guarded channels differ')
        result[path]={}
        for name,a in lhs[path].items():
            b=rhs[path][name]
            n=40 if name.endswith('acceleration') else 39
            la,lb=np.asarray(a['magnitudes'],float),np.asarray(b['magnitudes'],float)
            if (a['limit']!=b['limit'] or la.shape!=(n,) or lb.shape!=(n,) or
                    not np.isfinite([la,lb]).all() or a['limit']<=0):
                raise ValueError('Wrong native margin series')
            result[path][name]=compare_magnitudes(la,lb,a['limit'])
    # The core GRIP_LIMIT is frozen by the original 40 mm/s contract.
    result['previous_actual_grip_seam']=compare_magnitudes(np.asarray([seam_left]),np.asarray([seam_right]),.040,cutoff=.040+1e-10)
    return result


def compare_magnitudes(a,b,limit,*,cutoff=None):
    cutoff = limit*(1+1e-10) if cutoff is None else cutoff
    ma,mb=limit-a,limit-b
    diff=np.abs(a-b)
    allowed=.1*np.minimum(np.abs(ma),np.abs(mb))
    bad=np.flatnonzero(diff>allowed)
    return dict(limit=limit,minimum_signed_margin=[float(ma.min()),float(mb.min())],
                min_margin_after_difference=float(np.min(allowed-diff)),first_margin_violation=int(bad[0]) if len(bad) else None,
                same_original_exceedance=bool(np.array_equal(a>cutoff,b>cutoff)),
                within_reserved_margin=not len(bad))


def recover_site(root, config_sha, mode, output):
    """CPU-only sidecar. Recover state and prior command from the pinned site's reader."""
    import hashlib
    from pathlib import Path
    from hopper_numeric_probe import ORDINALS, PANEL, pin, record
    from compare_phase0_numeric import evidence
    from scripts.sla.hopper_site import pinned

    cfg = pin(root / 'CONFIG.json', config_sha)
    if cfg['site'] == 'core':
        from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
        from openpi_client.image_tools import resize_with_pad
        source = pin(Path(cfg['receiver_config']), cfg['receiver_sha256'])
        target = pinned(source['input'])['target_D1']
        data = AdaptTargetDataset(target['legacy'], target['captures'], resize_with_pad=resize_with_pad)
        data.bind_state(target['statistics'])
        panel = pin(Path(source['panel']['path']), PANEL)
    elif cfg['site'] == 'hopper':
        from scripts.sla.hopper_panels import FixedPanels
        panels = FixedPanels(Path(cfg['view'])/'assets.verified', Path(cfg['eval_assets']))
        panel, data = panels.formal, panels.datasets['target']
    else:
        raise ValueError('Unknown site')
    ids = {'ordinary2':ORDINALS[:2], 'ordinary14':ORDINALS[2:], 'ordinary16':ORDINALS}[mode]
    folder = root/mode
    if json.loads((folder/'RESULT.json').read_bytes())['completed'] != len(ids):
        raise ValueError('Incomplete observation set')
    rows = {}
    for ordinal in ids:
        rec, a = evidence(folder, ordinal)
        fixed = panel['formal']['target'][ordinal]
        index = fixed['index']
        context = panel['target_contexts'][str(index)]
        previous = data.reference(index)
        if (fixed['noise_id'] != rec['noise_id'] or index != rec['index'] or
                data.identity(index) != fixed['identity'] or previous != context['previous_command']):
            raise ValueError('Wrong fixed observation or actual command history')
        state = data.raw(index)[0]
        values = per_point(a['physical'],state,np.asarray(context['policy_from_world']),
                           np.asarray(context['reference']),previous,rec)
        rows[str(ordinal)] = dict(artifact=rec['artifact_sha256'],paths=values,
                                  seam=rec['seam']['first_seam_speed_m_s'])
    record(output,dict(config_sha256=config_sha,mode=mode,site=cfg['site'],rows=rows))
    return hashlib.sha256(output.read_bytes()).hexdigest()


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('config_sha')
    parser.add_argument('mode',choices=['ordinary2','ordinary14','ordinary16'])
    parser.add_argument('private_output',type=Path)
    args=parser.parse_args()
    import os
    os.umask(0o077)
    print(recover_site(args.root,args.config_sha,args.mode,args.private_output))
