"""Describe already independently scored episodes; never run a model or simulator."""
import argparse,json
from pathlib import Path
import numpy as np
from contract import ROOT,sha,write


def report(point):
    rows=[];proofs={}
    for panel in ('TRAIN','DEV','CONFIRM_A','CONFIRM_B','ACQUISITION'):
        source=ROOT/'source/evidence'/f'{point}_{panel}.json'
        if not source.exists():continue
        cpu=ROOT/'audit'/f'{point}_{panel}_CPU.json';checked=json.loads(cpu.read_text())
        assert checked['status']=='PASS_INDEPENDENT_CPU_CONTACT_SCORE_CLOCK_INSTALLATION_AND_PAIRED_INITIAL_INPUTS'
        assert checked['source_sha256']==sha(source)
        original=json.loads(source.read_text());assert len(original['rows'])==len(checked['rows'])
        proofs[panel]=dict(source_sha256=sha(source),CPU_sha256=sha(cpu))
        for r,a in zip(original['rows'],checked['rows']):
            assert (r['task'],r['layout'],r['persistent_success'])==(a['task'],a['layout'],a['persistent_success'])
            folder=Path(r['folder']);result=json.loads((folder/'RESULT.json').read_text())
            assert sha(folder/'RESULT.json')==a['source_result_sha256']
            for name,h in result['files'].items():assert sha(folder/name)==h
            with np.load(folder/'commands.npz') as f:holds=int(f['ik_hold'].sum())
            queries=json.loads((folder/'QUERIES.json').read_text());gates=[q['gate'] for q in queries if 'gate' in q]
            predicates=a['score']['subpredicates'];hold=predicates[r['task']+'_hold']
            first=hold['first_true_s'];loss=next((e['time_s'] for e in hold['changes'] if not e['value'] and first is not None and e['time_s']>=first),None)
            videos=[dict(path=str(p),sha256=sha(p)) for p in folder.glob('*.mp4')]
            rows.append(dict(task=r['task'],layout=r['layout'],panel=panel,status=r['status'],complete_32s=a['score']['complete_32s'],
                persistent_success=r['persistent_success'],history=a['score']['history'],
                stages={n:predicates[k] for n,k in dict(grasp='grasp',lift='lift',transfer='horizontal',place='zone' if r['task']=='T1' else 'original_stack',release='not_grasped',withdraw='TCP_clear',hold=r['task']+'_hold').items()},
                first_task_hold_s=first,first_task_hold_loss_s=loss,hold_required=41,hold_observed=a['score']['hold_observed'],hold_pass=a['score']['hold_pass'],
                reference_to_installed_XYZ_RMS_m=a['reference_to_installed_XYZ_RMS_m'],installed_to_same_issued_achieved_XYZ_RMS_m=a['installed_to_same_issued_achieved_XYZ_RMS_m'],
                IK_holds=holds,gate_rejects=sum(g['decision']=='REJECT' for g in gates),model_queries=len(queries),videos=videos,video_status='ORIGINAL_MODEL_VIDEO' if videos else 'NOT_YET_RENDERED',source_result_sha256=a['source_result_sha256']))
    assert rows,'No actual CPU-closed panel exists yet'
    value=dict(status='ACTUAL_CPU_CLOSED_EPISODE_FUNNEL_NOT_NEW_QUALIFICATION',point=point,proofs=proofs,rows=rows,model_queries_added=0,physics_added=0,selected='NONE',deployment='NONE')
    write(ROOT/'r3/WP2'/point/'EPISODE_FUNNEL.json',value)
    route=ROOT/'qualification'/f'{point}_ROUTE.json'
    returned=dict(status='ACTUAL_CHECKPOINT_RETURN',point=point,proofs=proofs,panels={p:dict(requested=sum(r['panel']==p for r in rows),success=sum(r['panel']==p and r['persistent_success'] for r in rows)) for p in proofs},route=json.loads(route.read_text()) if route.exists() else 'OWNER_ROUTE_PENDING',selected='NONE',deployment='NONE')
    write(ROOT/'r3/WP2'/point/'CHECKPOINT_RETURN.json',returned)
    return returned


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',required=True);a=p.parse_args();print(json.dumps(report(a.point)))
