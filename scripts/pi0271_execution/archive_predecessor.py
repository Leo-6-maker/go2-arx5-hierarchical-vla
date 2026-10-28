"""Finish PI0270 preservation with the existing qualified archive/readback function."""
import argparse
import ast
import importlib.util
import os
from pathlib import Path

BASE = Path('/home/liuyu-nfs/Projects/go2-arx5')
ROOT = BASE/'PI0270'
RAW = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270')
SOURCE_SHA = '7bc5044261252c588394b3dfc6796ca92ef123efcd4500bd3a4e5f23d917dd26'


def prepared():
    path=BASE/'PI0269/code/archive_results.py'
    spec=importlib.util.spec_from_file_location('pi0270_reused_archive',path)
    mature=importlib.util.module_from_spec(spec);spec.loader.exec_module(mature)
    assert mature.sha(path)==SOURCE_SHA
    os.environ.update(HF_HOME=str(ROOT/'runtime/hf_online'),HF_XET_CACHE=str(ROOT/'runtime/hf_online/xet'),HF_HUB_OFFLINE='0')
    mature.ROOT=mature.R1=ROOT;mature.RAW=RAW
    tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    roots=next(n for n in node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='roots' for t in n.targets))
    replacement=ast.parse("""roots = {
        'AGG4096': [(RAW/'virtual/AGG4096','virtual/AGG4096'), (RAW/'offline/AGG4096','offline/AGG4096')],
        'newfit': [(RAW/'newfit','newfit'),(RAW/'noise-newfit','noise-newfit')],
        'component': [(RAW/'component/AGG4096','component/AGG4096')],
        'collection': [(RAW/'collection','collection'),(RAW/'labels','labels'),(RAW/'codec','codec'),(RAW/'noise','noise')],
        'summary': []}[part]""").body[0]
    node.body[node.body.index(roots)]=replacement
    changed=[]
    for n in ast.walk(node):
        if isinstance(n,ast.Constant) and isinstance(n.value,str):
            if n.value=='evidence/PI0269/prefix-results-20261003/':n.value='evidence/PI0270/learner-recovery-results-20261004/';changed.append('prefix')
            elif n.value=='PI0269 actual fixed prefix results ':n.value='PI0270 actual fixed recovery results ';changed.append('message')
    assert sorted(changed)==['message','prefix']
    scope=vars(mature).copy();scope['admitted']=admitted
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope)
    return mature,scope['main']


def admitted(part):
    # ponytail: reuse complete-source admission, packing and anonymous readback; only roots differ.
    import sys
    sys.path.insert(0,str(ROOT/'code'))
    from admission import read,sha
    audit=read(ROOT/'evidence/MAIN_TRAINING_AUDIT.json')
    assert audit['status']=='PASS_AGG4096_REUSED_CONTROL_TRAINING_INTEGRITY'
    if part=='collection':
        assert read(ROOT/'evidence/COLLECTION_PROCESS_COMPLETE.json')['status']=='ALL_FIXED_COLLECTION_FAMILY_PROCESSES_EXIT_ZERO'
        cpu=read(ROOT/'evidence/FULL_COLLECTION_AND_SUPPORT_CPU_RETURN.json')
        assert cpu['status']=='FULL_FIXED_COLLECTOR_AND_NEW_SUPPORT_CPU_RETURN_NOT_AGG_QUALITY' and cpu['cases']==192
        assert read(ROOT/'evidence/TRAJECTORY_CPU_collection_192.json')['cases']==192
        proposals=read(ROOT/'design/LABEL_PROPOSAL_CARD.json')['proposals'];assert len(proposals)==6784
        for row in proposals:
            p=RAW/'codec'/row['source_lane']/row['source_episode_key']/f"{row['query_index']:03d}.json"
            value=read(p)
            if value.get('source',{}).get('sample'):
                assert sha(value['source']['sample'])==value['source']['sample_sha256']
        return
    from quality_gate import admitted as quality_admitted
    quality_admitted()
    if part=='summary':
        assert read(ROOT/'evidence/QUALITY_QUERY_LANES_COMPLETE.json')['status']=='ALL_FIXED_NEW_QUALITY_QUERY_LANES_COMPLETE'
        assert read(RAW/'delivery/PRIMARY_VERDICT.json')['status']=='PRIMARY_EVALUATION_COMPLETE'
        assert read(RAW/'delivery/CPU_REPLAY.json')['status']=='PASS_SAVED_ARRAY_REPLAY'
        video=read(RAW/'delivery/AGG4096-VIDEO_RECEIPT.json')
        assert sha(RAW/'delivery/PI0270_AGG4096_VIRTUAL.mp4')==video['sha256']
        for earlier in ('collection','AGG4096','newfit','component'):
            assert read(ROOT/'evidence'/('RESULT_ARCHIVE_'+earlier+'.json'))['status']=='PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS'
        return
    from quality_evaluate import complete
    if part=='AGG4096':
        card=ROOT/'design/FIXED_OFFLINE_CARD.json';data=read(card)
        records=[RAW/'offline/AGG4096'/domain/r['uid']/'record.json' for domain,key in
                 (('fit','fit_rows'),('factor','factor_rows'),('legacy','legacy_rows')) for r in data[key]]
        assert len(records)==968 and complete(records,'AGG4096',card)
        families=[f['family_id'] for f in read(BASE/'PI0267/design/SCENE_CARD.json')['families'] if f['split']=='NEW_SIM_EVAL']
        records=[RAW/'virtual/AGG4096'/f'{f}-{t}-c{c}-{b}'/'result.json' for f in families
                 for t in ('M1','M3') for c in (0,1) for b in ('E0','E1')]
        assert len(records)==64 and complete(records,'AGG4096',ROOT/'design/EVALUATION_CARD.json')
    elif part=='newfit':
        card=ROOT/'design/NEW_FIT192.json';rows=read(card)['rows'];assert len(rows)==192
        for point in ('PREFIX4096','AGG4096'):
            assert complete([RAW/'newfit'/point/r['input_id']/'record.json' for r in rows],point,card)
    else:
        assert part=='component';rows=read(ROOT/'design/COMPONENT_CARD.json')['rows'];assert len(rows)==32
        assert complete([RAW/'component/AGG4096'/r['case']/'result.json' for r in rows],'AGG4096',ROOT/'design/EVALUATION_CARD.json',component=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('part',choices=('collection','AGG4096','newfit','component','summary'))
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--self-check',action='store_true');a=parser.parse_args()
    mature,run=prepared()
    if a.self_check:
        assert 'stream=True' in (BASE/'PI0269/code/archive_results.py').read_text()
        print('PASS_QUALIFIED_PACK_AND_ANONYMOUS_ALL_MEMBER_READBACK_SOURCE_NO_UPLOAD')
    elif a.check_only:admitted(a.part);print('PASS_COMPLETE_PI0270_ARCHIVE_PART_NO_UPLOAD',a.part)
    else:run(a.part)
