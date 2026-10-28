"""Compatibility views of already frozen evaluation cards; no new cases or noise."""
from pathlib import Path
from prepare import BASE,ROOT,RELEASE,read,sha,write


def main():
    master=read(ROOT/'design/EVALUATION_CARD.json');views={}
    for source,target in [('MAIN_CARD.json','RUNTIME_EVALUATION_CARD.json'),
                          ('COMPONENT_CARD.json','RUNTIME_COMPONENT_CARD.json'),
                          ('OFFLINE_CARD.json','RUNTIME_OFFLINE_CARD.json')]:
        assert sha(ROOT/'design'/source)==master['cards'][source]
        wrapped=read(ROOT/'design'/source);assert sha(wrapped['source_path'])==wrapped['source_sha256']
        value=dict(wrapped['frozen_original'],pi0273_release=RELEASE,pi0273_source_view_sha256=sha(ROOT/'design'/source))
        if source=='MAIN_CARD.json':
            value.update(model_points=master['points'],new_planned_episodes=128,
                caps=dict(inference_attempts=18432,inference_heads=73728,new_virtual_starts=208,total_new_frame_pairs=15360),
                source_membership_card_sha256=sha(ROOT/'design/EVALUATION_CARD.json'))
        elif source=='COMPONENT_CARD.json':value.update(points=master['points'],new_PI0273_proposals=64)
        else:value.update(model_points=master['points'],new_PI0273_rows=1936)
        write(ROOT/'design'/target,value);views[target]=sha(ROOT/'design'/target)
    support=read(ROOT/'design/SUPPORT_CARD.json');assert sha(ROOT/'design/SUPPORT_CARD.json')==master['cards']['SUPPORT_CARD.json']
    newfit=BASE/'PI0270/design/NEW_FIT192.json';noise=BASE/'PI0270/design/NEWFIT_QUERY_CARD.json'
    assert sha(newfit)==support['source_early_sha256']
    write(ROOT/'design/NEW_FIT192.json',read(newfit))
    write(ROOT/'design/NEWFIT_QUERY_CARD.json',dict(read(noise),points=master['points'],source_sha256=sha(noise)))
    late_noise=[n for r in support['late'] for n in r['noise']];assert len(late_noise)==592
    write(ROOT/'design/LATE_QUERY_CARD.json',dict(points=master['points'],noise=late_noise,conditions=296,queries_per_point=592))
    checked={}
    for name in ('RUNTIME_EVALUATION_CARD.json','RUNTIME_COMPONENT_CARD.json','RUNTIME_OFFLINE_CARD.json','NEWFIT_QUERY_CARD.json','LATE_QUERY_CARD.json'):
        for row in read(ROOT/'design'/name)['noise']:
            p=Path(row['file'])
            if str(p) not in checked:checked[str(p)]=sha(p)
            assert checked[str(p)]==row['sha256']
    offline=read(ROOT/'design/RUNTIME_OFFLINE_CARD.json')
    assert sum(len(offline[k]) for k in ('fit_rows','factor_rows','legacy_rows'))==968
    for row in [r for k in ('fit_rows','factor_rows','legacy_rows') for r in offline[k]]+support['early']+support['late']:
        if 'input' in row:assert sha(row['input'])==row['input_sha256']
    paths=[ROOT/'design'/n for n in (*views,'NEW_FIT192.json','NEWFIT_QUERY_CARD.json','LATE_QUERY_CARD.json')]
    write(ROOT/'evidence/QUALITY_MEMBERSHIP_READBACK.json',dict(status='PASS_FROZEN_MEMBERSHIP_INPUTS_AND_EXISTING_NOISE_BYTES',
        files=[dict(path=str(p),sha256=sha(p)) for p in paths],existing_noise_files=checked,
        inherited_memberships_unchanged=True,new_random_packets=0,model_queries=0,renders=0,physics=0))
    print('PASS_PI0273_EXISTING_FIXED_QUALITY_MEMBERSHIPS',len(checked),flush=True)


if __name__=='__main__':main()
