"""Fixed saved-RGB contact sheets for R3 H1 manual review; no renderer call."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pi0260_r2 import ROOT, read, sha

OUT=ROOT/'R3/image_review'


def cue_margin(row):
    check=row['check']
    if not check['pass_necessary']:return -1.
    counts=check['counts'];required=check['required']
    return min(max(counts[k].values())/v for k,v in required.items())


def source_rows(name):
    record=read(ROOT/'dataset'/(name+'.json'))
    family=name.rsplit('-',2)[0]
    observations=read(ROOT/'proposals'/family/name/'observations.json')
    checks={r['index']:r['check'] for r in observations['rows']}
    return [dict(r,check=checks[r['index']]) for r in record['rows']]


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    selected=read(ROOT/'dataset/codec_receipt.json')['selected_post_codec']
    manifest=[]
    for task in ('M1','M3'):
        samples=[]
        for split in ('TRAIN','KIN-VAL'):
            families=selected[split]
            for family in (families[0],families[-1]):
                for command in (0,1):
                    name=f'{family}-{task}-c{command}'
                    active=[r for r in source_rows(name) if r['active'] and r['usable']]
                    for target in (0,active[-1]['time']/2,active[-1]['time']):
                        row=min(active,key=lambda r:abs(r['time']-target))
                        samples.append((split,'fixed_phase',name,row))
            candidates=[]
            for family in families:
                for command in (0,1):
                    name=f'{family}-{task}-c{command}'
                    candidates.extend((cue_margin(r),name,r) for r in source_rows(name) if r['active'] and r['usable'])
            for _,name,row in sorted(candidates,key=lambda x:(x[0],x[1],x[2]['index']))[:2]:
                samples.append((split,'worst_required_cue',name,row))
        assert len(samples)==28
        image=Image.new('RGB',(4*408,7*238),(18,21,27));draw=ImageDraw.Draw(image)
        for n,(split,reason,name,row) in enumerate(samples):
            x=(n%4)*408;y=(n//4)*238
            with np.load(ROOT/row['frame']) as z:
                for offset,stream in ((0,'body'),(204,'wrist')):
                    frame=Image.fromarray(z[stream+'_rgb']).resize((192,192),Image.Resampling.NEAREST)
                    image.paste(frame,(x+offset,y+31))
            draw.text((x+3,y+3),f'{split} {reason} {name[-12:]} t={row["time"]:.2f}',fill=(230,230,230))
            draw.text((x+3,y+224),f'actual 96x96 RGB, cue margin {cue_margin(row):.2f}',fill=(230,230,230))
            manifest.append(dict(task=task,split=split,reason=reason,sequence=name,index=row['index'],
                                 time=row['time'],frame=row['frame'],frame_sha256=row['frame_sha256'],cue_margin=cue_margin(row)))
        path=OUT/(task+'_contact.png')
        assert not path.exists()
        image.save(path)
    (OUT/'manifest.json').write_text(json.dumps(dict(scope='28 source RGB pairs per task: fixed first/last family and 0/mid/last active phase, plus two minimum required-cue rows per split',
                                                 samples=manifest,contact_sha256={t:sha(OUT/(t+'_contact.png')) for t in ('M1','M3')}),indent=2)+'\n')
    print(json.dumps(dict(status='CONTACT_SHEETS_READY_FOR_MANUAL_REVIEW',samples=len(manifest),files=[str(OUT/(t+'_contact.png')) for t in ('M1','M3')])))


if __name__=='__main__':run()
