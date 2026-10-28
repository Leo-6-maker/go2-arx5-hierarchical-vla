"""The original memberships plus the registered field; no query or new noise."""
import shutil,sys
from pathlib import Path
ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0274');OLD=ROOT.parent/'PI0273';RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0274')
sys.path.insert(0,str(OLD/'code'));from quality_gate import read,sha
sys.path[:0]=[str(ROOT.parent/'PI0266/code'),str(ROOT.parent/'PI0250/runtime/reference-317a/src')];from pi0255_motion3 import write
def main():
    for name in ('RUNTIME_EVALUATION_CARD.json','RUNTIME_COMPONENT_CARD.json','RUNTIME_OFFLINE_CARD.json','NEW_FIT192.json','NEWFIT_QUERY_CARD.json','SUPPORT_CARD.json','LATE_QUERY_CARD.json','BROAD_READOUT_R1.json'):
        src=OLD/'design'/name;dst=ROOT/'design'/name
        if dst.exists():assert dst.read_bytes()==src.read_bytes()
        else:shutil.copy2(src,dst)
    binding=OLD/'model_bindings/BROAD5120.json';(ROOT/'model_bindings').mkdir(exist_ok=True);shutil.copy2(binding,ROOT/'model_bindings/BROAD5120.json')
    stats=ROOT/'contracts/ROT/source/dataset/statistics.json';stats.parent.mkdir(parents=True,exist_ok=True)
    if not stats.exists():shutil.copy2(OLD/'contracts/ROT/source/dataset/statistics.json',stats)
    noise=[];pose=read(ROOT/'design/POSE_CARD.json')
    for n in pose['noise']:
        assert sha(n['file'])==n['file_sha256']
        noise.append(dict(file=n['file'],sha256=n['file_sha256'],keys=[dict(name=n['name'],noise_sha256=n['noise_sha256'])],cell=n['cell'],j=n['j']))
    write(ROOT/'design/FIELD_QUERY_CARD.json',dict(points=['BROAD5120','EQUIL5120'],noise=noise,poses=1248,queries=4992,source_pose_card_sha256=sha(ROOT/'design/POSE_CARD.json')))
    inherited=[dict(path=str(ROOT/'design'/n),sha256=sha(ROOT/'design'/n),source_path=str(OLD/'design'/n)) for n in ('RUNTIME_EVALUATION_CARD.json','RUNTIME_COMPONENT_CARD.json','RUNTIME_OFFLINE_CARD.json','NEW_FIT192.json','NEWFIT_QUERY_CARD.json','SUPPORT_CARD.json','LATE_QUERY_CARD.json')]
    write(ROOT/'design/QUALITY_SCOPE.json',dict(points={'field':['BROAD5120','EQUIL5120'],'original_panels':['EQUIL5120']},inherited_cards=inherited,
        existing_BROAD_original_panels_reused=True,caps=dict(attempts=14336,heads=57344,starts=104,frame_pairs=9216),planned_max_queries=13784,model_queries=0))
    print('PASS_FROZEN_NEW_FIELD_AND_ORIGINAL_QUALITY_MEMBERSHIPS',flush=True)
if __name__=='__main__':main()
