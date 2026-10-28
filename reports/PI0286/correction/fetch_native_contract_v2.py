import base64,hashlib,json,sys
from pathlib import Path
here=Path(__file__).resolve().parent;sys.path.insert(0,str(here.parents[2]/'scripts/pi0276_execution'))
from coordinator import remote
script='''import base64,hashlib,json
from pathlib import Path
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286')
names=['correction/C1_PLAN_NATIVE_CONTRACT_V2.json','correction/NATIVE_CONTRACT_V2_DERIVATION_RECEIPT.json','correction/NATIVE_CONTRACT_V2_BOUNDARY_CHECK.json','correction/implementation_erratum/SINGLE_FILE_AND_SEAL_UPDATE.json','correction/implementation_erratum/CODE_SEAL_PRE_NATIVE_CONTRACT_V2.json','design/CODE_SEAL.json']
files={n:base64.b64encode((r/n).read_bytes()).decode() for n in names}
p=r/'correction/BANK_NATIVE_CONTRACT_V2.json';b=p.read_bytes();d=json.loads(b)
summary={k:d[k] for k in ('schema','status','valid_branches','valid_slots','reclassified_preclip_only_slots','physical_success_branches','source_contract_codec_valid_branches','original_V1_bank_sha256','training_admission','optimizer_launch_allowed')}
summary.update(bank_path=str(p),bank_sha256=hashlib.sha256(b).hexdigest(),bank_bytes=len(b),
slots=[{k:x.get(k) for k in ('slot','status','sealed_V1_status','maxima','result_sha256','takeover_tick')}|{'failed_H40_count':len(x.get('failures',[])),'sealed_V1_failed_H40_count':len(x.get('sealed_V1_failures_preserved',[])),'physical_success':x.get('status') in ('PASS_FRESH_SUFFIX_STRICT_CODEC','FAIL_SUPERVISION_CODEC_PRESERVE_RAW_STOP_BRANCH')} for x in d['all_requested_slot_reports']])
print(json.dumps(dict(files=files,bank_summary=summary)))
'''
value=json.loads(remote('core',script));dest=here/'native_contract_v2_receipts';dest.mkdir(exist_ok=True)
receipts={}
for name,s in value['files'].items():
    b=base64.b64decode(s);filename=Path(name).name
    if name=='design/CODE_SEAL.json':filename='CODE_SEAL_NATIVE_CONTRACT_V2.json'
    p=dest/filename
    if p.exists():assert p.read_bytes()==b
    else:p.write_bytes(b)
    receipts[name]=dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
summary=value['bank_summary'];assert summary['bank_sha256']=='f2c7b43af6edc42795ee0595ed0a9c9f05f440369945d5b43bfc38419e21cb8d'
(here/'BANK_NATIVE_CONTRACT_V2_COMPACT.json').write_text(json.dumps(summary,indent=2)+'\n')
(here/'NATIVE_CONTRACT_V2_LOCAL_RECEIPTS.json').write_text(json.dumps(receipts,indent=2)+'\n')
print(json.dumps(dict(valid_branches=summary['valid_branches'],bank_bytes=summary['bank_bytes'],receipts=len(receipts))))
