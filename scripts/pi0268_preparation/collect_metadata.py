"""Relay only completed training text logs/configs; no weights or remote writes on Hopper."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

REMOTE = r'''
from pathlib import Path
import hashlib,io,json,sys,tarfile
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0267')
files=[]
for arm in ('N','R'):
 for name in ('sample_audit.jsonl','S4_audit.jsonl','full_loss.jsonl','update_gradient_norms.jsonl','S4_first_update_gradients.json'):
  files.append((r/arm/'main/training'/name,arm+'/'+name))
 for name in ('statistics.json','processor_config.json','trainer_state.json'):
  files.append((r/arm/'main/training/checkpoint-4096'/name,arm+'/checkpoint-config/'+name))
files += [(r/'source/dataset/statistics.json','actual_statistics.json'),(r/'EXECUTION_FILESET.json','EXECUTION_FILESET.json')]
manifest=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for path,name in files:
  assert path.is_file() and not path.is_symlink()
  value=path.read_bytes();before=hashlib.sha256(value).hexdigest()
  row=dict(source=str(path),path=name,bytes=len(value),sha256=before)
  item=tarfile.TarInfo(name);item.size=len(value);tar.addfile(item,io.BytesIO(value))
  assert hashlib.sha256(path.read_bytes()).hexdigest()==before,'SOURCE_CHANGED'
  manifest.append(row)
 value=json.dumps(dict(source='Completed original PI0267 main logs/configs',files=manifest,weights_read=0,model_queries=0),indent=2).encode()
 item=tarfile.TarInfo('SOURCE_FILES.json');item.size=len(value);tar.addfile(item,io.BytesIO(value))
'''


def main():
    destination=Path(__file__).resolve().parents[3]/'preparation/source_metadata'
    destination.mkdir(parents=True,exist_ok=True)
    archive=destination/'completed_training_logs.tar.gz'
    if not archive.exists():
        result=subprocess.run(['ssh','nus-hopper','python3 -u -'],input=REMOTE.encode(),capture_output=True,check=True)
        assert len(result.stdout)<100*2**20
        archive.write_bytes(result.stdout)
    with tarfile.open(archive) as tar:
        receipt=json.load(tar.extractfile('SOURCE_FILES.json'))
        for row in receipt['files']:
            value=tar.extractfile(row['path']).read()
            assert len(value)==row['bytes'] and hashlib.sha256(value).hexdigest()==row['sha256']
    print(json.dumps(dict(archive=str(archive),compressed_bytes=archive.stat().st_size,files=len(receipt['files']),weights_read=0)))


if __name__=='__main__':main()
