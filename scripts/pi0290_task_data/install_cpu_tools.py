"""Isolated tokenizer/HF archival dependencies, no model imports or loading."""
import argparse
from pathlib import Path
import subprocess
from go2arx5_vla.task_data.resource import Budget
from go2arx5_vla.task_data.io import write

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
budget=Budget(root)
r=subprocess.run([str(root/'runtime/venv/bin/python'),'-m','pip','install','tokenizers==0.22.2','huggingface_hub==0.36.2'],capture_output=True,text=True)
write(root/'evidence/CPU_DEPENDENCIES.json',dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,GR00T=0,physics=0))
budget.finish_cpu('CPU_TOKENIZER_HF_DEPENDENCIES');print(r.stdout,r.stderr);raise SystemExit(r.returncode)
