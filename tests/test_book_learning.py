import pytest
import os,subprocess,sys
from pathlib import Path
from scripts.pi0294.multitask.book_learning import recipe,BookSampler


def test_new_book_recipe_is_distinct_and_preserves_controlled_comparison():
    r=recipe()
    assert r['parent']=='QUALIFIED_E1_512_NOT_RAW_OR_E2'
    assert r['original_E1_E2_recipes_data_R0_unchanged'] and not r['DEV_for_optimization']
    assert r['max_steps']==128 and r['per_device_train_batch_size']*r['gradient_accumulation_steps']==32
    assert r['shared_DiT_frozen'] and r['checkpoint_steps']==[128]


def test_finite_training_sampler_exact_absolute_cursor_and_resume():
    data=list(range(3005));full=list(BookSampler(data));suffix=list(BookSampler(data,start=64))
    assert full[64:]==suffix and [c for c,_ in full]==list(range(4096))
    assert set(i for _,i in full[:3005])==set(range(3005))
    sampler=BookSampler(data,start=64)
    for c,i in suffix[::127]:assert sampler.index_at(c)==i
    with pytest.raises(IndexError):sampler.index_at(4096)


def test_standalone_entry_bootstraps_before_import_from_registered_output_directory(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/pi0294/multitask/book_learning.py'
    env={k:v for k,v in os.environ.items()if k!='PYTHONPATH'}
    result=subprocess.run([sys.executable,'-B',str(script),'--help'],cwd=tmp_path,env=env,text=True,capture_output=True)
    assert result.returncode==0 and '--manifest-sha256'in result.stdout
    manifest=tmp_path/'RUN_MANIFEST.json';manifest.write_text('{}')
    result=subprocess.run([sys.executable,'-B',str(script),'--root',str(tmp_path),'--manifest-sha256','0'*64],cwd=tmp_path,env=env,text=True,capture_output=True)
    assert result.returncode!=0 and 'BOOK_ENTRY_SEALED_MANIFEST_BEFORE_PROJECT_IMPORT'in result.stderr
    assert 'ModuleNotFoundError'not in result.stderr
