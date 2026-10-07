"""The metadata exception accepts the bound run and rejects mismatched identities."""
import ast,copy
from pathlib import Path
source=Path(__file__).with_name('panel.py');tree=ast.parse(source.read_text())
node=next(n for n in tree.body if getattr(n,'name',None)=='snapshot_identity')
release='83af9099f3bab9a4ac08eaa85380c2cfd6351f34';scope=dict(RELEASE=release)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
ready=dict(point='COVER93512',release='1a20200ff066ef5a003650110cd20106f9a4bf64',resumable=False,arm='COVER93',step=512,source_full_trainer='/run/COVER93/main/training/checkpoint-512')
receipt=dict(status='PASS_LEGACY_CREATOR_METADATA_RECONCILIATION',governing_release=release,point='COVER93512',original_ready_sha256='r',readout_sha256='c',training_run_sha256='u',creator_sha256='eb790dbea1e8fc69bb328369aeadc32917e41bae5500d1e4f44f27aab93c55cf')
run=dict(release=release,arm='COVER93',phase='main',limit=4096,route='FAST_CACHE2',root='/run')
contract=dict(governing_release=release,arm='COVER93',initial_weights_revision='LOCAL_FOCUS4096@93e500e8ecb6a719d4a53be09d23b35bdeeb83fcc61e75416247a2c0380f4604')
args=['COVER93512',ready,receipt,run,contract,'r','c','u'];scope['snapshot_identity'](*args)
for index,key in [(1,'point'),(1,'source_full_trainer'),(2,'original_ready_sha256'),(2,'creator_sha256'),(3,'release'),(4,'initial_weights_revision')]:
 bad=copy.deepcopy(args);bad[index][key]='WRONG'
 try:scope['snapshot_identity'](*bad)
 except AssertionError:continue
 raise AssertionError((index,key))
print('PASS_BOUND_METADATA_EXCEPTION_AND_SIX_NEGATIVES')
