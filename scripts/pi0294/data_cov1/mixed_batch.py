"""The mature CPU reader/collator, now mixing distinct qualified parents."""
from pathlib import Path

from scripts.pi0292.durable import read
from scripts.pi0294.e1.scope import STATS
from scripts.pi0294.fd1 import sha
from scripts.pi0294.native_dataset import NativeDataset
from scripts.pi0294.pv1.export import batch_inventory
from .scope import sealed_file


def qualify(bindings,processor,official_source,cosmos,guard=lambda:None):
    if len(bindings)!=2 or len({b['id'] for b in bindings})!=2:
        raise PermissionError('DATA_COV1_BOUNDED_CROSS_EPISODE_BATCH_NEEDS_TWO_DISTINCT_PARENTS')
    processed=[];identities=[];raw_hashes=set()
    for b in bindings:
        guard();root=Path(b['export_root']);receipt=root/'EXPORT_RECEIPT.json'
        sealed_file(receipt,b['export_receipt_sha256']);row=read(receipt)
        if (row['id']!=b['id'] or not row['admitted_for_training']
                or row['status']!='PASS_SAVED_TEACHER_EXPORT_AND_OFFICIAL_CHAIN'
                or row['codec']['status']!='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP'
                or row['split']!='TRAIN' or row['nominal_clearance_m']!=.004
                or not row['full_task_success'] or not row['independent_raw_admitted']
                or row['file']!='numeric/'+row['id']+'.npz'):
            raise PermissionError('DATA_COV1_ONLY_FULLY_QUALIFIED_TRAIN_PARENTS')
        if row['source_raw_manifest_sha256'] in raw_hashes:
            raise PermissionError('DATA_COV1_DISTINCT_PARENTS_NOT_RELABELED_RAW_DATA')
        raw_hashes.add(row['source_raw_manifest_sha256'])
        sealed_file(root/'statistics.json',STATS);sealed_file(root/row['file'],row['sha256'])
        manifest=read(root/'DATASET_MANIFEST.json')
        if len(manifest['episodes'])!=1 or manifest['episodes'][0]['id']!=row['id']:
            raise PermissionError('DATA_COV1_EXACT_SINGLE_PARENT_READER_ROOT')
        dataset=NativeDataset(root,official_source,cosmos,for_optimization=True,
            raw_bindings={row['id']:str(root/'raw'/row['id'])})
        dataset.processor=processor
        if len(dataset)!=601:raise PermissionError('DATA_COV1_COMPLETE_601_NATIVE_WINDOWS')
        for anchor in (0,80,160,320,480,600):
            guard();processed.append(dataset[anchor]['processed'])
            identities.append(dict(id=row['id'],anchor=anchor,receipt_sha256=sha(receipt)))
    batch=batch_inventory(processor.collator(processed),len(processed))
    repeated=batch_inventory(processor.collator(processed),len(processed))
    if batch!=repeated:raise ValueError('DATA_COV1_CROSS_PARENT_COLLATOR_MUST_BE_DETERMINISTIC')
    return dict(status='PASS_ACTUAL_CROSS_PARENT_OFFICIAL_CPU_BATCH',parent_count=len(bindings),
        actual_parent_anchor_rows=identities,batch=batch,deterministic_repeat=True,
        distinct_parent_ids=sorted({b['id'] for b in bindings}),
        model_forward=False,neural_queries=0,new_physics=0,new_constructors=0,
        does_not_replace_each_parent_full_raw_and_601_codec_audit=True)


def qualify_all(bindings,processor,official_source,cosmos,guard=lambda:None):
    if len(bindings)<2 or len({b['id'] for b in bindings})!=len(bindings):
        raise PermissionError('DATA_COV1_DISTINCT_BANK_PARENTS_FOR_BOUNDED_MIXED_BATCHES')
    pairs=[bindings[i:i+2] for i in range(0,len(bindings)-1,2)]
    if len(bindings)%2:pairs.append(bindings[-2:])
    batches=[qualify(pair,processor,official_source,cosmos,guard) for pair in pairs]
    covered={identifier for row in batches for identifier in row['distinct_parent_ids']}
    if covered!={b['id'] for b in bindings}:raise ValueError('DATA_COV1_EVERY_PARENT_MUST_ENTER_A_CROSS_PARENT_BATCH')
    return dict(status='PASS_EVERY_PARENT_IN_BOUNDED_CROSS_PARENT_OFFICIAL_CPU_BATCHES',
        distinct_parent_count=len(covered),maximum_parents_per_batch=2,maximum_samples_per_batch=12,
        odd_final_parent_pairs_with_previous_parent_not_an_extra_demo=True,batches=batches,
        model_forward=False,new_physics=0,new_constructors=0,neural_queries=0)
