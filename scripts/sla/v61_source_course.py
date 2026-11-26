"""Bind a sealed Corelab bank to the R4 parent and four native training readers."""

from functools import partial
import hashlib
import json

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v61_source_bank import (AUX_COEFFICIENT, BATCH, ENDPOINTS, PLAN, PINS, SEED, UPDATES,
                             source_identity, target_identity)
from v6_dual_course import DualSourcePool, four_domain_adamw, initialize_single, reduced_fm


MAIN = "502d890d3ff314ac42f4c79c360942b9b626bf1fcc4b6fc488fbab15ba508361"
R4 = "f1f5a994005a9fc831934a156ca823d7e4b67794ce2533d72e087c2aeb29b9cb"
BANK_RECEIPT = "1ba0092a0c982c51e785849b86332bfc9c3db636bad6a82cab97548bed027891"
# The inherited Hopper reader consumes the portable R200 package, while the bank pins its Corelab copy.
R200_PORTABLE_MANIFEST = "e09f90169d988e8dbe39921aaa10611babf5d7227a7088ddf6d88a5e4b3d558b"
R200_CONTENT = "8fb693b5c2521587e4807a92a6947efa78be7819299ab8e35a0c379504df5e55"
MICRO = dict(target=16, single=16, robodojo=40, libero=16)
SINGLE_CURSOR = 10_000_000


def r200_content_sha(manifest):
    content = {key: value for key, value in manifest.items()
               if key not in ("data_site_binding", "portable_parent_manifest_sha256")}
    content["episodes"] = [{key: value for key, value in row.items() if key != "raw_path"}
                           for row in content["episodes"]]
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_bank(root):
    record = read(pin(root / "ACTUAL.json", BANK_RECEIPT))
    require(record["plan"] == PLAN and record["optimizer_visits"] == 0
            and record["endpoints"] == list(ENDPOINTS) and record["manifest_sha256"] == PINS
            and record["domain_batch"] == BATCH and record["auxiliary_coefficient"] == AUX_COEFFICIENT,
            "Wrong sealed V6.1 bank")
    with np.load(pin(root / "BANK.npz", record["bank_file_sha256"]), allow_pickle=False) as payload:
        banks = {name: payload[name] for name in BATCH}
        slots, rates = payload["auxiliary_slots"], payload["rates"]
    require(all(rows.shape == (UPDATES, BATCH[name]) and rows.dtype == np.int64
                for name, rows in banks.items()) and slots.shape == (UPDATES, 4)
            and rates.shape == (UPDATES,) and np.isfinite(rates).all() and np.all(rates > 0),
            "SOURCE bank shape or schedule changed")
    return record, banks, slots, rates


def checked_readers(record, banks, slots, target, single, old, new, libero):
    identities = dict(target=target_identity(target), single=source_identity(single), dual_old=source_identity(old),
                      dual_new=source_identity(new), libero=source_identity(libero))
    require(identities == record["reader_order_sha256"], "A live training dataset reordered its admitted rows")
    pool = DualSourcePool(old, new)
    datasets = dict(target=target, single=single, robodojo=pool, libero=libero)
    require(all(0 <= rows.min() <= rows.max() < len(datasets[name]) for name, rows in banks.items())
            and np.all(banks["robodojo"][:, :40] < pool.base_length)
            and np.all(banks["robodojo"][:, 40:] >= pool.base_length), "Scheduled visits differ from admission")
    full = np.asarray([target.support(i).all() for i in range(len(target))])
    auxiliary_rows = np.take_along_axis(banks["target"], slots, axis=1)
    require(full[auxiliary_rows].all() and np.all(np.diff(np.sort(auxiliary_rows, axis=1), axis=1) > 0),
            "Four distinct actual H40 examples are required for each update")
    return datasets


def bind(custody, short, prepared, corrected, extension, single_root, bank_root):
    from openpi_client.image_tools import resize_with_pad
    from v4_long_engine import LongLearner
    from v4_long_inputs import inputs
    from v6_reanchored_teacher_labels import bind_dataset
    from v6_robodojo_reader import ExpansionDataset
    from v6_ville_reader import SingleArmReader

    predecessor = custody / "HOPPER-ADAPT20K-V1/MAIN/checkpoints/phase03800"
    record, banks, slots, rates = read_bank(bank_root)
    datasets, tokenizer, _, _, auxiliary, _, _ = inputs(custody, short, prepared, predecessor)
    target, old, libero = (datasets[name] for name in ("target", "robodojo", "libero"))
    command_path = pin(short / "incoming/V3-MAIN-CONSUMED-DATA-02/MANIFEST.json", PINS["target_commands"])
    bind_dataset(target, corrected, PINS["target_corrected"], command_path)
    pin(prepared / "native_aux-01/MANIFEST.json", PINS["target_aux"])
    require(old.manifest_sha in (PINS["dual_old"], R200_PORTABLE_MANIFEST)
            and r200_content_sha(old.manifest) == R200_CONTENT,
            "The historical RoboDojo source manifest or its admitted contents changed")
    ready = read(extension / "READY.json")
    require(ready["hopper_readback"] and ready["newly_admitted_train"] == 107
            and ready["manifest_sha256"] == PINS["dual_new"], "The fixed RoboDojo input was not admitted")
    raw = dict(path=str(extension / "RAW_BINDING.private.json"), sha256=ready["raw_binding_sha256"])
    new = ExpansionDataset(extension, PINS["dual_new"], raw_binding=raw, resize_with_pad=resize_with_pad)
    single_proof = read(pin(single_root / "CORE_READER.json",
                            "b643899475ff509419fbc730c87e4922cb003203ac0aaa8cbe3a463cca7bad6b"))
    require(single_proof["manifest_sha256"] == PINS["single"]
            and single_proof["by_split"]["TRAIN"]["episodes"] == 30, "The fixed Ville input was not admitted")
    single = SingleArmReader(single_root, PINS["single"], single_root / "raw",
                             split="TRAIN", resize=resize_with_pad)
    datasets = checked_readers(record, banks, slots, target, single, old, new, libero)

    def factory(graphs, parameters, frozen, rng):
        graphs = dict(graphs, single=graphs["target"])
        parameters = dict(parameters, single=initialize_single(parameters["target"]))
        return LongLearner(graphs=graphs, parameters=parameters, frozen=frozen, datasets=datasets,
                           tokenizer=tokenizer, banks=banks, auxiliary_slots=slots, auxiliary=auxiliary,
                           cursors=dict(target=0, single=SINGLE_CURSOR, robodojo=0, libero=0), rates=rates,
                           rng=rng, batch_by_domain=BATCH,
                           gradient_join=partial(reduced_fm, batch=BATCH, aux_weight=AUX_COEFFICIENT),
                           domain_roles=dict(target="target", single="target", robodojo="robodojo", libero="libero"),
                           support_domains=("target", "single"), optimizer_factory=four_domain_adamw)

    return factory, datasets, banks, slots, auxiliary, record


def corrected_parent(custody, short, run):
    import jax
    import orbax.checkpoint as ocp
    from flax.traverse_util import flatten_dict
    from v3_main_place_learn import endpoint_files
    from v4_long_checkpoint import load_parent

    endpoint = run / "STEP0200"
    manifest = pin(run / "STEP0200_SEALED.json", R4)
    receipt = read(manifest)
    terminal = read(run / "TERMINAL.json")
    contract = run / "RUN_CONTRACT.json"
    require(receipt["step"] == terminal["updates"] == 200
            and terminal["status"] == "FORMAL_RETENTION_200"
            and terminal["role"] == "R4_REEXPRESSED_COMMANDS"
            and terminal["contract_sha256"] == sha(contract)
            and read(contract)["reexpressed_geometry_manifest_sha256"] == PINS["target_corrected"]
            and endpoint_files(endpoint) == receipt["files"], "The complete R4 parent failed transport custody")
    main = custody / "HOPPER-ADAPT20K-V1/MAIN/checkpoints/phase03800"
    graphs, parameters, frozen, backbone = load_parent(custody, main, MAIN, short / "incoming/reference")
    with ocp.PyTreeCheckpointer() as reader:
        meta = reader.metadata(endpoint).tree
        require(set(meta) == {"parameters", "optimizer", "step", "rng"}, "Incomplete parent tree")
        # The incoming optimizer is verified by the sealed file list and the independent Corelab restore.
        field = meta["parameters"]
        args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), field)
        values = reader.restore(endpoint, args=ocp.args.PyTreeRestore(
            item={"parameters": field}, restore_args={"parameters": args}, transforms={}))["parameters"]
    require(set(values) == set(parameters) == {"shared", "target", "robodojo", "libero"},
            "The R4 parent must contain every existing domain")
    for name, value in values.items():
        supplied = flatten_dict(value)
        original = flatten_dict(parameters[name].to_pure_dict())
        require(supplied.keys() == original.keys() and all(
            supplied[k].shape == old.shape and supplied[k].dtype == old.dtype and np.isfinite(supplied[k]).all()
            for k, old in original.items()), "R4 parameter partition changed")
        parameters[name].replace_by_pure_dict(value)
    return graphs, parameters, frozen, dict(backbone, initial_r4_receipt_sha256=R4, new_optimizer=True,
                                           parent_parameters=45, source_rng_seed=SEED)
