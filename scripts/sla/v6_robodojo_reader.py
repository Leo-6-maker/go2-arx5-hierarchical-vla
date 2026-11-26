"""Bind relocated V6 public RoboDojo files to the existing native codec."""

from pathlib import Path

from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_data import RobodojoDataset
from go2arx5_vla.sla_adapter.source_native_v6 import SourceDataset

from v6_ville_pilot import need


class ExpansionDataset(RobodojoDataset):
    def __init__(self, root, manifest_sha, *, raw_binding, resize_with_pad):
        # The V6 release has its own manifest. Keep the established native
        # window decoder and inherited relocated raw-path hash checks.
        SourceDataset.__init__(self, root, manifest_sha, split="TRAIN", resize_with_pad=resize_with_pad)
        need(self.manifest["recipe"] == "SLA_V6_EXISTING_ROBODOJO_CODEC_EXTENSION"
             and self.manifest["test_labels_read"] is False and self.manifest["target_commands"] is False
             and self.manifest["horizon"] == 50 and self.manifest["nominal_frequency_Hz"] == 30,
             "Expected the exact V6 recorded-reference ABI")
        binding = read_pin(raw_binding)
        need(binding["manifest_sha256"] == manifest_sha and binding["split"] == "TRAIN"
             and set(binding["raw"]) == set(self.episodes), "V6 raw population differs from original bytes")
        for key, row in self.episodes.items():
            raw = binding["raw"][key]
            need(raw["sha256"] == row["raw_sha256"]
                 and Path(raw["file"]).parts == ("raw", f"{row['raw_sha256']}.hdf5"),
                 "V6 raw member left its registered archive")
        self.raw_binding = binding["raw"]
        self.verified_raw = set()
