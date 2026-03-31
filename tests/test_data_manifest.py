from __future__ import annotations

import json

from data.foldbench import build_manifest_dataframe, derive_targets, filter_complete_records, summarize_manifest


def test_build_manifest_dataframe_and_summary(tmp_path):
    json_path = tmp_path / "fb_protein.json"
    msa_root = tmp_path / "foldbench_msas"
    cif_root = tmp_path / "reference_structures"

    msa_root.mkdir()
    cif_root.mkdir()
    (msa_root / "7abc_A").mkdir()
    (cif_root / "7abc-assembly1_1.cif").write_text("data_test\n", encoding="utf-8")

    payload = {
        "queries": {
            "7ABC": {
                "chains": [
                    {
                        "chain_ids": ["A", "1"],
                        "sequence": "ACDE",
                    }
                ]
            },
            "7XYZ": {
                "chains": [
                    {
                        "chain_ids": ["A"],
                        "sequence": "MNPQRS",
                    }
                ]
            },
        }
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest_df = build_manifest_dataframe(
        json_path=json_path,
        msa_root=msa_root,
        cif_root=cif_root,
    )

    assert len(manifest_df) == 2
    assert derive_targets(manifest_df) == ["7abc_A", "7xyz_A"]

    complete_df = filter_complete_records(manifest_df)
    assert len(complete_df) == 1
    assert complete_df.iloc[0]["query_name"] == "7ABC"

    summary = summarize_manifest(manifest_df)
    assert summary["records"] == 2
    assert summary["complete_records"] == 1
    assert summary["sequence_length"]["max"] == 6
