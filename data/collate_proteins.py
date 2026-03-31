from __future__ import annotations

import torch

from data.dataloaders import AA_VOCAB


def collate_proteins(batch):
    batch_size = len(batch)
    max_length = max(item["seq_tokens"].shape[0] for item in batch)
    max_msa_depth = max(item["msa_tokens"].shape[0] for item in batch)

    pad_token = AA_VOCAB["-"]

    seq_tokens = torch.full((batch_size, max_length), pad_token, dtype=torch.long)
    seq_mask = torch.zeros((batch_size, max_length), dtype=torch.float32)

    msa_tokens = torch.full((batch_size, max_msa_depth, max_length), pad_token, dtype=torch.long)
    msa_mask = torch.zeros((batch_size, max_msa_depth, max_length), dtype=torch.float32)

    coords_n = torch.zeros((batch_size, max_length, 3), dtype=torch.float32)
    coords_ca = torch.zeros((batch_size, max_length, 3), dtype=torch.float32)
    coords_c = torch.zeros((batch_size, max_length, 3), dtype=torch.float32)

    valid_res_mask = torch.zeros((batch_size, max_length), dtype=torch.float32)
    valid_backbone_mask = torch.zeros((batch_size, max_length), dtype=torch.float32)

    dist_map = torch.zeros((batch_size, max_length, max_length), dtype=torch.float32)
    pair_mask = torch.zeros((batch_size, max_length, max_length), dtype=torch.float32)
    backbone_pair_mask = torch.zeros((batch_size, max_length, max_length), dtype=torch.float32)

    torsion_true = torch.zeros((batch_size, max_length, 3, 2), dtype=torch.float32)
    torsion_mask = torch.zeros((batch_size, max_length, 3), dtype=torch.float32)

    ids = []
    msa_chain_ids = []
    matched_chain_ids = []
    sequence_strs = []
    match_identity = torch.zeros(batch_size, dtype=torch.float32)

    for index, item in enumerate(batch):
        length = item["seq_tokens"].shape[0]
        msa_depth = item["msa_tokens"].shape[0]

        seq_tokens[index, :length] = item["seq_tokens"]
        seq_mask[index, :length] = 1.0

        msa_tokens[index, :msa_depth, :length] = item["msa_tokens"]
        msa_mask[index, :msa_depth, :length] = item["msa_mask"]

        coords_n[index, :length] = item["coords_n"]
        coords_ca[index, :length] = item["coords_ca"]
        coords_c[index, :length] = item["coords_c"]

        valid_res_mask[index, :length] = item["valid_res_mask"]
        valid_backbone_mask[index, :length] = item["valid_backbone_mask"]

        dist_map[index, :length, :length] = item["dist_map"]
        pair_mask[index, :length, :length] = (
            item["valid_res_mask"][:, None] * item["valid_res_mask"][None, :]
        )
        backbone_pair_mask[index, :length, :length] = (
            item["valid_backbone_mask"][:, None] * item["valid_backbone_mask"][None, :]
        )

        torsion_true[index, :length] = item["torsion_true"]
        torsion_mask[index, :length] = item["torsion_mask"]

        ids.append(item["id"])
        msa_chain_ids.append(item["msa_chain_id"])
        matched_chain_ids.append(item["matched_chain_id"])
        sequence_strs.append(item["sequence_str"])
        match_identity[index] = item["match_identity"]

    return {
        "id": ids,
        "msa_chain_id": msa_chain_ids,
        "matched_chain_id": matched_chain_ids,
        "match_identity": match_identity,
        "sequence_str": sequence_strs,
        "seq_tokens": seq_tokens,
        "seq_mask": seq_mask,
        "msa_tokens": msa_tokens,
        "msa_mask": msa_mask,
        "coords_n": coords_n,
        "coords_ca": coords_ca,
        "coords_c": coords_c,
        "dist_map": dist_map,
        "valid_res_mask": valid_res_mask,
        "valid_backbone_mask": valid_backbone_mask,
        "pair_mask": pair_mask,
        "backbone_pair_mask": backbone_pair_mask,
        "torsion_true": torsion_true,
        "torsion_mask": torsion_mask,
    }
