from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data.dataloaders import extract_chain_sequences_and_backbone, pairwise_distances, read_a3m
from data.foldbench import load_manifest_dataframe, summarize_manifest


def manifest_summary(manifest_csv: str | Path) -> dict:
    manifest_df = load_manifest_dataframe(manifest_csv)
    return summarize_manifest(manifest_df)


def msa_preview(a3m_path: str | Path, limit: int = 5) -> dict:
    sequences = read_a3m(a3m_path, max_msa_seqs=limit)
    return {
        "path": str(Path(a3m_path).expanduser()),
        "num_sequences_previewed": len(sequences),
        "sequence_lengths": [len(sequence) for sequence in sequences],
        "sequences": sequences,
    }


def compute_distance_map(cif_path: str | Path, chain_id: str) -> np.ndarray:
    chain_data = extract_chain_sequences_and_backbone(cif_path)
    if chain_id not in chain_data:
        raise KeyError(f"Chain '{chain_id}' not found in {cif_path}")

    coords_ca = chain_data[chain_id]["coords_ca"]
    coords_ca = np.nan_to_num(coords_ca, nan=0.0)
    distance_map = pairwise_distances(torch.tensor(coords_ca, dtype=torch.float32))
    return distance_map.numpy()


def save_distance_map_figure(cif_path: str | Path, chain_id: str, output_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    distance_map = compute_distance_map(cif_path=cif_path, chain_id=chain_id)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(distance_map)
    plt.colorbar(label="Distance (A)")
    plt.title(f"CA distance map - chain {chain_id}")
    plt.xlabel("Residue index")
    plt.ylabel("Residue index")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Foldbench data artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_cmd = subparsers.add_parser("manifest-summary", help="Print a summary of a CSV manifest.")
    summary_cmd.add_argument("--manifest-csv", required=True, type=str)

    msa_cmd = subparsers.add_parser("msa-preview", help="Preview sequences stored in an A3M file.")
    msa_cmd.add_argument("--a3m-path", required=True, type=str)
    msa_cmd.add_argument("--limit", default=5, type=int)

    dmap_cmd = subparsers.add_parser("distance-map", help="Render a CA distance map from an mmCIF file.")
    dmap_cmd.add_argument("--cif-path", required=True, type=str)
    dmap_cmd.add_argument("--chain-id", required=True, type=str)
    dmap_cmd.add_argument("--output", required=True, type=str)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "manifest-summary":
        print(manifest_summary(args.manifest_csv))
        return

    if args.command == "msa-preview":
        print(msa_preview(args.a3m_path, limit=args.limit))
        return

    if args.command == "distance-map":
        output = save_distance_map_figure(args.cif_path, args.chain_id, args.output)
        print(f"[viz] saved distance map to {output}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
