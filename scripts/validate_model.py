"""CLI for model instantiation, forward smoke checks, and validation tests.

This script creates the configured AlphaFold2 model and loss stack, runs a
synthetic forward-plus-loss smoke test, and can dispatch the repository test
suite so the high-level model wiring can be validated from one place.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import subprocess
import sys

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import (
    build_ideal_backbone_local,
    build_loss_from_config,
    build_model_from_config,
    choose_device,
    count_trainable_parameters,
    load_yaml_config,
    make_synthetic_batch,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate AlphaFold2 model wiring and tests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    instantiate_cmd = subparsers.add_parser("instantiate", help="Instantiate the configured model and print a summary.")
    instantiate_cmd.add_argument("--config", type=str, default="config/experiments/af2_poc.yaml")
    instantiate_cmd.add_argument("--device", type=str, default=None)

    smoke_cmd = subparsers.add_parser("forward-smoke", help="Run a synthetic forward and loss smoke test.")
    smoke_cmd.add_argument("--config", type=str, default="config/experiments/af2_poc.yaml")
    smoke_cmd.add_argument("--device", type=str, default=None)
    smoke_cmd.add_argument("--batch-size", type=int, default=1)
    smoke_cmd.add_argument("--msa-depth", type=int, default=4)
    smoke_cmd.add_argument("--seq-len", type=int, default=16)
    smoke_cmd.add_argument("--num-recycles", type=int, default=0)

    pytest_cmd = subparsers.add_parser("pytest", help="Run validation tests through pytest.")
    pytest_cmd.add_argument("--target", action="append", default=None, help="Specific pytest target. Repeatable.")
    pytest_cmd.add_argument(
        "--pytest-arg",
        action="append",
        default=None,
        help="Extra raw pytest arg. Repeatable. Use forms like --pytest-arg=-q.",
    )

    all_cmd = subparsers.add_parser("all", help="Run instantiate, forward-smoke, and pytest in sequence.")
    all_cmd.add_argument("--config", type=str, default="config/experiments/af2_poc.yaml")
    all_cmd.add_argument("--device", type=str, default=None)
    all_cmd.add_argument("--batch-size", type=int, default=1)
    all_cmd.add_argument("--msa-depth", type=int, default=4)
    all_cmd.add_argument("--seq-len", type=int, default=16)
    all_cmd.add_argument("--num-recycles", type=int, default=0)
    all_cmd.add_argument("--target", action="append", default=None)
    all_cmd.add_argument("--pytest-arg", action="append", default=None)

    return parser.parse_args(argv)


def run_instantiate(args: argparse.Namespace) -> None:
    config = load_yaml_config(args.config)
    device = choose_device(args.device)
    model = build_model_from_config(config, device=device)
    criterion = build_loss_from_config(config, device=device)

    print(
        {
            "device": device,
            "model_class": model.__class__.__name__,
            "criterion_class": criterion.__class__.__name__,
            "trainable_parameters": count_trainable_parameters(model),
            "model_config": config.get("model", {}),
            "loss_config": config.get("loss", {}),
        }
    )


def run_forward_smoke(args: argparse.Namespace) -> dict[str, object]:
    config = load_yaml_config(args.config)
    device = choose_device(args.device)
    model = build_model_from_config(config, device=device)
    criterion = build_loss_from_config(config, device=device)
    model.eval()

    batch = make_synthetic_batch(
        config,
        batch_size=args.batch_size,
        msa_depth=args.msa_depth,
        seq_len=args.seq_len,
        device=device,
    )
    ideal_backbone_local = build_ideal_backbone_local(config, device=device)

    with torch.no_grad():
        outputs = model(
            seq_tokens=batch["seq_tokens"],
            msa_tokens=batch["msa_tokens"],
            seq_mask=batch["seq_mask"],
            msa_mask=batch["msa_mask"],
            ideal_backbone_local=ideal_backbone_local,
            num_recycles=args.num_recycles,
        )
        losses = criterion(outputs, batch)

    summary = {
        "device": device,
        "num_recycles": int(args.num_recycles),
        "output_shapes": {
            key: list(value.shape)
            for key, value in outputs.items()
            if torch.is_tensor(value)
        },
        "finite_outputs": {
            key: bool(torch.isfinite(value).all().item())
            for key, value in outputs.items()
            if torch.is_tensor(value)
        },
        "losses": {key: float(value.detach().item()) for key, value in losses.items()},
    }
    print(summary)
    return summary


def run_pytest(args: argparse.Namespace) -> None:
    targets = args.target or ["tests"]
    extra_args = args.pytest_arg or []
    command = [sys.executable, "-m", "pytest", "--capture=no", *targets, *extra_args]
    print(f"[scripts.validate_model] running: {' '.join(command)}")
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.command == "instantiate":
        run_instantiate(args)
        return

    if args.command == "forward-smoke":
        run_forward_smoke(args)
        return

    if args.command == "pytest":
        run_pytest(args)
        return

    if args.command == "all":
        run_instantiate(args)
        run_forward_smoke(args)
        run_pytest(args)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
