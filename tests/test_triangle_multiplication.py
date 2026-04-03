"""Test triangle multiplication blocks with shared silent-runner helpers and pairwise fixtures."""

import copy
import math

import torch
import torch.nn as nn


from model.triangle_multiplication import *
from tests.test_helpers import (
    assert_close,
    assert_finite_tensor,
    assert_scalar_finite,
    assert_shape,
    finalize_test_results,
    run_test_silent,
)


# =========================================================
# Fake pair batch
# =========================================================
torch.manual_seed(11)


def make_fake_pair_batch(
    B=2,
    L=250,
    c_z=128,
    device="cpu",
    dtype=torch.float32,
):
    z = torch.randn(B, L, L, c_z, device=device, dtype=dtype)

    pair_mask = torch.ones(B, L, L, device=device, dtype=dtype)

    # padding realista al final en filas/columnas
    for b in range(B):
        cut = torch.randint(low=int(0.7 * L), high=L + 1, size=(1,)).item()
        pair_mask[b, cut:, :] = 0.0
        pair_mask[b, :, cut:] = 0.0

    return {
        "z": z,
        "pair_mask": pair_mask,
    }


def permute_pair_tensor(z, perm):
    return z[:, perm][:, :, perm, :]


def permute_pair_mask(pair_mask, perm):
    return pair_mask[:, perm][:, :, perm]


# =========================================================
# Generic tests for triangle multiplication modules
# =========================================================
def test_triangle_output_shape(module, batch):
    module.eval()
    with torch.no_grad():
        out = module(batch["z"], batch["pair_mask"])

    B, L, _, _ = batch["z"].shape
    c_z = batch["z"].shape[-1]
    assert_shape(out, (B, L, L, c_z), "triangle_out")


def test_triangle_output_finite(module, batch):
    module.eval()
    with torch.no_grad():
        out = module(batch["z"], batch["pair_mask"])
    assert_finite_tensor(out, "triangle_out")


def test_triangle_deterministic_eval(module, batch):
    module.eval()
    with torch.no_grad():
        out1 = module(batch["z"], batch["pair_mask"])
        out2 = module(batch["z"], batch["pair_mask"])
    assert_close(out1, out2, name="deterministic_eval")


def test_triangle_mask_all_ones_matches_unmasked(module, batch):
    module.eval()
    B, L, _, _ = batch["z"].shape
    all_ones = torch.ones(B, L, L, device=batch["z"].device, dtype=batch["z"].dtype)

    with torch.no_grad():
        out_masked = module(batch["z"], all_ones)
        out_unmasked = module(batch["z"], None)

    assert_close(
        out_masked, out_unmasked,
        atol=1e-5, rtol=1e-5,
        name="mask_all_ones_matches_unmasked"
    )


def test_triangle_all_zero_mask_gives_zero_output(module, batch):
    module.eval()
    B, L, _, _ = batch["z"].shape
    zero_mask = torch.zeros(B, L, L, device=batch["z"].device, dtype=batch["z"].dtype)

    with torch.no_grad():
        out = module(batch["z"], zero_mask)

    assert_close(out, torch.zeros_like(out), atol=1e-7, rtol=1e-6, name="all_zero_mask_zero_output")


def test_triangle_output_zero_on_masked_positions(module, batch):
    module.eval()
    with torch.no_grad():
        out = module(batch["z"], batch["pair_mask"])

    masked = (batch["pair_mask"] == 0).unsqueeze(-1)
    assert_close(
        out.masked_select(masked),
        torch.zeros_like(out.masked_select(masked)),
        atol=1e-7,
        rtol=1e-6,
        name="masked_positions_are_zero"
    )


def test_triangle_gradient_flow(module, batch):
    module.train()

    for p in module.parameters():
        if p.grad is not None:
            p.grad.zero_()

    out = module(batch["z"], batch["pair_mask"])
    loss = out.mean()

    assert_scalar_finite(loss, "loss")
    loss.backward()

    got_grad = False
    for name, p in module.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} did not receive gradient"
            assert torch.isfinite(p.grad).all(), f"{name} gradient has NaN/Inf"
            got_grad = True

    assert got_grad, "No parameter got gradients"


def test_triangle_input_sensitivity(module, batch):
    module.eval()

    z2 = batch["z"].clone()
    valid_idx = (batch["pair_mask"] > 0).nonzero(as_tuple=False)
    assert valid_idx.numel() > 0, "No valid positions in pair_mask"

    # perturbar varias posiciones válidas
    for row in valid_idx[:16]:
        b, i, j = row.tolist()
        z2[b, i, j, :] += 0.5

    with torch.no_grad():
        out1 = module(batch["z"], batch["pair_mask"])
        out2 = module(z2, batch["pair_mask"])

    diff = (out1 - out2).abs().max().item()
    assert diff > 1e-8, f"Output did not change enough, max diff={diff}"


def test_triangle_permutation_equivariance(module, batch):
    """
    Si permutamos simultáneamente los residuos en ambos ejes de z y pair_mask,
    la salida debe permutarse de la misma forma.
    """
    module.eval()
    L = batch["z"].shape[1]
    perm = torch.randperm(L, device=batch["z"].device)

    z_perm = permute_pair_tensor(batch["z"], perm)
    mask_perm = permute_pair_mask(batch["pair_mask"], perm)

    with torch.no_grad():
        out = module(batch["z"], batch["pair_mask"])
        out_perm = module(z_perm, mask_perm)

    out_expected = permute_pair_tensor(out, perm)

    assert_close(
        out_perm, out_expected,
        atol=1e-5, rtol=1e-5,
        name="permutation_equivariance"
    )


def test_triangle_masked_input_perturbation_has_no_effect(module, batch):
    """
    Si perturbamos únicamente posiciones con pair_mask=0, la salida no debería cambiar.
    """
    module.eval()

    z2 = batch["z"].clone()
    masked_idx = (batch["pair_mask"] == 0).nonzero(as_tuple=False)

    if masked_idx.numel() == 0:
        return

    for row in masked_idx[:16]:
        b, i, j = row.tolist()
        z2[b, i, j, :] += 10.0

    with torch.no_grad():
        out1 = module(batch["z"], batch["pair_mask"])
        out2 = module(z2, batch["pair_mask"])

    assert_close(
        out1, out2,
        atol=1e-6, rtol=1e-5,
        name="masked_input_perturbation_no_effect"
    )


# =========================================================
# Structural relation between Outgoing and Incoming
# =========================================================
def test_triangle_outgoing_incoming_transpose_equivalence_with_constant_gate(
    batch, c_z=128, c_hidden=128, dropout=0.0, device="cpu"
):
    """
    Test estructural correcto para estas implementaciones:

    Si Outgoing e Incoming comparten pesos, dropout=0 y además hacemos
    el output_gate constante (sigmoid(0)=0.5), entonces sí debe cumplirse:

        Incoming(z, mask) == Outgoing(z^T, mask^T)

    porque la única fuente que rompía la equivalencia era el gate dependiente
    de z_norm[i,j] vs z_norm[j,i].
    """
    outgoing = TriangleMultiplicationOutgoing(
        c_z=c_z, c_hidden=c_hidden, dropout=dropout
    ).to(device)
    incoming = TriangleMultiplicationIncoming(
        c_z=c_z, c_hidden=c_hidden, dropout=dropout
    ).to(device)

    incoming.load_state_dict(copy.deepcopy(outgoing.state_dict()))

    # neutralizar el gate final en ambos módulos
    with torch.no_grad():
        outgoing.output_gate.weight.zero_()
        outgoing.output_gate.bias.zero_()
        incoming.output_gate.weight.zero_()
        incoming.output_gate.bias.zero_()

    outgoing.eval()
    incoming.eval()

    z = batch["z"]
    mask = batch["pair_mask"]

    z_t = z.transpose(1, 2)
    mask_t = mask.transpose(1, 2)

    with torch.no_grad():
        out_in = incoming(z, mask)
        out_out = outgoing(z_t, mask_t)

    assert_close(
        out_in, out_out,
        atol=1e-5, rtol=1e-5,
        name="incoming_outgoing_transpose_equivalence_with_constant_gate"
    )


# =========================================================
# Silent orchestrators
# =========================================================
def run_triangle_outgoing_test_suite(
    c_z=128,
    c_hidden=128,
    dropout=0.1,
    B=2,
    L=250,
    device="cpu",
):
    batch = make_fake_pair_batch(B=B, L=L, c_z=c_z, device=device)
    module = TriangleMultiplicationOutgoing(
        c_z=c_z,
        c_hidden=c_hidden,
        dropout=dropout,
    ).to(device)

    tests = [
        ("output_shape", lambda: test_triangle_output_shape(module, batch)),
        ("output_finite", lambda: test_triangle_output_finite(module, batch)),
        ("deterministic_eval", lambda: test_triangle_deterministic_eval(module, batch)),
        ("mask_all_ones_matches_unmasked", lambda: test_triangle_mask_all_ones_matches_unmasked(module, batch)),
        ("all_zero_mask_gives_zero_output", lambda: test_triangle_all_zero_mask_gives_zero_output(module, batch)),
        ("output_zero_on_masked_positions", lambda: test_triangle_output_zero_on_masked_positions(module, batch)),
        ("gradient_flow", lambda: test_triangle_gradient_flow(module, batch)),
        ("input_sensitivity", lambda: test_triangle_input_sensitivity(module, batch)),
        ("permutation_equivariance", lambda: test_triangle_permutation_equivariance(module, batch)),
        ("masked_input_perturbation_has_no_effect", lambda: test_triangle_masked_input_perturbation_has_no_effect(module, batch)),
    ]

    results = [run_test_silent(name, fn) for name, fn in tests]
    finalize_test_results(results, suite_name="TriangleMultiplicationOutgoing")
    return module, batch, results


def run_triangle_incoming_test_suite(
    c_z=128,
    c_hidden=128,
    dropout=0.1,
    B=2,
    L=250,
    device="cpu",
):
    batch = make_fake_pair_batch(B=B, L=L, c_z=c_z, device=device)
    module = TriangleMultiplicationIncoming(
        c_z=c_z,
        c_hidden=c_hidden,
        dropout=dropout,
    ).to(device)

    tests = [
        ("output_shape", lambda: test_triangle_output_shape(module, batch)),
        ("output_finite", lambda: test_triangle_output_finite(module, batch)),
        ("deterministic_eval", lambda: test_triangle_deterministic_eval(module, batch)),
        ("mask_all_ones_matches_unmasked", lambda: test_triangle_mask_all_ones_matches_unmasked(module, batch)),
        ("all_zero_mask_gives_zero_output", lambda: test_triangle_all_zero_mask_gives_zero_output(module, batch)),
        ("output_zero_on_masked_positions", lambda: test_triangle_output_zero_on_masked_positions(module, batch)),
        ("gradient_flow", lambda: test_triangle_gradient_flow(module, batch)),
        ("input_sensitivity", lambda: test_triangle_input_sensitivity(module, batch)),
        ("permutation_equivariance", lambda: test_triangle_permutation_equivariance(module, batch)),
        ("masked_input_perturbation_has_no_effect", lambda: test_triangle_masked_input_perturbation_has_no_effect(module, batch)),
    ]

    results = [run_test_silent(name, fn) for name, fn in tests]
    finalize_test_results(results, suite_name="TriangleMultiplicationIncoming")
    return module, batch, results


def run_triangle_joint_structural_test_suite(
    c_z=128,
    c_hidden=128,
    B=2,
    L=250,
    device="cpu",
):
    batch = make_fake_pair_batch(B=B, L=L, c_z=c_z, device=device)

    tests = [
        (
            "incoming_outgoing_transpose_equivalence",
            lambda: test_triangle_outgoing_incoming_transpose_equivalence_with_constant_gate(
                batch=batch,
                c_z=c_z,
                c_hidden=c_hidden,
                dropout=0.0,
                device=device,
            ),
        ),
    ]

    results = [run_test_silent(name, fn) for name, fn in tests]
    finalize_test_results(results, suite_name="TriangleMultiplicationJoint")
    return batch, results
