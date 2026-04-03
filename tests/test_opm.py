"""Test the outer product mean module with shared AlphaFold-style tensor assertions."""

import copy
import traceback
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from tests.test_helpers import *
from model.outer_product_mean import *
# =========================================================
# Tests for OuterProductMean
# =========================================================
def test_opm_output_shape(module, batch):
    module.eval()
    with torch.no_grad():
        z = module(batch["m"], batch["msa_mask"])

    B, N, L, _ = batch["m"].shape
    c_z = module.output_linear.out_features
    assert_shape(z, (B, L, L, c_z), "z_update")


def test_opm_output_finite(module, batch):
    module.eval()
    with torch.no_grad():
        z = module(batch["m"], batch["msa_mask"])
    assert_finite_tensor(z, "z_update")


def test_opm_deterministic_eval(module, batch):
    module.eval()
    with torch.no_grad():
        z1 = module(batch["m"], batch["msa_mask"])
        z2 = module(batch["m"], batch["msa_mask"])
    assert_close(z1, z2, name="deterministic_eval")


def test_opm_mask_all_ones_matches_unmasked(module, batch):
    module.eval()
    B, N, L, _ = batch["m"].shape
    all_ones = torch.ones(B, N, L, device=batch["m"].device, dtype=batch["m"].dtype)

    with torch.no_grad():
        z_masked = module(batch["m"], all_ones)
        z_unmasked = module(batch["m"], None)

    assert_close(
        z_masked, z_unmasked,
        atol=1e-5, rtol=1e-5,
        name="masked_vs_unmasked_all_ones"
    )


def test_opm_invariant_to_msa_permutation(module, batch):
    """
    OuterProductMean promedia sobre la dimensión N del MSA,
    así que permutar el orden de las secuencias no debería cambiar el resultado.
    """
    module.eval()
    B, N, L, _ = batch["m"].shape
    perm = torch.randperm(N, device=batch["m"].device)

    m_perm = batch["m"][:, perm]
    mask_perm = batch["msa_mask"][:, perm]

    with torch.no_grad():
        z1 = module(batch["m"], batch["msa_mask"])
        z2 = module(m_perm, mask_perm)

    assert_close(z1, z2, atol=1e-5, rtol=1e-5, name="msa_permutation_invariance")


def test_opm_ignores_fully_masked_extra_sequences(module, batch):
    """
    Si agregamos secuencias extra totalmente mask=0, no deberían alterar el promedio.
    """
    module.eval()
    B, N, L, c_m = batch["m"].shape
    extra = 7

    m_extra = torch.randn(B, extra, L, c_m, device=batch["m"].device, dtype=batch["m"].dtype)
    mask_extra = torch.zeros(B, extra, L, device=batch["m"].device, dtype=batch["m"].dtype)

    m_aug = torch.cat([batch["m"], m_extra], dim=1)
    mask_aug = torch.cat([batch["msa_mask"], mask_extra], dim=1)

    with torch.no_grad():
        z_base = module(batch["m"], batch["msa_mask"])
        z_aug = module(m_aug, mask_aug)

    assert_close(
        z_base, z_aug,
        atol=1e-5, rtol=1e-5,
        name="ignore_fully_masked_extra_sequences"
    )


def test_opm_gradient_flow(module, batch):
    module.train()

    for p in module.parameters():
        if p.grad is not None:
            p.grad.zero_()

    z = module(batch["m"], batch["msa_mask"])
    loss = z.mean()

    assert_scalar_finite(loss, "loss")
    loss.backward()

    got_grad = False
    for name, p in module.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} did not receive gradient"
            assert torch.isfinite(p.grad).all(), f"{name} gradient has NaN/Inf"
            got_grad = True

    assert got_grad, "No parameter got gradients"


def test_opm_output_changes_if_input_changes(module, batch):
    module.eval()

    m2 = batch["m"].clone()
    mask = batch["msa_mask"]

    # buscar una posición válida
    idx = (mask > 0).nonzero(as_tuple=False)
    b, n, l = idx[0].tolist()

    m2[b, n, l, :] += 1.0

    with torch.no_grad():
        z1 = module(batch["m"], batch["msa_mask"])
        z2 = module(m2, batch["msa_mask"])

    diff = (z1 - z2).abs().max().item()
    assert diff > 1e-8, f"Output did not change enough, max diff={diff}"


def test_opm_all_zero_mask_gives_constant_pairwise_output(module, batch):
    """
    Si todo el msa_mask = 0, entonces a=b=0 y el outer=0 en todos los pares.
    Después de output_linear, el resultado puede NO ser cero por el bias,
    pero sí debe ser constante a través de (i,j).
    """
    module.eval()
    B, N, L, _ = batch["m"].shape
    zero_mask = torch.zeros(B, N, L, device=batch["m"].device, dtype=batch["m"].dtype)

    with torch.no_grad():
        z = module(batch["m"], zero_mask)  # [B, L, L, c_z]

    # todos los pares (i,j) deberían ser iguales entre sí para cada batch
    ref = z[:, :1, :1, :]         # [B,1,1,c_z]
    ref = ref.expand_as(z)
    assert_close(z, ref, atol=1e-6, rtol=1e-5, name="all_zero_mask_constant_output")


def test_opm_special_case_outer_transpose_symmetry_when_a_equals_b(module, batch):
    module2 = copy.deepcopy(module)
    module2.eval()

    with torch.no_grad():
        module2.linear_b.weight.copy_(module2.linear_a.weight)
        module2.linear_b.bias.copy_(module2.linear_a.bias)

        m = module2.layer_norm(batch["m"])
        a = module2.linear_a(m)
        b = module2.linear_b(m)

        mask = batch["msa_mask"].unsqueeze(-1)
        a = a * mask
        b = b * mask

        outer = torch.einsum("bnic,bnjd->bijcd", a, b)

        pair_mask = torch.einsum("bni,bnj->bij", batch["msa_mask"], batch["msa_mask"])
        outer = outer / (pair_mask[..., None, None] + 1e-8)

    # propiedad correcta: outer[i,j,c,d] = outer[j,i,d,c]
    outer_rhs = outer.transpose(1, 2).transpose(-1, -2)
    assert torch.allclose(outer, outer_rhs, atol=1e-5, rtol=1e-5), \
        "outer does not satisfy transpose symmetry when linear_a == linear_b"


def test_opm_single_valid_sequence_matches_manual_formula(module, batch):
    """
    Caso estructural fuerte:
    si solo una secuencia del MSA está activa y las demás tienen mask=0,
    el promedio sobre N colapsa al outer product de esa secuencia.
    """
    module.eval()
    B, N, L, c_m = batch["m"].shape
    c_hidden = module.linear_a.out_features

    m = batch["m"].clone()
    mask = torch.zeros_like(batch["msa_mask"])
    mask[:, 0, :] = 1.0

    with torch.no_grad():
        z = module(m, mask)

        # reconstrucción manual
        m_ln = module.layer_norm(m)
        a = module.linear_a(m_ln)
        b = module.linear_b(m_ln)

        a = a * mask.unsqueeze(-1)
        b = b * mask.unsqueeze(-1)

        outer = torch.einsum("bnic,bnjd->bijcd", a, b)
        pair_mask = torch.einsum("bni,bnj->bij", mask, mask)
        outer = outer / (pair_mask[..., None, None] + 1e-8)
        outer = outer.reshape(B, L, L, c_hidden * c_hidden)
        z_manual = module.output_linear(outer)

    assert_close(z, z_manual, atol=1e-6, rtol=1e-5, name="single_valid_sequence_manual_match")


def run_outer_product_mean_test_suite(
    c_m=256,
    c_hidden=32,
    c_z=128,
    B=2,
    N_msa=128,
    L=250,
    device="cpu",
):
    batch = make_fake_msa_batch(
        B=B,
        N_msa=N_msa,
        L=L,
        c_m=c_m,
        device=device,
    )

    module = OuterProductMean(c_m=c_m, c_hidden=c_hidden, c_z=c_z).to(device)

    tests = [
        ("output_shape", lambda: test_opm_output_shape(module, batch)),
        ("output_finite", lambda: test_opm_output_finite(module, batch)),
        ("deterministic_eval", lambda: test_opm_deterministic_eval(module, batch)),
        ("mask_all_ones_matches_unmasked", lambda: test_opm_mask_all_ones_matches_unmasked(module, batch)),
        ("msa_permutation_invariance", lambda: test_opm_invariant_to_msa_permutation(module, batch)),
        ("ignores_fully_masked_extra_sequences", lambda: test_opm_ignores_fully_masked_extra_sequences(module, batch)),
        ("gradient_flow", lambda: test_opm_gradient_flow(module, batch)),
        ("input_sensitivity", lambda: test_opm_output_changes_if_input_changes(module, batch)),
        ("all_zero_mask_constant_output", lambda: test_opm_all_zero_mask_gives_constant_pairwise_output(module, batch)),
        ("special_case_outer_transpose_symmetry_when_a_equals_b",
         lambda: test_opm_special_case_outer_transpose_symmetry_when_a_equals_b(module, batch)),
        ("single_valid_sequence_manual_match", lambda: test_opm_single_valid_sequence_matches_manual_formula(module, batch)),
    ]

    results = [run_test_silent(name, fn) for name, fn in tests]
    finalize_test_results(results, suite_name="OuterProductMean")

    return module, batch, results