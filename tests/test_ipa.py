"""Test invariant point attention behavior using shared silent-runner helpers and synthetic frames."""

import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.invariant_point_attention import *
from tests.test_helpers import (
    assert_close,
    assert_finite_tensor,
    assert_scalar_finite,
    assert_shape,
    finalize_test_results,
    run_test_silent,
)


# =========================================================
# Geometry helpers
# =========================================================
def random_rotation_matrices(B, L, device="cpu", dtype=torch.float32):
    x = torch.randn(B, L, 3, 3, device=device, dtype=dtype)
    q, r = torch.linalg.qr(x)

    d = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    d = torch.where(d == 0, torch.ones_like(d), d)
    q = q * d.unsqueeze(-2)

    det = torch.det(q)
    flip = (det < 0).to(dtype).view(B, L, 1)
    q[..., :, 2] = q[..., :, 2] * (1.0 - 2.0 * flip)

    return q


def apply_global_rigid_to_frames(R, t, Rg, tg):
    """
    Residue frames:
        R' = Rg @ R
        t' = Rg @ t + tg
    R:  [B,L,3,3]
    t:  [B,L,3]
    Rg: [B,3,3]
    tg: [B,3]
    """
    R_new = torch.einsum("bxy,blyz->blxz", Rg, R)
    t_new = torch.einsum("bxy,bly->blx", Rg, t) + tg[:, None, :]
    return R_new, t_new


def identity_frames(B, L, device="cpu", dtype=torch.float32):
    R = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).repeat(B, L, 1, 1)
    t = torch.zeros(B, L, 3, device=device, dtype=dtype)
    return R, t


# =========================================================
# Fake IPA batch
# =========================================================
torch.manual_seed(11)


def make_fake_ipa_batch(
    B=2,
    L=250,
    c_s=256,
    c_z=128,
    device="cpu",
    dtype=torch.float32,
):
    s = torch.randn(B, L, c_s, device=device, dtype=dtype)
    z = torch.randn(B, L, L, c_z, device=device, dtype=dtype)
    R = random_rotation_matrices(B, L, device=device, dtype=dtype)
    t = torch.randn(B, L, 3, device=device, dtype=dtype)

    mask = torch.ones(B, L, device=device, dtype=dtype)
    for b in range(B):
        cut = torch.randint(low=int(0.7 * L), high=L + 1, size=(1,)).item()
        mask[b, cut:] = 0.0

    return {
        "s": s,
        "z": z,
        "R": R,
        "t": t,
        "mask": mask,
    }


# =========================================================
# Tests for IPA
# =========================================================
def test_ipa_output_shapes(module, batch):
    module.eval()
    with torch.no_grad():
        s_update, attn = module(
            batch["s"], batch["z"], batch["R"], batch["t"], batch["mask"]
        )

    B, L, c_s = batch["s"].shape
    H = module.num_heads

    assert_shape(s_update, (B, L, c_s), "s_update")
    assert_shape(attn, (B, H, L, L), "attn")


def test_ipa_outputs_finite(module, batch):
    module.eval()
    with torch.no_grad():
        s_update, attn = module(
            batch["s"], batch["z"], batch["R"], batch["t"], batch["mask"]
        )

    assert_finite_tensor(s_update, "s_update")
    assert_finite_tensor(attn, "attn")


def test_ipa_deterministic_eval(module, batch):
    module.eval()
    with torch.no_grad():
        s1, a1 = module(batch["s"], batch["z"], batch["R"], batch["t"], batch["mask"])
        s2, a2 = module(batch["s"], batch["z"], batch["R"], batch["t"], batch["mask"])

    assert_close(s1, s2, name="s_update_deterministic")
    assert_close(a1, a2, name="attn_deterministic")


def test_ipa_mask_all_ones_matches_unmasked(module, batch):
    module.eval()
    B, L, _ = batch["s"].shape
    all_ones = torch.ones(B, L, device=batch["s"].device, dtype=batch["s"].dtype)

    with torch.no_grad():
        s_masked, a_masked = module(batch["s"], batch["z"], batch["R"], batch["t"], all_ones)
        s_unmasked, a_unmasked = module(batch["s"], batch["z"], batch["R"], batch["t"], None)

    assert_close(s_masked, s_unmasked, atol=1e-5, rtol=1e-5, name="s_masked_vs_unmasked")
    assert_close(a_masked, a_unmasked, atol=1e-5, rtol=1e-5, name="a_masked_vs_unmasked")


def test_ipa_all_zero_mask_gives_zero_s_update(module, batch):
    module.eval()
    B, L, _ = batch["s"].shape
    zero_mask = torch.zeros(B, L, device=batch["s"].device, dtype=batch["s"].dtype)

    with torch.no_grad():
        s_update, attn = module(batch["s"], batch["z"], batch["R"], batch["t"], zero_mask)

    assert_close(
        s_update,
        torch.zeros_like(s_update),
        atol=1e-7,
        rtol=1e-6,
        name="all_zero_mask_zero_s_update",
    )
    assert_finite_tensor(attn, "attn_all_zero_mask")


def test_ipa_output_zero_on_masked_positions(module, batch):
    module.eval()
    with torch.no_grad():
        s_update, _ = module(batch["s"], batch["z"], batch["R"], batch["t"], batch["mask"])

    masked = (batch["mask"] == 0).unsqueeze(-1)
    assert_close(
        s_update.masked_select(masked),
        torch.zeros_like(s_update.masked_select(masked)),
        atol=1e-7,
        rtol=1e-6,
        name="masked_positions_zero",
    )


def test_ipa_attention_rows_sum_to_one_on_valid_queries(module, batch):
    module.eval()
    with torch.no_grad():
        _, attn = module(batch["s"], batch["z"], batch["R"], batch["t"], batch["mask"])

    # attn: [B,H,L,L], softmax over last dim
    row_sums = attn.sum(dim=-1)  # [B,H,L]
    valid_queries = batch["mask"].unsqueeze(1).expand_as(row_sums) > 0

    assert_close(
        row_sums[valid_queries],
        torch.ones_like(row_sums[valid_queries]),
        atol=1e-5,
        rtol=1e-5,
        name="attn_rows_sum_to_one",
    )


def test_ipa_gradients_finite(module, batch):
    module.train()

    for p in module.parameters():
        if p.grad is not None:
            p.grad.zero_()

    s_update, attn = module(batch["s"], batch["z"], batch["R"], batch["t"], batch["mask"])
    loss = s_update.mean() + attn.mean()

    assert_scalar_finite(loss, "loss")
    loss.backward()

    got_grad = False
    for name, p in module.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} did not receive gradient"
            assert torch.isfinite(p.grad).all(), f"{name} gradient has NaN/Inf"
            got_grad = True

    assert got_grad, "No parameter got gradients"


def test_ipa_input_sensitivity_to_s(module, batch):
    module.eval()
    s2 = batch["s"].clone()
    valid_idx = (batch["mask"] > 0).nonzero(as_tuple=False)
    assert valid_idx.numel() > 0, "No valid positions in mask"

    for row in valid_idx[:16]:
        b, i = row.tolist()
        s2[b, i, :] += 0.5

    with torch.no_grad():
        s1, _ = module(batch["s"], batch["z"], batch["R"], batch["t"], batch["mask"])
        s2_out, _ = module(s2, batch["z"], batch["R"], batch["t"], batch["mask"])

    diff = (s1 - s2_out).abs().max().item()
    assert diff > 1e-8, f"s_update did not change enough, max diff={diff}"


def test_ipa_input_sensitivity_to_z(module, batch):
    module.eval()
    z2 = batch["z"].clone()
    z2[:, :8, :8, :] += 0.5

    with torch.no_grad():
        s1, a1 = module(batch["s"], batch["z"], batch["R"], batch["t"], batch["mask"])
        s2, a2 = module(batch["s"], z2, batch["R"], batch["t"], batch["mask"])

    diff_s = (s1 - s2).abs().max().item()
    diff_a = (a1 - a2).abs().max().item()
    assert diff_s > 1e-8 or diff_a > 1e-8, (
        f"IPA output did not change enough after perturbing z, "
        f"max diff s={diff_s}, attn={diff_a}"
    )


def test_ipa_input_sensitivity_to_frames(module, batch):
    module.eval()
    t2 = batch["t"].clone()
    valid_idx = (batch["mask"] > 0).nonzero(as_tuple=False)
    assert valid_idx.numel() > 0, "No valid positions in mask"

    for row in valid_idx[:16]:
        b, i = row.tolist()
        t2[b, i, :] += torch.tensor([0.3, -0.2, 0.1], device=t2.device, dtype=t2.dtype)

    with torch.no_grad():
        s1, a1 = module(batch["s"], batch["z"], batch["R"], batch["t"], batch["mask"])
        s2, a2 = module(batch["s"], batch["z"], batch["R"], t2, batch["mask"])

    diff_s = (s1 - s2).abs().max().item()
    diff_a = (a1 - a2).abs().max().item()
    assert diff_s > 1e-8 or diff_a > 1e-8, (
        f"IPA output did not change enough after perturbing frames, "
        f"max diff s={diff_s}, attn={diff_a}"
    )


def test_ipa_global_rigid_invariance(module, batch):
    """
    IPA debe ser invariante a una transformación rígida global aplicada
    a todos los frames residuo:
        R_i' = Rg R_i
        t_i' = Rg t_i + tg

    Porque las distancias entre puntos globales se preservan y luego el
    point_out_global se vuelve a llevar al frame local de cada residuo.
    """
    module.eval()

    B, L, _ = batch["s"].shape
    Rg = random_rotation_matrices(B, 1, device=batch["s"].device, dtype=batch["s"].dtype).squeeze(1)
    tg = torch.randn(B, 3, device=batch["s"].device, dtype=batch["s"].dtype)

    R2, t2 = apply_global_rigid_to_frames(batch["R"], batch["t"], Rg, tg)

    with torch.no_grad():
        s1, a1 = module(batch["s"], batch["z"], batch["R"], batch["t"], batch["mask"])
        s2, a2 = module(batch["s"], batch["z"], R2, t2, batch["mask"])

    assert_close(s1, s2, atol=1e-5, rtol=1e-5, name="s_update_global_rigid_invariance")
    assert_close(a1, a2, atol=1e-5, rtol=1e-5, name="attn_global_rigid_invariance")


def test_ipa_identity_frames_runs_fine(module, batch):
    module.eval()
    B, L, _ = batch["s"].shape
    R_id, t_zero = identity_frames(B, L, device=batch["s"].device, dtype=batch["s"].dtype)

    with torch.no_grad():
        s_update, attn = module(batch["s"], batch["z"], R_id, t_zero, batch["mask"])

    assert_finite_tensor(s_update, "s_update_identity_frames")
    assert_finite_tensor(attn, "attn_identity_frames")


def test_ipa_rotations_are_valid_in_batch(batch):
    R = batch["R"]
    I = torch.eye(3, device=R.device, dtype=R.dtype).view(1, 1, 3, 3)

    RtR = torch.matmul(R.transpose(-1, -2), R)
    assert_close(RtR, I.expand_as(RtR), atol=1e-5, rtol=1e-5, name="R_orthonormal")

    det = torch.det(R)
    assert_close(det, torch.ones_like(det), atol=1e-5, rtol=1e-5, name="det_R_plus_one")


# =========================================================
# Silent orchestrator
# =========================================================
def run_invariant_point_attention_test_suite(
    c_s=256,
    c_z=128,
    num_heads=8,
    c_hidden=32,
    num_qk_points=4,
    num_v_points=8,
    B=2,
    L=250,
    device="cpu",
):
    batch = make_fake_ipa_batch(
        B=B,
        L=L,
        c_s=c_s,
        c_z=c_z,
        device=device,
    )

    module = InvariantPointAttention(
        c_s=c_s,
        c_z=c_z,
        num_heads=num_heads,
        c_hidden=c_hidden,
        num_qk_points=num_qk_points,
        num_v_points=num_v_points,
    ).to(device)

    tests = [
        ("rotations_valid_in_batch", lambda: test_ipa_rotations_are_valid_in_batch(batch)),
        ("output_shapes", lambda: test_ipa_output_shapes(module, batch)),
        ("outputs_finite", lambda: test_ipa_outputs_finite(module, batch)),
        ("deterministic_eval", lambda: test_ipa_deterministic_eval(module, batch)),
        ("mask_all_ones_matches_unmasked", lambda: test_ipa_mask_all_ones_matches_unmasked(module, batch)),
        ("all_zero_mask_gives_zero_s_update", lambda: test_ipa_all_zero_mask_gives_zero_s_update(module, batch)),
        ("output_zero_on_masked_positions", lambda: test_ipa_output_zero_on_masked_positions(module, batch)),
        ("attention_rows_sum_to_one_on_valid_queries", lambda: test_ipa_attention_rows_sum_to_one_on_valid_queries(module, batch)),
        ("gradients_finite", lambda: test_ipa_gradients_finite(module, batch)),
        ("input_sensitivity_to_s", lambda: test_ipa_input_sensitivity_to_s(module, batch)),
        ("input_sensitivity_to_z", lambda: test_ipa_input_sensitivity_to_z(module, batch)),
        ("input_sensitivity_to_frames", lambda: test_ipa_input_sensitivity_to_frames(module, batch)),
        ("global_rigid_invariance", lambda: test_ipa_global_rigid_invariance(module, batch)),
        ("identity_frames_runs_fine", lambda: test_ipa_identity_frames_runs_fine(module, batch)),
    ]

    results = [run_test_silent(name, fn) for name, fn in tests]
    finalize_test_results(results, suite_name="InvariantPointAttention")
    return module, batch, results
