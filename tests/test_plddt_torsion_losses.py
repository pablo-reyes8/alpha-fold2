"""Validate pLDDT and torsion losses across masking, invariance, and gradient checks."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.losses.pLDDT_loss import * 
from model.losses.torsion_loss import * 

torch.manual_seed(17)

# ============================================================
# Helpers
# ============================================================
def assert_close(x, y, atol=1e-5, rtol=1e-5, msg=""):
    if not torch.allclose(x, y, atol=atol, rtol=rtol):
        max_err = (x - y).abs().max().item()
        raise AssertionError(f"{msg} | max_err={max_err}")

def assert_scalar_finite(x, msg=""):
    assert torch.is_tensor(x), f"{msg} | expected tensor"
    assert x.ndim == 0, f"{msg} | expected scalar tensor, got shape {tuple(x.shape)}"
    assert torch.isfinite(x), f"{msg} | scalar is not finite"

def random_global_rotations(B, device="cpu", dtype=torch.float32):
    x = torch.randn(B, 3, 3, device=device, dtype=dtype)
    q, r = torch.linalg.qr(x)

    d = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    d = torch.where(d == 0, torch.ones_like(d), d)
    q = q * d.unsqueeze(-2)

    det = torch.det(q)
    flip = (det < 0).to(dtype).view(B, 1)
    q[..., :, 2] = q[..., :, 2] * (1.0 - 2.0 * flip)

    return q

def apply_global_rigid_to_points(x, Rg, tg):
    """
    x:  [B, L, 3]
    Rg: [B, 3, 3]
    tg: [B, 3]
    """
    return torch.einsum("bij,blj->bli", Rg, x) + tg[:, None, :]

def make_fake_plddt_batch(B=2, L=12, num_bins=50, device="cpu", dtype=torch.float32):
    x_true = torch.randn(B, L, 3, device=device, dtype=dtype)
    x_pred = x_true + 0.3 * torch.randn(B, L, 3, device=device, dtype=dtype)

    logits = torch.randn(B, L, num_bins, device=device, dtype=dtype)

    mask = torch.ones(B, L, device=device, dtype=dtype)
    if L >= 3:
        mask[0, -2:] = 0.0

    return x_pred, x_true, logits, mask

def make_fake_torsion_batch(B=2, L=10, T=7, device="cpu", dtype=torch.float32):
    torsion_true = torch.randn(B, L, T, 2, device=device, dtype=dtype)
    torsion_true = torsion_true / torch.linalg.norm(
        torsion_true, dim=-1, keepdim=True
    ).clamp_min(1e-8)

    torsion_pred = torsion_true + 0.2 * torch.randn(B, L, T, 2, device=device, dtype=dtype)

    torsion_mask = torch.ones(B, L, T, device=device, dtype=dtype)
    if L >= 2:
        torsion_mask[0, -2:, :] = 0.0

    return torsion_pred, torsion_true, torsion_mask

# ============================================================
# pLDDT LOSS TESTS
# ============================================================
def test_plddt_loss_scalar_and_finite():
    B, L, num_bins = 2, 12, 50
    x_pred, x_true, logits, mask = make_fake_plddt_batch(B=B, L=L, num_bins=num_bins)

    loss_fn = PlddtLoss(num_bins=num_bins, inclusion_radius=15.0)
    loss = loss_fn(logits, x_pred, x_true, mask=mask)

    assert_scalar_finite(loss, "pLDDT loss")

def test_plddt_loss_perfect_geometry_low_loss_with_correct_logits():
    """
    Build exact supervision targets from the same recipe as the loss,
    then place a very large logit on the correct bin.
    """
    B, L, num_bins = 2, 10, 50
    x_true = torch.randn(B, L, 3)
    x_pred = x_true.clone()
    mask = torch.ones(B, L)

    loss_fn = PlddtLoss(num_bins=num_bins, inclusion_radius=15.0)

    with torch.no_grad():
        diff_true = x_true[:, :, None, :] - x_true[:, None, :, :]
        diff_pred = x_pred[:, :, None, :] - x_pred[:, None, :, :]

        d_true = torch.sqrt((diff_true ** 2).sum(dim=-1) + loss_fn.eps)
        d_pred = torch.sqrt((diff_pred ** 2).sum(dim=-1) + loss_fn.eps)
        dist_error = torch.abs(d_pred - d_true)

        pair_mask = mask[:, :, None] * mask[:, None, :]
        not_self = 1.0 - torch.eye(L, dtype=x_true.dtype).unsqueeze(0)
        within_radius = (d_true < loss_fn.inclusion_radius).to(x_true.dtype)
        valid_pairs = pair_mask * not_self * within_radius

        s_05 = (dist_error < 0.5).to(x_true.dtype)
        s_10 = (dist_error < 1.0).to(x_true.dtype)
        s_20 = (dist_error < 2.0).to(x_true.dtype)
        s_40 = (dist_error < 4.0).to(x_true.dtype)
        pair_score = 0.25 * (s_05 + s_10 + s_20 + s_40)

        numer = (pair_score * valid_pairs).sum(dim=-1)
        denom = valid_pairs.sum(dim=-1).clamp_min(1.0)
        true_lddt = numer / denom
        true_score = 100.0 * true_lddt

        target_bins = torch.floor(true_score * num_bins / 100.0).long()
        target_bins = torch.clamp(target_bins, min=0, max=num_bins - 1)

    logits = torch.full((B, L, num_bins), -8.0, dtype=x_true.dtype)
    logits.scatter_(-1, target_bins.unsqueeze(-1), 8.0)

    loss = loss_fn(logits, x_pred, x_true, mask=mask)

    assert loss.item() < 1e-3, f"Correct pLDDT logits should give very low loss, got {loss.item()}"

def test_plddt_loss_wrong_logits_higher_than_correct_logits():
    B, L, num_bins = 2, 10, 50
    x_true = torch.randn(B, L, 3)
    x_pred = x_true + 0.2 * torch.randn(B, L, 3)
    mask = torch.ones(B, L)

    loss_fn = PlddtLoss(num_bins=num_bins, inclusion_radius=15.0)

    with torch.no_grad():
        diff_true = x_true[:, :, None, :] - x_true[:, None, :, :]
        diff_pred = x_pred[:, :, None, :] - x_pred[:, None, :, :]

        d_true = torch.sqrt((diff_true ** 2).sum(dim=-1) + loss_fn.eps)
        d_pred = torch.sqrt((diff_pred ** 2).sum(dim=-1) + loss_fn.eps)
        dist_error = torch.abs(d_pred - d_true)

        pair_mask = mask[:, :, None] * mask[:, None, :]
        not_self = 1.0 - torch.eye(L, dtype=x_true.dtype).unsqueeze(0)
        within_radius = (d_true < loss_fn.inclusion_radius).to(x_true.dtype)
        valid_pairs = pair_mask * not_self * within_radius

        s_05 = (dist_error < 0.5).to(x_true.dtype)
        s_10 = (dist_error < 1.0).to(x_true.dtype)
        s_20 = (dist_error < 2.0).to(x_true.dtype)
        s_40 = (dist_error < 4.0).to(x_true.dtype)
        pair_score = 0.25 * (s_05 + s_10 + s_20 + s_40)

        numer = (pair_score * valid_pairs).sum(dim=-1)
        denom = valid_pairs.sum(dim=-1).clamp_min(1.0)
        true_lddt = numer / denom
        true_score = 100.0 * true_lddt

        target_bins = torch.floor(true_score * num_bins / 100.0).long()
        target_bins = torch.clamp(target_bins, min=0, max=num_bins - 1)

    logits_good = torch.full((B, L, num_bins), -6.0)
    logits_good.scatter_(-1, target_bins.unsqueeze(-1), 6.0)

    wrong_bins = (target_bins + 1) % num_bins
    logits_bad = torch.full((B, L, num_bins), -6.0)
    logits_bad.scatter_(-1, wrong_bins.unsqueeze(-1), 6.0)

    loss_good = loss_fn(logits_good, x_pred, x_true, mask=mask)
    loss_bad = loss_fn(logits_bad, x_pred, x_true, mask=mask)

    assert loss_bad.item() > loss_good.item(), "Wrong pLDDT logits should have larger loss"

def test_plddt_loss_global_rigid_invariance():
    """
    Since only pairwise distances matter, applying the same global rigid transform
    to x_pred and x_true should leave the loss unchanged.
    """
    B, L, num_bins = 2, 11, 50
    x_pred, x_true, logits, mask = make_fake_plddt_batch(B=B, L=L, num_bins=num_bins)

    loss_fn = PlddtLoss(num_bins=num_bins, inclusion_radius=15.0)
    loss1 = loss_fn(logits, x_pred, x_true, mask=mask)

    Rg = random_global_rotations(B, device=x_true.device, dtype=x_true.dtype)
    tg = torch.randn(B, 3, device=x_true.device, dtype=x_true.dtype)

    x_pred2 = apply_global_rigid_to_points(x_pred, Rg, tg)
    x_true2 = apply_global_rigid_to_points(x_true, Rg, tg)

    loss2 = loss_fn(logits, x_pred2, x_true2, mask=mask)

    assert_close(loss1, loss2, atol=1e-5, rtol=1e-5, msg="pLDDT loss not rigid-invariant")

def test_plddt_loss_masked_residues_do_not_contribute():
    B, L, num_bins = 2, 12, 50
    x_pred, x_true, logits, mask = make_fake_plddt_batch(B=B, L=L, num_bins=num_bins)

    loss_fn = PlddtLoss(num_bins=num_bins, inclusion_radius=15.0)
    loss_ref = loss_fn(logits, x_pred, x_true, mask=mask)

    x_pred2 = x_pred.clone()
    x_true2 = x_true.clone()
    logits2 = logits.clone()

    masked = (mask == 0)
    x_pred2[masked] += 100.0
    x_true2[masked] -= 100.0
    logits2[masked] = 1e4 * torch.randn_like(logits2[masked])

    loss_corrupted = loss_fn(logits2, x_pred2, x_true2, mask=mask)

    assert_close(
        loss_ref,
        loss_corrupted,
        atol=1e-5,
        rtol=1e-5,
        msg="Masked residues should not affect pLDDT loss"
    )

def test_plddt_loss_no_valid_neighbors_gives_zero():
    """
    Put residues so far apart that no pair lies within inclusion_radius.
    Then no residue has valid supervision and the loss should be 0.
    """
    B, L, num_bins = 2, 8, 50
    x_true = torch.zeros(B, L, 3)
    x_true[:, :, 0] = torch.arange(L, dtype=torch.float32).view(1, L) * 100.0
    x_pred = x_true.clone()
    logits = torch.randn(B, L, num_bins)
    mask = torch.ones(B, L)

    loss_fn = PlddtLoss(num_bins=num_bins, inclusion_radius=15.0)
    loss = loss_fn(logits, x_pred, x_true, mask=mask)

    assert_scalar_finite(loss, "pLDDT no-neighbor loss")
    assert_close(loss, torch.zeros_like(loss), atol=1e-8, msg="No valid neighbors should give zero loss")

def test_plddt_loss_gradients_finite():
    B, L, num_bins = 2, 10, 50
    x_pred, x_true, logits, mask = make_fake_plddt_batch(B=B, L=L, num_bins=num_bins)
    logits = logits.clone().detach().requires_grad_(True)

    loss_fn = PlddtLoss(num_bins=num_bins, inclusion_radius=15.0)
    loss = loss_fn(logits, x_pred, x_true, mask=mask)

    assert_scalar_finite(loss, "pLDDT gradient loss")
    loss.backward()

    assert logits.grad is not None, "pLDDT logits got no gradient"
    assert torch.isfinite(logits.grad).all(), "pLDDT logits grad has NaN/Inf"

# ============================================================
# TORSION LOSS TESTS
# ============================================================
def test_torsion_loss_scalar_and_finite():
    B, L, T = 2, 10, 7
    torsion_pred, torsion_true, torsion_mask = make_fake_torsion_batch(B=B, L=L, T=T)

    loss_fn = TorsionLoss()
    loss = loss_fn(torsion_pred, torsion_true, torsion_mask=torsion_mask)

    assert_scalar_finite(loss, "Torsion loss")

def test_torsion_loss_perfect_prediction_zero():
    B, L, T = 2, 9, 7
    _, torsion_true, torsion_mask = make_fake_torsion_batch(B=B, L=L, T=T)

    loss_fn = TorsionLoss()
    loss = loss_fn(torsion_true, torsion_true, torsion_mask=torsion_mask)

    assert_close(loss, torch.zeros_like(loss), atol=1e-8, msg="Perfect torsion prediction should give zero loss")

def test_torsion_loss_known_value_opposite_vectors():
    """
    If pred = -true for unit vectors in R^2,
    squared error per torsion is ||u - (-u)||^2 = ||2u||^2 = 4.
    """
    B, L, T = 2, 6, 7
    torsion_true = torch.randn(B, L, T, 2)
    torsion_true = torsion_true / torch.linalg.norm(
        torsion_true, dim=-1, keepdim=True
    ).clamp_min(1e-8)

    torsion_pred = -torsion_true
    torsion_mask = torch.ones(B, L, T)

    loss_fn = TorsionLoss()
    loss = loss_fn(torsion_pred, torsion_true, torsion_mask=torsion_mask)

    expected = torch.tensor(4.0, dtype=loss.dtype, device=loss.device)
    assert_close(loss, expected, atol=1e-5, rtol=1e-5, msg="Opposite unit torsion vectors should give loss 4")

def test_torsion_loss_masked_entries_do_not_contribute():
    B, L, T = 2, 8, 7
    torsion_pred, torsion_true, torsion_mask = make_fake_torsion_batch(B=B, L=L, T=T)

    loss_fn = TorsionLoss()
    loss_ref = loss_fn(torsion_pred, torsion_true, torsion_mask=torsion_mask)

    torsion_pred2 = torsion_pred.clone()
    torsion_true2 = torsion_true.clone()

    masked = (torsion_mask == 0)
    torsion_pred2[masked] += 100.0
    torsion_true2[masked] -= 100.0

    loss_corrupted = loss_fn(torsion_pred2, torsion_true2, torsion_mask=torsion_mask)

    assert_close(
        loss_ref,
        loss_corrupted,
        atol=1e-5,
        rtol=1e-5,
        msg="Masked torsions should not affect loss"
    )

def test_torsion_loss_all_zero_mask_gives_zero():
    B, L, T = 2, 7, 7
    torsion_pred, torsion_true, _ = make_fake_torsion_batch(B=B, L=L, T=T)
    zero_mask = torch.zeros(B, L, T)

    loss_fn = TorsionLoss()
    loss = loss_fn(torsion_pred, torsion_true, torsion_mask=zero_mask)

    assert_scalar_finite(loss, "Torsion all-zero-mask loss")
    assert_close(loss, torch.zeros_like(loss), atol=1e-8, msg="All-zero torsion mask should give zero loss")

def test_torsion_loss_matches_manual_computation():
    B, L, T = 2, 5, 7
    torsion_pred, torsion_true, torsion_mask = make_fake_torsion_batch(B=B, L=L, T=T)

    loss_fn = TorsionLoss()
    loss = loss_fn(torsion_pred, torsion_true, torsion_mask=torsion_mask)

    sq_error = ((torsion_pred - torsion_true) ** 2).sum(dim=-1)
    manual = (sq_error * torsion_mask).sum() / torsion_mask.sum().clamp_min(1.0)

    assert_close(loss, manual, atol=1e-6, rtol=1e-6, msg="Torsion loss mismatch with manual formula")

def test_torsion_loss_gradients_finite():
    B, L, T = 2, 9, 7
    torsion_pred, torsion_true, torsion_mask = make_fake_torsion_batch(B=B, L=L, T=T)
    torsion_pred = torsion_pred.clone().detach().requires_grad_(True)

    loss_fn = TorsionLoss()
    loss = loss_fn(torsion_pred, torsion_true, torsion_mask=torsion_mask)

    assert_scalar_finite(loss, "Torsion gradient loss")
    loss.backward()

    assert torsion_pred.grad is not None, "torsion_pred got no gradient"
    assert torch.isfinite(torsion_pred.grad).all(), "torsion_pred grad has NaN/Inf"

# ============================================================
# RUNNER
# ============================================================
def run_plddt_and_torsion_loss_tests():

    # pLDDT
    test_plddt_loss_scalar_and_finite()
    test_plddt_loss_perfect_geometry_low_loss_with_correct_logits()
    test_plddt_loss_wrong_logits_higher_than_correct_logits()
    test_plddt_loss_global_rigid_invariance()
    test_plddt_loss_masked_residues_do_not_contribute()
    test_plddt_loss_no_valid_neighbors_gives_zero()
    test_plddt_loss_gradients_finite()

    # Torsion
    test_torsion_loss_scalar_and_finite()
    test_torsion_loss_perfect_prediction_zero()
    test_torsion_loss_known_value_opposite_vectors()
    test_torsion_loss_masked_entries_do_not_contribute()
    test_torsion_loss_all_zero_mask_gives_zero()
    test_torsion_loss_matches_manual_computation()
    test_torsion_loss_gradients_finite()
