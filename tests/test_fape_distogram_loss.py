"""Validate FAPE and distogram losses across invariances, masking, and gradient flow."""

from __future__ import annotations

import torch

from model.losses.distogram_loss import DistogramLoss
from model.losses.fape_loss import FAPELoss

torch.manual_seed(7)

# ============================================================
# Helpers
# ============================================================
def assert_close(x, y, atol=1e-5, rtol=1e-5, msg=""):
    if not torch.allclose(x, y, atol=atol, rtol=rtol):
        max_err = (x - y).abs().max().item()
        raise AssertionError(f"{msg} | max_err={max_err}")

def assert_scalar_finite(x, msg=""):
    assert torch.is_tensor(x), f"{msg} | expected tensor"
    assert x.ndim == 0, f"{msg} | expected scalar tensor, got {tuple(x.shape)}"
    assert torch.isfinite(x), f"{msg} | scalar is not finite"

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

def apply_global_rigid_to_frames_and_points(R, t, x, Rg, tg):
    """
    R:  [B,L,3,3]
    t:  [B,L,3]
    x:  [B,L,3]
    Rg: [B,3,3]
    tg: [B,3]
    """
    R_new = torch.einsum("bij,bljk->blik", Rg, R)
    t_new = torch.einsum("bij,blj->bli", Rg, t) + tg[:, None, :]
    x_new = torch.einsum("bij,blj->bli", Rg, x) + tg[:, None, :]
    return R_new, t_new, x_new

def make_fake_fape_batch(B=2, L=10, device="cpu", dtype=torch.float32):
    R_true = random_rotation_matrices(B, L, device=device, dtype=dtype)
    t_true = torch.randn(B, L, 3, device=device, dtype=dtype)
    x_true = t_true.clone()  # como PoC simple

    mask = torch.ones(B, L, device=device, dtype=dtype)
    mask[0, max(0, L - 2):] = 0.0

    return R_true, t_true, x_true, mask

def make_fake_distogram_batch(B=2, L=12, num_bins=64, device="cpu", dtype=torch.float32):
    x_true = torch.randn(B, L, 3, device=device, dtype=dtype)
    mask = torch.ones(B, L, device=device, dtype=dtype)
    mask[1, max(0, L - 3):] = 0.0
    logits = torch.randn(B, L, L, num_bins, device=device, dtype=dtype)
    return x_true, mask, logits

# ============================================================
# FAPE tests
# ============================================================
def test_fape_perfect_prediction_near_zero():
    B, L = 2, 10
    R_true, t_true, x_true, mask = make_fake_fape_batch(B=B, L=L)

    loss_fn = FAPELoss(length_scale=10.0, clamp_distance=10.0)
    loss = loss_fn(R_true, t_true, x_true, R_true, t_true, x_true, mask=mask)

    assert_scalar_finite(loss, "FAPE perfect loss")
    assert loss.item() < 2e-5, f"FAPE should be near 0 for perfect prediction, got {loss.item()}"

def test_fape_increases_when_prediction_worsens():
    B, L = 2, 10
    R_true, t_true, x_true, mask = make_fake_fape_batch(B=B, L=L)

    loss_fn = FAPELoss(length_scale=10.0, clamp_distance=10.0)

    loss0 = loss_fn(R_true, t_true, x_true, R_true, t_true, x_true, mask=mask)

    t_pred_small = t_true + 0.15 * torch.randn_like(t_true)
    x_pred_small = x_true + 0.15 * torch.randn_like(x_true)
    loss1 = loss_fn(R_true, t_pred_small, x_pred_small, R_true, t_true, x_true, mask=mask)

    t_pred_big = t_true + 1.0 * torch.randn_like(t_true)
    x_pred_big = x_true + 1.0 * torch.randn_like(x_true)
    loss2 = loss_fn(R_true, t_pred_big, x_pred_big, R_true, t_true, x_true, mask=mask)

    assert loss1.item() > loss0.item(), "Small perturbation should increase FAPE"
    assert loss2.item() >= loss1.item(), "Larger perturbation should not reduce FAPE"

def test_fape_global_rigid_invariance():
    """
    FAPE debe ser invariante si aplicamos la misma transformación rígida global
    a predicción y verdad.
    """
    B, L = 2, 9
    R_true, t_true, x_true, mask = make_fake_fape_batch(B=B, L=L)

    R_pred = R_true.clone()
    t_pred = t_true + 0.3 * torch.randn_like(t_true)
    x_pred = x_true + 0.3 * torch.randn_like(x_true)

    loss_fn = FAPELoss(length_scale=10.0, clamp_distance=10.0)
    loss1 = loss_fn(R_pred, t_pred, x_pred, R_true, t_true, x_true, mask=mask)

    Rg = random_global_rotations(B, device=t_true.device, dtype=t_true.dtype)
    tg = torch.randn(B, 3, device=t_true.device, dtype=t_true.dtype)

    R_pred2, t_pred2, x_pred2 = apply_global_rigid_to_frames_and_points(R_pred, t_pred, x_pred, Rg, tg)
    R_true2, t_true2, x_true2 = apply_global_rigid_to_frames_and_points(R_true, t_true, x_true, Rg, tg)

    loss2 = loss_fn(R_pred2, t_pred2, x_pred2, R_true2, t_true2, x_true2, mask=mask)

    assert_close(loss1, loss2, atol=1e-5, rtol=1e-5, msg="FAPE not invariant to global rigid transform")

def test_fape_masked_positions_do_not_contribute():
    B, L = 2, 10
    R_true, t_true, x_true, mask = make_fake_fape_batch(B=B, L=L)

    R_pred = R_true.clone()
    t_pred = t_true.clone()
    x_pred = x_true.clone()

    # Corrompemos sólo posiciones enmascaradas
    masked_idx = (mask == 0)
    t_pred[masked_idx] += 100.0
    x_pred[masked_idx] += 100.0

    loss_fn = FAPELoss(length_scale=10.0, clamp_distance=10.0)
    loss = loss_fn(R_pred, t_pred, x_pred, R_true, t_true, x_true, mask=mask)

    assert loss.item() < 2e-5, f"Masked corruption should not affect FAPE, got {loss.item()}"

def test_fape_all_zero_mask_gives_zero():
    B, L = 2, 8
    R_true, t_true, x_true, _ = make_fake_fape_batch(B=B, L=L)
    zero_mask = torch.zeros(B, L, dtype=t_true.dtype, device=t_true.device)

    R_pred = R_true.clone()
    t_pred = t_true + 5.0 * torch.randn_like(t_true)
    x_pred = x_true + 5.0 * torch.randn_like(x_true)

    loss_fn = FAPELoss(length_scale=10.0, clamp_distance=10.0)
    loss = loss_fn(R_pred, t_pred, x_pred, R_true, t_true, x_true, mask=zero_mask)

    assert_scalar_finite(loss, "FAPE all-zero-mask loss")
    assert_close(loss, torch.zeros_like(loss), atol=1e-8, msg="All-zero mask should give zero FAPE")

def test_fape_clamp_effect():
    B, L = 2, 10
    R_true, t_true, x_true, mask = make_fake_fape_batch(B=B, L=L)

    R_pred = R_true.clone()
    t_pred = t_true + 1000.0
    x_pred = x_true + 1000.0

    loss_small_clamp = FAPELoss(length_scale=10.0, clamp_distance=1.0)(
        R_pred, t_pred, x_pred, R_true, t_true, x_true, mask=mask
    )
    loss_big_clamp = FAPELoss(length_scale=10.0, clamp_distance=100.0)(
        R_pred, t_pred, x_pred, R_true, t_true, x_true, mask=mask
    )

    assert loss_big_clamp.item() >= loss_small_clamp.item(), "Larger clamp should not reduce max reachable loss here"
    assert loss_small_clamp.item() <= 1.0 / 10.0 + 1e-5, "Clamp=1 and length_scale=10 should cap normalized error near 0.1"

def test_fape_gradients_finite():
    B, L = 2, 9
    R_true, t_true, x_true, mask = make_fake_fape_batch(B=B, L=L)

    R_pred = R_true.clone().detach()
    t_pred = (t_true + 0.2 * torch.randn_like(t_true)).clone().detach().requires_grad_(True)
    x_pred = (x_true + 0.2 * torch.randn_like(x_true)).clone().detach().requires_grad_(True)

    loss_fn = FAPELoss(length_scale=10.0, clamp_distance=10.0)
    loss = loss_fn(R_pred, t_pred, x_pred, R_true, t_true, x_true, mask=mask)

    assert_scalar_finite(loss, "FAPE gradient loss")
    loss.backward()

    assert t_pred.grad is not None, "t_pred got no gradient"
    assert x_pred.grad is not None, "x_pred got no gradient"
    assert torch.isfinite(t_pred.grad).all(), "t_pred grad has NaN/Inf"
    assert torch.isfinite(x_pred.grad).all(), "x_pred grad has NaN/Inf"

# ============================================================
# Distogram tests
# ============================================================
def test_distogram_loss_scalar_and_finite():
    B, L, num_bins = 2, 12, 64
    x_true, mask, logits = make_fake_distogram_batch(B=B, L=L, num_bins=num_bins)

    loss_fn = DistogramLoss(num_bins=num_bins, min_bin=2.0, max_bin=22.0)
    loss = loss_fn(logits, x_true, mask=mask)

    assert_scalar_finite(loss, "Distogram loss")

def test_distogram_perfect_logits_low_loss():
    """
    Construimos logits muy altos en el bin correcto.
    La pérdida debe ser cercana a 0.
    """
    B, L, num_bins = 2, 11, 64
    x_true, mask, _ = make_fake_distogram_batch(B=B, L=L, num_bins=num_bins)

    loss_fn = DistogramLoss(num_bins=num_bins, min_bin=2.0, max_bin=22.0)

    with torch.no_grad():
        diff = x_true[:, :, None, :] - x_true[:, None, :, :]
        d_true = torch.sqrt((diff ** 2).sum(dim=-1) + loss_fn.eps)
        target_bins = torch.bucketize(d_true, loss_fn.boundaries)

    logits = torch.full((B, L, L, num_bins), -8.0, dtype=x_true.dtype, device=x_true.device)
    logits.scatter_(-1, target_bins.unsqueeze(-1), 8.0)

    loss = loss_fn(logits, x_true, mask=mask)

    assert loss.item() < 1e-3, f"Perfect distogram logits should give very low loss, got {loss.item()}"

def test_distogram_wrong_logits_higher_than_correct_logits():
    B, L, num_bins = 2, 10, 64
    x_true, mask, _ = make_fake_distogram_batch(B=B, L=L, num_bins=num_bins)
    loss_fn = DistogramLoss(num_bins=num_bins, min_bin=2.0, max_bin=22.0)

    with torch.no_grad():
        diff = x_true[:, :, None, :] - x_true[:, None, :, :]
        d_true = torch.sqrt((diff ** 2).sum(dim=-1) + loss_fn.eps)
        target_bins = torch.bucketize(d_true, loss_fn.boundaries)

    logits_good = torch.full((B, L, L, num_bins), -6.0, dtype=x_true.dtype, device=x_true.device)
    logits_good.scatter_(-1, target_bins.unsqueeze(-1), 6.0)

    wrong_bins = (target_bins + 1) % num_bins
    logits_bad = torch.full((B, L, L, num_bins), -6.0, dtype=x_true.dtype, device=x_true.device)
    logits_bad.scatter_(-1, wrong_bins.unsqueeze(-1), 6.0)

    loss_good = loss_fn(logits_good, x_true, mask=mask)
    loss_bad = loss_fn(logits_bad, x_true, mask=mask)

    assert loss_bad.item() > loss_good.item(), "Wrong-target logits should have larger loss"

def test_distogram_masked_positions_do_not_contribute():
    B, L, num_bins = 2, 12, 64
    x_true, mask, logits = make_fake_distogram_batch(B=B, L=L, num_bins=num_bins)

    loss_fn = DistogramLoss(num_bins=num_bins, min_bin=2.0, max_bin=22.0)

    loss_ref = loss_fn(logits, x_true, mask=mask)

    logits2 = logits.clone()
    invalid_pairs = (mask[:, :, None] * mask[:, None, :]) == 0
    logits2[invalid_pairs] = 1e4 * torch.randn_like(logits2[invalid_pairs])

    loss_corrupted = loss_fn(logits2, x_true, mask=mask)

    assert_close(loss_ref, loss_corrupted, atol=1e-5, rtol=1e-5, msg="Masked pairs should not affect distogram loss")

def test_distogram_all_zero_mask_gives_zero():
    B, L, num_bins = 2, 9, 64
    x_true = torch.randn(B, L, 3)
    logits = torch.randn(B, L, L, num_bins)
    zero_mask = torch.zeros(B, L)

    loss_fn = DistogramLoss(num_bins=num_bins, min_bin=2.0, max_bin=22.0)
    loss = loss_fn(logits, x_true, mask=zero_mask)

    assert_scalar_finite(loss, "Distogram all-zero-mask loss")
    assert_close(loss, torch.zeros_like(loss), atol=1e-8, msg="All-zero mask should give zero distogram loss")

def test_distogram_symmetric_logits_same_loss_after_transpose():
    """
    Como la verdad geométrica es simétrica en (i,j), transponer logits en i,j
    no debería cambiar la pérdida si logits también se transpone como par.
    """
    B, L, num_bins = 2, 8, 64
    x_true, mask, logits = make_fake_distogram_batch(B=B, L=L, num_bins=num_bins)

    loss_fn = DistogramLoss(num_bins=num_bins, min_bin=2.0, max_bin=22.0)

    loss1 = loss_fn(logits, x_true, mask=mask)
    loss2 = loss_fn(logits.transpose(1, 2), x_true, mask=mask)

    assert_close(loss1, loss2, atol=1e-5, rtol=1e-5, msg="Distogram loss should be symmetric under pair transpose")

def test_distogram_gradients_finite():
    B, L, num_bins = 2, 10, 64
    x_true, mask, logits = make_fake_distogram_batch(B=B, L=L, num_bins=num_bins)
    logits = logits.clone().detach().requires_grad_(True)

    loss_fn = DistogramLoss(num_bins=num_bins, min_bin=2.0, max_bin=22.0)
    loss = loss_fn(logits, x_true, mask=mask)

    assert_scalar_finite(loss, "Distogram gradient loss")
    loss.backward()

    assert logits.grad is not None, "Distogram logits got no gradient"
    assert torch.isfinite(logits.grad).all(), "Distogram logits grad has NaN/Inf"

def run_fape_and_distogram_tests():

    # FAPE
    test_fape_perfect_prediction_near_zero()
    test_fape_increases_when_prediction_worsens()
    test_fape_global_rigid_invariance()
    test_fape_masked_positions_do_not_contribute()
    test_fape_all_zero_mask_gives_zero()
    test_fape_clamp_effect()
    test_fape_gradients_finite()

    # Distogram
    test_distogram_loss_scalar_and_finite()
    test_distogram_perfect_logits_low_loss()
    test_distogram_wrong_logits_higher_than_correct_logits()
    test_distogram_masked_positions_do_not_contribute()
    test_distogram_all_zero_mask_gives_zero()
    test_distogram_symmetric_logits_same_loss_after_transpose()
    test_distogram_gradients_finite()

