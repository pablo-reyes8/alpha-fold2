"""Validate AlphaFold output heads for shapes, masking rules, determinism, and gradients."""

import torch
import torch.nn as nn

from model.alphafold2_heads import *
from model.torsion_head import *

torch.manual_seed(123)

def assert_close(x, y, atol=1e-5, rtol=1e-5, msg=""):
    if not torch.allclose(x, y, atol=atol, rtol=rtol):
        max_err = (x - y).abs().max().item()
        raise AssertionError(f"{msg} | max_err={max_err}")

def test_single_projection():
    B, N, L, c_m, c_s = 2, 16, 37, 256, 256
    m = torch.randn(B, N, L, c_m)

    mod = SingleProjection(c_m=c_m, c_s=c_s)
    mod.eval()

    s = mod(m)

    assert s.shape == (B, L, c_s), f"Bad shape: {s.shape}"
    assert torch.isfinite(s).all(), "SingleProjection has non-finite values"

    # check that it is really using first row only
    m2 = m.clone()
    m2[:, 1:] = torch.randn_like(m2[:, 1:]) * 1000.0
    s2 = mod(m2)

    assert_close(s, s2, atol=1e-5, msg="SingleProjection should depend only on first MSA row")

def test_plddt_head():
    B, L, c_s = 2, 51, 256
    num_bins = 50
    s = torch.randn(B, L, c_s)

    mod = PlddtHead(c_s=c_s, hidden=256, num_bins=num_bins)
    mod.eval()

    logits, plddt = mod(s)

    assert logits.shape == (B, L, num_bins), f"Bad logits shape: {logits.shape}"
    assert plddt.shape == (B, L), f"Bad plddt shape: {plddt.shape}"

    assert torch.isfinite(logits).all(), "pLDDT logits contain non-finite values"
    assert torch.isfinite(plddt).all(), "pLDDT contains non-finite values"

    # Since pLDDT is expectation over bins in [0,100], it must stay in [0,100]
    assert (plddt >= 0.0).all(), "pLDDT has values < 0"
    assert (plddt <= 100.0).all(), "pLDDT has values > 100"

    # softmax sanity
    probs = torch.softmax(logits, dim=-1)
    sums = probs.sum(dim=-1)
    assert_close(sums, torch.ones_like(sums), atol=1e-5, msg="Softmax probs do not sum to 1")

def test_distogram_head():
    B, L, c_z, num_bins = 2, 41, 128, 64
    z = torch.randn(B, L, L, c_z)

    mod = DistogramHead(c_z=c_z, num_bins=num_bins)
    mod.eval()

    logits = mod(z)

    assert logits.shape == (B, L, L, num_bins), f"Bad shape: {logits.shape}"
    assert torch.isfinite(logits).all(), "Distogram logits contain non-finite values"

    # symmetry check
    assert_close(
        logits,
        logits.transpose(1, 2),
        atol=1e-5,
        msg="Distogram logits should be symmetric in residue pair indices"
    )

def test_torsion_resblock():
    B, L, dim = 2, 29, 256
    x = torch.randn(B, L, dim)

    mod = TorsionResBlock(dim=dim, dropout=0.0)
    mod.eval()

    y = mod(x)

    assert y.shape == x.shape, f"Bad shape: {y.shape}"
    assert torch.isfinite(y).all(), "TorsionResBlock contains non-finite values"

    # residual block should not collapse everything
    diff = (y - x).abs().mean().item()
    assert diff > 0.0, "TorsionResBlock seems to produce no update"

def test_torsion_head_zero_init_behavior():
    B, L, c_s, n_torsions = 2, 33, 256, 7
    s_initial = torch.randn(B, L, c_s)
    s_final = torch.randn(B, L, c_s)

    mod = TorsionHead(
        c_s=c_s,
        hidden=256,
        n_torsions=n_torsions,
        num_res_blocks=2,
        dropout=0.0
    )
    mod.eval()

    torsions = mod(s_initial, s_final)

    assert torsions.shape == (B, L, n_torsions, 2), f"Bad shape: {torsions.shape}"
    assert torch.isfinite(torsions).all(), "TorsionHead outputs non-finite values"

    # Because final layer is zero-initialized, initial output should be exactly zero
    assert_close(
        torsions,
        torch.zeros_like(torsions),
        atol=1e-7,
        msg="TorsionHead should start near exactly zero because output layer is zero-init"
    )

def test_torsion_head_mask():
    B, L, c_s, n_torsions = 2, 20, 256, 7
    s_initial = torch.randn(B, L, c_s)
    s_final = torch.randn(B, L, c_s)

    mask = torch.ones(B, L)
    mask[0, -4:] = 0.0

    mod = TorsionHead(
        c_s=c_s,
        hidden=256,
        n_torsions=n_torsions,
        num_res_blocks=2,
        dropout=0.0
    )
    mod.eval()

    torsions = mod(s_initial, s_final, mask=mask)

    assert torsions.shape == (B, L, n_torsions, 2), f"Bad shape: {torsions.shape}"
    assert torch.isfinite(torsions).all(), "Masked torsions contain non-finite values"

    assert_close(
        torsions[0, -4:],
        torch.zeros_like(torsions[0, -4:]),
        atol=1e-7,
        msg="Masked torsions should be zero"
    )

def test_torsion_head_norm_after_breaking_zero_init():
    B, L, c_s, n_torsions = 2, 24, 256, 7
    s_initial = torch.randn(B, L, c_s)
    s_final = torch.randn(B, L, c_s)

    mod = TorsionHead(
        c_s=c_s,
        hidden=256,
        n_torsions=n_torsions,
        num_res_blocks=2,
        dropout=0.0
    )

    # perturb final layer so output is not all zero
    with torch.no_grad():
        mod.output.weight.normal_(mean=0.0, std=0.02)
        mod.output.bias.normal_(mean=0.0, std=0.02)

    mod.eval()
    torsions = mod(s_initial, s_final)

    assert torsions.shape == (B, L, n_torsions, 2)
    assert torch.isfinite(torsions).all(), "TorsionHead outputs non-finite values"

    norms = torch.linalg.norm(torsions, dim=-1)  # [B,L,n_torsions]

    # Only check unit norm where output is actually non-zero numerically
    # after perturbing, almost all should be unit norm
    assert_close(
        norms,
        torch.ones_like(norms),
        atol=1e-4,
        msg="Torsion vectors are not normalized to unit norm"
    )

def assert_scalar_finite(x, msg=""):
    assert torch.is_tensor(x), f"{msg} | expected tensor scalar"
    assert x.ndim == 0, f"{msg} | expected scalar tensor, got shape {tuple(x.shape)}"
    assert torch.isfinite(x), f"{msg} | scalar is not finite"

def test_single_projection_deterministic_eval():
    B, N, L, c_m, c_s = 2, 16, 37, 256, 256
    m = torch.randn(B, N, L, c_m)

    mod = SingleProjection(c_m=c_m, c_s=c_s)
    mod.eval()

    with torch.no_grad():
        s1 = mod(m)
        s2 = mod(m)

    assert_close(s1, s2, atol=1e-6, msg="SingleProjection eval is not deterministic")

def test_single_projection_gradients_finite():
    B, N, L, c_m, c_s = 2, 16, 37, 256, 256
    m = torch.randn(B, N, L, c_m)

    mod = SingleProjection(c_m=c_m, c_s=c_s)
    mod.train()

    s = mod(m)
    loss = (s ** 2).mean()

    assert_scalar_finite(loss, msg="SingleProjection loss not finite")
    loss.backward()

    got_grad = False
    for name, p in mod.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} got no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} gradient has NaN/Inf"
            got_grad = True

    assert got_grad, "No SingleProjection parameter received gradient"

def test_plddt_head_deterministic_eval():
    B, L, c_s = 2, 51, 256
    num_bins = 50
    s = torch.randn(B, L, c_s)

    mod = PlddtHead(c_s=c_s, hidden=256, num_bins=num_bins)
    mod.eval()

    with torch.no_grad():
        logits1, plddt1 = mod(s)
        logits2, plddt2 = mod(s)

    assert_close(logits1, logits2, atol=1e-6, msg="PlddtHead logits not deterministic in eval")
    assert_close(plddt1, plddt2, atol=1e-6, msg="PlddtHead plddt not deterministic in eval")

def test_plddt_head_uniform_logits_give_mean_bin_center():
    """
    If logits are uniform across bins, pLDDT should equal the mean of bin centers,
    which for evenly spaced centers in [0,100] is exactly 50.
    """
    B, L, c_s = 2, 13, 256
    num_bins = 50
    s = torch.randn(B, L, c_s)

    mod = PlddtHead(c_s=c_s, hidden=256, num_bins=num_bins)

    with torch.no_grad():
        for layer in mod.mlp:
            if isinstance(layer, nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()

    mod.eval()
    _, plddt = mod(s)

    expected = torch.full_like(plddt, 50.0)
    assert_close(plddt, expected, atol=1e-5, msg="Uniform logits should give pLDDT=50")

def test_plddt_head_input_sensitivity():
    B, L, c_s = 2, 31, 256
    num_bins = 50
    s1 = torch.randn(B, L, c_s)
    s2 = s1.clone()
    s2[:, :5, :] += 0.5

    mod = PlddtHead(c_s=c_s, hidden=256, num_bins=num_bins)
    mod.eval()

    with torch.no_grad():
        logits1, plddt1 = mod(s1)
        logits2, plddt2 = mod(s2)

    diff_logits = (logits1 - logits2).abs().max().item()
    diff_plddt = (plddt1 - plddt2).abs().max().item()

    assert diff_logits > 1e-8 or diff_plddt > 1e-8, (
        f"PlddtHead output did not change enough after input perturbation; "
        f"diff_logits={diff_logits}, diff_plddt={diff_plddt}"
    )

def test_plddt_head_gradients_finite():
    B, L, c_s = 2, 27, 256
    num_bins = 50
    s = torch.randn(B, L, c_s)

    mod = PlddtHead(c_s=c_s, hidden=256, num_bins=num_bins)
    mod.train()

    logits, plddt = mod(s)
    loss = logits.mean() + plddt.mean()

    assert_scalar_finite(loss, msg="PlddtHead loss not finite")
    loss.backward()

    got_grad = False
    for name, p in mod.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} got no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} gradient has NaN/Inf"
            got_grad = True

    assert got_grad, "No PlddtHead parameter received gradient"

def test_distogram_head_deterministic_eval():
    B, L, c_z, num_bins = 2, 41, 128, 64
    z = torch.randn(B, L, L, c_z)

    mod = DistogramHead(c_z=c_z, num_bins=num_bins)
    mod.eval()

    with torch.no_grad():
        logits1 = mod(z)
        logits2 = mod(z)

    assert_close(logits1, logits2, atol=1e-6, msg="DistogramHead eval is not deterministic")

def test_distogram_head_input_symmetrization_equivalence():
    """
    Because the head symmetrizes internally, feeding z or 0.5*(z+z^T)
    should produce exactly the same output.
    """
    B, L, c_z, num_bins = 2, 25, 128, 64
    z = torch.randn(B, L, L, c_z)
    z_sym = 0.5 * (z + z.transpose(1, 2))

    mod = DistogramHead(c_z=c_z, num_bins=num_bins)
    mod.eval()

    with torch.no_grad():
        logits1 = mod(z)
        logits2 = mod(z_sym)

    assert_close(
        logits1,
        logits2,
        atol=1e-6,
        msg="DistogramHead not equivalent on z vs symmetrized z"
    )

def test_distogram_head_input_sensitivity():
    B, L, c_z, num_bins = 2, 31, 128, 64
    z1 = torch.randn(B, L, L, c_z)
    z2 = z1.clone()
    z2[:, :4, :4, :] += 0.5

    mod = DistogramHead(c_z=c_z, num_bins=num_bins)
    mod.eval()

    with torch.no_grad():
        logits1 = mod(z1)
        logits2 = mod(z2)

    diff = (logits1 - logits2).abs().max().item()
    assert diff > 1e-8, f"DistogramHead output did not change enough after perturbing z; diff={diff}"

def test_distogram_head_gradients_finite():
    B, L, c_z, num_bins = 2, 29, 128, 64
    z = torch.randn(B, L, L, c_z)

    mod = DistogramHead(c_z=c_z, num_bins=num_bins)
    mod.train()

    logits = mod(z)
    loss = logits.mean()

    assert_scalar_finite(loss, msg="DistogramHead loss not finite")
    loss.backward()

    got_grad = False
    for name, p in mod.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} got no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} gradient has NaN/Inf"
            got_grad = True

    assert got_grad, "No DistogramHead parameter received gradient"

def test_torsion_resblock_deterministic_eval():
    B, L, dim = 2, 29, 256
    x = torch.randn(B, L, dim)

    mod = TorsionResBlock(dim=dim, dropout=0.1)
    mod.eval()

    with torch.no_grad():
        y1 = mod(x)
        y2 = mod(x)

    assert_close(y1, y2, atol=1e-6, msg="TorsionResBlock eval is not deterministic")

def test_torsion_resblock_gradients_finite():
    B, L, dim = 2, 29, 256
    x = torch.randn(B, L, dim)

    mod = TorsionResBlock(dim=dim, dropout=0.0)
    mod.train()

    y = mod(x)
    loss = (y ** 2).mean()

    assert_scalar_finite(loss, msg="TorsionResBlock loss not finite")
    loss.backward()

    got_grad = False
    for name, p in mod.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} got no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} gradient has NaN/Inf"
            got_grad = True

    assert got_grad, "No TorsionResBlock parameter received gradient"

def test_torsion_head_deterministic_eval():
    B, L, c_s, n_torsions = 2, 21, 256, 7
    s_initial = torch.randn(B, L, c_s)
    s_final = torch.randn(B, L, c_s)

    mod = TorsionHead(
        c_s=c_s,
        hidden=256,
        n_torsions=n_torsions,
        num_res_blocks=2,
        dropout=0.1
    )
    mod.eval()

    with torch.no_grad():
        t1 = mod(s_initial, s_final)
        t2 = mod(s_initial, s_final)

    assert_close(t1, t2, atol=1e-6, msg="TorsionHead eval is not deterministic")

def test_torsion_head_all_zero_mask_gives_zero():
    B, L, c_s, n_torsions = 2, 20, 256, 7
    s_initial = torch.randn(B, L, c_s)
    s_final = torch.randn(B, L, c_s)
    mask = torch.zeros(B, L)

    mod = TorsionHead(
        c_s=c_s,
        hidden=256,
        n_torsions=n_torsions,
        num_res_blocks=2,
        dropout=0.0
    )
    mod.eval()

    torsions = mod(s_initial, s_final, mask=mask)

    assert_close(
        torsions,
        torch.zeros_like(torsions),
        atol=1e-7,
        msg="All-zero mask should force torsion output to zero"
    )

def test_torsion_head_input_sensitivity_after_breaking_zero_init():
    B, L, c_s, n_torsions = 2, 24, 256, 7
    s_initial_1 = torch.randn(B, L, c_s)
    s_final_1 = torch.randn(B, L, c_s)

    s_initial_2 = s_initial_1.clone()
    s_final_2 = s_final_1.clone()
    s_final_2[:, :4, :] += 0.5

    mod = TorsionHead(
        c_s=c_s,
        hidden=256,
        n_torsions=n_torsions,
        num_res_blocks=2,
        dropout=0.0
    )

    with torch.no_grad():
        mod.output.weight.normal_(mean=0.0, std=0.02)
        mod.output.bias.normal_(mean=0.0, std=0.02)

    mod.eval()
    tors1 = mod(s_initial_1, s_final_1)
    tors2 = mod(s_initial_2, s_final_2)

    diff = (tors1 - tors2).abs().max().item()
    assert diff > 1e-8, f"TorsionHead output did not change enough after perturbing inputs; diff={diff}"

def test_torsion_head_gradients_finite_after_breaking_zero_init():
    B, L, c_s, n_torsions = 2, 18, 256, 7
    s_initial = torch.randn(B, L, c_s)
    s_final = torch.randn(B, L, c_s)
    mask = torch.ones(B, L)

    mod = TorsionHead(
        c_s=c_s,
        hidden=256,
        n_torsions=n_torsions,
        num_res_blocks=2,
        dropout=0.0
    )

    with torch.no_grad():
        mod.output.weight.normal_(mean=0.0, std=0.02)
        mod.output.bias.normal_(mean=0.0, std=0.02)

    mod.train()
    torsions = mod(s_initial, s_final, mask=mask)
    loss = torsions.mean()

    assert_scalar_finite(loss, msg="TorsionHead loss not finite")
    loss.backward()

    got_grad = False
    for name, p in mod.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} got no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} gradient has NaN/Inf"
            got_grad = True

    assert got_grad, "No TorsionHead parameter received gradient"

def run_all_head_tests():

    # SingleProjection
    test_single_projection()
    test_single_projection_deterministic_eval()
    test_single_projection_gradients_finite()

    # PlddtHead
    test_plddt_head()
    test_plddt_head_deterministic_eval()
    test_plddt_head_uniform_logits_give_mean_bin_center()
    test_plddt_head_input_sensitivity()
    test_plddt_head_gradients_finite()

    # DistogramHead
    test_distogram_head()
    test_distogram_head_deterministic_eval()
    test_distogram_head_input_symmetrization_equivalence()
    test_distogram_head_input_sensitivity()
    test_distogram_head_gradients_finite()

    # TorsionResBlock
    test_torsion_resblock()
    test_torsion_resblock_deterministic_eval()
    test_torsion_resblock_gradients_finite()

    # TorsionHead
    test_torsion_head_zero_init_behavior()
    test_torsion_head_mask()
    test_torsion_head_all_zero_mask_gives_zero()
    test_torsion_head_norm_after_breaking_zero_init()
    test_torsion_head_deterministic_eval()
    test_torsion_head_input_sensitivity_after_breaking_zero_init()
    test_torsion_head_gradients_finite_after_breaking_zero_init()

