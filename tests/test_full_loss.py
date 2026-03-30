
from model.alphafold2_full_loss import *


torch.manual_seed(11)

def test_plddt_loss():
    B, L, num_bins = 2, 12, 50

    x_true = torch.randn(B, L, 3)
    x_pred = x_true.clone()
    mask = torch.ones(B, L)
    mask[0, -2:] = 0.0

    logits = torch.randn(B, L, num_bins)

    loss_fn = PlddtLoss(num_bins=num_bins, inclusion_radius=15.0)
    loss = loss_fn(logits, x_pred, x_true, mask=mask)

    print("PlddtLoss:", loss.item())
    assert torch.isfinite(loss), "PlddtLoss is not finite"
    assert loss.ndim == 0, "PlddtLoss should be scalar"


def test_torsion_loss():
    B, L, T = 2, 10, 7

    torsion_true = torch.randn(B, L, T, 2)
    torsion_true = torsion_true / torch.linalg.norm(
        torsion_true, dim=-1, keepdim=True
    ).clamp_min(1e-8)

    torsion_pred = torsion_true.clone()
    torsion_mask = torch.ones(B, L, T)
    torsion_mask[0, -3:, :] = 0.0

    loss_fn = TorsionLoss()
    loss = loss_fn(torsion_pred, torsion_true, torsion_mask)

    print("TorsionLoss perfect prediction:", loss.item())
    assert torch.isfinite(loss), "TorsionLoss is not finite"
    assert loss.item() < 1e-7, "TorsionLoss should be ~0 for perfect prediction"

    torsion_pred2 = torsion_pred + 0.1 * torch.randn_like(torsion_pred)
    loss2 = loss_fn(torsion_pred2, torsion_true, torsion_mask)

    print("TorsionLoss perturbed:", loss2.item())
    assert loss2.item() > loss.item(), "Perturbed torsion loss should be larger"


def random_unit_vectors(shape, device="cpu", dtype=torch.float32):
    x = torch.randn(*shape, device=device, dtype=dtype)
    x = x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-8)
    return x


def test_alphafold_loss_orchestrator():
    B, L = 2, 16
    num_dist_bins = 64
    num_plddt_bins = 50
    T = 7

    coords_ca = torch.randn(B, L, 3)
    coords_n = coords_ca + random_unit_vectors((B, L, 3))
    coords_c = coords_ca + random_unit_vectors((B, L, 3))

    valid_res_mask = torch.ones(B, L)
    valid_backbone_mask = torch.ones(B, L)

    valid_res_mask[1, -4:] = 0.0
    valid_backbone_mask[1, -4:] = 0.0

    out = {
        "R": torch.eye(3).view(1, 1, 3, 3).repeat(B, L, 1, 1),
        "t": torch.randn(B, L, 3),
        "distogram_logits": torch.randn(B, L, L, num_dist_bins),
        "plddt_logits": torch.randn(B, L, num_plddt_bins),
        "torsions": torch.randn(B, L, T, 2),
    }

    out["torsions"] = out["torsions"] / torch.linalg.norm(
        out["torsions"], dim=-1, keepdim=True
    ).clamp_min(1e-8)

    torsion_true = torch.randn(B, L, T, 2)
    torsion_true = torsion_true / torch.linalg.norm(
        torsion_true, dim=-1, keepdim=True
    ).clamp_min(1e-8)

    torsion_mask = torch.ones(B, L, T)
    torsion_mask[0, -2:, :] = 0.0

    batch = {
        "coords_n": coords_n,
        "coords_ca": coords_ca,
        "coords_c": coords_c,
        "valid_res_mask": valid_res_mask,
        "valid_backbone_mask": valid_backbone_mask,
        "torsion_true": torsion_true,
        "torsion_mask": torsion_mask,
    }

    loss_fn = AlphaFoldLoss()
    losses = loss_fn(out, batch)

    for k, v in losses.items():
        print(k, v.item() if torch.is_tensor(v) and v.ndim == 0 else v)

    assert "loss" in losses
    assert torch.isfinite(losses["loss"]), "Total AlphaFold loss is not finite"
    assert losses["loss"].ndim == 0, "Total loss should be scalar"

    for name in ["fape_loss", "dist_loss", "plddt_loss", "torsion_loss"]:
        assert name in losses, f"Missing {name}"
        assert torch.isfinite(losses[name]), f"{name} is not finite"


test_plddt_loss()
test_torsion_loss()
test_alphafold_loss_orchestrator()
print("All new loss tests passed.")