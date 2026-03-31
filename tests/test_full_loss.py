from model.alphafold2_full_loss import *

def move_batch_to_device(batch, device: str):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

torch.manual_seed(11)

def random_unit_vectors(shape, device="cpu", dtype=torch.float32):
    x = torch.randn(*shape, device=device, dtype=dtype)
    x = x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-8)
    return x


def test_real_batch_plddt_loss(loader, device="cpu"):
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    B, L, _ = batch["coords_ca"].shape
    num_bins = 50

    x_true = batch["coords_ca"]
    x_pred = x_true.clone()                      # caso perfecto-ish
    mask = batch["valid_res_mask"]
    logits = torch.randn(B, L, num_bins, device=device)

    loss_fn = PlddtLoss(num_bins=num_bins, inclusion_radius=15.0)
    loss = loss_fn(logits, x_pred, x_true, mask=mask)

    print("PlddtLoss (real batch):", loss.item())
    assert torch.isfinite(loss), "PlddtLoss is not finite"
    assert loss.ndim == 0, "PlddtLoss should be scalar"


def test_real_batch_torsion_loss(loader, device="cpu"):
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    torsion_true = batch["torsion_true"]        # [B,L,3,2]
    torsion_mask = batch["torsion_mask"]        # [B,L,3]

    torsion_pred = torsion_true.clone()
    loss_fn = TorsionLoss()
    loss = loss_fn(torsion_pred, torsion_true, torsion_mask)

    print("TorsionLoss perfect prediction (real batch):", loss.item())
    assert torch.isfinite(loss), "TorsionLoss is not finite"
    assert loss.item() < 1e-7, "TorsionLoss should be ~0 for perfect prediction"

    torsion_pred2 = torsion_true + 0.1 * torch.randn_like(torsion_true)
    torsion_pred2 = torsion_pred2 / torch.linalg.norm(
        torsion_pred2, dim=-1, keepdim=True
    ).clamp_min(1e-8)

    loss2 = loss_fn(torsion_pred2, torsion_true, torsion_mask)

    print("TorsionLoss perturbed (real batch):", loss2.item())
    assert torch.isfinite(loss2), "Perturbed TorsionLoss is not finite"
    assert loss2.item() > loss.item(), "Perturbed torsion loss should be larger"


def test_real_batch_alphafold_loss_orchestrator(loader, device="cpu"):
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    B, L = batch["seq_tokens"].shape
    num_dist_bins = 64
    num_plddt_bins = 50
    T = batch["torsion_true"].shape[2]   # debería ser 3

    # out fake, pero consistente con el batch real
    out = {
        "R": torch.eye(3, device=device).view(1, 1, 3, 3).repeat(B, L, 1, 1),
        "t": batch["coords_ca"].clone(),   # usar CA real como predicción fácil
        "distogram_logits": torch.randn(B, L, L, num_dist_bins, device=device),
        "plddt_logits": torch.randn(B, L, num_plddt_bins, device=device),
        "torsions": batch["torsion_true"].clone(),  # predicción perfecta para torsiones
    }

    loss_fn = AlphaFoldLoss(
        fape_length_scale=10.0,
        fape_clamp_distance=10.0,
        dist_num_bins=num_dist_bins,
        dist_min_bin=2.0,
        dist_max_bin=22.0,
        plddt_num_bins=num_plddt_bins,
        plddt_inclusion_radius=15.0,
        w_fape=0.5,
        w_dist=0.3,
        w_plddt=0.01,
        w_torsion=0.01,
    )

    losses = loss_fn(out, batch)

    for k, v in losses.items():
        if torch.is_tensor(v) and v.ndim == 0:
            print(k, v.item())
        else:
            print(k, v)

    assert "loss" in losses
    assert torch.isfinite(losses["loss"]), "Total AlphaFold loss is not finite"
    assert losses["loss"].ndim == 0, "Total loss should be scalar"

    for name in ["fape_loss", "dist_loss", "plddt_loss", "torsion_loss"]:
        assert name in losses, f"Missing {name}"
        assert torch.isfinite(losses[name]), f"{name} is not finite"

    # como pusimos torsiones perfectas, debería ser casi cero
    assert losses["torsion_loss"].item() < 1e-7, "torsion_loss should be ~0 for perfect torsion prediction"

    # chequeos extra de shape para estar tranquilos
    assert out["R"].shape == (B, L, 3, 3)
    assert out["t"].shape == (B, L, 3)
    assert out["distogram_logits"].shape == (B, L, L, num_dist_bins)
    assert out["plddt_logits"].shape == (B, L, num_plddt_bins)
    assert out["torsions"].shape == batch["torsion_true"].shape
    assert T == 3, f"Expected 3 torsions, got {T}"


# =========================
# Run all
# =========================
device = 'cpu'
test_real_batch_plddt_loss(loader, device=device)
test_real_batch_torsion_loss(loader, device=device)
test_real_batch_alphafold_loss_orchestrator(loader, device=device)
print("All real-batch loss tests passed.")