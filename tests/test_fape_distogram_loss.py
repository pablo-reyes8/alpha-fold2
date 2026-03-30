import torch 
from model.losses.fape_loss import * 
from model.losses.pLDDT_loss import * 
from model.losses.distogram_loss import * 
from model.losses.torsion_loss import * 
from model.losses.loss_helpers import *

torch.manual_seed(7)

def test_fape_loss():
    B, L = 2, 10

    # True frames
    R_true = torch.eye(3).view(1, 1, 3, 3).repeat(B, L, 1, 1)
    t_true = torch.randn(B, L, 3)

    # PoC: x_true = C_alpha coords = t_true
    x_true = t_true.clone()

    # Pred identical to true
    R_pred = R_true.clone()
    t_pred = t_true.clone()
    x_pred = x_true.clone()

    mask = torch.ones(B, L)
    mask[0, -2:] = 0.0

    loss_fn = FAPELoss(length_scale=10.0, clamp_distance=10.0)
    loss = loss_fn(R_pred, t_pred, x_pred, R_true, t_true, x_true, mask=mask)

    print("FAPE perfect-prediction loss:", loss.item())
    assert torch.isfinite(loss), "FAPE loss is not finite"
    assert loss.item() < 2e-5, "FAPE should be near 0 for perfect prediction"

    # perturb prediction
    t_pred2 = t_true + 0.5 * torch.randn(B, L, 3)
    x_pred2 = t_pred2.clone()
    loss2 = loss_fn(R_pred, t_pred2, x_pred2, R_true, t_true, x_true, mask=mask)

    print("FAPE perturbed loss:", loss2.item())
    assert loss2.item() > loss.item(), "Perturbed FAPE should be larger"


def test_distogram_loss():
    B, L, num_bins = 2, 12, 64
    x_true = torch.randn(B, L, 3)
    mask = torch.ones(B, L)
    mask[1, -3:] = 0.0

    logits = torch.randn(B, L, L, num_bins)

    loss_fn = DistogramLoss(num_bins=num_bins, min_bin=2.0, max_bin=22.0)
    loss = loss_fn(logits, x_true, mask=mask)

    print("Distogram loss:", loss.item())
    assert torch.isfinite(loss), "Distogram loss is not finite"
    assert loss.ndim == 0, "Distogram loss should be scalar"


test_fape_loss()
test_distogram_loss()
print("All loss tests passed.")