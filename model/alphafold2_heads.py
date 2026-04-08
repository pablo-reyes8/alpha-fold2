"""Prediction heads built on top of the shared AlphaFold representations.

The classes in this module project internal sequence or pair features into the
single representation, pLDDT logits, distogram logits, masked-MSA logits, and
an optional predicted-TM head used for confidence-style reporting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleProjection(nn.Module):
    """
    Project MSA representation to single representation.
    AF2-like simplification: use first row of MSA and project to c_s.
    """
    def __init__(self, c_m=256, c_s=256):
        super().__init__()
        self.ln = nn.LayerNorm(c_m)
        self.linear = nn.Linear(c_m, c_s)

    def forward(self, m):
        # first sequence in MSA as target row
        s = m[:, 0]                  # [B, L, c_m]
        s = self.linear(self.ln(s))  # [B, L, c_s]
        return s


class PlddtHead(nn.Module):
    def __init__(self, c_s=256, hidden=256, num_bins=50):
        super().__init__()
        self.num_bins = num_bins
        self.ln = nn.LayerNorm(c_s)
        self.mlp = nn.Sequential(
            nn.Linear(c_s, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_bins))

    def forward(self, s):
        logits = self.mlp(self.ln(s))                # [B, L, num_bins]
        probs = torch.softmax(logits, dim=-1)
        bin_centers = torch.linspace(
            100.0 / self.num_bins / 2,
            100.0 - 100.0 / self.num_bins / 2,
            self.num_bins,
            device=s.device,
            dtype=s.dtype)

        plddt = (probs * bin_centers).sum(dim=-1)    # [B, L]
        return logits, plddt


class DistogramHead(nn.Module):
    def __init__(self, c_z=128, num_bins=64):
        super().__init__()
        self.num_bins = num_bins
        self.ln = nn.LayerNorm(c_z)
        self.linear = nn.Linear(c_z, num_bins)

    def forward(self, z):
        z_sym = 0.5 * (z + z.transpose(1, 2))        # symmetrize
        logits = self.linear(self.ln(z_sym))         # [B, L, L, num_bins]
        return logits


class MaskedMsaHead(nn.Module):
    def __init__(self, c_m=256, num_classes=23):
        super().__init__()
        self.num_classes = num_classes
        self.ln = nn.LayerNorm(c_m)
        self.linear = nn.Linear(c_m, num_classes)

    def forward(self, m):
        logits = self.linear(self.ln(m))
        return logits


def compute_predicted_tm_score(
    tm_logits: torch.Tensor,
    *,
    residue_mask: torch.Tensor | None = None,
    bin_centers: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """AlphaFold pTM lower bound from pairwise error logits.

    Parameters
    ----------
    tm_logits : [B, L, L, num_bins]
        Logits over aligned-error bins derived from the final pair representation.
    residue_mask : [B, L], optional
        Valid residues to include in the domain / chain subset.
    bin_centers : [num_bins], optional
        Representative error values for each aligned-error bin.
    eps : float
        Small numerical constant.
    """

    if tm_logits.ndim != 4:
        raise ValueError(f"tm_logits must have shape [B, L, L, C], got {tuple(tm_logits.shape)}")

    batch_size, length, _, num_bins = tm_logits.shape
    if bin_centers is None:
        if num_bins <= 1:
            bin_width = 0.5
        else:
            bin_width = 31.5 / float(num_bins - 1)
        bin_centers = torch.arange(num_bins, device=tm_logits.device, dtype=tm_logits.dtype)
        bin_centers = bin_width * (bin_centers + 0.5)
    else:
        bin_centers = bin_centers.to(device=tm_logits.device, dtype=tm_logits.dtype)
        if bin_centers.numel() != num_bins:
            raise ValueError(
                f"bin_centers must have {num_bins} entries, got {bin_centers.numel()}"
            )

    if residue_mask is None:
        residue_mask = torch.ones(batch_size, length, device=tm_logits.device, dtype=tm_logits.dtype)
    else:
        residue_mask = residue_mask.to(device=tm_logits.device, dtype=tm_logits.dtype)

    num_res = residue_mask.sum(dim=-1).clamp_min(1.0)
    d0 = 1.24 * torch.clamp(num_res, min=19.0).sub(15.0).pow(1.0 / 3.0) - 1.8
    d0 = d0.clamp_min(0.5)

    probs = F.softmax(tm_logits, dim=-1)
    tm_kernel = 1.0 / (1.0 + (bin_centers.view(1, 1, 1, -1) / (d0.view(-1, 1, 1, 1) + eps)) ** 2)
    expected_tm = (probs * tm_kernel).sum(dim=-1)

    per_alignment = (expected_tm * residue_mask[:, None, :]).sum(dim=-1) / num_res.view(-1, 1)
    per_alignment = per_alignment.masked_fill(residue_mask <= 0, float("-inf"))

    ptm = per_alignment.max(dim=-1).values
    has_valid = residue_mask.sum(dim=-1) > 0
    ptm = torch.where(has_valid, ptm, torch.zeros_like(ptm))
    return ptm


class TMHead(nn.Module):
    def __init__(self, c_z=128, num_bins=64, max_error=31.5):
        super().__init__()
        self.num_bins = int(num_bins)
        self.max_error = float(max_error)
        self.ln = nn.LayerNorm(c_z)
        self.linear = nn.Linear(c_z, self.num_bins)

        if self.num_bins <= 1:
            bin_width = 0.5
        else:
            bin_width = self.max_error / float(self.num_bins - 1)
        bin_centers = bin_width * (torch.arange(self.num_bins, dtype=torch.float32) + 0.5)
        self.register_buffer("bin_centers", bin_centers, persistent=False)

    def compute_ptm(self, tm_logits, residue_mask=None):
        return compute_predicted_tm_score(
            tm_logits,
            residue_mask=residue_mask,
            bin_centers=self.bin_centers,
        )

    def forward(self, z, residue_mask=None):
        logits = self.linear(self.ln(z))
        ptm = self.compute_ptm(logits, residue_mask=residue_mask)
        return logits, ptm
