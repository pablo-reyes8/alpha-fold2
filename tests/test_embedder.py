
import torch
import torch.nn as nn

from model.input_embedder import *

def test_input_embedder_shapes():

    B = 2
    N_msa = 128
    L = 275
    n_tokens = 23
    pad_idx = 0

    seq_tokens = torch.randint(1, n_tokens, (B, L))
    msa_tokens = torch.randint(1, n_tokens, (B, N_msa, L))

    seq_mask = torch.ones(B, L, dtype=torch.float32)
    msa_mask = torch.ones(B, N_msa, L, dtype=torch.float32)

    # meter algunos pads para probar mask
    seq_tokens[0, -10:] = pad_idx
    seq_mask[0, -10:] = 0.0

    msa_tokens[0, :, -10:] = pad_idx
    msa_mask[0, :, -10:] = 0.0

    model = InputEmbedder(
        n_tokens=n_tokens,
        c_m=256,
        c_z=128,
        c_s=256,
        max_relpos=32,
        pad_idx=pad_idx)

    m, z = model(
        seq_tokens=seq_tokens,
        msa_tokens=msa_tokens,
        seq_mask=seq_mask,
        msa_mask=msa_mask,)

    # --------------------------
    # shape checks
    # --------------------------
    assert m.shape == (B, N_msa, L, 256), f"m shape incorrect: {m.shape}"
    assert z.shape == (B, L, L, 128), f"z shape incorrect: {z.shape}"

    # --------------------------
    # finiteness
    # --------------------------
    assert torch.isfinite(m).all(), "m has non-finite values"
    assert torch.isfinite(z).all(), "z has non-finite values"

    # --------------------------
    # masked positions ~ zero
    # --------------------------
    assert torch.allclose(
        m[0, :, -10:, :],
        torch.zeros_like(m[0, :, -10:, :]),
        atol=1e-5
    ), "MSA masked positions are not zeroed"

    assert torch.allclose(
        z[0, -10:, :, :],
        torch.zeros_like(z[0, -10:, :, :]),
        atol=1e-5
    ), "Pair masked rows are not zeroed"

    assert torch.allclose(
        z[0, :, -10:, :],
        torch.zeros_like(z[0, :, -10:, :]),
        atol=1e-5
    ), "Pair masked cols are not zeroed"

    print("OK: InputEmbedder passed shape and mask tests.")
    print("m shape:", m.shape)
    print("z shape:", z.shape)