#!/usr/bin/env python3
"""
Generate new diagrams from a trained diffusion model
"""
import os, argparse, torch, sys
from pathlib import Path
import matplotlib.pyplot as plt
from diffusers import UNet2DModel

# --- import helpers from the trainer ---
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.train_diffusion_unet import prepare_sched, save_grid

# --- sampling ---
@torch.no_grad()
def p_sample_loop(model, sched, shape, device, T, out_dir, tag="manual"):
    betas = sched["betas"]
    alphas = 1 - betas
    a_cum = torch.cumprod(alphas, dim=0)
    a_prev = torch.cat([torch.tensor([1.0], device=device), a_cum[:-1]], dim=0)
    img = torch.randn(shape, device=device)
    for t in reversed(range(T)):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps = model(img, t_tensor).sample
        a_t, a_cum_t, b_t = alphas[t], a_cum[t], betas[t]
        mean = (img - ((1 - a_t) / torch.sqrt(1 - a_cum_t)) * eps) / torch.sqrt(a_t)
        if t > 0:
            img = mean + torch.sqrt(b_t) * torch.randn_like(img)
        else:
            img = mean
    img = (img.clamp(-1, 1) + 1) / 2.0
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"samples_{tag}.png")
    save_grid(img, path)
    return path

def main(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = UNet2DModel(
        sample_size=a.image_size, in_channels=3, out_channels=3,
        layers_per_block=2, block_out_channels=(64,128,256,256),
        down_block_types=("DownBlock2D","DownBlock2D","DownBlock2D","AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D")
    ).to(device)

    print("Loading checkpoint:", a.ckpt)
    state = torch.load(a.ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    sched = prepare_sched(a.timesteps, device)
    out_path = p_sample_loop(
        model, sched,
        shape=(a.n_samples, 3, a.image_size, a.image_size),
        device=device, T=a.timesteps,
        out_dir=a.out_dir, tag="gen"
    )
    print("✅ Saved generated diagrams →", out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to trained .pt file")
    p.add_argument("--out_dir", default="outputs/generated")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--timesteps", type=int, default=400)
    p.add_argument("--n_samples", type=int, default=16)
    a = p.parse_args()
    main(a)
