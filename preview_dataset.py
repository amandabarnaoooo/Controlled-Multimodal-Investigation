import argparse, os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.utils.seed import set_seed
from src.data.mathverse_images import load_mathverse_local

def save_image_grid(tensor_batch: torch.Tensor, path: str):
    """
    tensor_batch: [B, 3, H, W] in [0,1]
    Saves a grid PNG without tripping on axes shapes.
    """
    import numpy as np
    b = tensor_batch.size(0)
    cols = min(8, b)
    rows = (b + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    # Normalize axes to a flat 1-D array of Axes
    axes = np.atleast_1d(axes).ravel()

    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i < b:
            img = tensor_batch[i].permute(1, 2, 0).detach().cpu().numpy()
            ax.imshow(img)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)

def main(args):
    set_seed(42)
    ds = load_mathverse_local(args.data_dir, args.image_size)
    if len(ds) == 0:
        raise RuntimeError(f"No images found in {args.data_dir}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    batch = next(iter(loader))
    imgs = batch["image"]  # [B,3,H,W], float32 [0,1]

    print("Batch tensor:", imgs.shape, imgs.dtype,
          "min:", float(imgs.min()), "max:", float(imgs.max()))
    out_png = os.path.join(args.out_dir, f"preview_{args.image_size}.png")
    save_image_grid(imgs, out_png)
    print(f"Saved preview grid → {out_png}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/mathverse_images")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out_dir", type=str, default="outputs/preview")
    args = p.parse_args()
    main(args)
