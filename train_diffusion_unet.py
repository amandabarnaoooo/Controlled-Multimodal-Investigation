#!/usr/bin/env python3
"""
Minimal diffusion trainer for MathVerse diagrams
Author: Amanda + GPT-5 reset version
"""
import os, argparse, random, numpy as np, torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from diffusers import UNet2DModel
import matplotlib.pyplot as plt

# ---------- utilities ----------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def save_grid(tensor, path):
    """Save a [B,3,H,W] tensor grid."""
    import numpy as np
    b = tensor.size(0)
    cols = min(8,b); rows = (b+cols-1)//cols
    fig,axs=plt.subplots(rows,cols,figsize=(cols*2,rows*2))
    axs=np.atleast_1d(axs).ravel()
    for i in range(rows*cols):
        axs[i].axis("off")
        if i<b:
            img=tensor[i].permute(1,2,0).detach().cpu().numpy()
            axs[i].imshow(img)
    os.makedirs(os.path.dirname(path),exist_ok=True)
    plt.tight_layout(); plt.savefig(path,dpi=150); plt.close(fig)

def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start,beta_end,T)

def prepare_sched(T,device):
    betas=linear_beta_schedule(T).to(device)
    alphas=1-betas
    a_cum=torch.cumprod(alphas,dim=0)
    return {
        "betas":betas,
        "sqrt_a_cum":torch.sqrt(a_cum),
        "sqrt_1ma_cum":torch.sqrt(1-a_cum)
    }

def add_noise(x0,t,sched):
    noise=torch.randn_like(x0)
    sqrt_a=sched["sqrt_a_cum"][t].reshape(-1,1,1,1)
    sqrt_1ma=sched["sqrt_1ma_cum"][t].reshape(-1,1,1,1)
    return sqrt_a*x0+sqrt_1ma*noise,noise

# ---------- training ----------
def main(a):
    set_seed(42)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform=transforms.Compose([
        transforms.Resize((a.image_size,a.image_size)),
        transforms.ToTensor(),
    ])
    ds=datasets.ImageFolder(a.data_dir,transform)
    dl=DataLoader(ds,batch_size=a.batch_size,shuffle=True,num_workers=a.workers)

    unet=UNet2DModel(
        sample_size=a.image_size,in_channels=3,out_channels=3,
        layers_per_block=2,
        block_out_channels=(64,128,256,256),
        down_block_types=("DownBlock2D","DownBlock2D","DownBlock2D","AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D")
    ).to(device)

    opt=torch.optim.AdamW(unet.parameters(),lr=a.lr)
    mse=nn.MSELoss()
    sched=prepare_sched(a.timesteps,device)

    out_ckpt=Path(a.out_dir)/"checkpoints"
    out_samples=Path(a.out_dir)/"samples"
    out_ckpt.mkdir(parents=True,exist_ok=True); out_samples.mkdir(parents=True,exist_ok=True)

    global_step=0
    for epoch in range(1,a.epochs+1):
        for img,_ in dl:
            img=img.to(device)
            img=(img-0.5)*2        # normalize to [-1,1]
            t=torch.randint(0,a.timesteps,(img.size(0),),device=device)
            x_t,noise=add_noise(img,t,sched)
            pred=unet(x_t,t).sample
            loss=mse(pred,noise)
            opt.zero_grad(); loss.backward(); opt.step()

            if global_step%a.log_every==0:
                print(f"Epoch {epoch} Step {global_step} Loss {loss.item():.4f}")
            if global_step%a.sample_every==0:
                with torch.no_grad():
                    z=torch.randn(8,3,a.image_size,a.image_size,device=device)
                    imgs=(z.clamp(-1,1)+1)/2
                    save_grid(imgs,str(out_samples/f"sample_e{epoch}_s{global_step}.png"))
            if global_step%a.save_every==0 and global_step>0:
                torch.save(unet.state_dict(),out_ckpt/f"unet_e{epoch}_s{global_step}.pt")
            global_step+=1
        torch.save(unet.state_dict(),out_ckpt/f"unet_e{epoch}.pt")
    print("✅ Training complete")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data_dir",type=str,default="data/mathverse_images")
    p.add_argument("--out_dir",type=str,default="outputs/diffusion")
    p.add_argument("--image_size",type=int,default=64)
    p.add_argument("--batch_size",type=int,default=64)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--timesteps",type=int,default=400)
    p.add_argument("--log_every",type=int,default=50)
    p.add_argument("--sample_every",type=int,default=500)
    p.add_argument("--save_every",type=int,default=1000)
    a=p.parse_args(); main(a)
