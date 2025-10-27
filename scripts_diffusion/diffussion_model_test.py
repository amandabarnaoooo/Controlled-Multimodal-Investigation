# Mathverse Diffusion Model Fine-Tuning and Image Generation Pipeline

"""

This script fine-tunes a diffusion-based U-Net model on the MathVerse “Vision Only” dataset. It Processes, adds noise via fwd diffussion, trains a model, and through reverse diffusion sampling generates new MathVerse-style images into a zip file.

"""
"""

Fine tuning a stable diffusion model using the mathverse mini dataset to generate realistic diagrams and images


"""

#Importing the required libraries and python functionalities

import os
import math
import random
from pathlib import Path
from PIL import Image
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from datasets import load_dataset
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, UNet2DModel#, DPMSolverMultistepScheduler
#from peft import LoraConfig, get_peft_model, PeftModel 
from accelerate import Accelerator
import matplotlib.pyplot as plt
import io
import zipfile
from datetime import datetime
import os

#from tqdm.auto import tqdm
"""

Parameter set up and device usage

"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_size = 512
num_timesteps = 1000
time_steps_reverse = 200

in_ch = 3
out_ch = 3

layer_per_block = 5

block_out_ch = (64, 128, 256, 512) 
down_blocks = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D")
up_blocks = ("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")


batch_size = 4


learning_rate = 1e-4
epochs = 10


num_images_to_create = 100
image_dir = "/data/ecau171/mathverse_proj/"

"""

Create a class to convert images to RGB 

"""


class convertRGB:
    def __call__(self, img):
        return img.convert("RGB")



"""

Develop the vision encoder to process images in later tasks


"""
# Define a small test U-Net for speed
unet = UNet2DModel(
    sample_size=img_size,          # image size (must match your transform)
    in_channels=in_ch,           # RGB
    out_channels=out_ch,          # predict noise for RGB channels
    layers_per_block=layer_per_block,      # keep small for testing
    block_out_channels=block_out_ch, # small number of channels
    down_block_types=down_blocks, # minimal downsampling
    up_block_types= up_blocks
)

unet = unet.to(device)

"""

Mathverse Dataset of vision only 

"""

ds_dict = load_dataset("AI4Math/MathVerse", "testmini")
ds = ds_dict["testmini"]

ds_vision = ds.filter(lambda x: x["problem_version"] == "Vision Only")
ds_images = ds_vision.remove_columns([col for col in ds_vision.column_names if col != "image"])



"""

Understanding the dataset

"""

modes = [img.mode for img in ds_images["image"]]
sizes = [img.size for img in ds_images["image"]]

unique_modes = set(modes)
unique_sizes = set(sizes)

print("Number of unique image modes:", len(unique_modes))
print("Number of unique image sizes:", len(unique_sizes))


"""

Transformation pipeline

"""

transform = transforms.Compose([
    convertRGB(),
    transforms.Resize((512, 512)),   # resize to fixed resolution
    transforms.ToTensor(),           # convert to tensor
    transforms.Normalize([0.5]*3, [0.5]*3)
])



"""

Transforming each image of the dataset

"""

def transform_examples(example):
    imgs = example["image"]

    if isinstance(imgs, list):
        example["image"] = [transform(img) for img in imgs]
    else:
        example["image"] = transform(imgs)

    return example

ds_processed = ds_images.with_transform(transform_examples)

# Single example of dataset created
"""
sample = ds_processed[0]["image"]
print(type(sample), sample.shape) # output: is <class 'torch.Tensor'> torch.Size([3, 64, 64])

plt.imshow(sample.permute(1, 2, 0))
plt.show()

"""


#Data Loader


train_loader = DataLoader(ds_processed, batch_size=batch_size, shuffle=True)

batch = next(iter(train_loader))
#print(batch["image"].shape)  # Expected: [8, 3, 256, 256]



"""

Creating required attributes for noise contribution

"""


def linear_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.01
    return torch.linspace(beta_start, beta_end, timesteps)

betas = linear_beta_schedule(num_timesteps)
alphas = 1. - betas
alpha_cum_prod = torch.cumprod(alphas, dim=0)


"""

XXXX

"""

def get_index_from_list(alpha_cumprod, t, x_shape):
    # a: 1D tensor length T
    # t: [B] long tensor of timesteps
    a = alpha_cumprod.to(t.device)
    out = a[t]  # shape [B]
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))



"""

Forward Diffusion - adding noise to the images

"""

def fwd_diff_sample(x_0, t, device="cuda"):
    noise = torch.randn_like(x_0)
    sqrt_alpha_cumprod_t = get_index_from_list(torch.sqrt(alpha_cum_prod), t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(torch.sqrt(1 - alpha_cum_prod), t, x_0.shape)
    return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise


"""
#Plotting the cumulative product of alphas across timestepss... i.e. how observable the image is at that point

plt.plot(alpha_cum_prod)
plt.title("Cumulative Product of Alphas (ᾱ_t)")
plt.xlabel("timestep")
plt.ylabel("ᾱ_t")
plt.show()

"""

"""
# Verifying using one sample the dataset has noise added to images

img_pil = ds_images[0]["image"]
x0 = transform(img_pil).unsqueeze(0).to(device)  # shape: (1, 3, 64, 64)

timesteps = [0, 100, 199]
noisy_images = []

for t in timesteps:
    t_tensor = torch.tensor([t], dtype=torch.long, device=device)
    xt, _ = fwd_diff_sample(x0, t_tensor, device=device)
    noisy_images.append(xt)


def show_images(noisy_images, timesteps):
    fig, axs = plt.subplots(1, len(timesteps), figsize=(15, 3))
    for i, img in enumerate(noisy_images):
        img_show = img[0].detach().cpu().permute(1, 2, 0)
        img_show = (img_show + 1) / 2  # unnormalize from [-1,1] → [0,1]
        axs[i].imshow(img_show)
        axs[i].set_title(f"t={timesteps[i]}")
        axs[i].axis("off")
    plt.show()

show_images(noisy_images, timesteps)
"""


"""

Defining the loss function


"""

def diff_loss(model, x_0, t, device="cuda"):
    
    x_t, noise = fwd_diff_sample(x_0, t, device = device)
    predicted_noise = model(x_t, t).sample
    
    loss = torch.nn.functional.mse_loss(predicted_noise, noise)
    return loss


"""
#Verify the loss is working using a small subset

#Get a small batch of real data
sample_batch = next(iter(train_loader))["image"].to(device)

# Random timesteps for each sample in batch
t = torch.randint(0, num_timesteps, (sample_batch.shape[0],), device=device).long()

# Compute loss
loss = diff_loss(unet, sample_batch, t, device=device)

print(f"✅ Diffusion loss computed successfully: {loss.item():.6f}")

"""



"""

Completing optimisation on training set

"""
optimiser = Adam(unet.parameters(), lr=learning_rate)


for epoch in range(epochs):
    for batch in train_loader:

        x_0 = batch["image"].to(device)

        t = torch.randint(0,num_timesteps, (x_0.shape[0], ), device=device).long()
        optimiser.zero_grad()
        loss = diff_loss(unet, x_0, t, device=device)

        loss.backward()
        optimiser.step()
    print(f"Epoch {epoch+1}/{epochs} ✅ | Loss: {loss.item():.6f}")



"""

Defining the reverse diffusion function


"""

@torch.no_grad()
def sample_timestep(x_t, t, model):
    """
    Single reverse diffusion step.
    x_t: current noisy image
    t: current timestep
    model: trained UNet
    """
    betas_t = get_index_from_list(betas, t, x_t.shape)
    sqrt_one_minus_alpha_cumprod_t = get_index_from_list(
        torch.sqrt(1 - alpha_cum_prod), t, x_t.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(
        torch.sqrt(1.0 / alphas), t, x_t.shape
    )

    # Predict the noise using the model
    predicted_noise = model(x_t, t).sample

    # Compute the mean of the posterior
    mean = sqrt_recip_alphas_t * (
        x_t - betas_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t
    )

    # Add noise except for t=0
    if t[0] > 0:
        noise = torch.randn_like(x_t)
    else:
        noise = torch.zeros_like(x_t)

    posterior_var_t = get_index_from_list(betas, t, x_t.shape)
    x_prev = mean + torch.sqrt(posterior_var_t) * noise
    return x_prev



"""

Generating images from pure noise

"""

@torch.no_grad()
def sample_image(model, num_timesteps, img_size, device="cuda"):
    # Start from pure Gaussian noise
    img = torch.randn((1, 3, img_size, img_size), device=device)
    
    # Reverse diffusion: from t=T → t=0
    for i in reversed(range(1, num_timesteps)):
        t = torch.tensor([i], device=device, dtype=torch.long)
        img = sample_timestep(img, t, model)

    # Clamp pixel values to valid range
    img = torch.clamp(img, -1.0, 1.0)
    return img

# Generate one image
sampled_img = sample_image(unet, num_timesteps=num_timesteps, img_size=img_size, device=device)
print("✅ Sampling complete:", sampled_img.shape)




"""

Generate and save new images created to a zip file i can access later

"""

def generate_and_zip_images(
    model=unet,
    num_images=10,
    img_size=512,
    num_timesteps=200,
    device="cuda",
    output_dir="/data/ecau171/mathverse_proj/",
):
    """
    Generates diffusion-based images and saves them all into a timestamped ZIP file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(output_dir, f"mathverse_generated_{timestamp}.zip")

    print(f"🚀 Generating {num_images} images ({img_size}x{img_size}) → {zip_path}")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for i in range(num_images):
            img = sample_image(model, num_timesteps=num_timesteps, img_size=img_size, device=device)

            # Convert tensor to PIL image
            img_show = img.squeeze(0).detach().cpu()
            img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min())
            img_np = (img_show.permute(1, 2, 0).numpy() * 255).astype("uint8")
            img_pil = Image.fromarray(img_np)

            # Save in memory and write to zip
            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            zipf.writestr(f"generated_{i+1}.png", img_bytes.read())

            if (i + 1) % 5 == 0 or i == num_images - 1:
                print(f"🖼️  Generated {i+1}/{num_images}")

    print(f"✅ All {num_images} images saved to: {zip_path}\n")
    return zip_path

zip_path = generate_and_zip_images(
    model=unet,
    num_images=num_images_to_create,
    img_size=img_size,
    num_timesteps=num_timesteps,
    device=device,
    output_dir = image_dir
)
print(f"Images archived at: {zip_path}")
