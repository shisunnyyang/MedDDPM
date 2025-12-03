# MedDDPM - Conditional Denoising Diffusion Probabilistic Models (DDPMs) for MRI Image Generation

This project implements a **Conditional Denoising Diffusion Probabilistic Model (DDPM)** to generate MRI-like images. The model supports conditioning via binary masks, enabling the generation of images with specific anatomical structures or pathologies.

## Project Structure

The Colab notebook is organized into the following sections:

1. **Environment Setup** – Installing required libraries  
2. **Model Definition** – Implementing the `Unet` architecture and `GaussianDiffusion` model  
3. **Data Preparation** – Organizing MRI images and building conditioning masks  
4. **Conditional Training** – Training DDPM with additional conditioning input  
5. **Image Generation** – Sampling new MRI-like images  
6. **Hyperparameter Tuning** – Adjusting key diffusion and training parameters  

## Setup and Installation

Install required Python packages:

```bash
pip install denoising_diffusion_pytorch
pip install kagglehub   # Optional: Only if downloading data from Kaggle
```

## Data Preparation

### 1. MRI Dataset

MRI images should be stored in:

```
/content/drive/MyDrive/ColabNotebooks/EnhanceMRIdata/brain
```

You may:

- Upload your own dataset manually  
- Download using KaggleHub (if configured)  

### 2. Conditioning Masks

Dummy binary masks are generated and stored in:

```
/content/drive/MyDrive/ColabNotebooks/EnhanceMRIdata/masks
```

These masks:

- Match the size `IMG_SIZE` (e.g., 128 × 128)  
- Are placeholders for real anatomical or pathology masks  

In real applications, replace dummy masks with segmentation maps (tumor masks, tissue labels, etc.).

## Model Architecture

The model uses a **modified U-Net** that accepts a conditioning mask as an additional input.  
Conditioning is applied by **concatenating** the binary mask with the noisy image before passing it through the U-Net.

Modifications include:

- Added `condition_channels` parameter in the U-Net constructor  
- Updated forward pass to combine `(image + condition)` inputs  

This allows the diffusion process to generate MRI images that follow the specified conditioning structure.

## Training the Model

Training uses the `Trainer` class from `denoising_diffusion_pytorch`.

Example:

```python
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

trainer = Trainer(
    diffusion,
    '/content/drive/MyDrive/ColabNotebooks/EnhanceMRIdata/brain',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 900,
    gradient_accumulate_every = 2,
    ema_decay = 0.995,
    amp = False,
    calculate_fid = True
)

trainer.train()
```

### Hyperparameter Tuning

Key trainable parameters include:

- `train_num_steps` – total optimization steps (increased to **1500** in the notebook)  
- `train_lr` – learning rate  
- `ema_decay` – smoothing factor for exponential moving average  

Adjusting these can significantly impact the final image quality.

## Generating Images

Sample new images using:

```python
sampled_images = diffusion.sample(batch_size = 16)
```

### Display the resulting images:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.ravel()

for i in range(sampled_images.shape[0]):
    image_np = sampled_images[i].permute(1, 2, 0).cpu().numpy()
    axes[i].imshow(image_np)
    axes[i].axis('off')
    axes[i].set_title(f"Generated Image {i+1}")

plt.tight_layout()
plt.show()
```

## Saving and Loading the Model

```python
torch.save(diffusion.state_dict(), 'diffusion_model.pth')
diffusion.load_state_dict(torch.load('diffusion_model.pth'))
```

## Further Enhancements

- Use real anatomical masks rather than dummy masks  
- Text conditioning using CLIP or other embeddings  
- Robust FID-based evaluation  
- Attention-augmented U-Nets  
- Training with larger datasets  

