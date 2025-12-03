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
You may change it to your own directory.

You may:

- Upload your own dataset manually  
- Download using KaggleHub (if configured)

### 1.1 Configuration For Importing Dataset Directly From Kaggle
 
1. Visit www.kaggle.com. Go to your profile and click on Settings.
1. Scroll to API section and Click Expire API Token to remove previous tokens.
1. Click on Create New API Token - It will download kaggle.json file on your machine.
1. Go to your Google Colab project file and run the following commands:
 
```
!pip install -q kaggle
from google.colab import files
files.upload()
# Choose the kaggle.json file that you downloaded
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
```
### 2. Conditioning Masks

Dummy binary masks are generated and stored in:

```
/content/drive/MyDrive/ColabNotebooks/EnhanceMRIdata/masks
```

You may change it to your own directory.

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

## Citation
```bibtex
@misc{ddpmgithub,
    key = {Denoising Diffusion Probabilistic Model, in Pytorch},
    url = {https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main},
    author ={lucidrains}
}

@misc{kazerouni2023diffusionmodelsmedicalimage,
      title={Diffusion Models for Medical Image Analysis: A Comprehensive Survey}, 
      author={Amirhossein Kazerouni and Ehsan Khodapanah Aghdam and Moein Heidari and Reza Azad and Mohsen Fayyaz and Ilker Hacihaliloglu and Dorit Merhof},
      year={2023},
      eprint={2211.07804},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2211.07804}, 
}

@misc{vivekananthan2024comparativeanalysisgenerativemodels,
      title={Comparative Analysis of Generative Models: Enhancing Image Synthesis with VAEs, GANs, and Stable Diffusion}, 
      author={Sanchayan Vivekananthan},
      year={2024},
      eprint={2408.08751},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.08751}, 
}

@misc{ho2020denoisingdiffusionprobabilisticmodels,
      title={Denoising Diffusion Probabilistic Models}, 
      author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
      year={2020},
      eprint={2006.11239},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2006.11239}, 
}

@misc{xiao2022tacklinggenerativelearningtrilemma,
      title={Tackling the Generative Learning Trilemma with Denoising Diffusion GANs}, 
      author={Zhisheng Xiao and Karsten Kreis and Arash Vahdat},
      year={2022},
      eprint={2112.07804},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2112.07804}, 
}

@misc{hu2022unsuperviseddenoisingretinaloct,
      title={Unsupervised Denoising of Retinal OCT with Diffusion Probabilistic Model}, 
      author={Dewei Hu and Yuankai K. Tao and Ipek Oguz},
      year={2022},
      eprint={2201.11760},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2201.11760}, 
}

@misc{yu2025petimagedenoisingtextguided,
      title={PET Image Denoising via Text-Guided Diffusion: Integrating Anatomical Priors through Text Prompts}, 
      author={Boxiao Yu and Savas Ozdemir and Jiong Wu and Yizhou Chen and Ruogu Fang and Kuangyu Shi and Kuang Gong},
      year={2025},
      eprint={2502.21260},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2502.21260}, 
}

@misc{brainmriimagesdataset,
    title ={Brain MRI Images},
    url = {https://www.kaggle.com/datasets/ashfakyeafi/brain-mri-images/data},
    author = {Ashfak Yeaki},
    year = {2023}
}

@misc{brainmrisegmentation,
    title = {Brain MRI segmentation},
    author ={Mateusz Buda},
    year = {2018},
    url = {https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation}
}

@misc{chung2022scorebaseddiffusionmodelsaccelerated,
      title={Score-based diffusion models for accelerated MRI}, 
      author={Hyungjin Chung and Jong Chul Ye},
      year={2022},
      eprint={2110.05243},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2110.05243}, 
}

@misc{fastmri,
    url = {https://fastmri.med.nyu.edu/},
    title = {Welcome to the fastMRI Dataset},
    author = {NYU Langone Health}
}

@misc{brats2020,
    url = {https://www.kaggle.com/datasets/awsaf49/brats2020-training-data},
    title = {Brain Tumor Segmentation (BraTS2020)},
    author = {AWSAF}
}

@misc{stanformri,
    url = {https://aimi.stanford.edu/shared-datasets},
    author = {Center for Artificial Intelligence in Medicine and Imaging},
    year = {2025}
}

@misc{sherbrooke3shell,
    url = {https://digital.lib.washington.edu/researchworks/items/e4d95d19-2d3b-49bd-ae0b-27c58080116c},
    author = {Rokem, Ariel},
    year = {2017-03-31}
}

@article{thereticaldm,
   title={Generative diffusion models: A survey of current theoretical developments},
   volume={608},
   ISSN={0925-2312},
   url={http://dx.doi.org/10.1016/j.neucom.2024.128373},
   DOI={10.1016/j.neucom.2024.128373},
   journal={Neurocomputing},
   publisher={Elsevier BV},
   author={Yeğin, Melike Nur and Amasyalı, Mehmet Fatih},
   year={2024},
   month=dec, pages={128373} }

@misc{jiang2025fastddpmfastdenoisingdiffusion,
      title={Fast-DDPM: Fast Denoising Diffusion Probabilistic Models for Medical Image-to-Image Generation}, 
      author={Hongxu Jiang and Muhammad Imran and Teng Zhang and Yuyin Zhou and Muxuan Liang and Kuang Gong and Wei Shao},
      year={2025},
      eprint={2405.14802},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2405.14802}, 
}

@INPROCEEDINGS{psnrssim,
  author={Horé, Alain and Ziou, Djemel},
  booktitle={2010 20th International Conference on Pattern Recognition}, 
  title={Image Quality Metrics: PSNR vs. SSIM}, 
  year={2010},
  volume={},
  number={},
  pages={2366-2369},
  keywords={PSNR;Degradation;Image quality;Additives;Transform coding;Sensitivity;Image coding;PSNR;SSIM;image quality metrics},
  doi={10.1109/ICPR.2010.579}}

@misc{ho2020denoisingdiffusionprobabilisticmodels,
      title={Denoising Diffusion Probabilistic Models}, 
      author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
      year={2020},
      eprint={2006.11239},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2006.11239}, 
}

@incollection{mridenoisefiltering,
author = {Hanafy M. Ali},
title = {MRI Medical Image Denoising by Fundamental Filters},
booktitle = {High-Resolution Neuroimaging - Basic Physical Principles and Clinical Applications},
publisher = {IntechOpen},
address = {London},
year = {2018},
editor = {Ahmet Mesrur Halefoğlu},
chapter = {7},
doi = {10.5772/intechopen.72427},
url = {https://doi.org/10.5772/intechopen.72427}
}

@misc{rethinkfid,
      title={Rethinking FID: Towards a Better Evaluation Metric for Image Generation}, 
      author={Sadeep Jayasumana and Srikumar Ramalingam and Andreas Veit and Daniel Glasner and Ayan Chakrabarti and Sanjiv Kumar},
      year={2024},
      eprint={2401.09603},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2401.09603}, 
}

@inproceedings{evaluationsuitabilityinceptionscore,
author = {Chan, Derrick Adrian and Sithungu, Siphesihle Philezwini},
title = {Evaluating the Suitability of Inception Score and Fr\'{e}chet Inception Distance as Metrics for Quality and Diversity in Image Generation},
year = {2025},
isbn = {9798400717437},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3708778.3708790},
doi = {10.1145/3708778.3708790},
abstract = {Variational Autoencoders (VAEs) have gained popularity as one of the main approaches for generating diverse and high-quality synthetic images. This study examines the suitability of evaluation metrics, specifically Inception Score and Fr\'{e}chet Inception Distance (FID), for assessing these images. Particularly, the study focuses on the generation of synthetic images based on the MNIST handwritten digits dataset. Through the use of VAE-generated MNIST image samples, the study analyses the abovementioned metrics alongside alternative methods that can be used to assess image quality and diversity. The findings made from the study reveal the strengths and limitations of each metric in evaluating image quality and diversity. This paper underscores the need for tailored metrics to enhance the evaluation of generative models, while specifically using the performance of a VAE as the domain of investigation.},
booktitle = {Proceedings of the 2024 7th International Conference on Computational Intelligence and Intelligent Systems},
pages = {79–85},
numpages = {7},
keywords = {Fr\'{e}chet Inception Distance, Image Generation, Inception Score, MNIST, Variational Autoencoders},
location = {
},
series = {CIIS '24}
}

@article{deeplearningbasedsysn,
title = {Deep learning based synthesis of MRI, CT and PET: Review and analysis},
journal = {Medical Image Analysis},
volume = {92},
pages = {103046},
year = {2024},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2023.103046},
url = {https://www.sciencedirect.com/science/article/pii/S1361841523003067},
author = {Sanuwani Dayarathna and Kh Tohidul Islam and Sergio Uribe and Guang Yang and Munawar Hayat and Zhaolin Chen},
keywords = {Medical image synthesis, Generative deep-learning models, Pseudo-CT, Synthetic MR, Synthetic PET},
abstract = {Medical image synthesis represents a critical area of research in clinical decision-making, aiming to overcome the challenges associated with acquiring multiple image modalities for an accurate clinical workflow. This approach proves beneficial in estimating an image of a desired modality from a given source modality among the most common medical imaging contrasts, such as Computed Tomography (CT), Magnetic Resonance Imaging (MRI), and Positron Emission Tomography (PET). However, translating between two image modalities presents difficulties due to the complex and non-linear domain mappings. Deep learning-based generative modelling has exhibited superior performance in synthetic image contrast applications compared to conventional image synthesis methods. This survey comprehensively reviews deep learning-based medical imaging translation from 2018 to 2023 on pseudo-CT, synthetic MR, and synthetic PET. We provide an overview of synthetic contrasts in medical imaging and the most frequently employed deep learning networks for medical image synthesis. Additionally, we conduct a detailed analysis of each synthesis method, focusing on their diverse model designs based on input domains and network architectures. We also analyse novel network architectures, ranging from conventional CNNs to the recent Transformer and Diffusion models. This analysis includes comparing loss functions, available datasets and anatomical regions, and image quality assessments and performance in other downstream tasks. Finally, we discuss the challenges and identify solutions within the literature, suggesting possible future directions. We hope that the insights offered in this survey paper will serve as a valuable roadmap for researchers in the field of medical image synthesis.}
}

@article{fidequation,
title = {The Fréchet distance between multivariate normal distributions},
journal = {Journal of Multivariate Analysis},
volume = {12},
number = {3},
pages = {450-455},
year = {1982},
issn = {0047-259X},
doi = {https://doi.org/10.1016/0047-259X(82)90077-X},
url = {https://www.sciencedirect.com/science/article/pii/0047259X8290077X},
author = {D.C Dowson and B.V Landau},
keywords = {Fréchet distance, multivariate normal distributions, covariance matrices},
abstract = {The Fréchet distance between two multivariate normal distributions having means μX, μY and covariance matrices ΣX, ΣY is shown to be given by d2 = |μX − μY|2 + tr(ΣX + ΣY − 2(ΣXΣY)12). The quantity d0 given by d02 = tr(ΣX + ΣY − 2(ΣXΣY)12) is a natural metric on the space of real covariance matrices of given order.}
}
```
