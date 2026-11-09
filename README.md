# Frequency-Guided Diffusion Purification

A PyTorch framework for neural network classifier adversarial resilience. Purges adversarial perturbations using a DDIM based diffusion pipeline based on frequency space guidance and fast Fourier transform truncated reconstruction.


## Overview

Adversarial perturbations typically concentrate in high-frequency domains, while global semantic structure resides in low-frequency bands. This project decouples adversarial defense from model training by purifying inputs before classification through four steps:

1. **Noise and x0 Prediction.** Initially, the potentially adversarial data point x is passed through a diffusion neural network, which predicts which pixels do not follow the distribution it was trained on. Utilizing a DDIM [38] sampler, a prediction of the image after many purification steps, x0 is returned at no extra compute cost.

2. **Biasing for Long Run Consistency** To allow many diffusion iterations without accuracy 
loss, x is quantized to get ˆx. xguide is computed by taking low frequency, structural data from ˆx and high frequency, noisy data from x0. The noise prediction is biased towards xguide using an analytic simplification of the guidance scheme devised by Wangetal to avoid the requirement of a conditionally trained diffusion model.

3. Denoising x is de-noised a single step using the biased noise prediction to get xbiased, which is passed onto the next diffusion iteration.

4. Majority VotingFinally,thecollectionofxguide values computed during intermediate denoising steps is passed through the classifier back-end, and a majority vote is performed on the resultant logits. Later xguide values are weighed more heavily as the signal-noise ratio of the scheduler converges.

## Repository Structure

* `diffusion_defence.py`: Core `nn.Module` wrappers (`DiffPure`, `DiffPure_x0v`, and the full `DiffPure_guided` pipeline).
* `guided_scheduler.py`: Custom `DDIMScheduler_guided` for injecting analytic conditioning gradients.
* `frequency_band_blending.py`: 2D FFT decomposition and circular mask reconstruction.
* `models.py`: Pretrained model and diffusion loaders for CIFAR-10, MNIST, and Fashion-MNIST.
* `torch_common_clean.py`: Dataset loaders, norm projections ($L_2, L_\infty$), and BPDA autograd functions.

## Installation

```bash
pip install torch torchvision diffusers transformers huggingface_hub torchattacks kagglehub numpy
```

## Quick Start

```python
import torch
from models import get_CIFAR10_classifier, get_CIAFAR10_diffusion_model, cnn_preproc
from diffusion_defence import DiffPure_guided

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained CIFAR-10 diffusion UNet and ResNet56 classifier
ddpm = get_CIAFAR10_diffusion_model(device=device)
classifier = get_CIFAR10_classifier(name="cifar10_resnet56", device=device)

# Initialize defense pipeline
defence = DiffPure_guided(
    ddpm=ddpm,
    classifier=classifier,
    steps=100,
    noise_add=100,
    classifier_norm_function=cnn_preproc,
    guidance_scale=2.0,
    _quant_level=6,
    _fft_r=0.1
).to(device)

# Purify and classify inputs (scaled in [0, 1])
x = torch.rand(4, 3, 32, 32).to(device)
logits = defence(x)
```

## Key Results (CIFAR-10)

Evaluated under full white-box settings (including BPDA and EOT) and black-box attacks:
* **FGSM / PGD**: >90% and ≥85% robust accuracy.
* **Square Attack (Black-Box)**: ≥93% robust accuracy.
* **Adaptive C&W + BPDA + EOT**: 68% compensated robust accuracy ($L_2 < 0.75$).
* **High-Frequency Adaptive PGD**: 79% robust accuracy.

## Report
Extensive report outlining the algorithm and evaulating against similar methods included (**report.pdf**)