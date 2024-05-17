# mnist-gan-pytorch
A GAN Model to generate handwritten MNIST Images

## Features
⚡Image Generation  
⚡Generative Adverserial Network (GAN)  
⚡Fully Connected Neural Network Layers  
⚡MNIST  
⚡PyTorch 

## Table of Contents
- [Introduction](#introduction) 
- [Objective](#objective)
- [Dataset](#dataset)
- [How To Use](#how-to-use)
- [Outputs](#outputs)

## Introduction
### Introductory Paragraph about GANs

Generative Adversarial Networks (GANs) are a class of artificial intelligence algorithms used in unsupervised machine learning. GANs consist of two neural networks, the generator and the discriminator, that are trained simultaneously through adversarial processes. The generator creates fake data samples that mimic real data, while the discriminator evaluates the authenticity of these samples. This interplay pushes both networks to improve, resulting in a generator capable of producing highly realistic data.

### Basic Math Behind GANs

A typical GAN network is shown below...<br><br>

<img src="https://github.com/dineshg20897/mnist-gan-pytorch/blob/main/assets/GAN.png?raw=true" width="800"><br><br>

The fundamental concept behind GANs involves a minimax game between the generator \( G \) and the discriminator \( D \). The generator \( G \) maps a noise vector \( z \) from a latent space to the data space to produce synthetic data samples \( G(z) \). The discriminator \( D \), on the other hand, receives a data sample and outputs a probability \( D(x) \) that the sample is real (from the true data distribution) or fake (from the generator). The objective function of a GAN is defined as:

```math
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
```

In this equation, $\( p_{\text{data}}(x) \)$ represents the real data distribution, and $\( p_z(z) \)$ is the distribution of the noise vector \( z \). The discriminator aims to maximize the probability of correctly identifying real and fake samples, while the generator strives to minimize the probability of the discriminator detecting its fakes. Through iterative training, both networks reach an equilibrium where the generator produces highly realistic data that the discriminator can no longer distinguish from real data.


## Objective

Our objective in this project is to develop a GAN and train it using the MNIST dataset, to enable the network to generate _fake_ MNIST digit images that closely resemble the images from the actual MNIST dataset.


## Dataset

The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The Images were normalized to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale level. It contains 60,000 training images and 10,000 testing images.


## How to use

1. Ensure the below-listed packages are installed.
    - `NumPy`
    - `matplotlib`
    - `torch`
    - `torchvision`
2. Download `GAN_using_PyTorch.ipynb` jupyter notebook from this repository.
3. Execute the notebook from start to finish in one go. If a GPU is available (recommended), it'll use it automatically; otherwise, it'll fall back to the CPU. 
4. Experiment with different hyperparameters – longer training would yield better results.


## Outputs

Here you can see the GAN Model getting progressively better at generating _fake_ images for every 10 epochs

<img src="https://github.com/dineshg20897/mnist-gan-pytorch/blob/main/assets/Output.png?raw=true" width="800">
