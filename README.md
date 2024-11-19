# Txt2Img Generation: Understanding Generative AI Models with Text Condition

This repository contains the code and resources for the poster presentation titled **"Txt2Img Generation: Understanding Generative AI Models with Text Condition"**, presented by undergraduate researchers at **2024 University of Seoul (Univ. of Seoul)**.

## Overview

Generative AI has opened new doors in creative applications such as text-to-image generation. This project explores and implements various generative models, focusing on text-conditional image generation techniques, including:
- Variational Autoencoders (VAEs)
- Generative Adversarial Networks (GANs)
- Diffusion Models (DDPM, Latent Diffusion Models)

Through rigorous experimentation, we evaluated the effectiveness of these models using key evaluation metrics such as FID (Fréchet Inception Distance) and IS (Inception Score). We also investigated challenges in training these models, particularly in high-resolution image generation and conditioning mechanisms.

---

## Features

- **Model Implementations**:
  - Python-based implementations of VAE, GAN, and Diffusion Models.
  - Integration of text conditions using embeddings and cross-attention mechanisms.

- **Evaluation Metrics**:
  - Code to compute FID and IS scores to assess model performance.
  - Experiment results to compare conditional and non-conditional generative models.

- **Challenges Addressed**:
  - Hyperparameter tuning for generative models.
  - Exploring variance schedules in diffusion models.
  - Generating high-resolution, text-aligned images.

---

## Repository Structure

```plaintext
├── models/              # Implementations of VAE, GAN, DDPM, and LDM
├── datasets/            # Scripts for loading and preprocessing datasets (e.g., MNIST, CIFAR-10)
├── evaluations/         # Metrics for evaluating model performance (FID, IS)
├── experiments/         # Experiment configurations and results
├── utils/               # Helper functions for training and visualization
├── notebooks/           # Jupyter notebooks for experimentation and visualization
└── README.md            # This file
