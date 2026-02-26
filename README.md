# Deep Generative Models — Theory to Practice

A **math-first, implementation-oriented** collection of Python notebooks and reference code for **Deep Generative Models (DGMs)**.  
This repository bridges **probabilistic foundations**, **optimization theory**, and **modern architectures** powering today’s image generators and Large Language Models (LLMs).

> _Learning generative models is like learning to cook without a recipe — divergences are taste tests, GANs are chef vs critic, diffusion sculpts order from noise, and alignment tunes the dish to human preference._

---

## 📌 Contents Overview

This repo covers **both theory and code**, progressing from first principles to state-of-the-art models.

---

## 🧠 Topics Covered

### 1️⃣ Mathematical & Probabilistic Foundations
- Probability distributions and density modeling  
- Latent variable models  
- Jensen’s Inequality and variational bounds  
- Law of Large Numbers  
- Manifold Hypothesis  
- Score-based modeling and Tweedie’s formula  

---

### 2️⃣ Divergence Measures & Optimal Transport
- $f$-divergence family  
  - KL-divergence  
  - Jensen–Shannon divergence  
  - Total Variation distance  
- Variational Divergence Minimization (VDM)  
- Optimal Transport theory  
- Wasserstein distance and geometry of distributions  
- Wasserstein GAN (WGAN) motivation and formulation  

---

### 3️⃣ Generative Adversarial Networks (GANs)
- Adversarial learning as saddle-point optimization  
- Generator–Discriminator dynamics  
- DCGAN architecture (convolutional GANs)  
- Conditional GANs  
- Training instabilities and mode collapse  
- GANs under the Manifold Hypothesis  

---

### 4️⃣ Variational Autoencoders (VAEs)
- Latent variable formulation  
- Evidence Lower Bound (ELBO) derivation  
- Reparameterization trick  
- Posterior collapse  
- Vector Quantized VAE (VQ-VAE)  
- Discrete vs continuous latent spaces  

---

### 5️⃣ Diffusion Models
- Forward noising process  
- Reverse denoising process  
- Denoising Diffusion Probabilistic Models (DDPM)  
- Denoising Diffusion Implicit Models (DDIM)  
- Score matching interpretation  
- Latent Diffusion Models (LDM)  

---

### 6️⃣ Auto-Regressive Models
- Autoregressive factorization of likelihood  
- Recurrent Neural Networks (RNNs)  
- Transformers  
- Self-attention mechanism  
- Multi-Head Attention  
- Causal masking and decoding  

---

### 7️⃣ State-Space Models (SSMs)
- Continuous and discrete state-space models  
- Linear SSM formulation  
- S4 (Structured State Space Sequence) models  
- Selective State-Space Models (Mamba)  
- Parallel scan algorithms vs convolutional kernels  

---

### 8️⃣ Model Evaluation & Practical Techniques
- Frechet Inception Distance (FID)  
- GAN inversion and latent editing  
- Unsupervised Domain Adaptation (UDA)  
- Teacher forcing for sequence models  

---

### 9️⃣ Model Alignment & Reinforcement Learning
- Alignment problem in LLMs  
- Reinforcement Learning from Human Feedback (RLHF)  
- Reward modeling  
- Bradley–Terry preference model  
- Proximal Policy Optimization (PPO)  
- Trust Region Policy Optimization (TRPO)  
- Direct Preference Optimization (DPO)  

---

### 🔟 Applications & Case Studies
- Large Language Models (ChatGPT, Claude, Gemini)  
- Image generation (DALL·E, Stable Diffusion)  
- Multimodal generative systems  

---

## 🛠️ Implementation Details
- Python + PyTorch based notebooks  
- Minimal, readable implementations  
- Emphasis on clarity over abstraction  
- Mathematical derivations paired with code  
- Reproducible experiments and visualizations  

---
