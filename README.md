# PTVAE.jl

This is the implementation for "Adapting deep generative approaches for getting synthetic data with realistic marginal distributions".

## Abstract

Synthetic data generation is of great interest in diverse applications, such as for privacy protection. Deep generative models, such as variational autoencoders (VAEs), are a popular approach for creating such synthetic datasets from original data. Despite the success of VAEs, there are limitations when it comes to the bimodal and skewed marginal distributions. These deviate from the unimodal symmetric distributions that are encouraged by the normality assumption typically used for the latent representations in VAEs. While there are extensions that assume other distributions for the latent space, this does not generally increase flexibility for data with many different distributions. Therefore, we propose a novel method, pre-transformation variational autoencoders (PTVAEs), to specifically address bimodal and skewed data, by employing pre-transformations at the level of original variables. Two types of transformations are used to bring the data close to a normal distribution by a separate parameter optimization for each variable in a dataset. We compare the performance of our method with other state-of-the-art methods for synthetic data generation. In addition to the visual comparison, we use a utility measurement for a quantitative evaluation. The results show that the PTVAE approach can outperform others in both bimodal and skewed data generation. Furthermore, the simplicity of the approach makes it usable in combination with other extensions of VAE.

## Main requirements

Julia: 1.3.0
