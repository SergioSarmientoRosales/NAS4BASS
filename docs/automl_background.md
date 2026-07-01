# AutoML Background

This page explains the main concepts in beginner-friendly language.

## Neural Architecture Search

Neural Architecture Search, or NAS, is a method for automatically exploring neural network designs instead of manually designing every architecture.

In this repository, a candidate architecture is represented as a code. The search algorithm changes that code, decodes it into a neural network design, evaluates it, and keeps promising candidates.

## BASS

BASS is a search space designed for super-resolution image restoration, where candidate architectures are represented through modular design choices.

Instead of choosing one fixed neural network manually, BASS lets the search process combine operations, channels, kernels, and repeated blocks.

## Super-Resolution Image Restoration

Super-resolution image restoration, or SRIR, tries to reconstruct a higher-resolution image from a lower-resolution input.

This is useful when the image has limited resolution or has been degraded by acquisition, compression, downsampling, blur, or noise.

## Zero-Cost Predictors

Zero-cost predictors estimate architecture quality without fully training every model, reducing computational cost.

They are not truly free, but they are much cheaper than training thousands of candidate networks. They usually inspect gradients, weights, activations, parameter counts, or other signals from an untrained or lightly initialized network.

## Why Efficient Proxies Matter

Training every candidate architecture is expensive; therefore, efficient proxies can help prioritize promising architectures.

In practice, this repository supports two broad evaluation modes:

- Model-based evaluation: trained surrogate models predict architecture quality.
- Zero-cost evaluation: metrics such as SynFlow-like scores estimate architecture potential without full training.

Both modes are tools for reducing search cost. They should be validated against fully trained results before making strong final claims.
