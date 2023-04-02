This repository serves as a re-implementation of several classical Magnetic Resonance Imaging (MRI) reconstruction algorithms that were proposed before the emergence of deep learning. The aim is to provide an accelerated implementation of these algorithms using GPU, making them easy to use and compare with current deep learning-based methods. The codes are written in Python and GPU acceleration is achieved through [PyTorch](https://pytorch.org/), while [TorchKbNUFFT](https://torchkbnufft.readthedocs.io/) is used to implement Non-Uniform Fast Fourier Transform (NUFFT).

---

## Algorithms

#### GRASP [1] ✅

$$
d = \underset{d}{\arg\min} ||F\cdot C \cdot d-m||^2_2+\lambda_1||S \cdot d||_1
$$

$d$ is the reconstructed image, $F$ is the NUFFT operator, $C$ is the coil sensitivity map, $m$ is the measured k-space data, $S$ is the sparsity operator (total variation), and $\lambda_1$ is the regularization parameter.

#### XD-GRASP [2] ✅

$$
d = \underset{d}{\arg\min} ||F\cdot C \cdot d-m||^2_2+\lambda_1||S_1 \cdot d||_1+\lambda_2||S_2 \cdot R \cdot d||_1
$$

$S_1, S_2$ are the sparsity operators applied along two different dimension (for cardiac: cardiac motion & respiratory motion)(for DCE: contrast-enhancement phase & respiratory motion ), $R$ is a reordering operator that reorders the k-space data.

#### RACER-GRASP

#### L+S

## Reference

[1] Feng, Li, et al. "Golden‐angle radial sparse parallel MRI: combination of compressed sensing, parallel imaging, and golden‐angle radial sampling for fast and flexible dynamic volumetric MRI." *Magnetic resonance in medicine* 72.3 (2014): 707-717.

[2] Feng, Li, et al. "XD‐GRASP: golden‐angle radial MRI with reconstruction of extra motion‐state dimensions using compressed sensing." *Magnetic resonance in medicine* 75.2 (2016): 775-788.
