This repository serves as a re-implementation of several classical Magnetic Resonance Imaging (MRI) reconstruction algorithms that were proposed before the emergence of deep learning. The aim is to provide an accelerated implementation of these algorithms using GPU, making them easy to use and compare with current deep learning-based methods. The codes are written in Python and GPU acceleration is achieved through [PyTorch](https://pytorch.org/), while [TorchKbNUFFT](https://torchkbnufft.readthedocs.io/) is used to implement Non-Uniform Fast Fourier Transform (NUFFT).

---

## Algorithms

#### GRASP [1] ✅

$$
x = \underset{x}{\arg\min} ||F\cdot C \cdot x-y_c||^2_2+\lambda_1||S \cdot x||_1
$$

$x$ is the reconstructed image, $F$ is the NUFFT operator, $C$ is the coil sensitivity map, $y_c$ is the measured k-space data on each coil, $S$ is the sparsity operator (total variation), and $\lambda_1$ is the regularization parameter.

#### XD-GRASP [2] ✅

$$
x = \underset{x}{\arg\min} ||F\cdot C \cdot x-y_c||^2_2+\lambda_1||S_1 \cdot x||_1+\lambda_2||S_2 \cdot R \cdot x||_1
$$

$S_1, S_2$ are the sparsity operators applied along two different dimension (for cardiac: cardiac motion & respiratory motion)(for DCE liver: contrast-enhancement phase & respiratory motion ), $R$ is a reordering operator that reorders the k-space data.

#### RACER-GRASP [3]

$$
x = \underset{x}{\arg\min} ||\frac{W}{R}(F\cdot C \cdot x-y_c)||^2_2+\lambda_1||S \cdot x||_1
$$

$W$ represents motion weights for different respiratory phases, $R$ represents the streak-ratio weights to adjust the contribution of each coil in the data consistency term.

#### L+S

## Reference

[1] Feng, Li, et al. "Golden‐angle radial sparse parallel MRI: combination of compressed sensing, parallel imaging, and golden‐angle radial sampling for fast and flexible dynamic volumetric MRI." *Magnetic resonance in medicine* 72.3 (2014): 707-717.

[2] Feng, Li, et al. "XD‐GRASP: golden‐angle radial MRI with reconstruction of extra motion‐state dimensions using compressed sensing." *Magnetic resonance in medicine* 75.2 (2016): 775-788.

[3] Feng, Li, et al. "RACER‐GRASP: respiratory‐weighted, aortic contrast enhancement‐guided and coil‐unstreaking golden‐angle radial sparse MRI." *Magnetic resonance in medicine* 80.1 (2018): 77-89.
