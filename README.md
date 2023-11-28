This repository serves as a re-implementation of several classical Magnetic Resonance Imaging (MRI) reconstruction algorithms that were proposed before the emergence of deep learning. The aim is to provide an accelerated implementation of these algorithms using GPU, making them easy to use and compare with current deep learning-based methods. The codes are written in Python and GPU acceleration is achieved through [PyTorch](https://pytorch.org/), while [TorchKbNUFFT](https://torchkbnufft.readthedocs.io/) is used to implement Non-Uniform Fast Fourier Transform (NUFFT).

---

## Reconstruction

#### GRASP [1] ✅ [[doc]](doc/grasp/README.md)

$$
x = \underset{x}{\arg\min} ||D^{\frac{1}{2}}(F\cdot C \cdot x-y_c)||^2_2+\lambda_1||S \cdot x||_1
$$

$x$ is the reconstructed image, $F$ is the NUFFT operator, $C$ is the coil sensitivity map, $y_c$ is the measured k-space data on each coil, $D$ is the variable density compensation, $S$ is the sparsity operator (total variation), and $\lambda_1$ is the regularization parameter.

#### XD-GRASP [2] ✅ [[doc]](doc/xd_grasp/README.md)

$$
x = \underset{x}{\arg\min} ||D^{\frac{1}{2}}(F\cdot C \cdot x-y_c)||^2_2+\lambda_1||S_1 \cdot x||_1+\lambda_2||S_2 \cdot R \cdot x||_1
$$

$S_1, S_2$ are the sparsity operators applied along two different dimension (for cardiac: cardiac motion & respiratory motion)(for DCE liver: contrast-enhancement phase & respiratory motion ), $R$ is a reordering operator that reorders the k-space data.

#### RACER-GRASP [3] ✅ [[doc]](doc/racer_grasp/README.md)

$$
x = \underset{x}{\arg\min} ||\frac{W}{R}D^{\frac{1}{2}}(F\cdot C \cdot x-y_c)||^2_2+\lambda_1||S \cdot x||_1
$$

$W$ represents motion weights for different respiratory phases, $R$ represents the streak-ratio weights to adjust the contribution of each coil in the data consistency term.

#### L+S [4] ✅ [[doc]](doc/l_s/README.md)

$$
\min_{L,S} \frac{1}{2} ||F\cdot C\cdot (L+S)-y_c||^2_2 + \lambda_L ||L||_* + \lambda_S ||TS||_1
$$

Matrix decomposition $x=L+S$, The L component captures the correlated background between frames and S captures the dynamic information. $||\cdot||_*$ is the nuclear-norm.


#### GRAPPA [5] ❌ [[doc]](doc/grappa/README.md)

$$
S_j(k_y-m \triangle k_y) = \sum_{l=1}^{L} \sum_{b=0}^{N_b-1} n(j,b,l,m) S_l(k_y-bA\triangle k_y)
$$

$S_j(k_y-m \triangle k_y)$ is the reconstructed data in coil $j$ at a line $m\triangle k_y$ offset from the normally acquired data using a blockwise reconstruction. $A$ represents the acceleration factor. $L$ is the number of coils, $N_b$ is the number of blocks used in the reconstruction, where a block is defined as a single acquired line and $A-1$ missing lines.  $n(j,b,l,m)$ is the GRAPPA kernel, where the index $l$ counts through the individual coils, the index $b$ counts through the individual reconstruction blocks

#### POCS ❌

## Coil Compression

#### GCC [6] ❌ [[doc]](doc/gcc/README.md)

## Sensitivity Map Estimation

#### ESPIRiT [7] ❌ [[doc]](doc/espirit/README.md)


## Reference

[1] Feng, Li, et al. "Golden‐angle radial sparse parallel MRI: combination of compressed sensing, parallel imaging, and golden‐angle radial sampling for fast and flexible dynamic volumetric MRI." *Magnetic resonance in medicine* 72.3 (2014): 707-717.

[2] Feng, Li, et al. "XD‐GRASP: golden‐angle radial MRI with reconstruction of extra motion‐state dimensions using compressed sensing." *Magnetic resonance in medicine* 75.2 (2016): 775-788.

[3] Feng, Li, et al. "RACER‐GRASP: respiratory‐weighted, aortic contrast enhancement‐guided and coil‐unstreaking golden‐angle radial sparse MRI." *Magnetic resonance in medicine* 80.1 (2018): 77-89.

[4] Otazo, Ricardo, Emmanuel Candes, and Daniel K. Sodickson. "Low‐rank plus sparse matrix decomposition for accelerated dynamic MRI with separation of background and dynamic components." *Magnetic resonance in medicine* 73.3 (2015): 1125-1136.

[5] Griswold, Mark A., et al. "Generalized autocalibrating partially parallel acquisitions (GRAPPA)." Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine 47.6 (2002): 1202-1210.

[6] Zhang, Tao, et al. "Coil compression for accelerated imaging with Cartesian sampling." Magnetic resonance in medicine 69.2 (2013): 571-582.

[7] Uecker, Martin, et al. "ESPIRiT—an eigenvalue approach to autocalibrating parallel MRI: where SENSE meets GRAPPA." Magnetic resonance in medicine 71.3 (2014): 990-1001.