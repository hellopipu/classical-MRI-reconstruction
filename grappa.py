import numpy as np
import scipy.linalg as la
import scipy.signal as sg
import scipy.fftpack as fp
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import view_as_windows
import numpy as np


def calibrate(AtA, kSize, nCoil, coil, lambda_, sampling=None):
    if sampling is None:
        sampling = np.ones([kSize[0], kSize[1], nCoil])

    dummyK = np.zeros((kSize[0], kSize[1], nCoil))
    dummyK[kSize[0]//2, kSize[1]//2, coil] = 1
    idxY = np.where(dummyK)
    sampling[idxY] = 0
    idxA = np.where(sampling)

    Aty = AtA[:, idxY]
    Aty = Aty[idxA]
    AtA = AtA[idxA, :]
    AtA = AtA[:, idxA]

    kernel = np.zeros_like(sampling)

    lambda_ = np.linalg.norm(AtA, 'fro') / AtA.shape[0] * lambda_

    rawkernel = np.linalg.inv(AtA + np.eye(AtA.shape[0]) * lambda_) @ Aty
    kernel[idxA] = rawkernel

    return kernel, rawkernel

def corrMatrix(kCalib, kSize):
    sx, sy, nCoil = kCalib.shape

    A = []

    for ni in range(nCoil):
        tmp = view_as_windows(kCalib[:,:,ni], kSize)
        tmp = tmp.reshape(-1, kSize[0]*kSize[1])
        A.append(tmp.T)

    A = np.concatenate(A, axis=1)
    AtA = np.dot(A.T, A)

    return AtA


def ARC(kData, AtA, kSize, c, lambda_):
    sx, sy, nCoil = kData.shape

    kData = np.pad(kData, ((0, kSize[0]-1), (0, kSize[1]-1), (0, 0)))

    dummyK = np.zeros((kSize[0], kSize[1], nCoil))
    dummyK[kSize[0]//2, kSize[1]//2, c] = 1
    idxy = np.where(dummyK)

    res = np.zeros((sx, sy))

    MaxListLen = 100
    LIST = np.zeros((kSize[0]*kSize[1]*nCoil, MaxListLen))
    KEY =  np.zeros((kSize[0]*kSize[1]*nCoil, MaxListLen))
    count = 0

    for y in range(sy):
        for x in range(sx):
            tmp = kData[x:x+kSize[0], y:y+kSize[1], :]
            pat = np.abs(tmp) > 0
            if pat[idxy] or np.sum(pat) == 0:
                res[x, y] = tmp[idxy]
            else:
                key = pat.flatten()
                idx = 0
                for nn in range(KEY.shape[1]):
                    if np.sum(key == KEY[:, nn]) == len(key):
                        idx = nn
                        break
                if idx == 0:
                    count += 1
                    kernel = calibrate(AtA, kSize, nCoil, c, lambda_, pat)  # Assuming calibrate is a defined function
                    KEY[:, count % MaxListLen] = key
                    LIST[:, count % MaxListLen] = kernel.flatten()
                else:
                    kernel = LIST[:, idx]
                res[x, y] = np.sum(kernel * tmp.flatten())
    return res

def GRAPPA(kdata,kcalib,kernel_size,lambd=0.1):
    # kdata: k-space data, shape: (kx,ky,coil)
    # kcalib: k-space calibration data, shape: (xx,yy,coil)
    # kernel_size: kernel size, shape: (kernel_x,kernel_y)
    # lambd: regularization parameter
    # return: GRAPPA reconstructed image


    # k-space data size
    fe, pe, coils = kdata.shape
    # k-space calibration data size
    AtA = corrMatrix(kcalib, kernel_size)
    recon_kdata = np.zeros_like(kdata)
    for i in range(coils):
        recon_kdata[:,:,i] = ARC(kdata, AtA, kernel_size, i, lambd)
    return recon_kdata
