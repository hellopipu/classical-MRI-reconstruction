'''
Feng L, Axel L, Chandarana H, Block KT, Sodickson DK, Otazo R.
XD-GRASP: Golden-angle radial MRI with reconstruction of
extra motion-state dimensions using compressed sensing
Magn Reson Med. 2016 Feb;75(2):775-88

(c) Li Feng, 2016, New York University
Li.Feng@nyumc.org

reimplemented in python by Bingyu Xin, 2023
'''

import os.path
from os.path import join
from scipy.io.matlab import loadmat
import torchkbnufft as tkbn
import torch
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import peakutils
from scipy.interpolate import UnivariateSpline
import time
import numpy as np


class Base():
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.prepare_for_xdgrasp()

    def cal_metric(self, gt, pred):
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        gt = np.abs(gt) / np.abs(gt).max()
        pred = np.abs(pred) / np.abs(pred).max()
        psnr = peak_signal_noise_ratio(gt, pred, data_range=gt.max())
        ssim = structural_similarity(gt, pred, data_range=gt.max())
        print('PSNR: %5.2f, SSIM: %5.4f' % (psnr, ssim))
        return psnr, ssim

    def save_results(self, recon_cs, filename):
        import SimpleITK as sitk
        folder = os.path.dirname(filename)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        ttxy = recon_cs.abs().squeeze().cpu().numpy().reshape(self.nt, self.ntres, self.oshape, self.oshape)  # shape (11, 4, 256, 256), # only save the magnitude
        ttxy = ttxy / ttxy.max()  # norm to 1
        # save different respiratory phases
        for i in range(self.ntres):
            img_sitk = sitk.GetImageFromArray(ttxy[:, i])
            sitk.WriteImage(img_sitk, filename.split('.nii.gz')[0] + '_res{}.nii.gz'.format(i))
        print(f'### Reconstucted image saved to {filename}')

    def compare_with_matlab_results(self, recon_nufft, recon_cs, show_img=False):
        print('### Compared with original matlab code result: ')
        data_path_nufft = 'data/grasp/data_gradding.mat'
        data_path_cs = 'data/grasp/data_xdgrasp.mat'

        folder = os.path.dirname(data_path_cs)

        if not os.path.isfile(data_path_cs):
            print('Download the matlab reconstruction data...')
            os.makedirs(folder, exist_ok=True)
            import urllib.request
            url_nufft = 'https://rutgers.box.com/shared/static/3e4uj4fqtx9oe4ry3svxtao4ov472u0n.mat'
            url_cs = 'https://rutgers.box.com/shared/static/l8myxgsx76px9xgryrjfdp2ou5kpxfjn.mat'
            urllib.request.urlretrieve(url_cs, data_path_cs)
            urllib.request.urlretrieve(url_nufft, data_path_nufft)
        else:
            print('The matlab reconstruction data exists.')

        recon_grasp_matlab = loadmat(data_path_cs)['data_xdgrasp']
        recon_nufft_matlab = loadmat(data_path_nufft)['data_gridding']

        print('NUFFT RECONSTRUCTION : ')
        self.cal_metric(recon_nufft.squeeze().cpu().numpy(),
                        recon_nufft_matlab.transpose(3, 2, 0, 1).reshape(self.nt*self.ntres, self.oshape, self.oshape))
        print('GRASP RECONSTRUCTION : ')
        self.cal_metric(recon_cs.squeeze().cpu().numpy(),
                        recon_grasp_matlab.transpose(3, 2, 0, 1).reshape(self.nt*self.ntres, self.oshape, self.oshape))

    def process_data(self, data):
        smaps, traj, kc, kdata, dcp = data['b1'], data['k'], data['kc'], data['kdata'], data['w']

        # norm smaps, from ESPIRiT paper
        smaps = smaps / np.sum(np.abs(smaps) ** 2, axis=-1, keepdims=True) ** 0.5  # h,w,coil

        output_shape = smaps.shape[0]

        nx, ntViews, nc = kdata.shape
        kdata_u1, traj_u1, w_u1, self.Res_Signal, peaks = self.sort_data(kc, kdata, traj, dcp)

        sam, tt, resp, spo, nc = kdata_u1.shape

        self.oshape = output_shape

        # to torch tensor
        op = tkbn.KbNufft((output_shape, output_shape)).cuda()
        op_adj = tkbn.KbNufftAdjoint((output_shape, output_shape)).cuda()
        toep_op = tkbn.ToepNufft().cuda()

        kdata = torch.from_numpy(kdata_u1.transpose(1, 2, 4, 0, 3).reshape(tt * resp, nc, sam * spo)).cuda()

        traj = np.stack([traj_u1.real, traj_u1.imag], -1)
        traj = torch.from_numpy(2 * np.pi * traj.transpose(1, 2, 4, 0, 3).reshape(tt * resp, 2, sam * spo)).cuda()

        dcp = torch.from_numpy(w_u1.transpose(1, 2, 0, 3).reshape(tt * resp, 1, sam * spo)).cuda()
        smaps = torch.from_numpy(smaps.transpose(2, 0, 1)[None]).cuda()

        kernel = tkbn.calc_toeplitz_kernel(traj, im_size=(output_shape, output_shape), weights=dcp, norm='ortho')

        # nufft recon
        print(' -- calculating gridding reconstruction')
        recon_nufft = op_adj(kdata * dcp, traj, smaps=smaps, norm='ortho')
        # l1 reg, hyper-parameter can be tuned
        _lambda_0 = 0.03 * recon_nufft.abs().max()
        _lambda_1 = 0.015 * recon_nufft.abs().max()

        return op, toep_op, kdata, traj, dcp, smaps, kernel, recon_nufft, _lambda_0, _lambda_1

    def prepare_for_xdgrasp(self):
        # load data
        print(' -- loading data')
        data_path = 'data/xdgrasp/data_DCE.mat'
        folder = os.path.dirname(data_path)
        if not os.path.isfile(data_path):
            os.makedirs(folder, exist_ok=True)
            import urllib.request
            print('Download the DCE liver data...')
            url = 'https://rutgers.box.com/shared/static/fcz0s665pffqgaz19smoddfa2br3ziha.mat'
            urllib.request.urlretrieve(url, data_path)
        else:
            print('The DCE liver data exists.')
        data = loadmat(data_path)

        print(' -- preprocessing data: sort respiratory signal, move array to pytorch tensor, calculate toep kernel')
        self.op, self.toep_op, self.kdata, self.traj, self.dcp, self.smaps, self.kernel, self.recon_nufft, self._lambda_0, self._lambda_1 = self.process_data(
            data)


class XDGRASP(Base):
    def __init__(self):
        super(XDGRASP, self).__init__()

    def tv(self, x, dim=0):
        if dim == 0:  # along time
            y = torch.cat([x[1::], x[-1::]], dim=0) - x
        else:  # along resp
            y = torch.cat([x[:, 1::], x[:, -1::]], dim=1) - x
        return y

    def adj_tv(self, x, dim=0):
        if dim == 0:  # along time
            y = torch.cat([x[0:1], x[0:-1]], dim=0) - x
            y[0] = -x[0]
            y[-1] = x[-2]
        else:  # along resp
            y = torch.cat([x[:, 0:1], x[:, 0:-1]], dim=1) - x
            y[:, 0] = -x[:, 0]
            y[:, -1] = x[:, -2]
        return y

    def grad(self, x, _l1Smooth=1e-15):
        # L2 norm part
        aah = []
        for i in range(self.kernel.shape[0]):
            aah.append(self.toep_op(x[i:i + 1], self.kernel[i:i + 1], smaps=self.smaps, norm='ortho'))
        aah = torch.cat(aah, dim=0)

        L2Grad = 2 * (aah - self.recon_nufft)

        x = x.view(self.nt, self.ntres, 1, self.oshape, self.oshape)
        # TV along time
        w = self.tv(x, 0)
        L1Grad_0 = self.adj_tv(w * (w.abs() ** 2 + _l1Smooth) ** -0.5, 0).view(self.nt * self.ntres, 1, self.oshape,
                                                                               self.oshape)
        # TV along time
        w = self.tv(x, 1)
        L1Grad_1 = self.adj_tv(w * (w.abs() ** 2 + _l1Smooth) ** -0.5, 1).view(self.nt * self.ntres, 1, self.oshape,
                                                                               self.oshape)

        # composite gradient
        g = L2Grad + self._lambda_0 * L1Grad_0 + self._lambda_1 * L1Grad_1
        return g

    def objective(self, x, dx, t, _l1Smooth=1e-15):
        x_ = x + t * dx
        # L2-norm part
        w = (self.op(x_, self.traj, smaps=self.smaps, norm='ortho') - self.kdata) * self.dcp ** 0.5
        L2Obj = (w.abs() ** 2).sum()

        x_ = x_.view(self.nt, self.ntres, 1, self.oshape, self.oshape)
        # TV along time
        w = self.tv(x_, 0)
        L1Obj_0 = ((w.abs() ** 2 + _l1Smooth) ** 0.5).sum()

        # TV along respiration
        w = self.tv(x_, 1)
        L1Obj_1 = ((w.abs() ** 2 + _l1Smooth) ** 0.5).sum()

        # objective function
        res = L2Obj + self._lambda_0 * L1Obj_0 + self._lambda_1 * L1Obj_1
        return res, L2Obj, self._lambda_0 * L1Obj_0, self._lambda_1 * L1Obj_1

    def CSL1NlCg_XDGRASP(self, x0, max_iter, gradToll=1e-8, maxlsiter=6, alpha=0.01, beta=0.6, print_detail=False):
        '''
        Non-linear Conjugate Gradient Algorithm
        :param x0: starting point (gridding images)
        :param max_iter: num of iterations
        :param gradToll: stopping criteria by gradient magnitude
        :param maxlsiter: line search parameter: max num of line-search iterations
        :param alpha: line search parameter
        :param beta: line search parameter
        :param print_detail: print line-search details
        :return: x, reconstructed images
        '''

        # starting point
        x = x0

        t0 = 1
        k = 0
        nite = max_iter - 1
        # compute g0  = grad(f(x))
        g0 = self.grad(x)
        dx = -g0
        # iterations
        while (1):
            # backtracking line-search
            f0, _, _, _ = self.objective(x, dx, 0)
            t = t0
            f1, mse_, tv0_, tv1_ = self.objective(x, dx, t)
            lsiter = 0
            ff = f0 - alpha * t * (torch.conj(g0) * dx).sum().abs()
            while (f1 > ff) and (lsiter < maxlsiter):
                lsiter = lsiter + 1
                t = t * beta
                f1, mse_, tv0_, tv1_ = self.objective(x, dx, t)
                ff = f0 - alpha * t * (torch.conj(g0) * dx).sum().abs()
                if print_detail:
                    print('----------------backtracking line-search: {}'.format((f1 - ff).item()))

            # control the number of line searches by adapting the initial step search
            t0 = t0 * beta if lsiter > 2 else t0
            t0 = t0 / beta if lsiter < 1 else t0

            # update x
            x = x + t * dx

            print(
                ' iter={}, cost {}, mse {}, tv0 {}, tv1 {}'.format(k, 1e5 * f1, 1e5 * mse_, 1e5 * tv0_, 1e5 * tv1_))
            if print_detail:
                print('----------------conjugate gradient calculation')
            # conjugate gradient calculation
            g1 = self.grad(x)
            bk = (g1.abs() ** 2).sum() / ((g0.abs() ** 2).sum() + torch.finfo(torch.float64).eps)
            g0 = g1
            dx = -g1 + bk * dx
            k = k + 1

            # stopping criteria
            if k > nite or torch.linalg.norm(dx) < gradToll:
                break
        return x

    def CoilClustering(self, d1, thresh):
        nviews, nc = d1.shape
        corrm = np.zeros((nc, nc))
        for i in range(nc):
            for j in range(nc):
                corrm[i, j] = np.corrcoef(d1[:, i], d1[:, j])[0, 1]
                corrm[j, i] = corrm[i, j]
        mask = np.zeros_like(corrm)
        mask[np.abs(corrm) > thresh] = 1

        u, s, vh = np.linalg.svd(mask, full_matrices=False)
        v1 = np.abs(u[:, 0])
        thresh2 = 0.1
        subgroup = np.zeros((nc))
        subgroup[v1 > thresh2] = 1

        dave = np.zeros((nviews))

        # adjust from the first coil
        subindex = np.where(subgroup == 1)[0]
        coilID = subgroup.copy()
        for c in range(nc):
            if subgroup[c] > 0:
                if corrm[subindex[0], c] > 0:
                    dave = dave + d1[:, c]
                else:
                    dave = dave - d1[:, c]
                    coilID[c] = -coilID[c]
        dave = dave / subgroup.sum()

        return dave, coilID

    def sort_data(self, kc, kdata, k, w):
        # generate the z-projection profiles
        # ZIP: Projection profiles along the Z dimension with interpolation.
        ZIP = np.abs(np.fft.fftshift(np.fft.ifft(kc, n=400, axis=0, norm='ortho'), axes=0))
        # Normalization of each projection in each coil element
        maxprof = np.max(ZIP, axis=0, keepdims=True)
        minprof = np.min(ZIP, axis=0, keepdims=True)
        ZIP = (ZIP - minprof) / (maxprof - minprof)
        # STEP 1: find the coil elements with good respiratory motion display from the late enhancement spokes
        ZIP1 = ZIP[:, 500::, :]
        nz, ntviews, nc = ZIP1.shape
        PCs = np.zeros((600, 2 * nc))
        kk = 0
        # Perform PCA on each coil element
        for i in range(nc):
            tmp = ZIP1[:, :, i]  # 400,600
            covariance = np.cov(tmp)
            V, tmp2 = np.linalg.eigh(covariance, 'U')
            V = V[::-1]
            tmp2 = tmp2[:, ::-1]  # 400,400
            PC = (tmp.transpose(1, 0) @ tmp2)
            for j in range(2):
                tmp3 = savgol_filter(PC[:, j], 6, 2)  # smooth(PC,j) #lowess_smooth(PC[:,j],6)
                tmp3 = (tmp3 - tmp3.min()) / (tmp3.max() - tmp3.min())
                PCs[:, i * 2 + j] = tmp3
        # Coil clusting to find the respiratory motion signal
        thresh = 0.97
        Res_Signal, cluster = self.CoilClustering(PCs, thresh);
        Res_Signal = (Res_Signal - Res_Signal.min()) / (Res_Signal.max() - Res_Signal.min())
        cluster = np.abs(cluster[::2] + cluster[1::2])
        # step 2 : Estimating respiratory motion from the "good" coil elements
        # perform PCA on the stack of "good" coil elements
        nz, ntviews, nc = ZIP.shape
        SI = ZIP[:, :, cluster != 0]
        SI = np.abs(SI.transpose(0, 2, 1).reshape(-1, ntviews))
        covariance = np.cov(SI)
        V, PC = np.linalg.eigh(covariance, 'U')
        V = V[::-1]  # 1200
        PC = PC[:, ::-1]
        SI = SI.T @ PC
        # Do some smoothing
        for i in range(3):
            SI[:, i] = savgol_filter(SI[:, i], 6, 2)
        Res_Signal = SI[:, 0];
        Res_Signal = -Res_Signal
        Res_Signal = (Res_Signal - Res_Signal.min()) / (Res_Signal.max() - Res_Signal.min())
        # Estimate the envelope of the signal (contrast enhancement + respiration)
        t = 16
        peak_indices = peakutils.indexes(Res_Signal, thres=0.5, min_dist=t)
        peaks = Res_Signal[peak_indices]

        # substract the estimated envelope
        smoothing_spline = UnivariateSpline(peak_indices, Res_Signal[peak_indices], s=0.015)
        ftmax = smoothing_spline(np.arange(len(Res_Signal)))
        Res_Signal = Res_Signal - ftmax

        peak_indices = peakutils.indexes(Res_Signal, thres=0.5, min_dist=t)
        peaks = Res_Signal[peak_indices]

        # save the final respiratory signal on the projections
        path_res_signal = 'output/xdgrasp/Res_Signal.png'
        dirname = os.path.dirname(path_res_signal)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        plt.figure(figsize=[20, 5])
        plt.imshow(np.abs(ZIP[:, :, 4]), 'gray')
        plt.plot(-Res_Signal * 150 + 100, 'r')
        plt.plot(peak_indices, -peaks * 150 + 100, 'bo')
        plt.xticks([]), plt.yticks([]), plt.title('projection profile')
        plt.savefig('output/xdgrasp/Res_Signal.png')

        # Data sorting
        nline = 100  # number of spoeks for each contrast-enhanced phase
        nt = ntviews // nline  # number of contrast-enhanced phases
        ntres = 4  # number of respiratory phases
        self.nt, self.ntres = nt, ntres
        nline2 = nline // ntres  # number of spokes in each phases after respiratory sorting

        sam, ntviews, nc = kdata.shape

        _index = np.argsort(Res_Signal.reshape(nt, nline), axis=1)[:, ::-1]
        kdata_u = kdata.reshape(sam, nt, nline, nc)
        k_u = k.reshape(sam, nt, nline)
        w_u = w.reshape(sam, nt, nline)

        kdata_u1 = np.zeros_like(kdata_u)
        k_u1 = np.zeros_like(k_u)
        w_u1 = np.zeros_like(w_u)
        for i in range(nt):
            kdata_u1[:, i] = kdata_u[:, i, _index[i]]
            k_u1[:, i] = k_u[:, i, _index[i]]
            w_u1[:, i] = w_u[:, i, _index[i]]

        kdata_u1 = kdata_u1.reshape(sam, nt, ntres, nline2, nc)
        k_u1 = k_u1.reshape(sam, nt, ntres, nline2)
        w_u1 = w_u1.reshape(sam, nt, ntres, nline2)
        return kdata_u1, k_u1, w_u1, Res_Signal, peaks

    def run(self, nround=3, max_iter=6, save_to_file=True, compare_matlab=False):
        '''
        run the XD-GRASP method. For reconstruction shape of (11,4,256,256), run time on A100: 30s, GPU usage: 8.5G
        :param nround: run the grasp algorithm nround times
        :param max_iter: the iterations of nonlinear conjugate gradient descent algorithm with backtracking line search in grasp algorithm
        :param save_to_file: save the zero-filled and grasp reconstructed images to nii.gz file
        :param compare_matlab: compare the python results with the original matlab code results
        :return: grasp reconstructed liver images, shape (28,1,384,384), pytorch tensor, type: torch.complex64
        '''
        print('Start XD-GRASP algorithm...')
        start_time = time.time()
        recon_cs = self.recon_nufft
        for i in range(nround):
            print(' -- XDGRASP Reconstruction Round {}'.format(i))
            recon_cs = self.CSL1NlCg_XDGRASP(recon_cs, max_iter)
        end_time = time.time()
        print('### Done! Running time: %.2f s' % (end_time - start_time))
        if save_to_file:
            self.save_results(self.recon_nufft, 'output/xdgrasp/DCE_nufft.nii.gz')
            self.save_results(recon_cs, 'output/xdgrasp/DCE_xdgrasp.nii.gz')
        if compare_matlab:
            #TODO: The low SSIM values observed in the NUFFT results may be attributed to differences
            # in the implementation of the respiratory signal calculation, which can result in variations
            # in the binning of the k-space data.
            # this code: NUFFT: PSNR: 28.00, SSIM: 0.5561;;; GRASP:PSNR: 33.59, SSIM: 0.9293
            self.compare_with_matlab_results(self.recon_nufft, recon_cs)
        return recon_cs


if __name__ == '__main__':
    with torch.no_grad():
        xdgrasp_method = XDGRASP()
        recon_xdgrasp = xdgrasp_method.run()
