'''
GRASP (Golden-angle RAdial Sparse Parallel MRI)
Combination of compressed sensing, parallel imaging and golden-angle
radial sampling for fast dynamic MRI.
This demo will reconstruct one slice of a contrast-enhanced liver scan.
Radial k-space data are continously acquired using the golden-angle
scheme and retrospectively sorted into a time-series of images using
a number of consecutive spokes to form each frame. In this example, 21
consecutive spokes are used for each frame, which provides 28 temporal
frames. The reconstructed image matrix for each frame is 384x384. The
undersampling factor is 384/21=18.2.

Li Feng, Ricardo Otazo, NYU, 2012

reimplement in python by Bingyu Xin, 2023
'''

import os
import time
from scipy.io.matlab import loadmat
import numpy as np
import torch
import torchkbnufft as tkbn


class Base():
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.prepare_for_grasp()

    def compare_with_matlab_results(self, recon_nufft, recon_cs, show_img=False):
        print('### Compared with original matlab code result: ')
        data_path_cs = 'data/grasp/liver_recon_cs.mat'
        data_path_nufft = 'data/grasp/liver_nufft_rec.mat'

        folder = os.path.dirname(data_path_cs)

        if not os.path.isfile(data_path_cs):
            print('Download the matlab reconstruction data...')
            os.makedirs(folder, exist_ok=True)
            import urllib.request
            url_nufft = 'https://rutgers.box.com/shared/static/qhhhwjju87zzp2k3akkcvkl73nkd1kw8.mat'
            url_cs = 'https://rutgers.box.com/shared/static/nxih5itvyaccyjx3uxew3lzdzpb38ywu.mat'
            urllib.request.urlretrieve(url_cs, data_path_cs)
            urllib.request.urlretrieve(url_nufft, data_path_nufft)
        else:
            print('The matlab reconstruction data exists.')

        recon_grasp_matlab = loadmat(data_path_cs)['recon_cs']
        recon_nufft_matlab = loadmat(data_path_nufft)['recon_nufft']

        print('- NUFFT RECONSTRUCTION : ')
        self.cal_metric(recon_nufft_matlab, recon_nufft.squeeze().permute(1, 2, 0).cpu().numpy()[::-1])
        print('- GRASP RECONSTRUCTION : ')
        self.cal_metric(recon_grasp_matlab, recon_cs.squeeze().permute(1, 2, 0).cpu().numpy()[::-1])

        if show_img:
            import matplotlib.pyplot as plt
            _slice = 10
            plt.subplot(221)
            plt.imshow(recon_cs[_slice, 0,].abs().cpu().numpy()[::-1], 'gray')  # grasp
            plt.subplot(222)
            plt.imshow(np.abs(recon_grasp_matlab[:, :, _slice]), 'gray')  # grasp matlab
            plt.subplot(223)
            plt.imshow(recon_nufft[_slice, 0].abs().cpu().numpy()[::-1], 'gray')  # nufft
            plt.subplot(224)
            plt.imshow(np.abs(recon_nufft_matlab[..., _slice]), 'gray')  # nufft matlab
            plt.show()

    def cal_metric(self, gt, pred):
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        gt = np.abs(gt) / np.abs(gt).max()
        pred = np.abs(pred) / np.abs(pred).max()
        psnr = peak_signal_noise_ratio(gt, pred, data_range=gt.max())
        ssim = structural_similarity(gt, pred, data_range=gt.max())
        print(f'PSNR: {psnr:.2f}, SSIM: {ssim:.4f}')

    def save_results(self, recon_cs, filename):
        import SimpleITK as sitk
        folder = os.path.dirname(filename)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        txy = recon_cs.abs().squeeze().cpu().numpy()  # only save the magnitude
        txy = txy / txy.max()  # norm to 1
        img_sitk = sitk.GetImageFromArray(txy[:, ::-1])
        sitk.WriteImage(img_sitk, filename)
        print(f'### Reconstucted image saved to {filename}')

    def prepare_for_grasp(self):
        # load data
        data_path = 'data/grasp/liver_data.mat'
        folder = os.path.dirname(data_path)
        if not os.path.isfile(data_path):
            os.makedirs(folder, exist_ok=True)
            import urllib.request
            print('Download the liver data...')
            url = 'https://rutgers.box.com/shared/static/pnjemhzp2pk18kgphr5ihkjp8ep50t67.mat'
            urllib.request.urlretrieve(url, data_path)
        else:
            print('The liver data exists.')
        data = loadmat(data_path)

        self.op, self.toep_op, self.kdata, self.traj, self.dcp, self.smaps, self.kernel, self.recon_nufft, self._lambda = self.process_data(
            data)

    def process_data(self, data):
        print('Preprocess the data...')
        kdata, traj, dcp, smaps = data['kdata'].astype(np.complex64), data['k'].astype(np.complex64), data['w'].astype(
            np.float32), data['b1'].astype(np.complex64)
        # norm smaps, from ESPIRiT paper
        smaps = smaps / np.sum(np.abs(smaps) ** 2, axis=2, keepdims=True) ** 0.5  # shape (384, 384, 12), (H,W,nc)

        output_shape = smaps.shape[0]
        nx, ntViews, nc = kdata.shape

        # define number of spokes to be used per frame (Fibonacci number)
        nspokes = 21
        # number of frames
        nt = int(ntViews // nspokes)
        kdata = kdata[:, 0:nt * nspokes].reshape(nx, nt, nspokes, nc).transpose(1, 3, 0,
                                                                                2)  # shape (28, 12, 768, 21), (T,nc,nx,ntViews)
        traj = 2 * np.pi * traj[:, 0:nt * nspokes].reshape(nx, nt, nspokes).transpose(1, 0, 2)  # shape (28, 768, 21)
        dcp = dcp[:, 0:nt * nspokes].reshape(nx, nt, nspokes).transpose(1, 0, 2)  # shape (28, 768, 21)
        # print(smaps.shape, kdata.shape, traj.shape, dcp.shape)

        # to torch tensor
        op = tkbn.KbNufft((output_shape, output_shape)).to(self.device)
        op_adj = tkbn.KbNufftAdjoint((output_shape, output_shape)).to(self.device)
        toep_op = tkbn.ToepNufft().to(self.device)

        kdata = torch.from_numpy(kdata.reshape(nt, nc, -1)).to(self.device)  # nt, nc, sam*spo
        traj = torch.from_numpy(traj.reshape(nt, -1)).to(self.device)
        traj = torch.view_as_real(traj).permute(0, 2, 1)  # nt, 2, sam*spo
        dcp = torch.from_numpy(dcp.reshape(nt, 1, -1)).to(self.device)  # nt, 1, sam*spo
        smaps = torch.from_numpy(smaps[None].transpose(0, 3, 1, 2)).to(self.device)
        kernel = tkbn.calc_toeplitz_kernel(traj, im_size=(output_shape, output_shape), weights=dcp, norm='ortho')

        # nufft recon
        recon_nufft = op_adj(kdata * dcp, traj, smaps=smaps, norm='ortho')  # shape (28,1,384,384)
        # l1 reg, hyper-parameter can be tuned
        _lambda = 0.25 * recon_nufft.abs().max()

        return op, toep_op, kdata, traj, dcp, smaps, kernel, recon_nufft, _lambda


class GRASP(Base):
    def __init__(self):
        super(GRASP, self).__init__()

    def tv(self, x):
        y = torch.cat([x[1::], x[-1::]], dim=0) - x
        return y

    def adj_tv(self, x):
        y = torch.cat([x[0:1], x[0:-1]], dim=0) - x
        y[0] = -x[0]
        y[-1] = x[-2]
        return y

    def grad(self, x, _l1Smooth=1e-15):
        # L2 norm part
        aah = []
        for i in range(self.kernel.shape[0]):
            aah.append(self.toep_op(x[i:i + 1], self.kernel[i:i + 1], smaps=self.smaps, norm='ortho'))
        aah = torch.cat(aah, dim=0)

        L2Grad = 2 * (aah - self.recon_nufft)
        # L1 norm part
        w = self.tv(x)
        L1Grad = self.adj_tv(w * (w.abs() ** 2 + _l1Smooth) ** -0.5)

        # composite gradient
        g = L2Grad + self._lambda * L1Grad
        return g

    def objective(self, x, dx, t, _l1Smooth=1e-15):
        x_ = x + t * dx
        # L2-norm part
        w = (self.op(x_, self.traj, smaps=self.smaps, norm='ortho') - self.kdata) * self.dcp ** 0.5
        L2Obj = (w.abs() ** 2).sum()

        # L1-norm part
        w = self.tv(x_)
        L1Obj = ((w.abs() ** 2 + _l1Smooth) ** 0.5).sum()
        # objective function
        res = L2Obj + self._lambda * L1Obj
        return res

    def CSL1NlCg(self, x0, max_iter, gradToll=1e-3, maxlsiter=150, alpha=0.01, beta=0.6, print_detail=False):
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
            f0 = self.objective(x, dx, 0)
            t = t0
            f1 = self.objective(x, dx, t)
            lsiter = 0
            ff = f0 - alpha * t * (torch.conj(g0) * dx).sum().abs()
            while (f1 > ff) and (lsiter < maxlsiter):
                lsiter = lsiter + 1
                t = t * beta
                f1 = self.objective(x, dx, t)
                ff = f0 - alpha * t * (torch.conj(g0) * dx).sum().abs()
                if print_detail:
                    print('----------------backtracking line-search: {}'.format((f1 - ff).item()))

            # control the number of line searches by adapting the initial step search
            t0 = t0 * beta if lsiter > 2 else t0
            t0 = t0 / beta if lsiter < 1 else t0

            # update x
            x = x + t * dx

            print(' iter={}, cost = {}'.format(k, f1))
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

    def run(self, nround=3, nite=8, save_to_file=True, compare_matlab=False):
        '''
        run the GRASP method. For reconstruction shape of (28,1,384,384), run time on A100: 30s, GPU usage: 7.5G
        :param nround: run the grasp algorithm nround times
        :param nite: the iterations of nonlinear conjugate gradient descent algorithm with backtracking line search in grasp algorithm
        :param save_to_file: save the zero-filled and grasp reconstructed images to nii.gz file
        :param compare_matlab: compare the python results with the original matlab code results
        :return: grasp reconstructed liver images, shape (28,1,384,384), pytorch tensor, type: torch.complex64
        '''
        print('Start GRASP algorithm...')
        start_time = time.time()
        recon_cs = self.recon_nufft
        for i in range(nround):
            print(f'Round = {i}: ')
            recon_cs = self.CSL1NlCg(recon_cs, nite)  # shape (28,1,384,384)
        end_time = time.time()
        print('### Done! Running time: %.2f s' % (end_time - start_time))
        if save_to_file:
            self.save_results(self.recon_nufft, 'output/grasp/liver_nufft.nii.gz')
            self.save_results(recon_cs, 'output/grasp/liver_grasp.nii.gz')
        if compare_matlab:
            # this code: NUFFT: PSNR: 60.83, SSIM: 0.9999;;; GRASP: PSNR: 35.44, SSIM: 0.9620
            self.compare_with_matlab_results(self.recon_nufft, recon_cs)
        return recon_cs


if __name__ == '__main__':
    with torch.no_grad():
        grasp_method = GRASP()
        recon_grasp = grasp_method.run()
