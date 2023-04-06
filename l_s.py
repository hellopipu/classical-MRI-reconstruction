'''
L+S reconstruction of undersampled dynamic MRI data using iterative
soft-thresholding of singular values of L and iterative clipping of
entries of S (temporal total variation on S)

L+S reconstruction of dynamic contrast-enhanced abdominal MRI acquired
with golden-angle radial sampling

Ricardo Otazo (2013)

reimplement in python by Bingyu Xin, Rutgers, 2023
'''

import os
import time
import numpy as np
import torch
import torchkbnufft as tkbn
from scipy.io.matlab import loadmat
import SimpleITK as sitk


class Base():
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.output_dir = 'output/l_s'
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def compare_with_matlab_results(self, recon_nufft, recon_cs, show_img=False):
        print('### Compared with original matlab code result: ')
        data_path_cs = 'data/l_s/recon_l_s.mat'
        data_path_nufft = 'data/l_s/recon_nufft.mat'

        folder = os.path.dirname(data_path_cs)

        if not os.path.isfile(data_path_cs):
            print('Download the matlab reconstruction data...')
            os.makedirs(folder, exist_ok=True)
            import urllib.request
            url_nufft = 'https://rutgers.box.com/shared/static/r7giyzfn4mc8yz8clgq3kzhmwb36zos6.mat'
            url_cs = 'https://rutgers.box.com/shared/static/rvfji0032rwljqg3rzcn8aw63bu6gwkr.mat'
            urllib.request.urlretrieve(url_cs, data_path_cs)
            urllib.request.urlretrieve(url_nufft, data_path_nufft)
        else:
            print('The matlab reconstruction data exists.')

        recon_l_s_matlab = loadmat(data_path_cs)['LplusS']
        recon_nufft_matlab = loadmat(data_path_nufft)['recon_nufft']

        print('- NUFFT RECONSTRUCTION : ')
        self.cal_metric(recon_nufft_matlab[::-1], recon_nufft.squeeze().permute(1, 2, 0).cpu().numpy()[::-1])
        print('- L+S RECONSTRUCTION : ')
        self.cal_metric(recon_l_s_matlab, recon_cs.squeeze().permute(1, 2, 0).cpu().numpy()[::-1])

        if show_img:
            import matplotlib.pyplot as plt
            _slice = 10
            plt.subplot(221)
            plt.imshow(recon_cs[_slice, 0,].abs().cpu().numpy()[::-1], 'gray')  # l_s
            plt.subplot(222)
            plt.imshow(np.abs(recon_l_s_matlab[:, :, _slice]), 'gray')  # l_s matlab
            plt.subplot(223)
            plt.imshow(recon_nufft[_slice, 0].abs().cpu().numpy()[::-1], 'gray')  # nufft
            plt.subplot(224)
            plt.imshow(np.abs(recon_nufft_matlab[::-1, :, _slice]), 'gray')  # nufft matlab
            plt.show()

    def cal_metric(self, gt, pred):
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        gt = np.abs(gt) / np.abs(gt).max()
        pred = np.abs(pred) / np.abs(pred).max()
        psnr = peak_signal_noise_ratio(gt, pred, data_range=gt.max())
        ssim = structural_similarity(gt, pred, data_range=gt.max())
        print(f'PSNR: {psnr:.2f}, SSIM: {ssim:.4f}')

    def save_results(self, recon_cs, filename):
        filename = os.path.join(self.output_dir, filename)

        txy = recon_cs.abs().squeeze().cpu().numpy()  # only save the magnitude
        txy = txy / txy.max()  # norm to 1
        img_sitk = sitk.GetImageFromArray(txy[:, ::-1])
        sitk.WriteImage(img_sitk, filename)
        print(f'### Reconstucted image saved to {filename}')

    def prepare_for_LplusS(self):
        # load data
        # data_path = 'data/l_s/cardiac_data.mat'
        data_path = 'data/l_s/abdomen_dce_ga.mat'
        folder = os.path.dirname(data_path)
        if not os.path.isfile(data_path):
            os.makedirs(folder, exist_ok=True)
            import urllib.request
            print(f'Download the abdomen data to {folder}...')
            # url = 'https://rutgers.box.com/shared/static/85glcqu9dz8lvqnl8h8go265x94nntn3.mat'
            url = 'https://rutgers.box.com/shared/static/r8m69t3252jkp60c8gbalu022xwpurpe.mat'
            urllib.request.urlretrieve(url, data_path)
        else:
            print(f'The abdomen data exists in {folder}')
        data = loadmat(data_path)

        return self.process_data(data)

    def process_data(self, data):
        print('Preprocess the data...')
        kdata, traj, dcp, smaps = data['kdata'].astype(np.complex64), data['k'].astype(np.complex64), data['w'].astype(
            np.float32), data['b1'].astype(np.complex64)
        # norm smaps, from ESPIRiT paper
        smaps = smaps / np.sum(np.abs(smaps) ** 2, axis=2, keepdims=True) ** 0.5  # shape (384, 384, 12), (H,W,nc)

        output_shape = smaps.shape[0]
        nx, ntviews, nc = kdata.shape

        # define number of spokes to be used per frame (Fibonacci number)
        nspokes = 21
        # number of frames
        nt = int(ntviews // nspokes)
        kdata = kdata[:, 0:nt * nspokes].reshape(nx, nt, nspokes, nc).transpose(1, 3, 0,
                                                                                2)  # shape (28, 12, 768, 21), (T,nc,nx,ntviews)
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
        # l1 reg, (Regularization parameters were empirically selected)
        lambda_L = 0.025
        lambda_S = 0.5 * recon_nufft.abs().max()

        return op, toep_op, kdata, traj, dcp, smaps, kernel, recon_nufft, lambda_L, lambda_S


class LplusS(Base):
    def __init__(self):
        super(LplusS, self).__init__()
        self.op, self.toep_op, self.kdata, self.traj, self.dcp, self.smaps, \
        self.kernel, self.recon_nufft, self.lambda_L, self.lambda_S = self.prepare_for_LplusS()

    def tv(self, x):
        y = torch.cat([x[1::], x[-1::]], dim=0) - x
        return y

    def adj_tv(self, x):
        y = torch.cat([x[0:1], x[0:-1]], dim=0) - x
        y[0] = -x[0]
        y[-1] = x[-2]
        return y

    def softThreshold(self, x, p):
        y = (x.abs() - p) * torch.sign(x) * torch.sign(x.abs() - p)
        return y

    def IST(self, x0, max_iter, gradToll=0.0025, print_detail=False):
        '''
        iterative soft-thresholding algorithm
        :param x0: starting point (gridding images)
        :param max_iter: num of iterations
        :param gradToll: stopping criteria by gradient magnitude
        :param maxlsiter: line search parameter: max num of line-search iterations
        :param alpha: line search parameter
        :param beta: line search parameter
        :param print_detail: print line-search details
        :return: x, reconstructed images
        '''
        nt, _, h, w = x0.shape

        M = x0
        S = torch.zeros_like(M)
        z = torch.zeros((nt, 1, h, w)).to(self.device)
        nite = max_iter - 1
        ite = 0
        while (1):
            ite = ite + 1
            # low-rank update
            M0 = M
            Ut, St, Vt = torch.linalg.svd((M0 - S).reshape(nt, -1),
                                          full_matrices=False)  # torch.Size([28, 28]) torch.Size([28]) torch.Size([28, 147456])
            St = self.softThreshold(St, St[0] * self.lambda_L)
            St = torch.diag_embed(St).type(torch.complex64)
            L = Ut @ St @ Vt
            L = L.reshape(nt, 1, h, w)
            # sparse update - tv using clipping
            z = z + 0.25 * self.tv(M - L)
            z = torch.sgn(z) * torch.clamp(z.abs(), min=-self.lambda_S / 2, max=self.lambda_S / 2)
            S = M - L - self.adj_tv(z)
            # data consistency
            pred = L + S
            aah = []
            for i in range(self.kernel.shape[0]):
                aah.append(self.toep_op(pred[i:i + 1], self.kernel[i:i + 1], smaps=self.smaps, norm='ortho'))
            aah = torch.cat(aah, dim=0)
            res = (aah - self.recon_nufft)
            M = L + S - res
            # print cost function anf solution update
            cost = (res.abs() ** 2).sum() + self.lambda_L * St.abs().sum() + self.lambda_S * self.tv(S).abs().sum()
            update = torch.linalg.norm((M - M0).reshape(-1), 2) / torch.linalg.norm(M0.reshape(-1), 2)
            print(f' iter {ite}: cost: {cost}, update: {update}')
            # stopping criteria
            if update < gradToll or ite > nite:
                break
        return L, S

    def run(self, nround=1, nite=21, save_to_file=True, compare_matlab=False):
        '''
        run the L+S method. For reconstruction shape of (28,1,384,384), run time on A100: 4.5s, GPU usage: 6.4G
        :param nround: run the L+S algorithm nround times
        :param nite: the iterations of nonlinear conjugate gradient descent algorithm with backtracking line search in l+s algorithm
        :param save_to_file: save the zero-filled and L+S reconstructed images to nii.gz file
        :param compare_matlab: compare the python results with the original matlab code results
        :return: L+S reconstructed liver images, shape (28,1,384,384), pytorch tensor, type: torch.complex64
        '''
        print('Start L+S algorithm...')
        start_time = time.time()
        recon_cs = self.recon_nufft
        for i in range(nround):
            print(f'Round = {i}: ')
            L, S = self.IST(recon_cs, nite)  # shape (28,1,384,384)
            recon_cs = L + S
        end_time = time.time()
        print('### Done! Running time: %.2f s' % (end_time - start_time))
        if save_to_file:
            self.save_results(self.recon_nufft, 'DCE_nufft.nii.gz')
            self.save_results(L, 'DCE_L.nii.gz')
            self.save_results(S, 'DCE_S.nii.gz')
            self.save_results(recon_cs, 'DCE_L_S.nii.gz')
        if compare_matlab:
            # this code: NUFFT: PSNR: 62.86, SSIM: 0.9999;;; L+S: PSNR: PSNR: 40.04, SSIM: 0.9626
            self.compare_with_matlab_results(self.recon_nufft, recon_cs)
        return recon_cs


if __name__ == '__main__':
    with torch.no_grad():
        l_s_method = LplusS()
        recon_l_s = l_s_method.run(compare_matlab=True)
