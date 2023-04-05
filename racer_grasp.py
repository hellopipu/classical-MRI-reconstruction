'''
This package demo demonstrates the reconstruction methods RACER-GRASP
as described in the following paper:

"RACER-GRASP: Respiratory-weighted, aortic contrast enhancement-guided
 and coil-unstreaking golden-angle radial sparse MRI"
Feng L, Huang C, Shanbhogue K, Sodickson DK, Chandarana H, Otazo R.
Magn Reson Med. 2018 Jul;80(1):77-89. doi: 10.1002/mrm.27002. Epub 2017 Nov 28.

This package is released for academic use only!
If any component in this package is helpful for your research, please cite our paper

For any further help or information, please contact
Li Feng, PhD
fenglibme@gmail.com
01/10/2019

reimplement in python by Bingyu Xin, Rutgers, 2023
'''

import os
import time
import numpy as np
import torch
import torchkbnufft as tkbn
import mat73
from scipy.io.matlab import loadmat
from scipy.linalg import eigh
import peakutils
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import urllib.request


class Base():
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.output_dir = 'output/racer_grasp'
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def compare_with_matlab_results(self, recon_nufft, recon_cs, show_img=False):
        print('### Compared with original matlab code result: ')
        data_path_cs = 'data/racer_grasp/liver_recon_cs.mat'
        data_path_nufft = 'data/racer_grasp/liver_nufft_rec.mat'

        folder = os.path.dirname(data_path_cs)

        if not os.path.isfile(data_path_cs):
            print('Download the matlab reconstruction data...')
            os.makedirs(folder, exist_ok=True)
            url_nufft = 'https://rutgers.box.com/shared/static/7iiqcefjvbhdibkfius70t17gcg4fpw7.mat'
            url_cs = 'https://rutgers.box.com/shared/static/dyhjuegbpa9jkajxbuld3dolhulr3v9p.mat'
            urllib.request.urlretrieve(url_cs, data_path_cs)
            urllib.request.urlretrieve(url_nufft, data_path_nufft)
        else:
            print('The matlab reconstruction data exists.')

        recon_racer_grasp_matlab = loadmat(data_path_cs)['recon_cs']
        recon_nufft_matlab = loadmat(data_path_nufft)['recon_nufft']

        print('- NUFFT RECONSTRUCTION : ')
        self.cal_metric(recon_nufft_matlab, recon_nufft.squeeze().permute(1, 2, 0).cpu().numpy())
        print('- GRASP RECONSTRUCTION : ')
        self.cal_metric(recon_racer_grasp_matlab, recon_cs.squeeze().permute(1, 2, 0).cpu().numpy())

        if show_img:
            import matplotlib.pyplot as plt
            _slice = 10
            plt.subplot(221)
            plt.imshow(recon_cs[_slice, 0,].abs().cpu().numpy(), 'gray')  # grasp
            plt.subplot(222)
            plt.imshow(np.abs(recon_grasp_matlab[:, :, _slice]), 'gray')  # grasp matlab
            plt.subplot(223)
            plt.imshow(recon_nufft[_slice, 0].abs().cpu().numpy(), 'gray')  # nufft
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
        img_sitk = sitk.GetImageFromArray(txy)
        sitk.WriteImage(img_sitk, filename)
        print(f'### Reconstucted image saved to {filename}')

    def prepare_for_racer_grasp(self):
        kdata_path = 'data/racer_grasp/kdata.mat'
        traj_path = 'data/racer_grasp/Trajectory.mat'
        zip_path = 'data/racer_grasp/ZIP.mat'
        b1_path = 'data/racer_grasp/b1.mat'
        folder = os.path.dirname(kdata_path)
        if not os.path.isfile(b1_path):
            os.makedirs(folder, exist_ok=True)
            import urllib.request
            print(f'Download the liver data to {folder}...')
            url1 = 'https://rutgers.box.com/shared/static/i205mx3mqb8z75adf1r86ob3ef3gi3ri.mat'
            urllib.request.urlretrieve(url1, kdata_path)
            url2 = 'https://rutgers.box.com/shared/static/06f87yq4aojx5z2lo9hlaixocjqrf1rs.mat'
            urllib.request.urlretrieve(url2, traj_path)
            url3 = 'https://rutgers.box.com/shared/static/79fq3h80ad0lk3f63af8zhdxdtm5kp7c.mat'
            urllib.request.urlretrieve(url3, zip_path)
            url4 = 'https://rutgers.box.com/shared/static/ea1jgvgtz6r16de3ivsffcanm81qr4pz.mat'
            urllib.request.urlretrieve(url4, b1_path)
        else:
            print(f'The liver data exists in {folder}')

        kdata = mat73.loadmat(kdata_path)['kdata'] # shape: (512, 1144, 12, 20)
        data= loadmat(traj_path)
        traj, dcp = data['Traj'], data['DensityComp']
        kc = loadmat(zip_path)['ZIP'] # center of kspace, shape: (38, 1144, 20)
        smaps = loadmat(b1_path)['b1'] # 256,256,12,8

        return self.process_data((kdata, traj, dcp,kc,smaps))

    def process_data(self, data):
        kdata, traj, dcp, kc, smaps = data
        kdata, traj, dcp, kc, smaps = kdata.astype(np.complex64), traj.astype(np.complex64), dcp.astype(np.float32), \
                                kc.astype(np.complex64), smaps.astype(np.complex64)
        # norm smaps, from ESPIRiT paper
        smaps = smaps / np.sum(np.abs(smaps) ** 2, axis=-1, keepdims=True) ** 0.5  # h,w,coil

        output_shape = smaps.shape[0]
        self.oshape = output_shape

        # RACER-GRASP data processing
        kdata_Unstreaking_CoilCompression = self.unstreaking(kdata, traj, dcp)
        res_signal = self.resp_motion_detection(kc)
        ace_signal = self.bolus_tracking(kdata_Unstreaking_CoilCompression, traj, smaps)
        kdata, traj, dcp, smaps, soft_weight = self.data_sorting(kdata_Unstreaking_CoilCompression, traj, dcp, smaps, res_signal)

        # to torch tensor
        op = tkbn.KbNufft((output_shape, output_shape)).to(self.device)
        op_adj = tkbn.KbNufftAdjoint((output_shape, output_shape)).to(self.device)
        toep_op = tkbn.ToepNufft().to(self.device)

        nx, nt, nline, nc = kdata.shape
        kdata = torch.tensor(kdata.transpose(1,3,0,2).reshape(nt,nc,nx*nline)).to(self.device) # nt, nc, nx*nline
        traj = torch.tensor(2*np.pi*traj.transpose(1,0,2).reshape( nt,nx*nline))
        traj = torch.view_as_real(traj).permute(0,2,1).to(self.device) # nt, 2, nx*nline
        dcp = torch.tensor(dcp.transpose(1, 0, 2).reshape(nt, 1,nx*nline)).to(self.device) # nt, 1, nx*nline
        smaps = torch.tensor(smaps[None].transpose(0, 3, 1, 2)).to(self.device) # 1, nc, h, w

        soft_weight = soft_weight.reshape(1,1,-1).to(self.device) # 1,1,nx*nline
        kernel = tkbn.calc_toeplitz_kernel(traj, im_size=(output_shape, output_shape), weights=dcp*soft_weight, norm='ortho')

        # print('debug: ', kdata.dtype, traj.dtype, dcp.dtype, smaps.dtype, soft_weight.dtype, kernel.dtype)
        # nufft recon
        recon_nufft = op_adj(kdata * dcp, traj, smaps=smaps, norm='ortho')  # shape (11,1,256,256)
        recon_nufft_2 = op_adj(kdata * dcp * soft_weight, traj, smaps=smaps, norm='ortho')  # shape (11,1,256,256)

        # l1 reg, (Regularization parameters were empirically selected)
        _lambda = 0.03 * recon_nufft.abs().max()

        return op, toep_op, kdata, traj, dcp, smaps, kernel, recon_nufft, recon_nufft_2, _lambda, res_signal, soft_weight


class RespMotionDetection(Base):
    def __init__(self):
        super(RespMotionDetection, self).__init__()

    def coilClustering(self, d1, thresh):
        '''
        automatically determine the dominant motion for dense coil arrays.
        Zhang, Tao, et al. "Robust self‐navigated body MRI using dense coil arrays." Magnetic resonance in medicine 76.1 (2016): 197-205.
        :param d1: navigator signal
        :param thresh: correlation threshold
        :return: dave: average navigator signal within the cluster
                coilID: mask indicating whether the coil element is selected in the cluster
        '''
        nviews, nc = d1.shape
        corrm = np.zeros((nc, nc))
        for i in range(nc):
            for j in range(nc):
                corrm[i, j] = np.corrcoef(d1[:, i], d1[:, j])[0, 1]
                corrm[j, i] = corrm[i, j]
        mask = np.zeros_like(corrm)
        mask[np.abs(corrm) > thresh] = 1

        u, s, vh = np.linalg.svd(mask.astype(np.float64), full_matrices=False)
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

    def projNorm(self, ZIP):
        nx, ntViews, nc = ZIP.shape
        # projection normalization
        maxprof = np.max(ZIP, axis=0, keepdims=True)
        minprof = np.min(ZIP, axis=0, keepdims=True)
        ZIP = (ZIP - minprof) / (maxprof - minprof)
        # smoothing
        for i in range(nx):
            for j in range(nc):
                ZIP[i, :, j] = savgol_filter(ZIP[i, :, j], 10, 2)
                # ZIP[i, :, j] = lowess(ZIP[i, :, j], range(ntViews), frac=6 / ntViews, return_sorted=False)
        return ZIP

    def montion_detection_step1(self, ZIP,n, save_fig=True):

        nx, ntViews, nc = ZIP.shape
        # Do PCA along each coil element and to get the first two PCs from each coil, as described in the paper
        PCs = np.zeros((n, 2 * nc))
        for i in range(nc):
            SI = ZIP[:, -n:, i]  # 400,800
            covariance = np.cov(SI)
            V, PC = np.linalg.eigh(covariance, 'U')
            V = V[::-1]
            PC = PC[:, ::-1]  # 400,400
            SI = (SI.T @ PC)

            for j in range(2):
                tmp = savgol_filter(SI[:, j], 10, 2)  # smooth(PC,j) #lowess_smooth(PC[:,j],6)
                tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
                PCs[:, i * 2 + j] = tmp
        thresh = 0.97
        res_signal, cluster = self.coilClustering(PCs, thresh)
        res_signal_post = (res_signal - res_signal.min()) / (res_signal.max() - res_signal.min())
        # find the "good" coil elements used for estimation of all motion later
        cluster = np.abs(cluster[::2] + cluster[1::2])
        coil = np.where(cluster)[0]
        print(
            f'-- Motion detection step 1: find the coil elements with good representation of respiratory motion: {coil}')
        if save_fig:
            plt.figure(figsize=[20, 10])
            plt.imshow(np.abs(ZIP[:, -n:, 14]), 'gray')
            plt.plot(-res_signal_post * 100 + 200,'r')
            plt.xticks([]), plt.yticks([]), plt.title('motion signal: Step 1')
            plt.savefig(os.path.join(self.output_dir, 'motion_signal_step1.png'))
        return coil, res_signal_post

    def montion_detection_step2(self, ZIP, coil, n, res_signal_post, save_fig=True):
        print('-- Motion detection step 2: estimate motion signal using PCA from the selected coil elements')
        nx, ntViews, nc = ZIP.shape
        # Do PCA along all the selected coils concatated together
        ZIP1 = ZIP[..., coil]
        SI = np.abs(ZIP1.transpose(0, 2, 1).reshape(-1, ntViews))
        covariance = np.cov(SI)
        V, PC = np.linalg.eigh(covariance, 'U')
        V = V[::-1]  # 1200
        PC = PC[:, ::-1]
        SI = SI.T @ PC
        npc = 5 # Consider the first 5 principal component only
        SI=SI[:,:npc]
        # Do some smoothing
        for i in range(npc):
            SI[:, i] = savgol_filter(SI[:, i], 10, 2)

        # check the correlation with the motion signal detected in the first step
        # calculate the correlation for the late phase only
        corrm = np.zeros((npc))
        for ch in range(npc):
            x1 = res_signal_post - np.mean(res_signal_post)
            x2 = SI[-n:, ch] - np.mean(SI[-n:, ch])
            cross_cov = np.correlate(x1, x2, mode='valid') / (np.std(x1) * np.std(x2) * len(x1))
            corrm[ch] = cross_cov[0]

            # corrm[ch] = np.correlate(res_signal_post, SI[-n:, ch], mode='valid')[0] / (
            #         np.linalg.norm(res_signal_post) * np.linalg.norm(SI[-n:, ch]))
        corrm_abs = np.abs(corrm)
        coil_index_PCA = np.argmax(corrm_abs)
        res_signal = SI[:, coil_index_PCA]

        # flip the signal if the corrlation coefficient is negative
        if corrm[coil_index_PCA] != corrm_abs[coil_index_PCA]:
            res_signal = -res_signal
        res_signal = (res_signal - res_signal.min()) / (res_signal.max() - res_signal.min())

        if save_fig:
            plt.figure(figsize=[20, 5])
            plt.imshow(np.abs(ZIP[:,:,14]),'gray')
            plt.plot(-res_signal * 100 + 200, 'r')
            plt.xticks([]), plt.yticks([]), plt.title('motion signal: Step 2')
            plt.savefig(os.path.join(self.output_dir, 'motion_signal_step2.png'))
        return SI, corrm, res_signal, ZIP1

    def motion_detection_step3(self, res_signal, ZIP1, save_fig=True):
        print('-- Motion detection step3: estimate the envelop of the signal and substract it')
        # estimate the peak positions
        peak_indices = peakutils.indexes(res_signal, thres=0.5, min_dist=6)
        # check whether the peak positions are reasonable or not (cancel the peaks that are not local maximum)
        II = 6
        peak_indices_new = []
        for i in range(len(peak_indices)):
            t1 = res_signal[peak_indices[i]]<=res_signal[max(peak_indices[i]-II,0):peak_indices[i]]
            t2 = res_signal[peak_indices[i]]<=res_signal[peak_indices[i]+1:min(peak_indices[i]+II+1,len(res_signal))]
            if t1.sum()+t2.sum()==0:
                peak_indices_new.append(peak_indices[i])
        peak_indices = peak_indices_new
        peaks = res_signal[peak_indices]
        # Do a fitting and substract the fitted signal
        smoothing_spline = UnivariateSpline(peak_indices, res_signal[peak_indices], s=0.015)
        ftmax = smoothing_spline(np.arange(len(res_signal)))
        res_signal = res_signal - ftmax
        peaks = peaks-ftmax[peak_indices]
        if save_fig:
            plt.figure(figsize=[20, 5])
            plt.imshow(np.abs(ZIP1[:, :, 0]), 'gray')
            plt.plot(-res_signal *150+70, 'r')
            plt.plot(peak_indices, -peaks *150+70, 'bo')
            plt.xticks([]), plt.yticks([]), plt.title('motion signal: Step 3')
            plt.savefig(os.path.join(self.output_dir, 'motion_signal_step3.png'))
        return res_signal

    def resp_motion_detection(self, kc, save_fig=True):
        '''
        the automatic detection of a respiratory motion signal
        :return:
        '''
        # ZIP: Projection profiles along the Z dimension with interpolation.
        kc = kc.transpose(1, 0, 2)  # (nx, ntViews, nc)
        ZIP = np.abs(np.fft.fftshift(np.fft.ifft(kc, n=400, axis=0, norm='ortho'), axes=0))
        # Normalization of each projection in each coil element
        ZIP = self.projNorm(ZIP)  # Normalization includes temporal smoothing
        if save_fig:
            # plot the projection profiles
            coil_idx = 14
            plt.figure(figsize=[20, 5])
            plt.imshow(np.abs(ZIP[:, :, coil_idx]), 'gray')
            plt.xticks([]), plt.yticks([]), plt.title('Projection profile')
            plt.savefig(os.path.join(self.output_dir, 'projection_profile.png'))

        # There are 3 steps to generate a respiratory motion signal, as shown below
        # the last 800 spokes were used as the late enhancement phase for motion detection as described in the paper
        n1 = 800
        ## STEP 1: find the coil elements with good representation of respiratory motion from the late enhancement spokes
        coil, res_signal_post = self.montion_detection_step1(ZIP, n1)
        ## STEP 2: Estimate motion signal using PCA from the concatated coil elements.
        ## Those coil elements were selected in the first step
        SI, corrm, res_signal, ZIP1 = self.montion_detection_step2(ZIP, coil, n1, res_signal_post)
        ## Step 3: You noticed that the signal is not flat, due to the contrast
        ## injection. So, now let's estimate the envelop of the signal and substract it
        res_signal = self.motion_detection_step3(res_signal, ZIP1)

        return res_signal


class DataSorting(RespMotionDetection):
    def __init__(self):
        super(DataSorting, self).__init__()

    def recon_for_unstreaking(self, op_adj, kdata, traj, dcp, n):
        nx, ntViews, nz, nc = kdata.shape
        # select the last n spokes
        kdata = kdata[:, -n:].reshape(-1, nz, nc).permute(1, 2, 0)  # nz, nc, n*nx
        traj = traj[:, -n:].reshape(1, -1, 2).permute(0, 2, 1)  # 1, 2, n*nx
        dcp = dcp[:, -n:].reshape(1, 1, -1)  # 1, 1, n*nx
        recon_nufft = op_adj(kdata * dcp, traj, smaps=None, norm='ortho')  # nz, nc, oshape, oshape
        return recon_nufft.abs()

    def unstreaking(self, kdata, traj, dcp, streaking_ratio_threshold= 1.3, save_fig=True):
        '''
        The calculation of streak ratio of each coil element and the unstreaking process for this sample data

        the concept of streak raio was introduced in the paper:
        Xue, Yiqun, et al. "Automatic coil selection for streak artifact reduction in radial MRI." MRM 2012
        :return:
        '''
        # threshold for streaking ratio was empirically set as 1.3 in the paper
        print(f'-- Unstreaking kdata based on the streak ratio of each coil element, threshold={streaking_ratio_threshold}...')
        kdata = torch.tensor(kdata).to(self.device) # 512, 1144, 12, 20
        traj = torch.tensor(2*np.pi*traj).to(self.device)
        traj = torch.view_as_real(traj) # 512, 1144, 2
        dcp = torch.tensor(dcp).to(self.device) # 512, 1144

        n1 = 1000  # using 1000 spokes to generate artifact free images
        n2 = 40  # using 40 spokes to generate images with streaking artifacts
        nx, ntViews, nz, nc = kdata.shape  # 512, 1144, 12, 20
        # Note that the third dimension is z, NOT kz. (A FFT was performed already)
        op_adj = tkbn.KbNufftAdjoint((nx, nx)).to(self.device)  # 2x FOV
        Ref = self.recon_for_unstreaking(op_adj, kdata, traj, dcp, n1)  # artifact-free image (nz, nc, oshape, oshape)
        # why *n1/n2? scaling to have the same scale as the Ref image
        Img = self.recon_for_unstreaking(op_adj, kdata, traj, dcp,
                                         n2) * n1 / n2  # image with streaks (nz, nc, oshape, oshape)

        # As described in the paper, the Diff image is calculated as the 2x FOV
        Diff = (Ref - Img).abs()

        # The Ref and Img are then cropped to the 1x FOV
        Ref = Ref[:, :, Ref.shape[2] // 4:Ref.shape[2] * 3 // 4, Ref.shape[3] // 4:Ref.shape[3] * 3 // 4]
        Img = Img[:, :, Img.shape[2] // 4:Img.shape[2] * 3 // 4, Img.shape[3] // 4:Img.shape[3] * 3 // 4]

        Img_sos = torch.sqrt((Img ** 2).sum(dim=1))  # the square root of sum of squares of the coil elements

        # calculating the streak ratio, and normalize
        StreakRatio = torch.linalg.vector_norm(Diff, dim=[0, 2, 3]) / torch.linalg.vector_norm(Ref, dim=[0, 2, 3])
        StreakRatio = StreakRatio / StreakRatio.min()

        if save_fig:
            # plot the streak ratio for each coil element
            plt.figure()
            plt.plot(StreakRatio.cpu().numpy())
            plt.axhline(y=streaking_ratio_threshold, color='r', linestyle='--', label='threshold=1.3')
            plt.title('Streak ratio for each coil element');
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, 'streak_ratio.png'))
        # find the coil elements whose streak ratio greater than 1.3 (empirically set in the paper)
        # unstreaking is performed only for these coils, as described in the paper
        StreakRatio[StreakRatio <= streaking_ratio_threshold] = 1.

        # Do the unstreaking
        kdata = kdata / StreakRatio[None, None, None, :]

        # reconstruct images again for comparison
        op_adj = tkbn.KbNufftAdjoint((nx // 2, nx // 2)).to(self.device)  # 1x FOV
        # why /2? scaling to have the same scale as 2x FOV image
        Img = self.recon_for_unstreaking(op_adj, kdata, traj, dcp,
                                         n2) * n1 / n2 / 2  # image with streaks (nz, nc, oshape, oshape)

        Img_unstreaked_sos = torch.sqrt((Img ** 2).sum(dim=1))  # the square root of sum of squares of the coil elements
        # Note that the images displayed below are before iterative reconstruction
        # Thus, they both have strearking artifacts.
        # However, note that the right image has significantly less streaks
        norm = max(Img_sos.max(), Img_unstreaked_sos.max())
        Img_sos = Img_sos / norm
        Img_unstreaked_sos = Img_unstreaked_sos / norm
        if save_fig:
            # compare the images before and after unstreaking
            show_slice = 7
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.title('Before unstreaking')
            plt.imshow(Img_sos[show_slice].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            plt.subplot(1, 2, 2)
            plt.title('After unstreaking')
            plt.imshow(Img_unstreaked_sos[show_slice].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            plt.savefig(os.path.join(self.output_dir, 'unstreaking.png'))

        # Now compress the unstreaked kpsace kdata
        ncc = 8  # compress to 8 coils
        D = kdata.reshape(-1, nc) #.type(torch.complex128)
        U, S, V = torch.linalg.svd(D, full_matrices=False)
        kdata = (D @ V[:, :ncc]).reshape(nx, ntViews, nz, ncc).type(torch.complex64).cpu().numpy()

        # Note that the SVD implementation in pytorch differs from MATLAB, which may cause discrepancies in the smaps(pre-loaded) and kdata.
        # Therefore, we will use the compressed kspace data generated from MATLAB in the following steps.
        # TODO: generate the compressed smaps by myself instead of using the pre-loaded one.
        kdata_compressed_path = 'data/racer_grasp/kdata_Unstreaking_CoilCompression.mat'
        if not os.path.isfile(kdata_compressed_path):
            url1 = 'https://rutgers.box.com/shared/static/b56zombyahki6rc5ojneqk8g30ubv5it.mat'
            urllib.request.urlretrieve(url1, kdata_compressed_path)
        kdata= loadmat(kdata_compressed_path)['kdata'].astype(np.complex64)
        return kdata

    def bolus_tracking(self,kdata, traj, smaps, save_fig=True):
        '''
        TODO: not implemented yet
        :param kdata:
        :param traj:
        :param smaps:
        :return:
        '''
        print('-- Bolus tracking skipped (not implemented yet)...')
        ace_signal = None
        return ace_signal

    def data_sorting(self, kdata, traj, dcp, smaps, res_signal, ace_signal=None):
        # total acquisition time, this information was obtained from the raw data header file.
        TA = 178
        # reconstruct only 1 slice
        kdata = kdata[:, :, 0, :]
        smaps = smaps[:, :, 0, :]
        nx, ntviews, nc = kdata.shape # 512, 1144, 8

        tempRes = 15
        ntres=4
        nline = int(tempRes//(TA/ntviews)) # num of spokes in each contrast phase, 96
        if nline%ntres != 0:
            nline = nline - nline%ntres + ntres
        nline2 = nline//ntres # num of spokes in each resp phase, 24
        nt = ntviews//nline # 11

        weight = [4**(-i) for i in range(ntres)]
        # data sorting
        traj = traj[:, :nt*nline].reshape(nx,nt,nline) # 512, 11, 96
        dcp = dcp[:, :nt*nline].reshape(nx,nt,nline) # 512, 11, 96
        kdata = kdata[:, :nt*nline, :].reshape(nx,nt,nline,nc) # 512,11, 96, 8
        res_signal_tmp = res_signal[:nt*nline].reshape( nt,nline) # 11, 96

        # sorting according to the respiratory motion signal
        _index = np.argsort(res_signal_tmp, axis=1)[:,::-1]
        for i in range(nt):
            kdata[:,i] = kdata[:,i, _index[i]]
            dcp[:, i] = dcp[:, i, _index[i]]
            traj[:,i] = traj[:, i, _index[i]]

        soft_weight = torch.zeros((nx,nline))
        for j in range(ntres):
            soft_weight[:,j*nline2:(j+1)*nline2] = weight[j]

        return kdata, traj, dcp, smaps, soft_weight



class RACER_GRASP(DataSorting):
    def __init__(self):
        super(RACER_GRASP, self).__init__()
        self.op, self.toep_op, self.kdata, self.traj, self.dcp, self.smaps, self.kernel, \
        self.recon_nufft, self.recon_nufft_2, self._lambda, self.res_signal, self.soft_weight = self.prepare_for_racer_grasp()

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

        L2Grad = 2 * (aah - self.recon_nufft_2)
        # L1 norm part
        w = self.tv(x)
        L1Grad = self.adj_tv(w * (w.abs() ** 2 + _l1Smooth) ** -0.5)

        # composite gradient
        g = L2Grad + self._lambda * L1Grad
        return g

    def objective(self, x, dx, t, _l1Smooth=1e-15):
        x_ = x + t * dx
        # L2-norm part
        w = (self.op(x_, self.traj, smaps=self.smaps, norm='ortho') - self.kdata) * self.dcp ** 0.5 * self.soft_weight
        L2Obj = (w.abs() ** 2).sum()

        # L1-norm part
        w = self.tv(x_)
        L1Obj = ((w.abs() ** 2 + _l1Smooth) ** 0.5).sum()
        # objective function
        res = L2Obj + self._lambda * L1Obj
        return res

    def CSL1NlCg(self, x0, max_iter, gradToll=1e-8, maxlsiter=6, alpha=0.01, beta=0.6, print_detail=False):
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

    def run(self, nround=3, nite=6, save_to_file=True, compare_matlab=False):
        '''
        run the RACER-GRASP method. For reconstruction shape of (11,1,256,256) (96spokes/slice), run time on A100: 9, GPU usage: 14G
        The iterative reconstruction part is a respiratory soft-weighted GRASP method.
        :param nround: run the grasp algorithm nround times
        :param nite: the iterations of nonlinear conjugate gradient descent algorithm with backtracking line search in grasp algorithm
        :param save_to_file: save the zero-filled and grasp reconstructed images to nii.gz file
        :param compare_matlab: compare the python results with the original matlab code results
        :return: RACER-GRASP reconstructed liver images, shape (11,1,256,256), pytorch tensor, type: torch.complex64
        '''
        print('Start RACER-GRASP algorithm...')
        start_time = time.time()
        recon_cs = self.recon_nufft
        for i in range(nround):
            print(f'Round = {i}: ')
            recon_cs = self.CSL1NlCg(recon_cs, nite)  # shape (11,1,256,256
        end_time = time.time()
        print('### Done! Running time: %.2f s' % (end_time - start_time))
        if save_to_file:
            self.save_results(self.recon_nufft, 'output/racer_grasp/liver_nufft.nii.gz')
            self.save_results(recon_cs, 'output/racer_grasp/liver_racer_grasp.nii.gz')
        if compare_matlab:
            # this code: NUFFT: PSNR: 69.63, SSIM: 1.0000;;; RACER-GRASP: PSNR: 38.46, SSIM: 0.9692
            self.compare_with_matlab_results(self.recon_nufft, recon_cs)
        return recon_cs


if __name__ == '__main__':
    with torch.no_grad():
        racer_grasp_method = RACER_GRASP()
        recon_grasp = racer_grasp_method.run()
