# circular correlation between two tensors

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
import matplotlib.pyplot as plt

# compute the max correlation value and the corresponding circular shift 
# calculate the correlation and relative rotation between a and b (from a to b, point cloud transformation) using fft method

class circorr2(nn.Module):

    '''
    Circularly Correlates two images. Images must be of the same size.
    '''

    def __init__(self, is_circular=True, zero_mean_normalize=False):
        super(circorr2, self).__init__()
        self.InstanceNorm = nn.InstanceNorm2d(1, affine=False, track_running_stats=False)
        self.is_circular = is_circular
        self.zero_mean_normalize = zero_mean_normalize


    def forward(self, x1, x2, scale=1):
        """
        Args:
            x1 (torch.Tensor): Batch of Img1 of dimensions [B, C, H, W].
            x2 (torch.Tensor): Batch of Img2 of dimensions [B, C, H, W].
        Returns:
            scores (torch.Tensor): The circular cross correlation scores for the pairs. The output shape is [B, 1].
        """
        
        assert x1.shape[2:] == x2.shape[2:], "Images must be of the same size."

        if self.is_circular:
            corr, scores, shifts = self.circular_corr(x1, x2, scale)
            return corr, scores, shifts

        else:
            scores = self.corr(x1, x2)   
            return scores


    def corr(self, x1, x2):
        b, c, h, w = x2.shape

        if self.zero_mean_normalize:
            x1 = self.InstanceNorm(x1)
            x2 = self.InstanceNorm(x2)
        else:
            x1 = F.normalize(x1, dim=(-2,-1))
            x2 = F.normalize(x2, dim=(-2,-1))

        scores = torch.matmul(x1.view(b, 1, c*h*w), x2.view(b, c*h*w, 1))
        scores /= (h*w*c)
        
        return scores


    def circular_corr(self, x1, x2, scale=1):
        b, c, h, w = x2.shape

        if self.zero_mean_normalize:
            x1 = self.InstanceNorm(x1)
            x2 = self.InstanceNorm(x2)
        else:
            x1 = F.normalize(x1, dim=(-2,-1))
            x2 = F.normalize(x2, dim=(-2,-1))

        a_fft = torch.fft.fft2(x1, dim=-2, norm="ortho")
        b_fft = torch.fft.fft2(x2, dim=-2, norm="ortho")
        corr = torch.fft.ifft2(a_fft*b_fft.conj(), dim=-2, norm="ortho")  
        corr = torch.sqrt(corr.real**2 + corr.imag**2 + 1e-15)
        # sum the correlation over the channels
        corr = torch.sum(corr, dim=-3)
        # sum the correlation over the width
        corr = torch.sum(corr, dim=-1)
        # self.plot_corr(corr[0].detach().cpu().numpy(), 'corr.png')
        if scale != 1:
            # apply curve smoothing
            corr = self.curve_smoothing(corr, scale)
            # self.plot_corr(corr[0].detach().cpu().numpy(), 'corr_smooth.png')
        # find the max correlation value as score and the corresponding circular shift
        score = torch.max(corr, dim=-1)[0]/(0.15*c*h*w)
        # shift the correlation to the center 
        corr = torch.fft.fftshift(corr, dim=-1)
        shift = corr.shape[-1]//2 - torch.argmax(corr, dim=-1)
        return corr, score, shift


    def curve_smoothing(self, F, S):
        # width of the cross-correlation curve F
        B = F.shape[0]
        W = F.shape[1]
        # perform FFT to get the frequency domain representation of the curve
        F = torch.fft.fft2(F, dim=-1, norm="ortho")
        # zero-pad the curve
        padded_F = torch.cat((F[:,:int(W/2)], torch.zeros(B,(S-1)*W).to(F.device), F[:,int(W/2):]), dim=-1)
        # perform inverse FFT to get the smoothed curve
        smoothed_F = torch.fft.ifft2(padded_F, dim=-1, norm="ortho")
        # take the real part of the smoothed curve
        # smoothed_F_mag = torch.sqrt(smoothed_F.real**2 + smoothed_F.imag**2 + 1e-15)
        smoothed_F = smoothed_F.real
        return smoothed_F


    def plot_corr(self, a, save_path=None):
        # plot the correlation curve
        plt.plot(a)
        # plt.show()
        if save_path is not None:
           plt.savefig(save_path)
        plt.close()
        
        
if __name__ == "__main__":
    BATCH_SIZE = 10
    NUM_CHANNELS = 3
    IMG_H = 64
    IMG_W = 64
    x1 = torch.randn(BATCH_SIZE, NUM_CHANNELS, IMG_H, IMG_W)
    x2 = torch.randn(BATCH_SIZE, NUM_CHANNELS, IMG_H, IMG_W)
    circorr = circorr2()
    scores,shifts = circorr(x1, x2)
    print('output shape', scores.shape, shifts.shape)
    print('scores', scores)
    print('shifts', shifts)