from __future__ import division

import os, sys, logging, h5py, hdf5storage, datetime
import numpy as np

import torch
import torch.nn as nn
from skimage.metrics import structural_similarity


# ------------ LOSSES ------------

class Loss_MRAE(nn.Module):
    def __init__(self, eps=1e-4):
        super(Loss_MRAE, self).__init__()
        # set eps to 1e-4 or 1e-3 for reasonable outputs
        self.eps = eps

    def forward(self, outputs, label):
        # print(f"label.shape: {label.shape}, outputs.shape: {outputs.shape}")
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / (label + self.eps)
        mrae = torch.mean(error.reshape(-1))
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.reshape(-1)))
        return rmse

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=1):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)

class Loss_PSNR_modified(nn.Module):
    def __init__(self, data_range=1.0, eps=1e-8):
        """
        PSNR Loss Module.

        Args:
            data_range (float): The maximum data range of the input images.
            eps (float): Small constant for numerical stability.
        """
        super(Loss_PSNR_modified, self).__init__()
        self.data_range = data_range
        self.eps = eps

    def forward(self, im_true, im_fake):
        """
        Forward pass to compute PSNR.

        Args:
            im_true (torch.Tensor): Ground truth image, shape (N, C, H, W).
            im_fake (torch.Tensor): Predicted image, shape (N, C, H, W).

        Returns:
            torch.Tensor: Scalar PSNR loss.
        """
        assert im_true.shape == im_fake.shape, "Input shapes must match"

        # Clamp input values to valid range
        im_true = torch.clamp(im_true, min=0.0, max=1.0) * self.data_range
        im_fake = torch.clamp(im_fake, min=0.0, max=1.0) * self.data_range

        # Compute MSE loss
        mse = torch.mean((im_true - im_fake) ** 2, dim=(1, 2, 3))  # Per batch MSE
        mse = mse + self.eps  # Add epsilon for numerical stability

        # Compute PSNR
        psnr = 10.0 * torch.log10((self.data_range ** 2) / mse)

        # Return average PSNR across batch
        return torch.mean(psnr)

class Loss_SID(nn.Module):
    def __init__(self, eps=1e-3):
        super(Loss_SID, self).__init__()
        self.eps = eps

    def forward(self, tensor_pred, tensor_gt):
        assert tensor_pred.shape == tensor_gt.shape

        tensor_output = torch.clamp(tensor_pred, min=self.eps, max=1)
        tensor_gt = torch.clamp(tensor_gt, min=self.eps, max=1)

        # compute terms for sid
        a1 = tensor_output * torch.log10(tensor_output / tensor_gt)
        a2 = tensor_gt * torch.log10(tensor_gt / tensor_output)

        # sum over spatial dimensions
        a1_sum = a1.sum(dim=3).sum(dim=2)
        a2_sum = a2.sum(dim=3).sum(dim=2)

        # calculate sid loss
        sid = torch.mean(torch.abs(a1_sum + a2_sum))

        # check for nan
        if torch.isnan(sid): raise ValueError("NaN detected in SID loss computation.")

        return sid

class Loss_SID_modified(nn.Module):
    def __init__(self, eps=1e-3):
        """
        Spectral Information Divergence (SID) Loss.
        
        Args:
            eps (float): Small constant to prevent division by zero and log(0). Default is 1e-3.
        """
        super(Loss_SID_modified, self).__init__()
        self.eps = eps

    def forward(self, tensor_pred, tensor_gt):
        """
        Forward pass to compute SID loss.

        Args:
            tensor_pred (torch.Tensor): Predicted spectra, shape (N, C, H, W).
            tensor_gt (torch.Tensor): Ground truth spectra, shape (N, C, H, W).

        Returns:
            torch.Tensor: Scalar SID loss.
        """
        assert tensor_pred.shape == tensor_gt.shape, "Input shapes must match"
        
        # Normalize spectra to ensure they sum to 1
        tensor_pred = tensor_pred / tensor_pred.sum(dim=1, keepdim=True)
        tensor_gt = tensor_gt / tensor_gt.sum(dim=1, keepdim=True)

        # Clamp values to avoid log instability
        tensor_pred = torch.clamp(tensor_pred, min=self.eps, max=1 - self.eps)
        tensor_gt = torch.clamp(tensor_gt, min=self.eps, max=1 - self.eps)

        # Compute SID terms
        a1 = tensor_pred * torch.log(tensor_pred / tensor_gt)
        a2 = tensor_gt * torch.log(tensor_gt / tensor_pred)

        # Sum over spectral dimensions (channels) and spatial dimensions
        sid = (a1 + a2).sum(dim=1).mean()

        # Check for NaNs
        if torch.isnan(sid).any(): raise ValueError("NaN detected in SID loss computation.")

        return sid

class Loss_SID_optimized(nn.Module):
    def __init__(self, eps=1e-4):
        """
        Optimized Spectral Information Divergence (SID) Loss.
        
        Args:
            eps (float): Small constant to prevent division by zero and log(0). Default is 1e-4.
        """
        super(Loss_SID_optimized, self).__init__()
        # set eps to 1e-4 or 1e-3 for reasonable outputs
        self.eps = eps

    def forward(self, tensor_pred, tensor_gt):
        """
        Forward pass to compute SID loss.

        Args:
            tensor_pred (torch.Tensor): Predicted spectra, shape (N, C, H, W).
            tensor_gt (torch.Tensor): Ground truth spectra, shape (N, C, H, W).

        Returns:
            torch.Tensor: Scalar SID loss.
        """
        assert tensor_pred.shape == tensor_gt.shape, "Input shapes must match"
        
        # Normalize spectra to ensure they sum to 1
        tensor_pred = tensor_pred / (tensor_pred.sum(dim=1, keepdim=True) + self.eps)
        tensor_gt = tensor_gt / (tensor_gt.sum(dim=1, keepdim=True) + self.eps)

        # Clamp values to avoid log instability
        tensor_pred = torch.clamp(tensor_pred, min=self.eps, max=1 - self.eps)
        tensor_gt = torch.clamp(tensor_gt, min=self.eps, max=1 - self.eps)

        # Compute SID terms
        log_ratio = torch.log(tensor_pred / tensor_gt)
        a1 = tensor_pred * log_ratio
        a2 = tensor_gt * -log_ratio

        # Sum over spectral dimensions (channels) and spatial dimensions
        sid = (a1 + a2).sum(dim=1).mean()

        # Check for NaNs
        if torch.isnan(sid).any(): 
            raise ValueError("NaN detected in SID loss computation.")

        return sid

class Loss_SAM(nn.Module):
    def __init__(self, eps=1e-7):
        super(Loss_SAM, self).__init__()
        self.eps = eps

    def forward(self, tensor_pred, tensor_gt):
        assert tensor_pred.shape == tensor_gt.shape
        # inner product
        dot = torch.sum(tensor_pred * tensor_gt, dim=1).view(-1)
        # norm calculations
        image = tensor_pred.reshape(-1, tensor_pred.shape[1])
        norm_original = torch.norm(image, p=2, dim=1)

        target = tensor_gt.reshape(-1, tensor_gt.shape[1])
        norm_reconstructed = torch.norm(target, p=2, dim=1)

        norm_product = (norm_original.mul(norm_reconstructed)).pow(-1)
        argument = dot.mul(norm_product)
        # for avoiding arccos(1)
        acos = torch.acos(torch.clamp(argument, min=-1+self.eps, max=1-self.eps))
        loss = torch.mean(acos)

        if torch.isnan(loss): raise ValueError("NaN detected in SAM loss computation.")
        return loss

class Loss_SAM_modified(nn.Module):
    def __init__(self, eps=1e-7):
        """
        Spectral Angle Mapper (SAM) Loss.

        Args:
            eps (float): Small constant for numerical stability. Default is 1e-7.
        """
        super(Loss_SAM_modified, self).__init__()
        self.eps = eps

    def forward(self, tensor_pred, tensor_gt):
        """
        Forward pass to compute SAM loss.

        Args:
            tensor_pred (torch.Tensor): Predicted spectra, shape (N, C, H, W).
            tensor_gt (torch.Tensor): Ground truth spectra, shape (N, C, H, W).

        Returns:
            torch.Tensor: Scalar SAM loss.
        """
        assert tensor_pred.shape == tensor_gt.shape, "Input shapes must match"

        # Flatten spatial dimensions (preserve batch and channel)
        tensor_pred = tensor_pred.flatten(start_dim=2)
        tensor_gt = tensor_gt.flatten(start_dim=2)

        # Compute dot product
        dot = torch.sum(tensor_pred * tensor_gt, dim=1)

        # Compute norms with numerical stability
        norm_pred = torch.norm(tensor_pred, p=2, dim=1) + self.eps
        norm_gt = torch.norm(tensor_gt, p=2, dim=1) + self.eps

        # Compute the cosine similarity (clamped)
        cos_similarity = dot / (norm_pred * norm_gt)
        cos_similarity = torch.clamp(cos_similarity, min=-1 + self.eps, max=1 - self.eps)

        # Compute SAM (arccos of cosine similarity)
        sam = torch.acos(cos_similarity)

        # Mean SAM loss across batch
        loss = torch.mean(sam)

        # Check for NaNs
        if torch.isnan(loss).any(): raise ValueError("NaN detected in SAM loss computation.")

        return loss

class Loss_SAM_optimized(nn.Module):
    def __init__(self, eps=1e-7):
        """
        Optimized Spectral Angle Mapper (SAM) Loss.

        Args:
            eps (float): Small constant for numerical stability. Default is 1e-7.
        """
        super(Loss_SAM_optimized, self).__init__()
        self.eps = eps

    def forward(self, tensor_pred, tensor_gt):
        """
        Forward pass to compute SAM loss.

        Args:
            tensor_pred (torch.Tensor): Predicted spectra, shape (N, C, H, W).
            tensor_gt (torch.Tensor): Ground truth spectra, shape (N, C, H, W).

        Returns:
            torch.Tensor: Scalar SAM loss.
        """
        assert tensor_pred.shape == tensor_gt.shape, "Input shapes must match"

        # Flatten spatial dimensions (preserve batch and channel)
        tensor_pred = tensor_pred.flatten(start_dim=2)
        tensor_gt = tensor_gt.flatten(start_dim=2)

        # Compute dot product
        dot = torch.sum(tensor_pred * tensor_gt, dim=1)

        # Compute norms with numerical stability
        norm_pred = torch.norm(tensor_pred, p=2, dim=1) + self.eps
        norm_gt = torch.norm(tensor_gt, p=2, dim=1) + self.eps

        # Compute the cosine similarity (clamped)
        cos_similarity = dot / (norm_pred * norm_gt)
        cos_similarity = torch.clamp(cos_similarity, min=-1 + self.eps, max=1 - self.eps)

        # Compute SAM (arccos of cosine similarity)
        sam = torch.acos(cos_similarity)

        # Mean SAM loss across batch
        loss = torch.mean(sam)

        # Check for NaNs
        if torch.isnan(loss).any(): 
            raise ValueError("NaN detected in SAM loss computation.")

        return loss

class Loss_SSIM(nn.Module):
    def __init__(self, max_value=1.0, as_db=False):
        super(Loss_SSIM, self).__init__()
        self.max_value = max_value
        self.as_db = as_db

    def forward(self, img_pred, img_gt):
        # Convert to NumPy if inputs are PyTorch tensors
        if isinstance(img_pred, torch.Tensor):
            img_pred = img_pred.detach().cpu().numpy()
        if isinstance(img_gt, torch.Tensor):
            img_gt = img_gt.detach().cpu().numpy()

        # Determine window size dynamically based on image dimensions
        min_dim = min(img_pred.shape[-2], img_pred.shape[-1])
        win_size = min(7, min_dim) if min_dim < 7 else 7

        # Calculate SSIM
        ssim = structural_similarity(img_gt, img_pred, data_range=self.max_value, channel_axis=0, win_size=win_size)

        # Return SSIM in dB if requested
        # calculate the structural simularity index measure in decibels (NumPy - Test Error)
        if self.as_db:
            ssim_db = -10 * np.log10(1 - ssim)
            return ssim_db

        return ssim
