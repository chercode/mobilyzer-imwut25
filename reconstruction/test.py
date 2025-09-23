from __future__ import division
from skimage.color import rgb2lab, deltaE_cie76

import os, sys, argparse, time, datetime
import numpy as np
import torch.backends
from tqdm import tqdm

import torch
from torchsummary import summary
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from architecture import *
from hsi_dataset_mobilyzer import TestDataset, ValidDataset
import utils, losses
from utils import AverageMeter

import losses
import utils_truthful as tru

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def str2bool(v):
    return v.lower() in ('true', '1', 'yes')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lossfunctions_considered = ["MRAE", "SAM", "SID"]

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument("--method", type=str, default="mst_plus_plus")
parser.add_argument("--compressed", type=str2bool, default="True")
parser.add_argument('--truthful', type=str2bool, default="True")
parser.add_argument('--norm', type=str2bool, default="False")
parser.add_argument('--save_mat', type=str2bool, default="True")
parser.add_argument('--cmf', type=str, default="average")
parser.add_argument('--nir_mean', type=int, default=940)
parser.add_argument("--bands", type=int, default=68)
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--patch_size", type=int, default=64)
parser.add_argument("--gpu_id", type=str, default="0")

parser.add_argument("--data_root", type=str, default="/media/sma318/TOSHIBA EXT/dataset_open/data/evoo/HSI")
parser.add_argument("--split_root", type=str, default="/media/sma318/TOSHIBA EXT/dataset_open/data/evoo/HSI")
parser.add_argument("--outf", type=str, default="/local-scratch/MobiTru/exp/exp_test/", help="path log files")
parser.add_argument("--pretrained_model_path", type=str, default="../models/HSI/dataset_open/2025_09_23_14_44_18/net_10epoch.pth")



opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.benchmark = True

compressed = opt.compressed

chosen_list = "test"

bands = opt.bands
if opt.truthful: bands -= 4

cmf = opt.cmf
nir_mean = opt.nir_mean
S = np.load(f"./helper_matrices/S_{cmf}_{nir_mean}.npy")
B = np.load(f"./helper_matrices/B_{cmf}_{nir_mean}.npy")
PS = np.load(f"./helper_matrices/PS_{cmf}_{nir_mean}.npy")
PB = np.load(f"./helper_matrices/PB_{cmf}_{nir_mean}.npy")

var_name = "cube"
with open(f"{opt.data_root}/split_txt/test_list.txt", "r") as fin:
    hyper_list = [line.replace('\n', '.mat') for line in fin]
hyper_list.sort()

print(f"len HS list: {len(hyper_list)}")
print(f"HS list: {hyper_list}")

print("\nloading dataset ...")
test_data = TestDataset(data_root=opt.data_root, split_root=opt.split_root, bgr2rgb=True, truthful=opt.truthful, S=S, norm=opt.norm)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
print("Test set samples: ", len(test_data))

criterion_mrae = losses.Loss_MRAE()
criterion_rmse = losses.Loss_RMSE()
criterion_sam = losses.Loss_SAM_optimized()
criterion_sid = losses.Loss_SID_optimized()
criterion_ssim = losses.Loss_SSIM()
criterion_psnr = losses.Loss_PSNR()
# criterion_psnr = losses.Loss_PSNR_modified()
if torch.cuda.is_available():
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_sam.cuda()
    criterion_sid.cuda()
    criterion_ssim.cuda()
    criterion_psnr.cuda()

def test(data_loader, model, save_mat=False):
    model.eval()
    losses_mrae, losses_rmse, losses_sam, losses_sid, losses_ssim, losses_psnr, losses_psnr_new = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    inference = AverageMeter()

    print("Reconstructing outputs ...")
    for idx, (input, target_hsi) in tqdm(enumerate(data_loader)):
        input = input.cuda()
        target_hsi = target_hsi.cuda()
        with torch.no_grad():
            start_time = time.time()
            
            if opt.truthful:
                output_null = model(input)
                target_null = torch.zeros_like(output_null).cuda()
                output_hsi = torch.zeros_like(target_hsi).cuda()

                for i in range(output_null.shape[0]):
                    _, _, output_hsi[i] = tru.null2spec_modified(rgbn=input[i], b=output_null[i], S=S, B=B, vmax=1, offset=True, norm=opt.norm)
                    _, _, target_null[i] = tru.spec2null_modified(hyper=target_hsi[i], PS=PS, PB=PB, B=B, vmax=1, offset=True)
            
            else:
                output_hsi = model(input)

            inference.update(time.time() - start_time)

        losses_mrae.update(criterion_mrae(output_hsi, target_hsi).data)
        losses_rmse.update(criterion_rmse(output_hsi, target_hsi).data)
        losses_sam.update(criterion_sam(output_hsi, target_hsi).data)
        losses_sid.update(criterion_sid(output_hsi, target_hsi).data)
        losses_ssim.update(criterion_ssim(output_hsi, target_hsi).data)
        losses_psnr.update(criterion_psnr(output_hsi, target_hsi).data)

        output_hsi = output_hsi.cpu().numpy() * 1.0
        output_hsi = np.transpose(np.squeeze(output_hsi), (1, 2, 0))
        output_hsi = np.clip(output_hsi, 0.0, 1.0)

        if save_mat:
            base_filename = os.path.splitext(os.path.basename(hyper_list[idx]))[0]
            mat_path = os.path.join(opt.outf, f"{base_filename}.mat")
            utils.save_matv73_modified(mat_path, var_name, output_hsi)
            print("Saved:", mat_path)

    return losses_mrae.avg, losses_rmse.avg, losses_sam.avg, losses_sid.avg, losses_ssim.avg, losses_psnr.avg, losses_psnr_new.avg, inference.avg, inference.stddev

if __name__ == "__main__":
    model_type = "ripetrack" if opt.compressed else "mobispec"
    model_type += "_tru" if opt.truthful else "_notru"

    # output path
    date_time = str(datetime.datetime.now())
    date_time = utils.time2file_name(date_time)
    dataset_name = opt.data_root.split("/")[-4]
    model_name = opt.pretrained_model_path.split("/")[-2]
    opt.outf = opt.outf + dataset_name + "/" + model_name + "/" + date_time
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)

    log_file = open(opt.outf + "_log.txt", "w")
    sys.stdout = utils.Tee(sys.stdout, log_file)

    start_time = time.time()
    start_process_time = time.process_time()

    cudnn.benchmark = True
    model = model_generator(opt.method, opt.pretrained_model_path, compressed, bands=bands).cuda()
    model = model.to(device)
    model.eval()

    mrae, rmse, sam, sid, ssim, psnr, psnr_new, inference_time_avg, inference_time_std = test(test_loader, model, save_mat=opt.save_mat)

    end_time = time.time()
    end_process_time = time.process_time()

    print("--------------------------------------------")
    print(f"Data loaded from: {opt.data_root}")
    print(f"Saved to: {opt.outf}")

    print(f"pretrained_model_path: {opt.pretrained_model_path}")
    print(f"Model adapted from {'RipeTrack' if compressed else 'MobiSpectral'}")
    if opt.truthful: 
        print(f"Truthful reconstruction is used")
        print(f"CMF: {opt.cmf} @ {opt.nir_mean} nm")
    else: print(f"Standard non-truthful reconstruction is used")
    print(f"Normalization: {opt.norm}")

    print(f"Number of parameters: {sum(param.numel() for param in model.parameters()):,}\n")
    print(f"MRAE: {mrae:.5f} | RMSE: {rmse:.5f} | PSNR: {psnr:.4f}\nSAM: {sam:.5f} | SID: {sid:.5f} | SSIM: {ssim:.5f}")
    print(f"Inference time = {inference_time_avg:.3f} +/- {inference_time_std:.3f}")
    print("--------------------------------------------")

    sys.stdout = sys.__stdout__
