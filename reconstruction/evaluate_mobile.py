import torch
import argparse
import torch.backends.cudnn as cudnn
import os, warnings
from architecture import *
import glob
import cv2, hdf5storage
import numpy as np
import itertools
import imageio
import time
import sys
import utils_truthful as tru
import utils
from utils import AverageMeter
from tqdm import tqdm


def center_crop_at(image, center_y, center_x, crop_size=512):
    """
    Crop a square patch of size `crop_size`x`crop_size` from `image`
    such that (center_y, center_x) is the center of the cropped region.
    """
    half_size = crop_size // 2

    # Calculate the start and end coordinates
    start_y = center_y - half_size
    end_y = start_y + crop_size
    start_x = center_x - half_size
    end_x = start_x + crop_size

    # Slice the image
    cropped = image[start_y:end_y, start_x:end_x]

    return cropped

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
np.random.seed(10)

def str2bool(v):
    return v.lower() in ('true', '1', 'yes')

# parser = argparse.ArgumentParser(description="SSR")
parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument("--method", type=str, default="mst_plus_plus")
parser.add_argument("--bands", type=int, default=68)
parser.add_argument('--cmf', type=str, default="average")
parser.add_argument('--norm', type=str2bool, default="False")
parser.add_argument('--nir_mean', type=int, default="940")
parser.add_argument('--compressed', type=str2bool, default="True")
parser.add_argument('--truthful', type=str2bool, default="True")
parser.add_argument("--ensemble_mode", type=str, default="mean")
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--phone", type=str, default="pixel", choices=["oneplus", "doogee", "ulefone", "pixel"], help="Phone model for dimension extraction")

parser.add_argument("--split_root", type=str, default=None)
parser.add_argument("--data_label", type=str, default="reconstruction")

parser.add_argument("--data_root", type=str, default=f"/media/sma318/TOSHIBA EXT/dataset_open/data/evoo/phone/origin/intrinsic/US/")
parser.add_argument("--outf", type=str, default="/media/sma318/TOSHIBA EXT/dataset_open/data/evoo/phone/origin/reconstructed/US/", help="path log files")
parser.add_argument("--pretrained_model_path", type=str, default="../models/HSI/dataset_open/2025_09_23_14_44_18/net_10epoch.pth")


opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

bands = opt.bands
truthful = opt.truthful
cmf = opt.cmf
nir_mean = opt.nir_mean
compressed = opt.compressed
data_label = opt.data_label

if truthful: bands -= 4

if "raw" in opt.data_root: img_type = "RAW"
elif "rgb" in opt.data_root: img_type = "RGB"
else: img_type = "RGB"

var_name = "cube"

# read sensitivity matrix and helper matrices
S = np.load(f"./helper_matrices/S_{cmf}_{nir_mean}.npy")
B = np.load(f"./helper_matrices/B_{cmf}_{nir_mean}.npy")
PS = np.load(f"./helper_matrices/PS_{cmf}_{nir_mean}.npy")
PB = np.load(f"./helper_matrices/PB_{cmf}_{nir_mean}.npy")


def temp():
    cudnn.benchmark = True
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    model = model_generator(method, pretrained_model_path, compressed, bands).cuda()
    test_path = os.path.join(opt.data_root)
    inference_time_avg, inference_time_std = evaluate_mobile(model, test_path, opt.outf)

def evaluate_mobile(model, test_path, save_path, save_mat=True, clean_track=False):
    # setup img list for saving results
    if opt.split_root is None:
        img_names = os.listdir(test_path)
    else:
        with open(f"{opt.split_root}/split_txt/{data_label}.txt", "r") as fin:
            img_names = [ line.strip() for line in fin ]
        fin.close()
    img_names.sort()
    # print(img_names)

    # all_imgs = glob.glob(os.path.join(opt.data_root, "*.mat"))
    all_imgs = glob.glob(os.path.join(test_path, "*_rgb.png"))
    all_imgs.sort()

    # img_path_name = [ path for path in all_imgs if os.path.basename(path).split("_")[0].split(".")[0] in img_names ]
    img_path_name = [p for p in all_imgs if os.path.basename(p) in img_names]
    print(f"number of images in folder: {len(img_path_name)}")

    # calculate inference time for each image
    inference = AverageMeter()
    for i in tqdm(range(len(img_path_name))):
        start_time = time.time()

        if clean_track: 
            # HS img
            hyper_path = img_path_name[i]
            cube = hdf5storage.loadmat(hyper_path,variable_names=["rad"])
            hyper = cube["rad"][:, :, 1:204:3]          # (512, 512, 68)
            vnir = hyper @ S                            # (512, 512, 4)
            vnir = np.float32(vnir)
        else:
            # RGB img
            rgb = imageio.imread(img_path_name[i])      # (512, 512, 3)
            rgb = np.float32(rgb)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

            nir = imageio.imread(img_path_name[i].replace("_rgb.jpg","_nir.jpg"))              # (512, 512)
            nir = np.float32(nir)
            nir = (nir - nir.min()) / (nir.max() - nir.min())
            if nir.ndim == 3: nir = nir[:, :, 0]
            # stack RGB and NIR
            vnir = np.dstack((rgb, nir))                 # (512, 512, 4)
        vnir = np.expand_dims(np.transpose(vnir, [2, 0, 1]), axis=0).copy()    # (1, 4, 512, 512)
        vnir = torch.from_numpy(vnir).float().cuda()                           # (1, 4, 512, 512)

        with torch.no_grad():
            output = forward_ensemble(vnir, model, opt.ensemble_mode)
        if truthful:
            _, _, output = tru.null2spec_modified(rgbn=vnir.squeeze(0), b=output.squeeze(0), S=S, B=B, vmax=1, offset=True, norm=opt.norm)
        else:
            output = np.squeeze(output, axis=0)

        output = output.cpu().numpy() * 1.0
        output = np.transpose(output, (1, 2, 0)) # (512, 512, 68)
        output = np.minimum(output, 1.0)
        output = np.maximum(output, 0)

        inference.update(time.time() - start_time)

        # save results
        if save_mat:
            mat_dir = os.path.join(save_path, img_path_name[i].split("_")[0] + ".mat")
            utils.save_matv73_modified(mat_dir, var_name, output)

    return inference.avg, inference.stddev


def apply_phone_transforms(rgb, nir, phone):
    """
    Apply phone-specific resize and crop transformations.
    
    Args:
        rgb: RGB image array
        nir: NIR image array
        phone: Phone model name ('oneplus', 'doogee', 'ulefone', 'pixel')
    
    Returns:
        Transformed rgb and nir images
    """
    if phone == "oneplus":
        rgb = cv2.resize(rgb, (1152, 1504), interpolation=cv2.INTER_LINEAR)
        nir = cv2.resize(nir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        rgb = center_crop_at(rgb, center_y=866, center_x=536, crop_size=512)
        nir = center_crop_at(nir, center_y=830, center_x=582, crop_size=512)
    
    elif phone == "doogee":
        rgb = cv2.resize(rgb, (1152, 1504), interpolation=cv2.INTER_LINEAR)
        nir = cv2.resize(nir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        rgb = center_crop_at(rgb, center_y=792, center_x=572, crop_size=512)
        nir = center_crop_at(nir, center_y=806, center_x=488, crop_size=512)
    
    elif phone == "ulefone":
        rgb = cv2.resize(rgb, (1152, 1504), interpolation=cv2.INTER_LINEAR)
        nir = cv2.resize(nir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        rgb = center_crop_at(rgb, center_y=750, center_x=622, crop_size=512)
        nir = center_crop_at(nir, center_y=784, center_x=524, crop_size=512)
    
    elif phone == "pixel":
        rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_LINEAR)
        rgb = center_crop_at(rgb, center_y=259, center_x=329, crop_size=128)
        nir = center_crop_at(nir, center_y=242, center_x=333, crop_size=128)
    
    return rgb, nir


def evaluate_mobile_modified(model, test_path, save_path, save_mat=True):
    # Get all RAW images (1_RAW.png, 2_RAW.png, etc.)
    # rgb_images = sorted(glob.glob(os.path.join(test_path, f"*_{img_type}.png")))
    rgb_images = sorted(glob.glob(os.path.join(test_path, f"*_RGB.png")))
    print(f"Found {len(rgb_images)} RGB images")
    
    # Calculate inference time for each image
    inference = AverageMeter()
    
    for rgb_path in tqdm(rgb_images):
        start_time = time.time()
        
        # Process RGB image
        rgb = imageio.imread(rgb_path)

        nir_path = rgb_path.replace(f"_RGB.png", "_NIR.png")

        nir = imageio.imread(nir_path)

        # Apply phone-specific transformations
        rgb, nir = apply_phone_transforms(rgb, nir, opt.phone)

        rgb = np.float32(rgb)
        nir = np.float32(nir)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        print(rgb)
        nir = (nir - nir.min()) / (nir.max() - nir.min())
        if nir.ndim == 3: 
            nir = nir[:, :, 0]
            
        # Stack RGB and NIR
        vnir = np.dstack((rgb, nir))
        vnir = np.expand_dims(np.transpose(vnir, [2, 0, 1]), axis=0).copy()
        vnir = torch.from_numpy(vnir).float().cuda()

        with torch.no_grad():
            output = forward_ensemble(vnir, model, opt.ensemble_mode)
        output = output.squeeze(0)
            
        if truthful:
            _, _, output = tru.null2spec_modified(rgbn=vnir.squeeze(0), b=output, S=S, B=B, vmax=1, offset=True, norm=opt.norm)
            
        output = output.cpu().numpy() * 1.0
        output = np.transpose(output, (1, 2, 0))
        output = np.minimum(output, 1.0)
        output = np.maximum(output, 0)

        inference.update(time.time() - start_time)

        # Save results
        if save_mat:
            base_id = os.path.basename(rgb_path).split('_')[0]
            mat_dir = os.path.join(save_path, f"{base_id}.mat")
            # mat_dir = os.path.join(save_path, f"{main_model}.mat")
            print(f"Saving results to {mat_dir}")
            utils.save_matv73_modified(mat_dir, var_name, output)

    return inference.avg, inference.stddev


def evaluate_mobile_modified_new(model, test_path, save_path, list_path, save_mat=True):
    # Load test list from external path
    with open(list_path, 'r') as f:
        test_list = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(test_list)} entries from {list_path}")

    # Prepare inference time meter
    inference = AverageMeter()

    for base_name in tqdm(test_list):
        start_time = time.time()

        # Construct paths for RGB and NIR images
        rgb_path = os.path.join(test_path, f"{base_name}_RGB.png")
        nir_path = os.path.join(test_path, f"{base_name}_NIR.png")

        # Load and normalize RGB
        rgb = imageio.imread(rgb_path).astype(np.float32)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

        # Load and normalize NIR
        nir = imageio.imread(nir_path).astype(np.float32)
        if nir.ndim == 3:
            nir = nir[:, :, 0]
        nir = (nir - nir.min()) / (nir.max() - nir.min())

        # Stack RGB + NIR
        vnir = np.dstack((rgb, nir))
        vnir = np.expand_dims(np.transpose(vnir, [2, 0, 1]), axis=0).copy()
        vnir = torch.from_numpy(vnir).float().cuda()

        with torch.no_grad():
            output = forward_ensemble(vnir, model, opt.ensemble_mode)

        output = output.squeeze(0)

        if truthful:
            _, _, output = tru.null2spec_modified(
                rgbn=vnir.squeeze(0), b=output, S=S, B=B, vmax=1, offset=True, norm=opt.norm
            )

        output = output.cpu().numpy()
        output = np.transpose(output, (1, 2, 0))
        output = np.clip(output, 0, 1)

        inference.update(time.time() - start_time)

        # Save output
        if save_mat:
            mat_path = os.path.join(save_path, f"{base_name}.mat")
            print(f"Saving to {mat_path}")
            utils.save_matv73_modified(mat_path, var_name, output)

    return inference.avg, inference.stddev

def forward_ensemble(x, forward_func, ensemble_mode = "mean"):
    def _transform(data, xflip, yflip, transpose, reverse=False):
        if not reverse:  # forward transform
            if xflip:
                data = torch.flip(data, [3])
            if yflip:
                data = torch.flip(data, [2])
            if transpose:
                data = torch.transpose(data, 2, 3)
        else:  # reverse transform
            if transpose:
                data = torch.transpose(data, 2, 3)
            if yflip:
                data = torch.flip(data, [2])
            if xflip:
                data = torch.flip(data, [3])
        return data

    outputs = []
    opts = itertools.product((False, True), (False, True), (False, True))
    for xflip, yflip, transpose in opts:
        data = x.clone()
        data = _transform(data, xflip, yflip, transpose)
        data = forward_func(data)
        outputs.append(
            _transform(data, xflip, yflip, transpose, reverse=True))
    if ensemble_mode == "mean":
        return torch.stack(outputs, 0).mean(0)
    elif ensemble_mode == "median":
        return torch.stack(outputs, 0).median(0)[0]


if __name__ == "__main__":

    cudnn.benchmark = True
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    model = model_generator(method, pretrained_model_path, compressed, bands).cuda()

    model_name = opt.pretrained_model_path.split("/")[-2]
    opt.outf = opt.outf 
    # + "/" + model_name
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)

    log_file = open(opt.outf + "_log.txt", "w")
    sys.stdout = utils.Tee(sys.stdout, log_file)

    print(f"saving to {opt.outf}")    
    print(f"Using phone model: {opt.phone}")

    # inference_time_avg, inference_time_std = evaluate_mobile_modified_new(model, test_path=opt.data_root, save_path=opt.outf, list_path=opt.list_path)
    # inference_time_avg, inference_time_std = evaluate_mobile_modified(model, test_path, opt.outf)
    inference_time_avg, inference_time_std = evaluate_mobile_modified(model, opt.data_root, opt.outf)

    print("--------------------------------------------")
    if compressed: print(f"Model adapted from RipeTrack")
    else: print(f"Standard model adapted from MobiSpectral")
    if truthful: print(f"Using truthful reconstruction")
    else: print(f"Using standard non-truthful reconstruction")
    print(f"Number of parameters: {sum(param.numel() for param in model.parameters()):,}\n")
    print(f"Inference time = {inference_time_avg:.3f} +/- {inference_time_std:.3f}")
    print("--------------------------------------------")
    sys.stdout = sys.__stdout__