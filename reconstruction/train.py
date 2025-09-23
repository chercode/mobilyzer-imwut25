from __future__ import division

import os, sys, argparse, time, datetime
import numpy as np
import torch.backends
import random

import torch
from torchsummary import summary
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from architecture import *
from hsi_dataset_mobilyzer import TrainDataset, ValidDataset
import utils, losses
from utils import AverageMeter

# import losses
# import utils_truthful_rgb as tru
import utils_truthful as tru

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def str2bool(v):
    return v.lower() in ('true', '1', 'yes')

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # Make CuDNN deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lossfunctions_considered = ["MRAE", "SAM", "SID"]
# lossfunctions_considered = ["MRAE"]

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument("--method", type=str, default="mst_plus_plus")
parser.add_argument("--compressed", type=str2bool, default="True")
parser.add_argument("--truthful", type=str2bool, default="True")
parser.add_argument("--norm", type=str2bool, default="False")
parser.add_argument("--cmf", type=str, default="average")
parser.add_argument("--nir_mean", type=int, default=940)
parser.add_argument("--bands", type=int, default=68)
parser.add_argument("--alpha", type=float, default=1)
parser.add_argument("--batch_size", type=int, default=20, help="batch size")
parser.add_argument("--end_epoch", type=int, default=10, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--patch_size", type=int, default=64, help="patch size")
parser.add_argument("--stride", type=int, default=64, help="stride")
parser.add_argument("--gpu_id", type=str, default="0", help="path log files")
parser.add_argument("--loss_null", type=str2bool, default="True", help="null loss")
parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")

parser.add_argument("--data_root", type=str, default="/media/sma318/TOSHIBA EXT/dataset_open/data/evoo/HSI")
parser.add_argument("--split_root", type=str, default="/media/sma318/TOSHIBA EXT/dataset_open/data/evoo/HSI")
parser.add_argument("--outf", type=str, default="../models/HSI/", help="path log files")
parser.add_argument("--pretrained_model_path", type=str, default=None)

opt = parser.parse_args()

# Set seed for reproducibility BEFORE any random operations
set_seed(opt.seed)

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.autograd.set_detect_anomaly(False)
# Comment out benchmark for reproducibility
# torch.backends.cudnn.benchmark = True

# load dataset
print("\nloading dataset ...")
bands = opt.bands
# if truthful reconstruction is used, the number of reconstructed bands is reduced
# -= 3 for RGB only
# -= 4 for RGB + NIR
if opt.truthful: bands -= 4

alpha = opt.alpha

# read sensitivity matrix and helper matrices
cmf = opt.cmf
nir_mean = opt.nir_mean
S = np.load(f"helper_matrices/S_{cmf}_{nir_mean}.npy")
B = np.load(f"helper_matrices/B_{cmf}_{nir_mean}.npy")
PS = np.load(f"helper_matrices/PS_{cmf}_{nir_mean}.npy")
PB = np.load(f"helper_matrices/PB_{cmf}_{nir_mean}.npy")


train_data = TrainDataset(data_root=opt.data_root, split_root=opt.split_root, crop_size=opt.patch_size, bgr2rgb=True, arg=True, stride=opt.stride, truthful=opt.truthful, S=S, norm=opt.norm)
print("Training set samples: ", len(train_data))

print("Iteration per epoch:", len(train_data))
val_data = ValidDataset(data_root=opt.data_root, split_root=opt.split_root, bgr2rgb=True, truthful=opt.truthful, S=S, norm=opt.norm)
print("Validation set samples: ", len(val_data))

# iterations
per_epoch_iteration = 1000
total_iteration = per_epoch_iteration*opt.end_epoch
compressed = opt.compressed
print("compressed: ", compressed)

# loss functions --> validation
criterion_rmse = losses.Loss_RMSE()
criterion_psnr = losses.Loss_PSNR()
# criterion_psnr = losses.Loss_PSNR_modified()
# loss functions --> training
criterion_mrae = losses.Loss_MRAE()
# criterion_sam = losses.Loss_SAM()
# criterion_sam = losses.Loss_SAM_modified()
criterion_sam = losses.Loss_SAM_optimized()
# criterion_sid = losses.Loss_SID()
# criterion_sid = losses.Loss_SID_modified()
criterion_sid = losses.Loss_SID_optimized()
criterions = (criterion_mrae, criterion_sam, criterion_sid)


# model
pretrained_model_path = opt.pretrained_model_path
method = opt.method
model = model_generator(method, pretrained_model_path, compressed=compressed, bands=bands).cuda()
print(f"Number of parameters: {sum(param.numel() for param in model.parameters()):,}")
# print(summary(model, (4, 64, 64)))
# utils.my_summary(MST_Plus_Plus(), 64, 64, 4, 1)

# output path
date_time = str(datetime.datetime.now())
date_time = utils.time2file_name(date_time)
dataset_name = opt.data_root.split("/")[-4]
opt.outf = opt.outf + dataset_name + "/" + date_time
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if torch.cuda.is_available():
    model.cuda()
    criterion_mrae.to(device)
    criterion_sam.to(device)
    criterion_sid.to(device)
    criterion_rmse.to(device)
    criterion_psnr.to(device)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)


# logging
log_dir = os.path.join(opt.outf, 'train.log')
logger = utils.initialize_logger(log_dir)

# resume
resume_file = opt.pretrained_model_path
if resume_file is not None:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        iteration = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

model.to(device)

def worker_init_fn(worker_id):
    """Initialize worker with different seed"""
    np.random.seed(opt.seed + worker_id)
    random.seed(opt.seed + worker_id)

def main():
    # Set deterministic behavior for CuDNN
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    iteration = 0
    record_loss = 1000
    while iteration<total_iteration:
        model.train()
        losses, losses_mrae, losses_sam, losses_sid = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        # Use deterministic DataLoader
        train_loader = DataLoader(
            dataset=train_data, 
            num_workers=2, 
            batch_size=opt.batch_size, 
            shuffle=True, 
            pin_memory=True, 
            drop_last=True,
            worker_init_fn=worker_init_fn,  # For reproducible data loading
            generator=torch.Generator().manual_seed(opt.seed)  # For reproducible shuffling
        )
        val_loader = DataLoader(
            dataset=val_data, 
            num_workers=2, 
            batch_size=1, 
            shuffle=False, 
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(opt.seed)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(train_loader.__len__())*opt.end_epoch/opt.batch_size, eta_min=1e-6)

        for i, (images, labels_hsi) in enumerate(train_loader):
            labels_hsi = torch.add(labels_hsi, 0.0001)
            labels_hsi = labels_hsi.cuda()
            images = images.cuda()
            images = Variable(images)
            labels_hsi = Variable(labels_hsi)
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            
            if opt.truthful:
                # label_hsi shape: (68, 512, 512)
                # output shape: (64, 512, 512)

                output_null = model(images)
                labels_null = torch.zeros_like(output_null).cuda()
                output_hsi = torch.zeros_like(labels_hsi).cuda()

                for i in range(output_null.shape[0]):
                    _, _, output_hsi[i] = tru.null2spec_modified(rgbn=images[i], b=output_null[i], S=S, B=B, vmax=1, offset=True, norm=opt.norm)
                    _, _, labels_null[i] = tru.spec2null_modified(hyper=labels_hsi[i], PS=PS, PB=PB, B=B, vmax=1, offset=True)
                            
                # mrae
                # compare null against null
                null_loss_mrae = criterion_mrae(output_null, labels_null)
                # compare hsi against hsi
                hsi_loss_mrae = criterion_mrae(output_hsi, labels_hsi)
                # weighted sum of losses
                # loss_mrae = alpha * null_loss_mrae + (1 - alpha) * hsi_loss_mrae

                if opt.loss_null:
                    loss_mrae = null_loss_mrae
                    loss_sam = torch.mul(criterion_sam(output_null, labels_null), 0.1) if "SAM" in lossfunctions_considered else torch.tensor(0)
                    loss_sid = torch.mul(criterion_sid(output_null, labels_null), 0.001) if "SID" in lossfunctions_considered else torch.tensor(0)
                else:
                    loss_mrae = hsi_loss_mrae
                    loss_sam = torch.mul(criterion_sam(output_hsi, labels_hsi), 0.1) if "SAM" in lossfunctions_considered else torch.tensor(0)
                    loss_sid = torch.mul(criterion_sid(output_hsi, labels_hsi), 0.001) if "SID" in lossfunctions_considered else torch.tensor(0)
                loss = loss_mrae + loss_sam + loss_sid

            else:
                output_hsi = model(images)
                loss_mrae = criterion_mrae(output_hsi, labels_hsi)
                loss_sam = torch.mul(criterion_sam(output_hsi, labels_hsi), 0.1) if "SAM" in lossfunctions_considered else torch.tensor(0)
                loss_sid = torch.mul(criterion_sid(output_hsi, labels_hsi), 0.001) if "SID" in lossfunctions_considered else torch.tensor(0)
                loss = loss_mrae + loss_sam + loss_sid
            

            loss.backward()
            # call step function 
            optimizer.step()
            scheduler.step()

            # record loss 
            # losses.update(loss.data)
            losses.update(loss.item())
            losses_mrae.update(loss_mrae.item())
            losses_sam.update(loss_sam.item())
            losses_sid.update(loss_sid.item())

            iteration = iteration+1
            if iteration % 20 == 0:
                print('[iter: %d/%d], lr = %.9f, train_losses.avg = %.5f'
                      % (iteration, total_iteration, lr, losses.avg))

            # validation
            if iteration % 1000 == 0:

                total_loss, mrae_loss, sam_loss, sid_loss, psnr_loss = validate(val_loader, model)
                print(f"Total_Loss: {total_loss:.5f} | MRAE: {mrae_loss:.5f} | SAM: {sam_loss/0.1:.5f} | SID: {sid_loss/0.001:.5f} | PSNR: {psnr_loss:.4f}")

                # Save model
                if torch.abs(torch.tensor(total_loss) - record_loss) < 0.01 or total_loss < record_loss or iteration % 5000 == 0:
                    print(f'Saving to {opt.outf}')
                    utils.save_checkpoint(opt.outf, (iteration // 1000), iteration, model, optimizer)
                    if total_loss < record_loss:
                        record_loss = total_loss

                # printing losses
                print(f"Iter[{iteration:06d}], Epoch[{iteration//1000:06d}], LR: {lr:.9f}, Test metrics: Total_Loss: {total_loss:.5f} | MRAE: {mrae_loss:.5f} | SAM: {sam_loss/0.1:.5f} | SID: {sid_loss/0.001:.5f} | PSNR: {psnr_loss:.4f}")

                logger.info(
                    f"Iteration [{iteration:06d}] | Epoch [{iteration // 1000:06d}] | LR: {lr:.9f}\n"
                    f"Train Metrics: Total Loss: {losses.avg:.5f} | MRAE: {losses_mrae.avg:.5f} | "
                    f"SAM: {losses_sam.avg:.5f} | SID: {losses_sid.avg:.5f}\n"
                    f"Test Metrics: Total Loss: {total_loss:.5f} | MRAE: {mrae_loss:.5f} | "
                    f"SAM: {sam_loss:.5f} | SID: {sid_loss:.5f} | PSNR: {psnr_loss:.4f}"
                )
                
    return 0

# Validate
def validate(val_loader, model):
    model.eval()
    
    losses, losses_mrae, losses_sam, losses_sid, losses_psnr = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    for i, (input, target_hsi) in enumerate(val_loader):
        target_hsi = torch.add(target_hsi, 0.0001)
        input = input.cuda()
        target_hsi = target_hsi.cuda()
        with torch.no_grad():

            if opt.truthful:
                # input shape: (4, 512, 512)
                # target shape: (68, 512, 512)
                # output shape: (64, 512, 512)

                output_null = model(input)
                target_null = torch.zeros_like(output_null).cuda()
                output_hsi = torch.zeros_like(target_hsi).cuda()

                for i in range(output_null.shape[0]):
                    _, _, output_hsi[i] = tru.null2spec_modified(rgbn=input[i], b=output_null[i], S=S, B=B, vmax=1, offset=True, norm=opt.norm)
                    _, _, target_null[i] = tru.spec2null_modified(hyper=target_hsi[i], PS=PS, PB=PB, B=B, vmax=1, offset=True)

                # mrae
                # compare null against null
                null_loss_mrae = criterion_mrae(output_null, target_null)
                # compare hsi against hsi
                hsi_loss_mrae = criterion_mrae(output_hsi, target_hsi)
                # weighted sum of losses
                # loss_mrae = alpha * null_loss_mrae + (1 - alpha) * hsi_loss_mrae

                if opt.loss_null:
                    loss_mrae = null_loss_mrae
                    loss_sam = torch.mul(criterion_sam(output_null, target_null), 0.1) if "SAM" in lossfunctions_considered else torch.tensor(0)
                    loss_sid = torch.mul(criterion_sid(output_null, target_null), 0.001) if "SID" in lossfunctions_considered else torch.tensor(0)
                    loss_psnr = criterion_psnr(output_null, target_null)
                else:
                    loss_mrae = hsi_loss_mrae
                    loss_sam = torch.mul(criterion_sam(output_hsi, target_hsi), 0.1) if "SAM" in lossfunctions_considered else torch.tensor(0)
                    loss_sid = torch.mul(criterion_sid(output_hsi, target_hsi), 0.001) if "SID" in lossfunctions_considered else torch.tensor(0)
                    loss_psnr = criterion_psnr(output_hsi, target_hsi)
                loss = loss_mrae + loss_sam + loss_sid


            else:
                output_hsi = model(input)
                loss_mrae = criterion_mrae(output_hsi, target_hsi)
                loss_sam = torch.mul(criterion_sam(output_hsi, target_hsi), 0.1) if "SAM" in lossfunctions_considered else torch.tensor(0)
                loss_sid = torch.mul(criterion_sid(output_hsi, target_hsi), 0.001) if "SID" in lossfunctions_considered else torch.tensor(0)
                loss = loss_mrae + loss_sam + loss_sid
                loss_psnr = criterion_psnr(output_hsi, target_hsi)

        # record loss
        losses.update(loss.item())
        losses_mrae.update(loss_mrae.item())
        losses_sam.update(loss_sam.item())
        losses_sid.update(loss_sid.item())
        losses_psnr.update(loss_psnr.item())

        if opt.truthful:
            result = output_null.cpu().numpy() * 1.0
            gt = target_null.cpu().numpy() * 1.0
        else:
            result = output_hsi.cpu().numpy() * 1.0
            gt = target_hsi.cpu().numpy() * 1.0
        
        result = np.transpose(np.squeeze(result), [1, 2, 0])
        result = np.minimum(result, 1.0)
        result = np.maximum(result, 0)

        gt = np.transpose(np.squeeze(gt), [1, 2, 0])
        gt = np.minimum(gt, 1.0)
        gt = np.maximum(gt, 0)

    return losses.avg, losses_mrae.avg, losses_sam.avg, losses_sid.avg, losses_psnr.avg


if __name__ == '__main__':
    
    log_file = open(opt.outf + '_log.txt', 'w')
    sys.stdout = utils.Tee(sys.stdout, log_file)  # Duplicate output

    if compressed: print(f"Model modification adapted from RipeTrack")
    else: print(f"Standard model adapted from MobiSpectral")

    if opt.truthful: 
        print(f"Truthful reconstruction is used")
        print(f"CMF: {opt.cmf} @ {opt.nir_mean} nm")
    else: print(f"Standard non-truthful reconstruction is used")
    
    print(f"Number of parameters: {sum(param.numel() for param in model.parameters()):,}")
    print(f"Loss functions considered: {lossfunctions_considered}")
    print(f"Null loss: {opt.loss_null}")
    print(f"Random seed: {opt.seed}")

    print(f"Saved to: {opt.outf}")
    print(f"alpha value: {opt.alpha}")
    print(f"Normalization: {opt.norm}")
    print(f"Data loaded from: {opt.data_root}")

    print("--------------------------------------------")

    start_time = time.time()
    start_process_time = time.process_time()

    main()
    print(torch.__version__)

    end_time = time.time()
    end_process_time = time.process_time()

    print("--------------------------------------------")
    if compressed: print(f"Model modification adapted from RipeTrack")
    else: print(f"Standard model adapted from MobiSpectral")
    
    if opt.truthful: 
        print(f"Truthful reconstruction is used")
        print(f"CMF: {opt.cmf} @ {opt.nir_mean} nm")
    else: print(f"Standard non-truthful reconstruction is used")

    print(f"Loss functions considered: {lossfunctions_considered}")
    print(f"Null loss: {opt.loss_null}")
    print(f"Random seed: {opt.seed}")

    print(f"Number of parameters: {sum(param.numel() for param in model.parameters()):,}")
    print(f"time.time() : {utils.time_format(end_time - start_time)}")
    print(f"time.process_time() : {utils.time_format(end_process_time - start_process_time)}")

    print(f"Saved to: {opt.outf}")
    print(f"alpha value: {opt.alpha}")
    print(f"Normalization: {opt.norm}")
    print(f"Data loaded from: {opt.data_root}")

    print("--------------------------------------------")
    sys.stdout = sys.__stdout__