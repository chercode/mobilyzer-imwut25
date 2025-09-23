from __future__ import division

import os, sys, logging, h5py, hdf5storage, datetime
import numpy as np

import torch
import torch.nn as nn
# from fvcore.nn import FlopCountAnalysis


class Tee:
    """ Write to multiple files. """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensures immediate output

    def flush(self):
        for f in self.files:
            f.flush()

def time_format(t):
    """
    Args:
        t (float) : time in seconds
    Returns:
        t (str) : time in HH:MM:SS format
    """
    return str(datetime.timedelta(seconds=t)).split('.')[0]

def my_summary(test_model, H = 64, W = 64, C = 68, N = 1):
    """
    Args:
        test_model (nn.Module) : model to be evaluated
        H (int) : height of input image
        W (int) : width of input image
        C (int) : number of channels of input image
        N (int) : batch size of input image
    """
    model = test_model.cuda()
    print(model)
    inputs = torch.randn((N, C, H, W)).cuda()
    flops = FlopCountAnalysis(model,inputs)
    n_param = sum([p.nelement() for p in model.parameters()])
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')

class AverageMeter(object):
    """ Computes and stores average, standard deviation, and current value. """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.arr = []
        self.stddev = 0

    def update(self, val, n=1):
        # Convert `val` to a scalar if it's a memoryview, tensor, or NumPy array
        if isinstance(val, memoryview):
            val = np.array(val).item()  # Convert memoryview to scalar
        elif torch.is_tensor(val):
            val = val.cpu().item()  # Convert PyTorch tensor to scalar
        elif isinstance(val, np.ndarray):
            val = val.item()  # Convert NumPy array to scalar
        self.val = val
        self.arr.append(val)
        # self.arr.append(val.cpu().item() if torch.is_tensor(val) else val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.stddev = np.std(np.array(self.arr))


# ------------ FOR TRAIN ------------

def initialize_logger(file_dir):
    """ Initializes the logger file which logs results of all models (Reconstruction and Classification). """
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    """ Save the model checkpoint with other variables as well. """
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close


# ------------ FOR TEST ------------

def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)

def save_matv73_modified(mat_name, var_name, var):
    """
    Save data to a MATLAB .mat file in v7.3 format.
    Args:
        mat_name (str): Path to save the .mat file.
        var_name (str): Name of the variable to store.
        var: Variable to save.
    """
    try:
        # Ensure the directory exists
        mat_dir = os.path.dirname(mat_name)
        if mat_dir and not os.path.exists(mat_dir):
            os.makedirs(mat_dir)

        # Add .mat extension if missing
        if not mat_name.endswith('.mat'):
            mat_name += '.mat'

        # Save the .mat file
        hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)
        # print(f"File saved successfully: {mat_name}")

    except OSError as e:
        raise OSError(f"Unable to create or open file '{mat_name}'. Check the directory path, permissions, and file name. Original error: {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while saving the .mat file: {e}")
