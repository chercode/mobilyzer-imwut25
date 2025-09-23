import scipy.linalg
import os, sys, h5py, cv2, random, hdf5storage, torch, datetime, h5py, scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import dataset
from imageio.v2 import imread
from skimage import io, color
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler
from skimage.color import rgb2lab, deltaE_cie76

np.random.seed(10)

def compute_rgb_nir_metrics(recon_hsi, target_hsi):
    recon_rgb = recon_hsi[..., :3]
    target_rgb = target_hsi[..., :3]

    recon_lab = rgb2lab(recon_rgb)
    target_lab = rgb2lab(target_rgb)
    delta_e_map = deltaE_cie76(recon_lab, target_lab)
    delta_e_mean = np.mean(delta_e_map)
    rgb_mae = np.mean(np.abs(recon_rgb - target_rgb))

    nir_mae = None
    if recon_hsi.shape[-1] > 3:
        recon_nir = recon_hsi[..., 3]
        target_nir = target_hsi[..., 3]
        nir_mae = np.mean(np.abs(recon_nir - target_nir))

    return delta_e_mean, rgb_mae, nir_mae

def compute_loss(output, labels, lossfunctions_considered, criterion_mrae, criterion_sam, criterion_sid):
    """
    Compute the total loss for hyperspectral image reconstruction.

    Args:
        output (torch.Tensor): Predicted spectra, shape (N, C, H, W).
        labels (torch.Tensor): Ground truth spectra, shape (N, C, H, W).
        lossfunctions_considered (list): List of loss functions to consider.
        criterion_mrae (callable): MRAE loss function.
        criterion_sam (callable): SAM loss function.
        criterion_sid (callable): SID loss function.

    Returns:
        torch.Tensor: Total loss.
    """
    # Compute MRAE loss
    loss_mrae = criterion_mrae(output, labels)

    # Compute SAM loss if considered
    loss_sam = torch.mul(criterion_sam(output, labels), 0.1) if "SAM" in lossfunctions_considered else torch.tensor(0.0, device=output.device)

    # Compute SID loss if considered
    loss_sid = torch.mul(criterion_sid(output, labels), 0.001) if "SID" in lossfunctions_considered else torch.tensor(0.0, device=output.device)

    # Total loss
    total_loss = loss_mrae + loss_sam + loss_sid

    return total_loss, loss_mrae, loss_sam, loss_sid


# ------------------------------------------------------------------------------
# COLOR CONVERSIONS
# ------------------------------------------------------------------------------

def XYZ2sRGB_exgamma(XYZ):
    """
    Convert an XYZ image to sRGB color space without gamma correction.
    
    Parameters:
        XYZ (numpy.ndarray): Input image in XYZ color space of shape (..., 3), where the last dimension is (X, Y, Z).
    
    Returns:
        numpy.ndarray: Converted image in sRGB color space with the same shape as input.
    """
    # Transformation matrix from XYZ to sRGB (from IEC_61966-2-1)
    M = np.array([
        [ 3.2406, -1.5372, -0.4986],
        [-0.9689,  1.8758,  0.0414],
        [ 0.0557, -0.2040,  1.0570]
    ])

    # Reshape input to (-1, 3) for matrix multiplication if it's an image
    original_shape = XYZ.shape
    XYZ_flat = XYZ.reshape(-1, 3)

    # Apply transformation
    sRGB_flat = np.dot(XYZ_flat, M.T)  # Using transposed matrix to match numpy dot product convention

    # Reshape back to original image shape
    sRGB = sRGB_flat.reshape(original_shape)

    return sRGB

def sRGB2XYZ_exgamma(sRGB):
    """
    Convert an sRGB image to CIE XYZ color space without gamma correction.
    
    Parameters:
        sRGB (numpy.ndarray): Input sRGB image of shape (..., 3), where the last dimension is (R, G, B).
    
    Returns:
        numpy.ndarray: Converted image in XYZ color space with the same shape as input.
    """
    # Inverse transformation matrix from sRGB to XYZ (from IEC_61966-2-1)
    M_inv = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]
    ])

    # Reshape input to (-1, 3) for matrix multiplication if it's an image
    original_shape = sRGB.shape
    sRGB_flat = sRGB.reshape(-1, 3)

    # Apply inverse transformation
    XYZ_flat = np.dot(sRGB_flat, M_inv.T)  # Using transposed matrix to match numpy dot product convention

    # Reshape back to original image shape
    XYZ = XYZ_flat.reshape(original_shape)

    return XYZ

def func(xyzbar_path, nir_wavelength):
    # read mat file for xyzbar
    xyzbar = scipy.io.loadmat(xyzbar_path)
    xyzbar = xyzbar['xyzbar']

# ------------------------------------------------------------------------------
# TRUTHFUL FUNCTIONS
# ------------------------------------------------------------------------------

def spec2null(hyper, PS, PB, B, vmax, offset=True):
    # get fundamental metamer rF
    rF = np.einsum('ij,jkl->ikl', PS, hyper)
    # get metameric black rB
    rB = np.einsum('ij,jkl->ikl', PB, hyper)
    # get null-coefficient image b
    b = np.einsum('ij,jkl->ikl', B.T, rB)
    # offset
    if offset: b = (b + vmax) / (2 * vmax)
    return rF, rB, b

def null2spec(rgbn, b, S, B, vmax=1, offset=True):
    # get fundamental metamer rF
    rF = S @ np.linalg.inv(S.T @ S)
    rF = np.einsum('ij,jkl->ikl', rF, rgbn)
    # offset
    if offset: b = vmax * (2 * b - 1)
    # get metameric black rB
    rB = np.einsum('ij,jkl->ikl', B, b)
    # get final hypercube
    hyper = rF + rB
    return rF, rB, hyper

# null2spec torch
def null2spec_torch(rgbn, b, S, B, vmax=1, offset=True):
    # move to torch
    # assert B.dtype == torch.float32, "B must be of type torch.float32"
    # assert b.dtype == torch.float32, "b must be of type torch.float32"
    S = torch.from_numpy(S).float().to(rgbn.device)
    B = torch.from_numpy(B).float().to(rgbn.device)
    b = b.float().to(rgbn.device)
    # get fundamental metamer rF
    rF = S @ torch.linalg.pinv(S.T @ S)
    rF = torch.einsum('ij,jkl->ikl', rF, rgbn)
    # offset
    if offset: b = vmax * (2 * b - 1)
    # get metameric black rB
    rB = torch.einsum('ij,jkl->ikl', B, b)
    # get final hypercube
    hyper = rF + rB
    return rF, rB, hyper

def spec2null_modified(hyper, PS, PB, B, vmax, offset=True):
    # Convert numpy arrays to torch tensors
    PS_torch = torch.tensor(PS, dtype=hyper.dtype, device=hyper.device)
    PB_torch = torch.tensor(PB, dtype=hyper.dtype, device=hyper.device)
    B_torch = torch.tensor(B.T, dtype=hyper.dtype, device=hyper.device)
    
    # Get fundamental metamer rF
    rF = torch.einsum('ij,jkl->ikl', PS_torch, hyper)
    # Get metameric black rB
    rB = torch.einsum('ij,jkl->ikl', PB_torch, hyper)
    # Get null-coefficient image b
    b = torch.einsum('ij,jkl->ikl', B_torch, rB)
    # Offset
    if offset: b = (b + vmax) / (2 * vmax)
    
    return rF, rB, b

def null2spec_modified(rgbn, b, S, B, vmax=1, offset=True, norm=False):
    # Convert numpy arrays to torch tensors
    S_torch = torch.tensor(S, dtype=rgbn.dtype, device=rgbn.device)
    B_torch = torch.tensor(B, dtype=rgbn.dtype, device=rgbn.device)
    
    # Get fundamental metamer rF
    S_inv = torch.linalg.inv(S_torch.T @ S_torch)
    rF = torch.einsum('ij,jkl->ikl', S_torch @ S_inv, rgbn)
    # Offset
    if offset: b = vmax * (2 * b - 1)
    # Get metameric black rB
    rB = torch.einsum('ij,jkl->ikl', B_torch, b)
    # Get final hypercube
    if norm:
        rF = normalize(rF)
        rB = normalize(rB)
        hyper = normalize(rF + rB)
    else:
        hyper = rF + rB
    
    return rF, rB, hyper

def get_target(S, r):
    """
    Args:
        S (68, 4) : sensitivity matrix
        r (68, HW) : reflectance
    Returns:
        r (68, HW) : target null-space projection img
    """
    PS, PN = get_projection(S)
    target = PN @ r.reshape(r.shape[0], -1)
    return target.reshape(r.shape)

def get_pred(S, a, bits=8, forward=True):
    """
    Args:
        S (68, 4) : sensitivity matrix
        a (64, HW) : null-space coefficient img
    Returns:
        r (68, HW) : predicted null-space projection img
    """
    N = get_null_basis(S)
    N = torch.tensor(N, dtype=a.dtype, device=a.device)
    pred = dnn_offset(a=a, bits=8, forward=True)
    pred = N @ pred.reshape(pred.shape[0], -1)
    return pred.reshape(pred.shape[0], a.shape[1], a.shape[2])

# ------------------------------------------------------------------------------
# TRUTHFUL FUNCTIONS
# ------------------------------------------------------------------------------

def get_hs_rgb(S, r, reshape=False):
    """
    Args:
        S (68, 4) : sensitivity matrix
        r (68, HW) : reflectance
    Returns:
        I (HW, 4) : RGB img
    """
    if r.shape[2] <= 4: 
        rgb = r @ S
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        return rgb
    if reshape: return (S.T @ r).reshape(S.shape[1], 512, 512)
    else: return S.T @ r

def get_null_coeff(N, r, offset=False, reshape=False, bits=8):
    """
    Args:
        N (68, 64) : null basis
        r (68, HW) : reflectance / hyperspectral img 
    Returns:
        a (64, HW) : null-space coefficient img
    """
    a = np.linalg.inv(N.T @ N) @ N.T @ r
    if offset: a = dnn_offset(a, bits=bits, forward=True)
    if reshape: a = a.reshape(N.shape[1], 512, 512)
    return a

def dnn_offset(a, factor=1, bits=8, forward=True):
    """
    Args:
        a (64, HW) : null-space coefficient img
        factor (int) : scaling constant, default = 1
        bits (int) : number of bits of input img, default = 8
    Returns:
        a (64, HW) : recentered null-space coefficient img
    """
    vmax = 2**bits - 1
    if forward: a_recenter = (a + factor * vmax) / (2 * factor * vmax)
    else: a_recenter = factor * vmax * (2 * a - 1)
    return a_recenter

    # return (a + B) / (2*B)

def get_cam_proj(S, I):
    """
    Args:
        S (68, 4) : sensitivity matrix
        I (4, HW) : RGB img
    Returns:
        r_parallel (68, HW) : camera subspace proection img
    """
    if isinstance(S, np.ndarray) and isinstance(I, np.ndarray):
        return S @ np.linalg.inv(S.T @ S) @ I
    elif isinstance(S, torch.Tensor) and isinstance(I, torch.Tensor):
        return S @ torch.linalg.inv(S.T @ S) @ I
    else:
        S = torch.tensor(S, dtype=I.dtype, device=I.device)
        return S @ torch.linalg.inv(S.T @ S) @ I

def get_null_proj(N, a):
    """
    Args:
        N (68, 64) : null basis
        a (64, HW) : null-space coefficient img
    Returns:
        r_perpendicular (68, HW) : null-space projection img
    """
    if (isinstance(N, np.ndarray) and isinstance(a, np.ndarray)) or (isinstance(N, torch.Tensor) and isinstance(a, torch.Tensor)):
        return N @ a
    else:
        N = torch.tensor(N, dtype=a.dtype, device=a.device)
        return N @ a

def get_hs_rec(S, I, N, a, offset=False, bits=8):
    """
    Args:
        S (68, 4) : sensitivity matrix
        I (4, HW) : RGB img
        N (68, 64) : null basis
        a (64, HW) : null-space coefficient img
    Returns:
        r (68, HW) : reflectance / hyperspectral img
    """
    r_parallel = get_cam_proj(S=S, I=I)
    # a = dnn_offset(a, bits=8, forward=True)
    r_perpendicular = get_null_proj(N=N, a=a)
    if offset: r_perpendicular = dnn_offset(a=r_perpendicular, bits=bits, forward=True)
    hs_rec = r_parallel + r_perpendicular
    # hs_rec = normalize(hs_rec)
    return hs_rec


# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def get_nir_sensitivity_matrix(range=np.arange(700, 1001, 2)[-35:], amp=2, mean=940, std=10):
    """
    Args:
        range (68,) : input range of gaussian basis
        amp (int) : gaussian amplitude, default = 2
        mean (int) : gaussian mean, default = 940
        std (int) : gaussian std, default = 10
    Returns:
        S_nir (68, 1) : sensitivity matrix
    """
    S_nir = amp * np.exp(-(range - mean)**2 / (2 * std**2)) # (68,)
    return S_nir.reshape((-1, 1))   # (68, 1)

def get_ciexyz_sensitivity_matrix(path, names, norm=True, return_w=False):
    """
    Args:
        path (str) : path to csv file containing ciexyz values
        names (str) : column names of csv file
        normalize (bool) : normalize values between 0 and 1, default = True
        return_w (bool) : return wavelength values, default = False
    Returns:
        S_ciexyz (68, 3) : sensitivity matrix
    """
    ciexyz = pd.read_csv(path, names=names)
    # picking 68 cols starting 400 nm
    ciexyz = ciexyz[ciexyz["Wavelength (nm)"] >= 400]
    ciexyz = ciexyz[:68]
    # normalizing values between 0 and 1
    if norm:
        ciexyz['Red'] = normalize(ciexyz['Red'])
        ciexyz['Green'] = normalize(ciexyz['Green'])
        ciexyz['Blue'] = normalize(ciexyz['Blue'])
    w = np.asarray(ciexyz["Wavelength (nm)"])                  # wavelengths values (68,)
    S_ciexyz = np.asarray(ciexyz.drop(["Wavelength (nm)"], axis=1)) # (68, 3)
    if return_w: return S_ciexyz, w
    return S_ciexyz

def get_sensitivity_matrix(ciexyz_path="./resources/ciexyz64.csv", names=['Wavelength (nm)', 'Red', 'Green', 'Blue'], amp=2, mean=940, std=10, stack_nir=True, norm=False):
    """
    Args:
        ciexyz_path (str) : path to csv file containing ciexyz values
        names (str) : column names of csv file
        amp (int) : gaussian amplitude, default = 2
        mean (int) : gaussian mean, default = 940
        std (int) : gaussian std, default = 10
    Returns:
        S (68, 4) : sensitivity matrix
    """
    S_ciexyz = get_ciexyz_sensitivity_matrix(ciexyz_path, names, norm=norm, return_w=False)    # (68, 3)
    S_nir = get_nir_sensitivity_matrix(range=np.arange(700, 1001, 2)[-68:], amp=amp, mean=mean, std=std)    # (68, 1)
    # print(S_ciexyz.shape, S_nir.shape)
    if stack_nir: return np.hstack((S_ciexyz, S_nir))    # (68, 4)
    else: return S_ciexyz

def get_projection(S):
    PS = S @ np.linalg.inv(S.T @ S) @ S.T
    PN = np.eye(S.shape[0]) - PS
    return PS, PN

def get_null_basis(S, num_of_basis=64, fn_scipy=True):
    """
    Args:
        S (68, 4) : sensitivity matrix
        num_of_basis (int) : number of basis, default = 64
    Returns:
        N (68, num_of_basis) : null basis / reduced sensitivity matrix
    """
    num_of_basis = S.shape[0] - S.shape[1]
    if fn_scipy: 
        N = scipy.linalg.null_space(S.T)
    else: 
        PS, PN = get_projection(S)                      # (68, 68)
        _, _, vt = np.linalg.svd(PN, full_matrices=False)
        N = vt.T[:, :num_of_basis]                       # (68, num_of_basis)
    return N

def normalize(x):
    try:
        norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    except:
        norm = (x - x.min()) / (x.max() - x.min())
    return norm

def reflip(x, transpose=True):
    if transpose: x = x.transpose(0, 2, 1)
    else:
        x = cv2.flip(x, 0)
        x = cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)
    return x


# ------------------------------------------------------------------------------
# EVALUATION HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def plot_spectral_signature(hyper_img, pixel_coords=None, color="b", img_title=None, save_path=None):
    """
    Plots the spectral signature of a hyperspectral image at a specific pixel.

    Args:
        hyper_img (numpy.ndarray): Hyperspectral image of shape (bands, height, width).
        pixel_coords (tuple): Coordinates of the pixel (row, col) to plot.
                              If None, the center pixel is used.

    Returns:
        None: Displays the plot.
    """
    # Validate input shape
    if len(hyper_img.shape) != 3:
        raise ValueError("Input hyperspectral image must have shape (bands, height, width).")
    
    bands, height, width = hyper_img.shape

    # Set default pixel to the center if not specified
    if pixel_coords is None:
        pixel_coords = (height // 2, width // 2)
    
    row, col = pixel_coords
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError(f"Pixel coordinates {pixel_coords} are out of bounds for image dimensions ({height}, {width}).")
    
    # Extract spectral signature for the specified pixel
    spectral_signature = hyper_img[:, row, col]
    
    # Plot the spectral signature
    plt.figure(figsize=(8, 5))
    plt.xlim(0, bands + 1)
    plt.ylim(-0.1, 1)
    plt.plot(range(1, bands + 1), spectral_signature, color=color, linewidth=5)
    if img_title is not None: plt.title(f"{img_title} at px({row}, {col})")
    plt.xlabel("Band Index")
    plt.ylabel("Reflectance/Intensity")
    plt.legend([f"px({row}, {col})"], loc="upper left")
    plt.grid(True)
    plt.tight_layout()

    # Save if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Visualization saved to {save_path}")

    plt.show()

def visualize_all_bands(cube, title="", save_path=None):
    """
    Visualize all bands of a hyperspectral cube in a grid layout.
    
    Args:
        cube (numpy.ndarray): Hyperspectral image cube with shape (height, width, n_bands).
        title (str): Title for the overall visualization.
        save_path (str): File path to save the visualization (optional).
    """
    # Get dimensions of the cube
    n_bands, height, width = cube.shape

    # Determine grid size (rows and cols)
    cols = 8
    rows = (n_bands + cols - 1) // cols  # Ceiling division for rows

    # Create a figure
    fig, axes = plt.subplots(rows, cols, figsize=(16, 2 * rows))
    if title is not None: fig.suptitle(title, fontsize=16)

    # Plot each band
    for i in range(rows * cols):
        ax = axes.flat[i]  # Access axes in flat iteration
        if i < n_bands:
            ax.imshow(cube[i, :, :], cmap="gray")
            ax.set_title(f"Band {i + 1}")
        ax.axis("off")  # Turn off axes for all plots

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to include title

    # Save if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Visualization saved to {save_path}")

    plt.show()

import os
import matplotlib.pyplot as plt

def visualize_all_bands_modified(cube, save_name, save_dir, title=None):
    """
    Visualize all bands of a hyperspectral cube in a compact grid layout and save the figure.
    
    Args:
        cube (numpy.ndarray): Hyperspectral image cube with shape (bands, height, width).
        id_name (str): Identifier for naming the saved file (e.g., original ID).
        save_dir (str): Directory where the visualization will be saved.
        title (str): Title for the visualization.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Cube dimensions
    n_bands, height, width = cube.shape

    # Grid layout
    cols = 10  # Wider layout (more compact)
    rows = (n_bands + cols - 1) // cols

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(20, 2.0 * rows))
    if title is not None: fig.suptitle(title, fontsize=18, weight='bold')

    # Plot each band
    for idx in range(rows * cols):
        ax = axes.flat[idx]
        if idx < n_bands:
            ax.imshow(cube[idx, :, :], cmap='gray')
            ax.set_title(f"B{idx+1}", fontsize=7, pad=2)
        ax.axis('off')

    # Adjust layout to be compact but readable
    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=1.0)

    # Save figure
    save_path = os.path.join(save_dir, f"{save_name}_mat.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)  # Important: close the figure to free memory

    print(f"✅ Visualization saved to {save_path}")

import os
import matplotlib.pyplot as plt

def visualize_rgb(cube, save_name, save_dir):
    """
    Visualize the RGB channels (first 3 channels) of a 4-channel image and save the figure.

    Args:
        cube (numpy.ndarray): 4-channel image (H, W, 4), channels assumed as R, G, B, NIR.
        save_name (str): Name to save the figure (without extension).
        save_dir (str): Directory to save the figure.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Take the first 3 channels (RGB)
    rgb_image = cube[:3, :, :]
    rgb_image = np.transpose(rgb_image, (1, 2, 0))

    # Clip values if necessary (optional: depending on input range)
    rgb_image = rgb_image.clip(0, 1)

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 5))  # Compact figure
    ax.imshow(rgb_image)
    ax.axis('off')

    # Tight layout
    plt.tight_layout(pad=0)

    # Save figure
    save_path = os.path.join(save_dir, f"{save_name}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)  # Close figure to free memory

    print(f"✅ RGB visualization saved to {save_path}")

def read_hs_rec(data_root):
    data = [ os.path.join(data_root, filename) for filename in os.listdir(data_root) ]
    data.sort()
    imgs = []
    for path in tqdm(data):
        if path.split('.')[-1] != 'mat': continue
        img = h5py.File(path)["cube"]
        img = np.transpose(img, (0, 2, 1))
        imgs.append(img)
    return np.array(imgs)

def read_hs_rec_modified(data_root):
    data = [ os.path.join(data_root, filename) for filename in os.listdir(data_root) ]
    data.sort()
    imgs = []
    for path in tqdm(data):
        if path.split('.')[-1] != 'mat': continue
        cube = hdf5storage.loadmat(path,variable_names=['cube'])
        # hs = cube['rad'][:,:,1:204:3]
        img = cube['cube']
        img = np.transpose(img, [2, 0, 1])
        imgs.append(img)
    return np.array(imgs)

# adapted from https://github.com/mobispectral/mobicom23_mobispectral/
def read_hs_orig(hyper_data_root, rgb_data_root, split_root, list_name="test", return_list=False, idx=None):
    orig_hs, orig_rgb_nir = [], []
    hyper_data_path = f'{hyper_data_root}/HS_GT/'
    rgb_data_path = f'{rgb_data_root}/RGBN/'
    with open(f'{split_root}/split_txt/{list_name}_list.txt', 'r') as fin:
        hs_list = [line.replace('\n', '.mat') for line in fin]
        rgb_list = [line.replace('.mat','_RGB.png') for line in hs_list]
        nir_list = [line.replace('.mat','_NIR940.png') for line in hs_list]
    hs_list.sort()
    rgb_list.sort()
    nir_list.sort()

    if return_list: return hs_list, rgb_list, nir_list

    if idx is None: idx = len(hs_list)
    for i in tqdm(range(len(hs_list))[:idx]):
        hs_path = hyper_data_path + hs_list[i]
        cube = hdf5storage.loadmat(hs_path,variable_names=['rad'])
        # hs = cube['rad'][:,:,1:204:3]
        hs = cube['rad']
        hs = np.transpose(hs, [2, 0, 1])
        orig_hs.append(hs)
        rgb_path = rgb_data_path + rgb_list[i]
        nir_path = rgb_data_path + nir_list[i]
        orig_rgb = normalize(plt.imread(rgb_path))
        orig_nir = normalize(plt.imread(nir_path))
        rgb_nir = np.dstack((orig_rgb, orig_nir))
        rgb_nir = np.transpose(rgb_nir, [2, 0, 1])
        orig_rgb_nir.append(rgb_nir)
    return np.array(orig_hs), np.array(orig_rgb_nir), hs_list


