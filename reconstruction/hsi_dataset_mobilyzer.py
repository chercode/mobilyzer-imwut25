from torch.utils.data import Dataset
import numpy as np
import random, cv2, h5py, hdf5storage, scipy
from imageio import imread
from tqdm import tqdm
import utils_truthful as tru


class TrainDataset(Dataset):
    def __init__(self, data_root, split_root, crop_size, S, arg=True, bgr2rgb=True, stride=8, truthful=False, norm=False):
        self.arg = arg
        self.stride = stride
        self.crop_size = crop_size
        self.hypers, self.vnir = [], []
        h, w = 512, 512  # img shape
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum

        hyper_data_path = f'{data_root}/Train_Spec/'
        bgr_data_path = f'{data_root}/Train_RGB/'
        # bgr_data_path = f'{data_root}/Train_XYZ/'
        # bgr_data_path = f'{data_root}/Train_RGB_intrinsic_anchored/'
        # bgr_data_path = f'{data_root}/Train_RGB_wb/'
        # bgr_data_path = f'{data_root}/Train_RGB_hr/'
        # bgr_data_path = f'{data_root}/Train_RGB_wb_gc/'
        # bgr_data_path = f'{data_root}/Train_RGB_dif_img_gc/'
        # bgr_data_path = f'{data_root}/Train_RGB_hr_gc/'

        # for debugging
        print(f"HSI data path : {hyper_data_path}")
        print(f"RGB data path : {bgr_data_path}")

        with open(f'{split_root}/split_txt/train_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n','.mat') for line in fin]
            nir_list = [line.replace('.mat','_NIR.jpg') for line in hyper_list]
            bgr_list = [line.replace('.mat','_RGB.jpg') for line in hyper_list]
            # bgr_list = [line.replace('.mat','_RGB_D_gc.png') for line in hyper_list]
            # bgr_list = [line.replace('.mat','_XYZ.png') for line in hyper_list]

        data_list = [hyper_list, nir_list, bgr_list]
        for data in data_list:
            data.sort()

        print(f'len(hyper) of MobiSpectral dataset: {len(hyper_list)}')
        print(f'len(bgr) of MobiSpectral dataset: {len(bgr_list)}')
        print(f"Reading Spectral scenes ...")

        for i in tqdm(range(len(hyper_list))):
            # HS imgs
            hyper_path = hyper_data_path + hyper_list[i]
            cube = hdf5storage.loadmat(hyper_path, variable_names=['rad'])
            hyper = cube['rad'][:, :, 1:204:3]          # (512, 512, 68)
            # RGBN imgs
            vnir = hyper @ S                            # (512, 512, 4)
            vnir = np.transpose(vnir, [2, 0, 1])        # (4, 512, 512)
            vnir = np.float32(vnir)
            if norm: 
                vnir = tru.normalize(vnir)
            # if truthful:
            #     self.vnir.append(vnir)
            self.vnir.append(vnir)

            hyper = np.transpose(hyper, [2, 0, 1])      # (68, 512, 512)
            self.hypers.append(hyper)
            # continue

            # NIR imgs
            nir_path = bgr_data_path + nir_list[i]
            nir = imread(nir_path)
            nir = np.float32(nir)
            if norm:
                nir = (nir - nir.min()) / (nir.max() - nir.min())

            # RGB imgs
            # read from illumination list
            illumination_list = [bgr_list]
            for img_list in illumination_list:
                path = bgr_data_path + img_list[i]
                img = imread(path)
                img = np.float32(img)
                if norm:
                    img = (img - img.min()) / (img.max() - img.min())
                # stack RGB and NIR
                img = np.dstack((img, nir))             # (512, 512, 4)
                img = np.transpose(img, [2, 0, 1])      # (4, 512, 512)
                # if not truthful:
                #     self.vnir.append(img)
                # self.vnir.append(img)

        self.img_num = len(self.hypers)
        self.length = self.patch_per_img * self.img_num

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr = self.vnir[img_idx]
        hyper = self.hypers[img_idx]
        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.patch_per_img*self.img_num

class ValidDataset(Dataset):
    def __init__(self, data_root, split_root, S, bgr2rgb=True, truthful=False, norm=False):
        self.hypers, self.vnir = [], []
        hyper_data_path = f'{data_root}/Train_Spec/'
        bgr_data_path = f'{data_root}/Train_RGB/'
        # bgr_data_path = f'{data_root}/Train_XYZ/'
        # bgr_data_path = f'{data_root}/Train_RGB_intrinsic_anchored/'
        # bgr_data_path = f'{data_root}/Train_RGB_wb/'
        # bgr_data_path = f'{data_root}/Train_RGB_hr/'
        # bgr_data_path = f'{data_root}/Train_RGB_wb_gc/'
        # bgr_data_path = f'{data_root}/Train_RGB_dif_img_gc/'
        # bgr_data_path = f'{data_root}/Train_RGB_hr_gc/'

        with open(f'{split_root}/split_txt/valid_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n','.mat') for line in fin]
            nir_list = [line.replace('.mat','_NIR.jpg') for line in hyper_list]
            bgr_list = [line.replace('.mat','_RGB.jpg') for line in hyper_list]
            # bgr_list = [line.replace('.mat','_RGB_D_gc.png') for line in hyper_list]
            # bgr_list = [line.replace('.mat','_XYZ.png') for line in hyper_list]

        data_list = [hyper_list, nir_list, bgr_list]
        for data in data_list:
            data.sort()

        print(f'len(hyper) of MobiSpectral dataset: {len(hyper_list)}')
        print(f'len(bgr) of MobiSpectral dataset: {len(bgr_list)}')
        print(f"Reading Spectral scenes ...")
        
        for i in tqdm(range(len(hyper_list))):
            # HS imgs
            hyper_path = hyper_data_path + hyper_list[i]
            cube = hdf5storage.loadmat(hyper_path, variable_names=['rad'])
            hyper = cube['rad'][:, :, 1:204:3]          # (512, 512, 68)
            # RGBN imgs
            vnir = hyper @ S                            # (512, 512, 4)
            vnir = np.transpose(vnir, [2, 0, 1])        # (4, 512, 512)
            vnir = np.float32(vnir)
            if norm:
                vnir = tru.normalize(vnir)
            # if truthful:
            #     self.vnir.append(vnir)
            self.vnir.append(vnir)

            hyper = np.transpose(hyper, [2, 0, 1])      # (68, 512, 512)
            self.hypers.append(hyper)
            # continue

            # NIR imgs
            nir_path = bgr_data_path + nir_list[i]
            nir = imread(nir_path)
            nir = np.float32(nir)
            if norm:
                nir = (nir - nir.min()) / (nir.max() - nir.min())

            # RGB imgs
            # read from illumination list
            illumination_list = [bgr_list]
            for img_list in illumination_list:
                path = bgr_data_path + img_list[i]
                img = imread(path)
                img = np.float32(img)
                if norm:
                    img = (img - img.min()) / (img.max() - img.min())
                # stack RGB and NIR
                img = np.dstack((img, nir))             # (512, 512, 4)
                img = np.transpose(img, [2, 0, 1])      # (4, 512, 512)
                # if not truthful:
                #     self.vnir.append(img)
                # self.vnir.append(img)

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgrn = self.vnir[idx]
        return np.ascontiguousarray(bgrn), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)



class TestDataset(Dataset):
    def __init__(self, data_root, split_root, S, bgr2rgb=True, truthful=False, norm=False):
        self.hypers, self.vnir = [], []
        hyper_data_path = f'{data_root}/Train_Spec/'
        bgr_data_path = f'{data_root}/Train_RGB/'
        # bgr_data_path = f'{data_root}/Train_XYZ/'
        # bgr_data_path = f'{data_root}/Train_RGB_intrinsic_anchored/'
        # bgr_data_path = f'{data_root}/Train_RGB_wb/'
        # bgr_data_path = f'{data_root}/Train_RGB_hr/'
        # bgr_data_path = f'{data_root}/Train_RGB_wb_gc/'
        # bgr_data_path = f'{data_root}/Train_RGB_dif_img_gc/'
        # bgr_data_path = f'{data_root}/Train_RGB_hr_gc/'

        with open(f'{split_root}/split_txt/test_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n','.mat') for line in fin]
            nir_list = [line.replace('.mat','_NIR.jpg') for line in hyper_list]
            bgr_list = [line.replace('.mat','_RGB.jpg') for line in hyper_list]
            # bgr_list = [line.replace('.mat','_RGB_D_gc.png') for line in hyper_list]
            # bgr_list = [line.replace('.mat','_XYZ.png') for line in hyper_list]

        data_list = [hyper_list, nir_list, bgr_list]
        for data in data_list:
            data.sort()

        print(f'len(hyper) of MobiSpectral dataset: {len(hyper_list)}')
        print(f'len(bgr) of MobiSpectral dataset: {len(bgr_list)}')
        print(f"Reading Spectral scenes ...")
        
        for i in tqdm(range(len(hyper_list))):
            # HS imgs
            hyper_path = hyper_data_path + hyper_list[i]
            cube = hdf5storage.loadmat(hyper_path, variable_names=['rad'])
            hyper = cube['rad'][:, :, 1:204:3]          # (512, 512, 68)
            # RGBN imgs
            vnir = hyper @ S                            # (512, 512, 4)
            vnir = np.transpose(vnir, [2, 0, 1])        # (4, 512, 512)
            vnir = np.float32(vnir)
            if norm:
                vnir = tru.normalize(vnir)
            # if truthful:
            #     self.vnir.append(vnir)
            self.vnir.append(vnir)

            hyper = np.transpose(hyper, [2, 0, 1])      # (68, 512, 512)
            self.hypers.append(hyper)
            # continue

            # NIR imgs
            nir_path = bgr_data_path + nir_list[i]
            nir = imread(nir_path)
            nir = np.float32(nir)
            if norm:
                nir = (nir - nir.min()) / (nir.max() - nir.min())

            # RGB imgs
            # read from illumination list
            illumination_list = [bgr_list]
            for img_list in illumination_list:
                path = bgr_data_path + img_list[i]
                img = imread(path)
                img = np.float32(img)
                if norm:
                    img = (img - img.min()) / (img.max() - img.min())
                # stack RGB and NIR
                img = np.dstack((img, nir))             # (512, 512, 4)
                img = np.transpose(img, [2, 0, 1])      # (4, 512, 512)
                # if not truthful:
                #     self.vnir.append(img)
                # self.vnir.append(img)

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgrn = self.vnir[idx]
        return np.ascontiguousarray(bgrn), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)

