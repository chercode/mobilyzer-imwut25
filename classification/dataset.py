# dataset.py
import os
from glob import glob
import numpy as np
import hdf5storage
from torch.utils.data import Dataset

def load_mat(mat_name, var_name):
    """Helper to load a single variable from a .mat file."""
    data = hdf5storage.loadmat(mat_name, variable_names=[var_name])
    return data[var_name]

class DatasetFromDirectory(Dataset):
    IMAGE_SIZE = 32    # we want 64×64 center patches
    BLOCK_SIZE = 1
    var_name   = "cube"
    labelslist = ["0", "1", "2", "3", "lf"]  # only these three subfolders

    def __init__(self, root, dataset_dir, fruit):
        self.root   = root
        self.fruit  = fruit
        self.labels = []
        self.mats   = []

        # scan only the three known class folders
        for directory in glob(os.path.join(root, dataset_dir, "*")):
            folder_name = os.path.basename(directory)
            if folder_name not in self.labelslist:
                continue

            for mat_file in glob(os.path.join(directory, "*.mat")):
                # one signature per pixel in the 64×64 patch
                patches_per_image = (self.IMAGE_SIZE // self.BLOCK_SIZE) ** 2
                for _ in range(patches_per_image):
                    self.labels.append(folder_name)
                    self.mats.append(mat_file)

        # reset iterators
        self.row_itr   = 0
        self.col_itr   = 0
        self.loaded_mat = None

    def __len__(self):
        return len(self.labels)

    def get_mat(self, mat_path):
        mat = load_mat(mat_path, self.var_name)
        # extract 64×64 center patch
        h, w, c = mat.shape
        ch, cw = h // 2, w // 2
        start_h = ch - (self.IMAGE_SIZE // 2)
        start_w = cw - (self.IMAGE_SIZE // 2)
        return mat[start_h:start_h+self.IMAGE_SIZE,
                   start_w:start_w+self.IMAGE_SIZE, :]

    def divide_mat(self, mat):
        return mat[self.row_itr:self.row_itr+self.BLOCK_SIZE,
                   self.col_itr:self.col_itr+self.BLOCK_SIZE, :]

    def __getitem__(self, index):
        # on first pixel of each image, load the new mat
        if self.row_itr == 0 and self.col_itr == 0:
            self.loaded_mat = self.get_mat(self.mats[index])

        # grab one BLOCK_SIZE×BLOCK_SIZE×68 signature
        sig = self.divide_mat(self.loaded_mat)

        # advance pixel iterator
        self.row_itr += self.BLOCK_SIZE
        if self.row_itr >= self.IMAGE_SIZE:
            self.row_itr = 0
            self.col_itr += self.BLOCK_SIZE
            if self.col_itr >= self.IMAGE_SIZE:
                self.col_itr = 0

        label = self.labels[index]
        label = self.labelslist.index(label)
        return sig, label
