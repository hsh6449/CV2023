import torch
from torch.utils.data import DataLoader, Dataset

import os
import numpy as np
import matplotlib.pyplot as plt

### 지워도됨 
class coarse_seg_dataset(Dataset):
    def __init__(self, split='train', seg_path='coarse_seg'):
        self.seg_path = seg_path
        self.split = split

    def __len__(self):
        return len(os.listdir(os.path.join(self.seg_path, self.split)))

    def __getitem__(self, idx):
        # load npz file and get logits
        seg = np.load(os.path.join(self.seg_path, f'{self.split}/{idx}.npy'))[0][0]
        seg = torch.from_numpy(seg)
        return seg