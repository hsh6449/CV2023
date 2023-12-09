import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset

from main import SegDataset, train_dataset, valid_dataset

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset_seg = SegDataset(split='train_dyspn', seg_path='affinity')
valid_dataset_seg = SegDataset(split='valid_dyspn', seg_path='affinity')
train_dataloader_seg = DataLoader(train_dataset_seg, batch_size=1, shuffle=False, num_workers=2)
valid_dataloader_seg = DataLoader(valid_dataset_seg, batch_size=1, shuffle=False, num_workers=2)

def eval(mode=True):
    
    result = []
    for i, (sample, seg) in enumerate(train_dataloader_seg):

        y = torch.tensor(sample["mask"]).to(device)
        seg = torch.tensor(seg).to(device)  

        tp, fp, fn, tn = smp.metrics.get_stats(seg.long(), y.long(), mode="binary")

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        result.append(per_image_iou.item())

        print(f"Train - Image {i}, IoU: {per_image_iou.item()}")
    print(f"Train MIOU - {np.mean(result)}")
    
    result = []
    for i, (sample, seg) in enumerate(valid_dataloader_seg):

        y = torch.tensor(sample["mask"]).to(device)
        seg = torch.tensor(seg).to(device)  

        tp, fp, fn, tn = smp.metrics.get_stats(seg.long(), y.long(), mode="binary")

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        print(f"Train - Image {i}, IoU: {per_image_iou.item()}")
        result.append(per_image_iou.item())

    print(f"Valid MIOU - {np.mean(result)}")
if __name__ == "__main__":
    eval()