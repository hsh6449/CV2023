import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

# import pytorch_lightning as pl
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from torch.utils.data import DataLoader

from model import Unet_affinity as Unet
from tqdm import tqdm

import segmentation_models_pytorch as smp
import os
import pdb

root = "." # PA3
# SimpleOxfordPetDataset.download(root)

train_dataset = SimpleOxfordPetDataset(root, "train")
valid_dataset = SimpleOxfordPetDataset(root, "valid")

print(f"Train size: {len(train_dataset)}")
print(f"val size: {len(valid_dataset)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_cpu = os.cpu_count()

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

model = Unet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_f = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
epochs = 3

# print(next(iter(train_dataloader)))


def train():
  for epoch in tqdm(range(epochs)):
      for idx, samples in enumerate(train_dataloader):
          optimizer.zero_grad()

          x = samples["image"].to(device).to(torch.float32)
          y = samples["mask"].to(device)

          logits = model(x)
          # loss = F.binary_cross_entropy_with_logits(logits, y)
          loss = loss_f(logits, y)
          loss.backward()
          optimizer.step()

          if idx % 10 == 0:
            print(f"Epoch: {epoch+1}, Iter: {idx}, Loss: {loss.item()}")

      for idx, samples in enumerate(valid_dataloader):
          x = samples["image"].to(device).to(torch.float32)
          y = samples["mask"].to(device)
          logits = model(x)
          loss = loss_f(logits, y)
          print(f"Epoch: {epoch+1}, Iter: {idx}, Val Loss: {loss.item()}")


def plot_img(x, y, y_pred):
    fig, ax = plt.subplots(1, 3, figsize=(15, 15))

    x = x.squeeze(0).to("cpu").detach().numpy()
    y = y.to("cpu").detach().numpy()
    ax[0].imshow(x.reshape(256,256,3))
    ax[1].imshow(y.reshape(256,256))
    ax[2].imshow(y_pred.reshape(256,256))
    plt.show()

def eval():
    ## make coase segmentation

    model.eval()

    for i , sample in enumerate(train_dataloader):
        x = torch.tensor(sample["image"], dtype=torch.float32).to(device)
        y = torch.tensor(sample["mask"]).to(device)

        y_pred = model(x)
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.squeeze(0)
        
        y_pred = y_pred.to("cpu").detach().numpy()
        y_pred = (y_pred > 0.3)

        np.save(f"coarse_seg/train/{i}.npy", y_pred)

    for i , sample in enumerate(valid_dataloader):
    # sample = valid_dataset[0]
        x = torch.tensor(sample["image"], dtype=torch.float32).to(device)
        y = torch.tensor(sample["mask"]).to(device)

        y_pred = model(x)
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.squeeze(0)
        
        y_pred = y_pred.to("cpu").detach().numpy()
        y_pred = np.array([[1 if element > 0 else 0 for element in row] for row in y_pred])

        np.save(f"coarse_seg/valid/{i}.npy", y_pred)

    # plt.imsave(f"PA3/unet_seg/{i}.png", y_pred)

    # plot_img(x, y, y_pred)

if __name__ == "__main__":
    train()
    torch.save(model.state_dict(),"unet_seg/unet_seg.pth")
    eval()