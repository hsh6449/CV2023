import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

import pytorch_lightning as pl
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from torch.utils.data import DataLoader

from model import Unet
from tqdm import tqdm
import os
import pdb

root = "."
# SimpleOxfordPetDataset.download(root)

train_dataset = SimpleOxfordPetDataset(root, "train")
valid_dataset = SimpleOxfordPetDataset(root, "valid")

print(f"Train size: {len(train_dataset)}")
print(f"val size: {len(valid_dataset)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_cpu = os.cpu_count()

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=n_cpu)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=n_cpu)

model = Unet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 10

print(next(iter(train_dataloader)))
def train():
  for epoch in tqdm(range(epochs)):
      for idx, (x, y) in enumerate(train_dataloader):
          optimizer.zero_grad()
          logits = model(x.to(device))
          loss = F.binary_cross_entropy_with_logits(logits, y)
          loss.backward()
          optimizer.step()
          print(f"Epoch: {epoch}, Iter: {idx}, Loss: {loss.item()}")

      for idx, (x, y) in enumerate(valid_dataloader):
          logits = model(x)
          loss = F.binary_cross_entropy_with_logits(logits, y)
          print(f"Epoch: {epoch}, Iter: {idx}, Val Loss: {loss.item()}")


def plot_img(x, y, y_pred):
    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].imshow(x)
    ax[1].imshow(y)
    ax[2].imshow(y_pred)
    plt.show()

def test():
    model.eval()
    x, y = valid_dataset[0]
    y_pred = model(x.unsqueeze(0))
    y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.squeeze(0)
    y_pred = y_pred.squeeze(0)
    y_pred = y_pred.detach().numpy()
    y_pred = y_pred > 0.5
    plot_img(x, y, y_pred)

if __name__ == "__main__":
    train()
    test()