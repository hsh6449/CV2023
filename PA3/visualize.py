import numpy as np
import matplotlib.pyplot as plt
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset

from torchvision.transforms.functional import to_pil_image

root = "."
train_dataset = SimpleOxfordPetDataset(root, "train")
valid_dataset = SimpleOxfordPetDataset(root, "valid")

def visual(num, split="train", c=True):
  if split == "train":
    dataset = train_dataset
  elif split == "valid":
    dataset = valid_dataset
  if c == True:
    mode = "cspn"
    segmentation = np.load(f"affinity/{split}/{num}.npy")
  else:
    mode = "dyspn"
    segmentation = np.load(f"affinity/{split}_{mode}/{num}.npy")
  images = dataset[num]["image"]
  mask =np.load(f"coarse_seg/{split}/{num}.npy")
  gt = dataset[num]["mask"]

  # Copy the original image for overlaying masks
  overlay_image = images.copy()
  overlay_image2 = images.copy()
  mask = mask.squeeze()
  # Set the pixels outside the mask to black
  overlay_image[:, mask == 0] = 0
  overlay_image2[:, segmentation[0] == 0] = 0

  plt.figure(figsize=(15, 5))
  plt.subplot(1, 4, 1)
  plt.title("Original image")
  plt.imshow(to_pil_image(images.transpose(1,2,0)))
  plt.subplot(1, 4, 2)
  plt.title("Coarse segmentation")
  plt.imshow(to_pil_image(overlay_image.transpose(1,2,0)), cmap="gray")
  plt.subplot(1, 4, 3)
  plt.title("After Affinity")
  plt.imshow(to_pil_image(overlay_image2.transpose(1,2,0)), cmap="gray")
  plt.subplot(1, 4, 4)
  plt.title("Ground truth")
  plt.imshow(gt.squeeze(), cmap="gray")
  plt.show()

def intermidiate_vis(num, iter, split="train", c=True):
  if split == "train":
    dataset = train_dataset
  elif split == "valid":
    dataset = valid_dataset

  images = dataset[num]["image"]
  mask =np.load(f"coarse_seg/{split}/{num}.npy")
  gt = dataset[num]["mask"]

  # Copy the original image for overlaying masks
  overlay_image = images.copy()
  mask = mask.squeeze()
  # Set the pixels outside the mask to black
  overlay_image[:, mask == 0] = 0

  plt.figure(figsize=(15, 5))
  plt.subplot(1, iter+2, 1)
  plt.title("Original image")
  plt.imshow(to_pil_image(images.transpose(1,2,0)))
  plt.subplot(1, iter+2, 2)
  plt.title("Coarse segmentation")
  plt.imshow(to_pil_image(overlay_image.transpose(1,2,0)), cmap="gray")
  for i in range(iter):
    if c == True:
      result = np.load(f"intermidiate_result/{split}/{num}.npy")
    else:
      result = np.load(f"intermidiate_result/{split}_dyspn/{num}.npy")
    plt.subplot(1, iter+2, i+3)
    plt.title(f"Iteration {i+1}")
    plt.imshow(to_pil_image(result[i][0].squeeze()))
  # plt.subplot(1, iter+2, 9)
  # plt.imshow(gt.squeeze(), cmap="gray")
  plt.show()

c = True
intermidiate_vis(354, 5, "valid", c=c) # valid
visual(354, "valid",c= c) # valid
# epoch_vis(0, "train")
