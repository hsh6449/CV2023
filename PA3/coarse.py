# import Library
import os
import torch
import numpy as np

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset

# import your model or define here
from model import Unet, CSPN

# If you want to use args, you can use
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# This dataset is for coarse segmentation 
# You don't need to use this dataset if you want to make your own dataset
# If you use this dataset, you have to save logits as npz file(corase segmentation)
# It is ok to use data augmentation for this dataset

def fix_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class SegDataset(torch.utils.data.Dataset): 
  def __init__(self, split='train', seg_path='coarse_seg'):

    if split == 'train':
        self.dataset = train_dataset
    elif split == 'val':
        self.dataset = valid_dataset

    self.seg_path = seg_path
    self.split = split

  def __len__(self): 
    return len(self.dataset)
    
  def __getitem__(self, idx): 
    dataset_dic = self.dataset[idx]
    # load npz file and get logits 
    seg = np.load(os.path.join(self.seg_path, f'{self.split}/{idx}.npy'))[0]
    seg = torch.from_numpy(seg)
    return dataset_dic, seg

### Download data
root = "."
# SimpleOxfordPetDataset.download(root)
n_cpu = os.cpu_count()

### For coarse segmentation
train_dataset = SimpleOxfordPetDataset(root, "train")
valid_dataset = SimpleOxfordPetDataset(root, "valid")

# train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=n_cpu)
# valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=n_cpu)
# print(f"Train size: {len(train_dataset)}")
# print(f"val size: {len(valid_dataset)}")

### For CSPN
train_dataset_seg = SegDataset(split='train')
valid_dataset_seg = SegDataset(split='valid')
train_dataloader_seg = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=n_cpu)
valid_dataloader_seg = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=n_cpu)
print(f"Train size: {len(train_dataset_seg)}")
print(f"val size: {len(valid_dataset_seg)}")


class Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
         # Define all the models you want to use here
         # TODO
        self.model = Unet()
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.outputs = []

    def forward(self, image):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        # This is for coarse segmentation
        #dic, seg_init = batch
        #image = dic['image']
        #input_batch = torch.cat((image, seg_init), dim=1)

        dic = batch
        image = dic['image'].to(torch.float32)
        input_batch = image

        assert image.ndim == 4

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = dic["mask"]
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(input_batch)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        # self.outputs = pred_mask

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        self.outputs.append({
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })

        return loss

    def shared_epoch_end(self, stage):

        tp = torch.cat([x["tp"] for x in self.outputs])
        fp = torch.cat([x["fp"] for x in self.outputs])
        fn = torch.cat([x["fn"] for x in self.outputs])
        tn = torch.cat([x["tn"] for x in self.outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def on_training_epoch_end(self):
        return self.shared_epoch_end("train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def on_test_epoch_end(self):
        return self.shared_epoch_end("test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

def eval():
    model.eval()
    
    model.to(device)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)

    for i, sample in enumerate(train_dataloader):
        x = torch.tensor(sample["image"], dtype=torch.float32).to(device)
        y = torch.tensor(sample["mask"]).to(device)
        y = y.squeeze(0)

        y_pred = model(x)
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.squeeze(0)

        y_pred_binary = (y_pred > 0.5).to(torch.float32)
        y_true_binary = (y > 0.0).to(torch.float32)

        tp, fp, fn, tn = smp.metrics.get_stats(y_pred_binary.long(), y_true_binary.long(), mode="binary")

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        print(f"Train - Epoch: {trainer.current_epoch}, Image {i}, IoU: {per_image_iou.item()}")
        np.save(f"coarse_seg/train/{i}.npy", y_pred_binary.to("cpu").detach().numpy())

    for i, sample in enumerate(valid_dataloader):
        x = torch.tensor(sample["image"], dtype=torch.float32).to(device)
        y = torch.tensor(sample["mask"]).to(device)
        y = y.squeeze(0)

        y_pred = model(x)
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.squeeze(0)

        y_pred_binary = (y_pred > 0.5).to(torch.float32)
        y_true_binary = (y > 0.5).to(torch.float32)

        tp, fp, fn, tn = smp.metrics.get_stats(y_pred_binary.long(), y_true_binary.long(), mode="binary")

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        print(f"Valid - Epoch: {trainer.current_epoch}, Image {i}, IoU: {per_image_iou.item()}")
        np.save(f"coarse_seg/valid/{i}.npy", y_pred_binary.to("cpu").detach().numpy())

if __name__ == "__main__":
    fix_seed(0)

    model = Model(args)
    trainer = pl.Trainer(
        # gpus=1,
        max_epochs=10,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )

    # eval()
    eval() 