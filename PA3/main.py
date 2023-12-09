# import Library
import os
import torch
import numpy as np

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset

# import your model or define here
from model import Unet_affinity, CSPN, DYSPN, Unet_d
import time 

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

class SegDataset(Dataset): 
  def __init__(self, split='train', seg_path='coarse_seg'):

    if (split == 'train') or (split == 'train_dyspn'):
        self.dataset = train_dataset
    elif (split == 'valid') or (split == 'valid_dyspn'):
        self.dataset = valid_dataset

    self.seg_path = seg_path
    self.split = split

  def __len__(self): 
    return len(self.dataset)
    
  def __getitem__(self, idx): 
    dataset_dic = self.dataset[idx]
    # load npz file and get logits 
    seg = np.load(os.path.join(self.seg_path, f'{self.split}/{idx}.npy'))
    seg = torch.from_numpy(seg)
    return dataset_dic, seg

root = "."
n_cpu = os.cpu_count()

train_dataset = SimpleOxfordPetDataset(root, "train")
valid_dataset = SimpleOxfordPetDataset(root, "valid")

### For CSPN
train_dataset_seg = SegDataset(split='train')
valid_dataset_seg = SegDataset(split='valid')
train_dataloader_seg = DataLoader(train_dataset_seg, batch_size=8, shuffle=True, num_workers=2)
valid_dataloader_seg = DataLoader(valid_dataset_seg, batch_size=1, shuffle=False, num_workers=2)
print(f"Train size: {len(train_dataset_seg)}")
print(f"val size: {len(valid_dataset_seg)}")


class Model(pl.LightningModule):
    def __init__(self, args, c = True):
        super().__init__()

        self.args = args
        # freeze backbone layers

        self.c = c
        if c == True: # c = True -> CSPN, c = False -> DYSPN
            self.model = Unet_affinity()
            # self.model = smp.Unet(
            #     encoder_name="resnet34",
            #     encoder_weights=None,
            #     in_channels=4,
            #     classes=9,
            # )
            self.refine = CSPN()
        else:
            self.model = Unet_d()
            self.refine = DYSPN()

        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.outputs = []
        self.num = 0
        self.mode = "train"

    def forward(self, image, seg, coarse_seg, iter, eval=False):
        
        if eval:
            # evaluation 할 때 각 iteration 결과 저장 
            if self.num == 3312:
                self.mode = "valid"
                self.num = 0
            
            result = []
            affinity = self.model(image)

            for i in range(iter):

                seg = self.refine(affinity, seg, coarse_seg,i)
                # temp = seg.sigmoid()
                # temp = (temp > 0.5).float()
                result.append(seg.to("cpu").detach().numpy())
            
            if self.c == True:
                np.save(f"intermidiate_result/{self.mode}/{self.num}.npy", np.array(result))
            else:
                np.save(f"intermidiate_result/{self.mode}_dyspn/{self.num}.npy", np.array(result)) # iteration result 저장 
            self.num += 1

            return seg
        
        else:
            affinity = self.model(image) 

            for i in range(iter):
                seg = self.refine(affinity, seg, coarse_seg, i)

            return seg

    def shared_step(self, batch, stage, eval=False):
        # This is for coarse segmentation
        #dic, seg_init = batch
        #image = dic['image']
        #input_batch = torch.cat((image, seg_init), dim=1)

        dic, seg = batch
        image = dic['image'].to(torch.float32)

        assert image.ndim == 4

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        assert seg.ndim == 4
        assert seg.max() <= 1.0 and seg.min() >= 0

        input_batch = torch.cat((image, seg), dim=1) # (b, 4, h, w)

        logits_mask = self.forward(input_batch, seg, seg, 6, eval)


        loss = self.loss_fn(logits_mask, dic["mask"])

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), dic["mask"].long(), mode="binary")

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

def eval(mode=True):
    model.eval()
    
    model.to(device)
    train_dataloader_seg = DataLoader(train_dataset_seg, batch_size=1, shuffle=False, num_workers=1)

    for i, (sample, seg) in enumerate(train_dataloader_seg):
        x = torch.tensor(sample["image"], dtype=torch.float32).to(device) # 1, 3, 256, 256
        y = torch.tensor(sample["mask"]).to(device) # 1, 1, 256, 256
        seg = torch.tensor(seg).to(device)  # 1, 256, 256

        input_batch = torch.cat((x, seg), dim=1) # 1, 4, 256, 256
        # seg = seg.squeeze(0)

        y_pred = model(input_batch, seg, seg, 5, eval=True)
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.squeeze(0)
        y = y.squeeze(0)

        y_pred_binary = (y_pred > 0.5).to(torch.float32)
        y_true_binary = (y > 0.0).to(torch.float32)

        tp, fp, fn, tn = smp.metrics.get_stats(y_pred_binary.long(), y_true_binary.long(), mode="binary")

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        print(f"Train - Epoch: {trainer.current_epoch}, Image {i}, IoU: {per_image_iou.item()}")

        if mode is True:
            np.save(f"affinity/train/{i}.npy", y_pred_binary.to("cpu").detach().numpy())
        if mode == False:
            np.save(f"affinity/train_dyspn/{i}.npy", y_pred_binary.to("cpu").detach().numpy())

    for i, (sample, seg) in enumerate(valid_dataloader_seg):
        x = torch.tensor(sample["image"], dtype=torch.float32).to(device)
        y = torch.tensor(sample["mask"]).to(device)
        seg = torch.tensor(seg).to(device)

        input_batch = torch.cat((x, seg), dim=1)
        # seg = seg.squeeze(0)

        y_pred = model(input_batch, seg, seg, 5, eval=True)
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.squeeze(0)
        y = y.squeeze(0)

        y_pred_binary = (y_pred > 0.5).to(torch.float32)
        y_true_binary = (y > 0.5).to(torch.float32)

        tp, fp, fn, tn = smp.metrics.get_stats(y_pred_binary.long(), y_true_binary.long(), mode="binary")

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        print(f"Valid - Epoch: {trainer.current_epoch}, Image {i}, IoU: {per_image_iou.item()}")

        if mode == "cspn":
            np.save(f"affinity/valid/{i}.npy", y_pred_binary.to("cpu").detach().numpy())
        if mode == "dyspn":
            np.save(f"affinity/valid_dyspn/{i}.npy", y_pred_binary.to("cpu").detach().numpy())

if __name__ == "__main__":
    fix_seed(0)

    c = True # c = True -> CSPN, c = False -> DYSPN

    model = Model(args, c=c)
    trainer = pl.Trainer(
        # gpus=1,
        max_epochs=10,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader_seg,
        val_dataloaders=valid_dataloader_seg,
    )

    eval(mode=c)