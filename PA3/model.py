import torch 
import torch.nn as nn

class Unet(nn.Module):
  def __init__(self):
    super(Unet, self).__init__()

    # Convolutional layers
    self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
    self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
    self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
    self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
    self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
    self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
    self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
    self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
    self.conv9 = nn.Conv2d(512, 1024, 3, padding=1)
    self.conv10 = nn.Conv2d(1024, 1024, 3, padding=1)

    # Max pooling layers
    self.pool = nn.MaxPool2d(2, 2)
    
    # Upsampling layers
    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    # Batch normalization layers
    self.batch_norm1 = nn.BatchNorm2d(64)
    self.batch_norm2 = nn.BatchNorm2d(128)
    self.batch_norm3 = nn.BatchNorm2d(256)
    self.batch_norm4 = nn.BatchNorm2d(512)
    self.batch_norm5 = nn.BatchNorm2d(1024)
    
    # Dropout layer
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    # Encoder
    x = self.conv1(x)
    x = self.batch_norm1(x)
    x = torch.relu(x)
    x = self.conv2(x)
    x = self.batch_norm1(x)
    x = torch.relu(x)
    x1 = self.pool(x)

    x = self.conv3(x1)
    x = self.batch_norm2(x)
    x = torch.relu(x)
    x = self.conv4(x)
    x = self.batch_norm2(x)
    x = torch.relu(x)
    x2 = self.pool(x)

    x = self.conv5(x2)
    x = self.batch_norm3(x)
    x = torch.relu(x)
    x = self.conv6(x)
    x = self.batch_norm3(x)
    x = torch.relu(x)
    x3 = self.pool(x)

    x = self.conv7(x3)
    x = self.batch_norm4(x)
    x = torch.relu(x)
    x = self.conv8(x)
    x = self.batch_norm4(x)
    x = torch.relu(x)
    x4 = self.pool(x)

    x = self.conv9(x4)
    x = self.batch_norm5(x)
    x = torch.relu(x)
    x = self.conv10(x)
    x = self.batch_norm5(x)
    x = torch.relu(x)

    # Decoder
    x = self.up(x)
    x = torch.cat((x, x4), dim=1)
    x = self.conv8(x)
    x = self.batch_norm4(x)
    x = torch.relu(x)
    x = self.conv7(x)
    x = self.batch_norm4(x)
    x = torch.relu(x)

    x = self.up(x)
    x = torch.cat((x, x3), dim=1)
    x = self.conv6(x)
    x = self.batch_norm3(x)
    x = torch.relu(x)
    x = self.conv5(x)
    x = self.batch_norm3(x)
    x = torch.relu(x)

    x = self.up(x)
    x = torch.cat((x, x2), dim=1)
    x = self.conv4(x)
    x = self.batch_norm2(x)
    x = torch.relu(x)
    x = self.conv3(x)
    x = self.batch_norm2(x)
    x = torch.relu(x)

    x = self.up(x)
    x = torch.cat((x, x1), dim=1)
    x = self.conv2(x)
    x = self.batch_norm1(x)
    x = torch.relu(x)
    x = self.conv1(x)
    x = self.batch_norm1(x)
    x = torch.relu(x)

    return x
  
