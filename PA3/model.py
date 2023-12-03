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

    # decoder convolution layers
    self.conv11 = nn.Conv2d(1536, 512, 3, padding=1)
    self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
    self.conv13 = nn.Conv2d(768, 256, 3, padding=1)
    self.conv14 = nn.Conv2d(256, 256, 3, padding=1)
    self.conv15 = nn.Conv2d(384, 128, 3, padding=1)
    self.conv16 = nn.Conv2d(128, 128, 3, padding=1)
    self.conv17 = nn.Conv2d(192, 64, 3, padding=1)
    self.conv18 = nn.Conv2d(64, 64, 3, padding=1)
    self.convlast = nn.Conv2d(64, 1, 1, padding=0)


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

    # x.shape = b, 3, 256, 256
    # Encoder
    x = self.conv1(x) # b, 64, 256, 256
    x = self.batch_norm1(x)
    x = torch.relu(x)
    x = self.conv2(x) # b, 64, 256, 256
    x = self.batch_norm1(x)
    x_res = torch.relu(x)
    x = self.pool(x_res) # b, 64, 128, 128

    x = self.conv3(x) # b, 128, 128, 128
    x = self.batch_norm2(x)
    x = torch.relu(x)
    x = self.conv4(x)
    x = self.batch_norm2(x)
    x_res2 = torch.relu(x)
    x = self.pool(x_res2)

    x = self.conv5(x)
    x = self.batch_norm3(x)
    x = torch.relu(x)
    x = self.conv6(x)
    x = self.batch_norm3(x)
    x_res3 = torch.relu(x)
    x = self.pool(x_res3)

    x = self.conv7(x)
    x = self.batch_norm4(x)
    x = torch.relu(x)
    x = self.conv8(x)
    x = self.batch_norm4(x)
    x_res4 = torch.relu(x)
    x = self.pool(x_res4)

    x = self.conv9(x)
    x = self.batch_norm5(x)
    x = torch.relu(x)
    x = self.conv10(x)
    x = self.batch_norm5(x)
    x = torch.relu(x) # b, 1024, 16, 16

    # Decoder
    x = self.up(x) # b, 1024, 32, 32
    x = torch.cat((x_res4, x), dim=1) # b, 1536, 32, 32
    x = self.conv11(x)
    x = self.batch_norm4(x)
    x = torch.relu(x)
    x = self.conv12(x) # b, 512, 32, 32
    x = self.batch_norm4(x)
    x = torch.relu(x)

    x = self.up(x)
    x = torch.cat((x_res3, x), dim=1)
    x = self.conv13(x)
    x = self.batch_norm3(x)
    x = torch.relu(x)
    x = self.conv14(x)
    x = self.batch_norm3(x)
    x = torch.relu(x)

    x = self.up(x)
    x = torch.cat((x_res2, x), dim=1)
    x = self.conv15(x)
    x = self.batch_norm2(x)
    x = torch.relu(x)
    x = self.conv16(x)
    x = self.batch_norm2(x)
    x = torch.relu(x)

    x = self.up(x)
    x = torch.cat((x_res, x), dim=1)
    x = self.conv17(x)
    x = self.batch_norm1(x)
    x = torch.relu(x)
    x = self.conv18(x)
    x = self.batch_norm1(x)
    x = torch.relu(x)

    x = self.convlast(x)

    return x
  
