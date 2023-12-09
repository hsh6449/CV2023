import torch 
import torch.nn as nn

class Unet(nn.Module):
  """
  For coarse segmentation
  """
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
  
class CSPN(nn.Module):
  def __init__(self):
    super(CSPN, self).__init__()

    self.unfold = nn.Unfold(kernel_size=3, padding=1)

  def forward(self, affinity, cur_seg, coarse_seg, i):

    b, c, h, w = affinity.shape
    # unfold cur_seg
    cur_seg_unfold = self.unfold(cur_seg)

    # replace coarse_seg
    cur_seg_unfold[:,4] = coarse_seg.reshape(b,-1)

    #sum
    output = cur_seg_unfold * affinity.reshape(b,9,-1)
    output = torch.sum(output, dim=1)
    output = output.reshape(b, 1, h, w)

    return output 
  
class DYSPN(nn.Module):
  def __init__(self):
    super(DYSPN, self).__init__()

    self.unfold = nn.Unfold(kernel_size=7, padding=3)
    self.iter = 0

  def forward(self, mat, cur_seg, coarse_seg, i):
      
      b, c, h, w = mat.shape
      self.iter = i

      # attention & affinity
      if self.iter == 0:
        attention = torch.sigmoid(mat[:,48:52,:,:]) # b, 4, h, w
      elif self.iter == 1:
        attention = torch.sigmoid(mat[:,52:56,:,:])
      elif self.iter == 2:
        attention = torch.sigmoid(mat[:,56:60,:,:])
      elif self.iter == 3:
        attention = torch.sigmoid(mat[:,60:64,:,:])
      elif self.iter == 4:
        attention = torch.sigmoid(mat[:,64:68,:,:])
      elif self.iter == 5:
        attention = torch.sigmoid(mat[:,68:72,:,:])

      affinity = mat[:,0:48,:,:] # b, 48, h, w
      
      # seperate affinity matrix
      affinity_24_1 = affinity[:,0:8,:,:]
      affinity_24_2 = affinity[:,13:15,:,:] 
      affinity_24_3 = affinity[:,20:22,:,:] 
      affinity_24_4 = affinity[:,27:29,:,:] 
      affinity_24_5 = affinity[:,34:36,:,:]
      affinity_24_6 = affinity[:,41:49,:,:]

      affinity_24 = torch.cat((affinity_24_1, affinity_24_2, affinity_24_3, affinity_24_4, affinity_24_5, affinity_24_6), dim=1)
      affinity_24 = affinity_24 * attention[:,3,:,:].reshape(b,1,h,w)
  

      affinity_16_1 = affinity[:,8:13,:,:]
      affinity_16_2 = affinity[:,15,:,:].reshape(b,1,h,w) 
      affinity_16_3 = affinity[:,19,:,:].reshape(b,1,h,w) 
      affinity_16_4 = affinity[:,22,:,:].reshape(b,1,h,w) 
      affinity_16_5 = affinity[:,26,:,:].reshape(b,1,h,w) 
      affinity_16_6 = affinity[:,29,:,:].reshape(b,1,h,w) 
      affinity_16_7 = affinity[:,33,:,:].reshape(b,1,h,w) 
      affinity_16_8 = affinity[:,36:41,:,:]
      
      affinity_16 = torch.cat((affinity_16_1, affinity_16_2, affinity_16_3, affinity_16_4, affinity_16_5, affinity_16_6, affinity_16_7, affinity_16_8), dim=1)
      affinity_16 = affinity_16 * attention[:,2,:,:].reshape(b,1,h,w)


      affinity_8_1 = affinity[:,16:19,:,:] * attention[:,1,:,:].reshape(b,1,h,w)
      affinity_8_2 = affinity[:,23,:,:].reshape(b,1,h,w) * attention[:,1,:,:].reshape(b,1,h,w)
      affinity_8_3 = affinity[:,25,:,:].reshape(b,1,h,w) * attention[:,1,:,:].reshape(b,1,h,w)
      affinity_8_4 = affinity[:,30:33,:,:] * attention[:,1,:,:].reshape(b,1,h,w)

      affinity_8 = torch.cat((affinity_8_1, affinity_8_2, affinity_8_3, affinity_8_4), dim=1)
      affinity_8 = affinity_8 * attention[:,1,:,:].reshape(b,1,h,w)


    
      
      affinity_1 = coarse_seg * attention[:,0,:,:].reshape(b,1,h,w)

      # sum
      Sij = torch.sum(torch.cat((affinity_24, affinity_16, affinity_8), dim=1), dim=1)
      Sij_2 = torch.sum(abs(torch.cat((affinity_24, affinity_16, affinity_8), dim=1)), dim=1) # scale
      
      # unfold cur_seg
      cur_seg_unfold = self.unfold(cur_seg) # b, 49, h*w

      # replace coarse_seg
      cur_seg_unfold[:,24] = coarse_seg.reshape(b,-1)

      epsilon = 1e-5

      #sum
      sum1 = cur_seg_unfold[:,1:,:] * Sij.reshape(b,1,-1) / (Sij_2.reshape(b,1,-1)+ epsilon) # 1, 49, h*w 
      sum2 = (1 - Sij/(Sij_2 + epsilon)).reshape(b,1,h,w) * affinity_1
      sum2 = sum2.reshape(b,1,-1)


      output = sum1 + sum2
      output = torch.sum(output, dim=1)
      output = output.reshape(b, 1, h, w)
  
      return output
  

class Unet_affinity(nn.Module):
  def __init__(self):
    super(Unet_affinity, self).__init__()

    # Convolutional layers
    self.conv1 = nn.Conv2d(4, 64, 3, padding=1)
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
    self.convlast = nn.Conv2d(64, 9, 1, padding=0)


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


class Unet_d(nn.Module):
  def __init__(self):
    super(Unet_d, self).__init__()

    # Convolutional layers
    self.conv1 = nn.Conv2d(4, 64, 3, padding=1)
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
    self.convlast = nn.Conv2d(64, 72, 1, padding=0)


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
    self.batch_norm6 = nn.BatchNorm2d(72)
    
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