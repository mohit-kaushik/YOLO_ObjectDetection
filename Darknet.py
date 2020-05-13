import torch
from Blocks import Conv_Block, FC_Block

'''
  Architecture implementation YOLOv1
'''

class YOLO_model(torch.nn.Module):
  def __init__(self):
    super(YOLO_model, self).__init__()
    # in_channels, out_channels, kernel_size, stride, padding
    self.conv1 = Conv_Block(3, 64, kernel_size=7, stride=2, padding=3)
    # "maxpool"
    self.conv2 = Conv_Block(64, 192, kernel_size=3, stride=1, padding=1)
    # "MAxpol"

    self.conv3 = Conv_Block(192,128, kernel_size=1, stride=1, padding=0)
    self.conv4 = Conv_Block(128,256, kernel_size=3, stride=1, padding=1)
    self.conv5 = Conv_Block(256,256, kernel_size=1, stride=1, padding=0)
    self.conv6 = Conv_Block(256,512, kernel_size=3, stride=1, padding=1)


    # "M"
    self.conv7 = Conv_Block(512,256, kernel_size=1, stride=1, padding=0)
    self.conv8 = Conv_Block(256,512, kernel_size=3, stride=1, padding=1)

    self.conv9 = Conv_Block(512,256, kernel_size=1, stride=1, padding=0)
    self.conv10 = Conv_Block(256,512, kernel_size=3, stride=1, padding=1)

    self.conv11 = Conv_Block(512,256, kernel_size=1, stride=1, padding=0)
    self.conv12 = Conv_Block(256,512, kernel_size=3, stride=1, padding=1)

