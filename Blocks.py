import torch

'''
  Basic Convolution block for YOLOv1
'''
class Conv_Block(torch.nn.Module):
  def __init__(self, in_filters, out_filters, **kwargs):
    super(Conv_Block, self).__init__()

    # BatchNorm already includes bias so no need in conv2d
    self.conv = torch.nn.Conv2d(in_filters, out_filters, bias=False, **kwargs)
    self.batch_norm = torch.nn.BatchNorm2d(out_filters)
    self.activation = torch.nn.LeakyReLU(0.1)

  def forward(self, x):
    x = self.conv(x)
    x = self.batch_norm(x)
    return self.activation(x)

'''
	Fully Connected block for Yolov1
'''
class FC_Block(torch.nn.Module):
  def __init__(self):
    super(FC_Block, self).__init__()
    self.flatten_input = torch.nn.Flatten()
    self.linear1 = torch.nn.Linear(7*7*1024, 4096)
    self.linear2 = torch.nn.Linear(4096, 7*7*30)
    self.activation = torch.nn.LeakyReLU(0.1)

  def forward(self, x):
    x = self.flatten_input(x)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    return x


