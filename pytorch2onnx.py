import torch.onnx
import torchvision
import math
import onnx
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16prune': [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256,'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        if vgg_name == 'VGG16':
           temp = 512
        else:
           temp = 256
        self.classifier = nn.Linear(temp, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.flatten(1)
#        out = out.view(out.size(0),-1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



# Update the input name and path for your PyTorch model
input_pytorch_model = '/home/pwq/pytorch-cifar/checkpoint/vggcifar16baselineckpt.pth'
model = VGG('VGG16')
#model = ResNet_small(Bottleneck, [3,4,6,3])
net = torch.load(input_pytorch_model)
model.load_state_dict(net['net'])
torch.save(model, './model.pth')

# Create input with the correct dimensions of the input of your model, these are default ImageNet inputs
dummy_model_input = Variable(torch.randn(1, 3, 32, 32))

# Change this path to the output name and path for the ONNX model
output_onnx_model = 'vggcifar16baselineckpt.onnx'

# load the PyTorch model
model = torch.load('./model.pth')
# export the PyTorch model as an ONNX protobuf
torch.onnx.export(model, dummy_model_input, output_onnx_model)

onnx_model = onnx.load('vggcifar16vaselineckpt.onnx')
passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
from onnx import optimizer
optimized_model = optimizer.optimize(onnx_model, passes)

onnx.save(optimized_model, 'vggcifar16baselineckpt.onnx')

