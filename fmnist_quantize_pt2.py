#!/usr/bin/python
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvf
import torchvision.datasets as tds
from cnn_model import MyCnn

from pytorch_nndct.apis import Inspector, torch_quantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_root='./datasets'

batchsize = 1
model = MyCnn(10, 16).cpu()
model.load_state_dict(torch.load("fmnist.pt"))

mnist_eval = tds.FashionMNIST(root=dataset_root, download=True, train=False, transform=tvt.ToTensor())

# Run `xdputil query` in the Kria docker to get the DPU parameters
target = "DPUCZDX8G_ISA1_B3136" # (or 0x101000016010406)
inspector = Inspector(target)
dummy_input = torch.randn(batchsize, 1, 28, 28).to(device).float()
inspector.inspect(model, (dummy_input,), device=device, output_dir="inspect", image_format="png")

loss_fn = torch.nn.CrossEntropyLoss().to(device)
quantizer_test = torch_quantizer(
    'test',
    model,
    dummy_input,
    device=device)
    # target=target)
quantizer_test.load_ft_param()
quantizer_test.quant_model(mnist_eval[0][0].unsqueeze(0))

quantizer_test.export_xmodel()
