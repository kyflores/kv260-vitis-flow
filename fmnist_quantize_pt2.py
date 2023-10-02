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

mnist_eval = tds.FashionMNIST(root=dataset_root, download=True, train=False,
    transform=tvt.ToTensor())
val_loader = tud.DataLoader(mnist_eval, batch_size=batchsize, shuffle=True)

# Run `xdputil query` in the Kria docker to get the DPU parameters
target = "DPUCZDX8G_ISA1_B3136" # (or 0x101000016010406)
inspector = Inspector(target)
dummy_input = torch.randn(batchsize, 1, 28, 28).to(device).float()
inspector.inspect(model, (dummy_input,), device=device, output_dir="inspect", image_format="png")

# def evaluate(model, val_loader, loss_fn):
#     correct = 0
#     losses = []
#     total = 0
#     for i, (images, target) in enumerate(val_loader):
#         with torch.no_grad():
#             images = images.to(device)
#             targets = target.to(device)
#             outs = model(images)

#             loss = loss_fn(outs, targets)
#             losses.append(loss)
#             total += images.shape[0]
#             for x in range(outs.shape[0]):
#                 preds = F.softmax(outs, dim=1)
#                 cls = preds[x].argmax()
#                 lbl = targets[x]
#                 if cls == lbl:
#                     correct += 1

#     return torch.tensor(loss).mean()

loss_fn = torch.nn.CrossEntropyLoss().to(device)
# Now, set everything to test mode and export the xmodel
# dataset_test = torch.utils.data.Subset(
    # mnist_eval, random.sample(range(0, len(mnist_eval)), 1))
# val_loader_test = tud.DataLoader(mnist_eval, batch_size=1, shuffle=True)
quantizer_test = torch_quantizer(
    'test',
    model,
    dummy_input,
    device=device,
    target=target)
quantizer_test.load_ft_param()
quantizer_test.quant_model(mnist_eval[0][0].unsqueeze(0))
# loss_test = evaluate(quantizer_test.quant_model, val_loader_test, loss_fn)

quantizer_test.export_xmodel()
