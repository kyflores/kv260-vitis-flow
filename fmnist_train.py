#!/usr/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
import torchvision.transforms.functional as tvf
import torchvision.datasets as tds
import torchvision.transforms as tvt
from cnn_model import MyCnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_root='./datasets';
batchsize = 32
epochs = 10
lr = 0.01

mnist_train = tds.FashionMNIST(root=dataset_root, download=True, train=True,
    transform=tvt.ToTensor())
mnist_eval = tds.FashionMNIST(root=dataset_root, download=True, train=False,
    transform=tvt.ToTensor())
train_loader = tud.DataLoader(mnist_train, batch_size=batchsize, shuffle=True)
val_loader = tud.DataLoader(mnist_eval, batch_size=batchsize, shuffle=True)

model = MyCnn(out_sz=10, ch=16).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr)
lossfn = nn.CrossEntropyLoss()
for epoch in range(epochs):
    for (images, target) in train_loader:
        optimizer.zero_grad()
        targets = target.to(device)
        outs = model(images.float().to(device))
        lossfn(outs, targets).backward()
        optimizer.step()

    losses = []
    for (images, target) in val_loader:
        with torch.no_grad():
            targets = target.to(device)
            outs = model(images.float().to(device))
            losses.append(lossfn(outs, targets))
    print("Epoch: {}, eval loss: {}".format(epoch, torch.Tensor(losses).mean().item()))

correct = 0
with torch.no_grad():
    for image, target in mnist_eval:
        if F.softmax(model(image.float().unsqueeze(0).to(device)), dim=1).argmax() == target:
            correct += 1

    print("{:.2f}% correct".format(100*correct/len(mnist_eval)))

torch.save(model.state_dict(), 'fmnist.pt')
dummy_input = torch.randn(1, 1, 28, 28).to(device)
torch.onnx.export(
    model, dummy_input,"fmnist.onnx", verbose=True,
    input_names=["input0"],
    output_names=["output0"])

