

import torch

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datasets import train_dataloader, test_dataloader
from helepers import train_loop, test_loop
from models import NeuralNetwork

learning_rate = 1e-3
batch_size = 64
epochs = 15

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# save the model

torch.save(model, 'model.pth')

# model = torch.load('model.pth')