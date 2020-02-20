import copy
import os
import time

from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from torch import optim
from torch.optim import lr_scheduler

from utils import ChestXrayDataset, ToTensor, LeftToRightFlip, RandomCrop, Resize, ColorJitter, RandomRotation
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from focal_loss import FocalLoss
from model import get_model

transform = transforms.Compose([LeftToRightFlip(0.5),
                                RandomRotation(angle=3, p=0.5),
                                Resize(224),
                                ColorJitter(p=0.5, color=0.1, contrast=0.1, brightness=0.1, sharpness=0.1),
                                RandomCrop(scale=210, p=0.5),
                                Resize(224),
                                ToTensor()])

dataset = ChestXrayDataset(csv_file=os.path.join('dataset', 'train+test', 'train+test.csv'),
                           root_dir=os.path.join('dataset', 'train+test'),
                           transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
dataset_size = len(dataset)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            # requires grad setting
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        # learning rate decay
        scheduler.step()

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'train', epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict)

    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model


model_ft = get_model()

# criterion = nn.CrossEntropyLoss()
criterion = FocalLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
# optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=0.01, momentum=0.9, alpha=0.99, eps=1e-8, centered=True,
#                              weight_decay=1e-3)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.3)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=12)
torch.save(model_ft, 'resnet50.pth')
