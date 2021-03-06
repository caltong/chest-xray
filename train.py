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

transform = {'train': transforms.Compose([LeftToRightFlip(0.5),
                                          RandomRotation(angle=3, p=0.5),
                                          Resize(224),
                                          ColorJitter(p=0.5, color=0.1, contrast=0.1, brightness=0.1, sharpness=0.1),
                                          RandomCrop(scale=210, p=0.5),
                                          Resize(224),
                                          ToTensor()]),
             'test': transforms.Compose([ToTensor()])}

datasets = {x: ChestXrayDataset(csv_file=os.path.join('dataset', x, x + '.csv'),
                                root_dir=os.path.join('dataset', x),
                                transform=transform[x])
            for x in ['train', 'test']}
dataloaders = {x: DataLoader(datasets[x], batch_size=32, shuffle=True)
               for x in ['train', 'test']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'test']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    train_history = []
    test_hittory = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                # inputs = inputs.to(device)
                # labels = labels.to(device)
                inputs = inputs.cuda()  # 多卡训练
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_history.append([epoch, epoch_acc])
            else:
                test_hittory.append([epoch, epoch_acc])

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print(train_history)
    print(test_hittory)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = get_model('pretrained')

# criterion = nn.CrossEntropyLoss()
criterion = FocalLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
# optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=0.01, momentum=0.9, alpha=0.99, eps=1e-8, centered=True,
#                              weight_decay=1e-3)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.3)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=24)
torch.save(model_ft, 'resnet50.pth')
