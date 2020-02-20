import torch.nn as nn
from torchvision import models


def get_model():
    model_ft = models.resnext101_32x8d(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 4)

    # model_ft = model_ft.to(device)
    model_ft = nn.DataParallel(model_ft)  # 多卡训练
    model_ft = model_ft.cuda()

    return model_ft
