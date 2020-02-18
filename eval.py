import pandas as pd
import os
from PIL import Image
import random
from skimage import io
import skimage
import torch
from torchvision import models
from torch import nn

# csv_path = os.path.join('dataset', 'train', 'train.csv')
# train_data_frame = pd.read_csv(csv_path, header=None, names=['name', 'label'])
# zero = train_data_frame[train_data_frame['label'] == '0']
# one = train_data_frame[train_data_frame['label'] == '1']
# two = train_data_frame[train_data_frame['label'] == '2']
# three = train_data_frame[train_data_frame['label'] == '3']
#
# for i in range(10):
#     index = random.randint(0, len(three))
#     image_path = os.path.join('dataset', 'train', three.iloc[index]['name'])
#     image = Image.open(image_path)
#     image.show()


# model = models.resnet18(pretrained=True)
# num_ftrs = model.fc.in_features
# # Here the size of each output sample is set to 2.
# # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model.fc = nn.Linear(num_ftrs, 4)
#
# # model = model.to(torch.device('cuda:0'))
# model = nn.DataParallel(model)  # 多卡训练
# model = model.cuda()
# model.load_state_dict(torch.load('resnet50.pth'))

model = torch.load('resnet50.pth')
csv_path = os.path.join('dataset', 'val', 'upload.csv')
val_frame = pd.read_csv(csv_path)

for i in range(len(val_frame)):

    image_path = val_frame.iloc[i][0]
    image_path = os.path.join('dataset', 'val', image_path)
    image = io.imread(image_path)
    if len(image.shape) < 3:  # RGB to grey
        image = skimage.color.gray2rgb(image)
    elif image.shape[-1] == 4:
        image = skimage.color.rgba2rgb(image)
    image = image.transpose((2, 0, 1))
    image = torch.tensor(image).unsqueeze(0).type(torch.cuda.FloatTensor)
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    val_frame.loc[i, 'labels'] = int(preds)
    print(i / len(val_frame))

val_frame.to_csv('test.csv')
