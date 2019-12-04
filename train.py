import torch
from torch.utils import data

from tqdm import tqdm

from dataset import Dataset
from visualize import visualize_boxes
from models import YOLO
from loss import YoloLoss

images_path = "/home/layely/Myprojects/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages"
txt_file = "voc_2012.txt"

channels, height, width = (3, 464, 464)
S = 7
B = 2
C = 20

# Split dataset
train = 0.6
val = 0.2
test = 0.2

epochs = 100
lr = 0.001
momentum = 0.9
weight_decay = 5e-4
opt = torch.optim.SGD
batch_size = 24

dataloader = Dataset(images_path, txt_file, train,
                     val, test, (height, width), seed=1)
train_dataset, val_dataset, test_dataset = dataloader.get_datasets()
train_generator = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

pos = 5
img1 = train_dataset.images[pos]
img1 = img1.permute(1, 2, 0)

target1 = train_dataset.labels[pos]

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# visualize_boxes(img1, target1)
model = YOLO((channels, height, width), S, B, C)

loss_func = YoloLoss(S, B, C)
optimizer = opt(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

cur_epoch = 0
for epoch in range(cur_epoch, epochs):
    accumulated_train_loss = []
    # Set model in trainng mode
    model.train()
    iteration = 0
    for batch_x, batch_y in tqdm(train_generator):
        print("batch shape:", batch_x.shape)

        # Forward
        preds = model(batch_x)

        # compute loss
        loss = loss_func(preds, batch_y)
        accumulated_train_loss.append(loss.item())

        # zero gradients
        optimizer.zero_grad()

        # backward
        loss.backward()

        # Step to update optimizer params
        optimizer.step()

        iteration += 1

    train_loss = sum(accumulated_train_loss) / len(accumulated_train_loss)
    print("Train loss: {}".format(train_loss))
