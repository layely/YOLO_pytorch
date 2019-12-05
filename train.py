import torch
from torch.utils import data
from torch.autograd import Variable

from tqdm import tqdm

from dataset import Dataset
from visualize import visualize_boxes
from models import YOLO
from loss import YoloLoss
from preprocessing import Preprocessing

images_path = "/home/layely/Myprojects/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages"
txt_file = "voc_2012.txt"

channels, height, width = (3, 448, 448)
S = 7
B = 2
C = 20

# Split dataset
train = 1.
val = 1.
test = 1.

epochs = 200
lr = 0.1
momentum = 0.9
weight_decay = 5e-4
opt = torch.optim.SGD
batch_size = 1

dataloader = Dataset(images_path, txt_file, train,
                     val, test, (height, width), seed=1)
train_dataset, val_dataset, test_dataset = dataloader.get_datasets()
train_generator = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

preprocess = Preprocessing()
pos = 0
img1 = train_dataset.images[pos].detach()
target1 = train_dataset.labels[pos]
visualize_boxes(img1, target1, "gt.jpg", preprocess)

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

model = YOLO((channels, height, width), S, B, C)

loss_func = torch.nn.MSELoss(reduction="mean") # YoloLoss(S, B, C)
optimizer = opt(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

cur_epoch = 0
for epoch in range(cur_epoch, epochs):
    accumulated_train_loss = []
    # Set model in trainng mode
    model.train()
    iteration = 0
    for batch_x, batch_y in tqdm(train_generator):
        # print("batch shape:", batch_x.shape)

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

        # print(preds.view((-1, 30)))
        iteration += 1

        if epoch % 10 == 0:
            name = "epoch" + str(epoch) + ".jpg"
            img = batch_x.detach().view((channels, height, width))
            pred = preds.detach().view((S, S, 30))
            visualize_boxes(img, pred, name, preprocess)

    train_loss = sum(accumulated_train_loss) / len(accumulated_train_loss)
    print("Train loss: {}".format(train_loss))
