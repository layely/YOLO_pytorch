import torch
from torch.utils import data
from torch.autograd import Variable

from tqdm import tqdm

from dataset import Dataset
from visualize import visualize_boxes, print_cell_with_objects
from models import YOLO
from loss import YoloLoss
from preprocessing import Preprocessing

device = torch.device(type='cuda')

images_path = "/home/layely/Myprojects/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages"
txt_file = "voc_2012.txt"

channels, height, width = (3, 448, 448)
S = 7 # SxS grid cells
B = 2 # Number of bounding boxes per cell
C = 20 # Number of classes

# Split dataset
train = 1.
val = 1.
test = 1.

epochs = 2000
lr = 0.001
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
img1 = train_dataset.images[pos]
target1 = train_dataset.labels[pos]
visualize_boxes(img1, target1, "gt.jpg", preprocess)

# Enable anomaly detection for debugging purpose
torch.autograd.set_detect_anomaly(True)

model = YOLO((channels, height, width), S, B, C)
model = model.to(device)

loss_func =  YoloLoss(S, B, C) #torch.nn.MSELoss(reduction="sum")
optimizer = opt(model.parameters(), momentum=momentum, lr=lr, weight_decay=weight_decay) # momentum=momentum

cur_epoch = 0
for epoch in range(cur_epoch, epochs):
    accumulated_train_loss = []
    # Set model in trainng mode
    iteration = 0
    for batch_x, batch_y in tqdm(train_generator):
        model.train()
        # print("batch shape:", batch_x.shape)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

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

        # print("gradients")
        # print([p.grad for p in model.parameters()])

        # model.eval()
        # preds = model(batch_x)
        if (epoch + 1) % 100 == 0:
            name = "predictions/epoch" + str(epoch + 1) + ".jpg"
            img = batch_x.clone().detach().view((channels, height, width))
            pred = preds.clone().detach().view((S, S, B * 5 + C))
            target = batch_y.clone().detach().view((S, S, B * 5 + C))
            visualize_boxes(img, pred, name, preprocess)
            print("------------------------------------------")
            print_cell_with_objects(target)
            print("-- -- -- ")
            print_cell_with_objects(pred)

    train_loss = sum(accumulated_train_loss) / len(accumulated_train_loss)
    print("Epoch: {} --- Train loss: {}".format(epoch + 1, train_loss))
