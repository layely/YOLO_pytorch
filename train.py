import torch
from torch.utils import data
from torch.autograd import Variable

from tqdm import tqdm
import time

from data import DataGenerator
from visualize import visualize_boxes, print_cell_with_objects
from models import YOLO
from loss import YoloLoss
from preprocessing import Preprocessing
from tb import Tensorboard
from utils import save_checkpoint

device = torch.device(type='cuda')
num_gpu = torch.cuda.device_count()

images_path = "/home/layely/Myprojects/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages"
txt_file = "voc_2012.txt"

channels, height, width = (3, 448, 448)
S = 7  # SxS grid cells
B = 2  # Number of bounding boxes per cell
C = 20  # Number of classes

# Data split proportions
train = .6
val = .2
test = .2

# Training hyperparameters
epochs = 200
lr = 0.0005
momentum = 0.9
weight_decay = 5e-4
opt = torch.optim.SGD
batch_size = 24 * num_gpu


# Image normalization parameters
# Note that images are squished to
# the range [0, 1] before normalization
mean = [0.485, 0.456, 0.406] # RGB - Imagenet means
std = [0.229, 0.224, 0.225] # RGB - Imagenet standard deviations

# Random color transformation
brightness = 0.4
saturation = 0.4
contrast = 0.4
hue = 0.1

preprocess = Preprocessing(mean, std, brightness, saturation, contrast, hue)

# Load dataset
dataloader = DataGenerator(images_path, txt_file, train,
                           val, test, (height, width),
                           S, B, C, preprocess)
train_dataset, val_dataset, test_dataset = dataloader.get_datasets()
train_generator = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=min(num_gpu*8, 16))
val_generator = data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=min(num_gpu*8, 16))

# This is just to check that labels are correctly encoded.
pos = 0
img1, target1 = train_dataset.__getitem__(pos)
visualize_boxes(img1, target1, "gt.jpg", preprocess)

# Enable anomaly detection for debugging purpose
# torch.autograd.set_detect_anomaly(True)

model = YOLO((channels, height, width), S, B, C)
if num_gpu > 1:
  print("Let's use", num_gpu, "GPUs!")
  model = torch.nn.DataParallel(model)
model = model.to(device)


# Init tensorboard for loss visualization
tb = Tensorboard()

loss_func = YoloLoss(S, B, C)  # torch.nn.MSELoss(reduction="sum")
optimizer = opt(model.parameters(), momentum=momentum, lr=lr,
                weight_decay=weight_decay)  # momentum=momentum

# Keep the loss of the best model
best_model_loss = None

start_time = time.ctime()

cur_epoch = 0
for epoch in range(cur_epoch, epochs):
    accumulated_train_loss = []

    start_timestamp = time.time()
    # Train
    model.train()
    iteration = 0
    for batch_x, batch_y in tqdm(train_generator):
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

    if (epoch + 1) % 10 == 0:
        for i in range(min(batch_size, 4)):
            name = "predictions/epoch{}_{}.jpg".format(epoch + 1, i)
            img = batch_x.clone().detach()[i].view((channels, height, width))
            pred = preds.clone().detach()[i].view((S, S, B * 5 + C))
            target = batch_y.clone().detach()[i].view((S, S, B * 5 + C))
            visualize_boxes(img, pred, name, preprocess)
            # print("------------------------------------------")
            # print_cell_with_objects(target)
            # print("-- -- -- ")
            # print_cell_with_objects(pred)

    # Validation
    model.eval()
    accumulated_val_loss = []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(val_generator):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward
            preds = model(batch_x)

            # compute loss
            val_loss = loss_func(preds, batch_y)
            accumulated_val_loss.append(val_loss)

    duration = time.time() - start_timestamp
    # Epoch losses
    train_loss = sum(accumulated_train_loss) / len(accumulated_train_loss)
    val_loss = sum(accumulated_val_loss) / len(accumulated_val_loss)
    print("*** **** Epoch: {} --- Train loss: {} --- Val loss: {} - duration: {}".format(epoch +
                                                                          1, train_loss, val_loss, duration))

    # Add to tensorboard
    tb.add_scalar("{}/Train loss".format(start_time), train_loss, epoch)
    tb.add_scalar("{}/Val loss".format(start_time), val_loss, epoch)

    if not best_model_loss or val_loss > best_model_loss:
        save_checkpoint(model, optimizer, epoch, loss)

# End of train
tb.close()
