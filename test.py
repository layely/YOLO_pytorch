import torch
from torch.utils import data

from tqdm import tqdm
import time

from data import DataGenerator
from visualize import draw_all_bboxes, print_cell_with_objects
from models import YOLO
from loss import YoloLoss
from preprocessing import Preprocessing
from utils import load_checkpoint, write_result_to_disk

device = torch.device(type='cuda')

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

batch_size = 1


# Image normalization parameters
# Note that images are squished to
# the range [0, 1] before normalization
mean = [0.485, 0.456, 0.406] # RGB - Imagenet means
std = [0.229, 0.224, 0.225] # RGB - Imagenet standard deviations

preprocess = Preprocessing(S, B, C, mean, std)

# Load dataset
dataloader = DataGenerator(images_path, txt_file, train,
                           val, test, (height, width),
                           S, B, C, preprocess)
train_dataset, val_dataset, test_dataset = dataloader.get_datasets()
test_generator = data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

model = YOLO((channels, height, width), S, B, C)
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
try:
    load_checkpoint(model)
except:
    "Try with a dataparallel model"
    model = torch.nn.DataParallel(model)
    load_checkpoint(model)
    model = model.module.to(device)

loss_func = YoloLoss(S, B, C)  # torch.nn.MSELoss(reduction="sum")

start_timestamp = time.time()

# Validation
model.eval()
accumulated_test_loss = []
it = 0
with torch.no_grad():
    for batch_x, batch_y in tqdm(test_generator):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward
        preds = model(batch_x)

        # compute loss
        test_loss = loss_func(preds, batch_y)
        accumulated_test_loss.append(test_loss)

        name = "test_results/images/test{}.jpg".format(it + 1)
        img = batch_x.clone().view((channels, height, width))
        pred = preds.clone().view((S, S, B * 5 + C))
        target = batch_y.clone().view((S, S, B * 5 + C))

        # Get numpy image in opencv format
        np_img = preprocess.post_process_image(img)
        # get bounding boxes: xyxy + class + confidence
        img_h, img_w, _ = np_img.shape
        target_bboxes = preprocess.decode_label(target, img_h, img_w)
        pred_bboxes = preprocess.decode_label(pred, img_h, img_w)

        color = (0, 255, 0)
        draw_all_bboxes(np_img, target_bboxes, preprocess, color, name)

        # write results to disk
        gt_file_name = "test_results/gt/test{}.txt".format(it + 1)
        write_result_to_disk(gt_file_name, target_bboxes, type="gt")
        pred_file_name = "test_results/pred/test{}.txt".format(it + 1)
        write_result_to_disk(pred_file_name, pred_bboxes, type="pred")

        it += 1

duration = time.time() - start_timestamp
# Epoch losses
test_loss = sum(accumulated_test_loss) / len(accumulated_test_loss)
print("*** **** --- Test loss: {} - duration: {}".format(test_loss, duration))
