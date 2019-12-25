import torch
from torch.utils import data
from torch.autograd import Variable

from tqdm import tqdm
import time

from data import Dataset
from visualize import draw_all_bboxes, print_cell_with_objects
from models import YOLO
from loss import YoloLoss
from preprocessing import Preprocessing
from tb import Tensorboard
from utils import save_checkpoint

def change_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    device = torch.device(type='cuda')
    num_gpu = torch.cuda.device_count()

    # Train and test files
    voc_2007 = "/home/layely/Myprojects/datasets/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages"
    voc_2012 = "/home/layely/Myprojects/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages"
    train_images_dir = [voc_2007, voc_2012]
    train_files = ["voc_2007.txt", "voc_2012.txt"]
    voc_2007_test = "/home/layely/Myprojects/datasets/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages"
    test_images_dir = [voc_2007_test]
    test_files = ["voc_2007_test.txt"]

    channels, height, width = (3, 448, 448)
    S = 7  # SxS grid cells
    B = 2  # Number of bounding boxes per cell
    C = 20  # Number of classes

    # Training hyperparameters
    epochs = 160
    lr = 0.01
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

    preprocess = Preprocessing(S, B, C, mean, std, brightness, saturation, contrast, hue)

    # Load dataset
    train_dataset = Dataset(train_images_dir, train_files, (height, width),
                                S, B, C, preprocess, random_transform=True)
    val_dataset = Dataset(test_images_dir, test_files, (height, width),
                                S, B, C, preprocess, random_transform=False)
    train_generator = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=min(num_gpu*8, 16))
    val_generator = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=min(num_gpu*8, 16))

    # This is just to check that labels are correctly encoded.
    for pos in range(20):
        img1, target1 = train_dataset.__getitem__(pos)
        gt_color = (0, 0, 255) #red

        # Get numpy image in opencv format
        np_img = preprocess.post_process_image(img1)
        # get bounding boxes: xyxy + class + confidence
        img_h, img_w, _ = np_img.shape
        target_bboxes = preprocess.decode_label(target1, img_h, img_w)

        draw_all_bboxes(np_img, target_bboxes, preprocess, gt_color, "images_transformed/gt_{}.jpg".format(pos))

    import os
    os._exit(1)

    # Enable anomaly detection for debugging purpose
    torch.autograd.set_detect_anomaly(True)

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
    best_model_loss = test = float("inf")

    start_time = time.ctime()
    print("### start_time:", start_time)

    cur_epoch = 0
    for epoch in range(cur_epoch, epochs):
        if epoch == 74 or epoch == 104:
            print("Changing learning from {} to {}...".format(lr, lr * 0.1))
            lr = lr * 0.1
            change_lr(optimizer, lr)

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

        # if (epoch + 1) % 10 == 0:
        #     for i in range(min(batch_size, 4)):
        #         name = "predictions/epoch{}_{}.jpg".format(epoch + 1, i)
        #         img = batch_x.clone().detach()[i].view((channels, height, width))
        #         pred = preds.clone().detach()[i].view((S, S, B * 5 + C))
        #         target = batch_y.clone().detach()[i].view((S, S, B * 5 + C))
        #         pred_color = (0,255,0) # green
        #         draw_all_bboxes(img, pred,preprocess, pred_color, name)
        #         # print("------------------------------------------")
        #         # print_cell_with_objects(target)
        #         # print("-- -- -- ")
        #         # print_cell_with_objects(pred)

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

        if val_loss < best_model_loss:
            print("saving model...")
            best_model_loss = val_loss
            if isinstance(model, torch.nn.DataParallel):
                save_checkpoint(model.module, optimizer, epoch, loss)
            else:
                save_checkpoint(model, optimizer, epoch, loss)

    # End of train
    tb.close()
