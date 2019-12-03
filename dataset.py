import torch
from torch.utils import data
import numpy as np
import glob
import cv2
import math
from tqdm import tqdm

class Data(data.Dataset):
    def __init__(self, cat_images, dog_images, device=None):
        self.images = cat_images + dog_images
        self.labels = [0.] * len(cat_images) + [1.0] * len(dog_images)
        self.len = len(self.labels)

        # convert to numpy arrays
        self.images = np.asarray(self.images)
        self.labels = np.asarray(self.labels)

        # convert to tensors
        self.images = torch.from_numpy(self.images).float()
        self.labels = torch.from_numpy(self.labels).float()

        # In pytorch, conv2D expect input shape to be in this
        # form: (batch_size, channels, height, weight).
        self.images = self.images.permute(0, 3, 1, 2)

        # Move to specified device if applicable.
        if device:
            self.images = self.images.to(device)
            self.labels = self.labels.to(device)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        X = self.images[index]
        y = self.labels[index]
        return X, y

class DataSplit():
    def __init__(self, images_path, txt_file, train, val, test, input_shape, seed=1):
        print("Preparing data...")

        self.images_path = images_path
        self.txt_file = txt_file
        self.train = train
        self.val = val
        self.test = test
        self.seed = seed

        # Load images
        self.images = []
        self.labels = [] # Each label is a list of (xmin, ymin, xmax, ymax, class)
        with open(txt_file, "r") as f:
            lines = f.read().splitlines()
            for line in lines[:1]:
                row = line.split(' ')
                self.images.append(images_path + "/" + row[0])
                labels = [row[n:n+5] for n in range(1, len(row), 5)]
                self.labels.append(labels)


        self.images = self.load_images(self.images)
        # print(self.images.shape)
        self.labels = self.encode_labels(self.labels, self.images)

        self.images = self.resize_images(self.images, input_shape)
        print(self.images.shape)

        # cat_images = glob.glob(dirpath + "/Cat/*.jpg")[:]


        # # Compute train, val and test portions
        # total = train + val + test
        # train = train/total
        # val = val/total
        # test = test/total

        # # compute number of images in train and val
        # train_size = round(train * len(cat_images))
        # val_size = round(val * len(cat_images))

        # # Split into train, val and test
        # print("\tSpliting dataset...")
        # self.train_cats = cat_images[:train_size]
        # self.train_dogs = dog_images[:train_size]
        # self.val_cats = cat_images[train_size:train_size + val_size]
        # self.val_dogs = dog_images[train_size:train_size + val_size]
        # self.test_cats = cat_images[train_size + val_size:]
        # self.test_dogs = dog_images[train_size + val_size:]

    def get_datasets(self, device=None):
        train_dataset = Data(self.train_cats, self.train_dogs, device)
        val_dataset = Data(self.val_cats, self.val_dogs, device)
        test_dataset = Data(self.test_cats, self.test_dogs, device)
        return train_dataset, val_dataset, test_dataset

    def load_images(self, paths):
        # Read images to memory
        # Doing this because I have enough ram space:)
        images = []
        for path in self.images:
            images.append(self.read_image(path))
        return np.asarray(images)

    def encode_labels(self, labels, images):
        S = 7
        B = 2
        C = 20

        ground_truth = np.zeros((len(labels), S, S, B * 5 + C))
        relative_labels = []
        for i,label in enumerate(labels):
            height, width, channels = images[i].shape
            for one_box_str in label:
                xmin, ymin, xmax, ymax, cla = [int(a) for a in one_box_str]
                x_abs = (xmin + xmax) / 2
                y_abs = (ymin + ymax) / 2
                w_abs = xmax - xmin
                h_abs = ymax - ymin

                x = x_abs / width
                y = y_abs / height
                w = w_abs / width
                h = h_abs / height

                confidence = 1.

                # encoded class
                encoded_class = self.encode_class(cla, C)

                # compute the grid row and column for this bbox
                cell_size = 1. / S # relative size
                grid_x = math.floor(x / cell_size)
                grid_y = math.floor(y / cell_size)

                target = ([x, y, w, h, confidence] * 2) + encoded_class # length = B * xywhc + num_classes
                ground_truth[i, grid_y, grid_x, :] = target
        return ground_truth

    def encode_class(self, a_class, num_class):
        encoded_class = [0] * num_class
        encoded_class[a_class] = 1
        return encoded_class


    def read_image(self, filepath):
        img = cv2.imread(filepath)
        return img

    def resize_images(self, images, input_shape):
        resized_images = []
        for img in images:
            resized_img = cv2.resize(img, input_shape)
            resized_images.append(resized_img)
        return np.asarray(resized_images)
