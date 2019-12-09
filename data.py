import torch
from torch.utils import data
import numpy as np
import glob
import cv2
import math
from tqdm import tqdm
import random


class Dataset(data.Dataset):
    def __init__(self, images, labels, S, B, C, preprocessing=None, random_transform=False):
        self.images = images
        self.labels = labels
        self.len = len(self.labels)
        self.S, self.B, self.C = S, B, C
        self.random_transform = random_transform

        # convert to numpy arrays
        self.images = np.asarray(self.images)
        self.labels = np.asarray(self.labels)

        # convert to tensors
        # self.images = torch.from_numpy(self.images).float()
        # self.labels = torch.from_numpy(self.labels).float()

        # In pytorch, conv2D expect input shape to be in this
        # form: (batch_size, channels, height, weight).
        # self.images = self.images.permute(0, 3, 1, 2)

        self.preprocessing = preprocessing

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        np_image = self.images[index]  # h,w,c
        label = self.labels[index]

        # Augmentation
        if self.random_transform:
            if random.random() > 0.5:
                np_image, label = self.flip_horizontal(np_image, label)
            if random.random() > 0.2
                np_image = self.preprocessing.random_color_transform(np_image)

        torch_image = self.preprocessing.ToTensor(np_image)
        X = self.preprocessing.normalize(torch_image)

        y = self.encode_labels(label)

        return X.float(), torch.from_numpy(y).float()

    def encode_labels(self, label):
        S = self.S
        B = self.B
        C = self.C

        ground_truth = np.zeros((S, S, B * 5 + C))
        for xmin, ymin, xmax, ymax, cla in label:
            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin

            # compute the grid row and column for this bbox
            cell_size = 1. / S  # relative size
            grid_x = math.floor(x / cell_size)
            grid_y = math.floor(y / cell_size)

            # x,y relative to the cell
            x = x - (grid_x * cell_size)
            y = y - (grid_y * cell_size)

            # Normalize x and y with respect to cell_size
            # so that they are in the range [0, 1]
            x = x / cell_size
            y = y / cell_size

            confidence = 1.
            encoded_class = self.encode_class(int(cla), C)

            target = ([x, y, w, h, confidence] * B) + \
                encoded_class  # length = B * xywhc + num_classes
            ground_truth[grid_y, grid_x, :] = target
        return ground_truth

    def flip_horizontal(self, img, label):
        """
            img: numpy array
            label: bounding boxes with relative coordinates
        """

        flipped_img = np.fliplr(img).copy()
        flipped_label = []
        for xmin, ymin, xmax, ymax, cla in label:
            box = [1 - xmax, ymin, 1 - xmin, ymax, cla]
            flipped_label.append(box)
        return flipped_img, flipped_label

    def encode_class(self, a_class, num_class):
        encoded_class = [0] * num_class
        encoded_class[a_class] = 1
        return encoded_class


class DataGenerator():
    def __init__(self, images_path, txt_file, train, val, test, input_shape, S=7, B=2, C=20, preprocessing=None):
        print("Preparing data...")

        self.images_path = images_path
        self.txt_file = txt_file
        self.train = train
        self.val = val
        self.test = test
        self.S = S
        self.B = B
        self.C = C

        self.preprocessing = preprocessing

        # Load images
        self.images = []
        # Each label is a list of (xmin, ymin, xmax, ymax, class)
        self.labels = []
        with open(txt_file, "r") as f:
            lines = f.read().splitlines()
            for i, line in enumerate(lines[:3]):
                row = line.split(' ')
                self.images.append(images_path + "/" + row[0])
                labels = [row[n:n+5] for n in range(1, len(row), 5)]
                self.labels.append(labels)

        print("\tLoading images...")
        self.images = self.load_images(self.images)

        print("\tConvert labels to relative coordinates...")
        self.labels = self.to_relative_coordinates(self.labels, self.images)

        print("\tResizing images...")
        self.images = self.resize_images(self.images, input_shape)

        # Compute train, val and test portions
        total = train + val + test
        train = train/total
        val = val/total
        test = test/total

        # # compute number of images in train and val
        train_size = round(train * len(self.images))
        val_size = round(val * len(self.images))

        # # Split into train, val and test
        print("\tSpliting dataset...")
        self.trainX = self.images[:train_size]
        self.trainY = self.labels[:train_size]
        self.valX = self.images[train_size:train_size + val_size]
        self.valY = self.labels[train_size:train_size + val_size]
        self.testX = self.images[train_size + val_size:]
        self.testY = self.labels[train_size + val_size:]

    def get_datasets(self):
        S = self.S
        B = self.B
        C = self.C

        train_dataset = Dataset(self.trainX, self.trainY,
                                S, B, C, self.preprocessing, random_transform=True)
        val_dataset = Dataset(self.valX, self.valY, S,
                              B, C, self.preprocessing)
        test_dataset = Dataset(self.testX, self.testY,
                               S, B, C, self.preprocessing)
        return train_dataset, val_dataset, test_dataset

    def to_relative_coordinates(self, labels, images):
        labels_with_rel_coord = []
        for i in tqdm(range(len(labels))):
            xyxy_list = labels[i]
            image = images[i]
            image_h, image_w, _ = image.shape
            rel_coords = []
            for one_box_str in xyxy_list:
                x1, y1, x2, y2, classe = [int(a) for a in one_box_str]
                rel_x1 = x1 / image_w
                rel_x2 = x2 / image_w
                rel_y1 = y1 / image_h
                rel_y2 = y2 / image_h
                rel_coords.append([rel_x1, rel_y1, rel_x2, rel_y2, classe])
            labels_with_rel_coord.append(rel_coords)
        return labels_with_rel_coord

    def load_images(self, paths):
        # Read images to memory
        # Doing this because I have enough ram space:)
        images = []
        for path in tqdm(self.images):
            img = self.read_image(path)
            img = self.preprocessing.BGR2RGB(img)
            images.append(img)
        return np.asarray(images)

    def read_image(self, filepath):
        img = cv2.imread(filepath)
        return img

    def resize_images(self, images, input_shape):
        resized_images = []
        for img in images:
            resized_img = cv2.resize(img, input_shape)
            resized_images.append(resized_img)
        return np.asarray(resized_images)
