import torch
from torch.utils import data
import numpy as np
import glob
import cv2
import math
from tqdm import tqdm


class Dataset(data.Dataset):
    def __init__(self, images, labels, preprocessing=None):
        self.images = images
        self.labels = labels
        self.len = self.labels.shape[0]

        # convert to numpy arrays
        self.images = np.asarray(self.images)
        self.labels = np.asarray(self.labels)

        # convert to tensors
        self.images = torch.from_numpy(self.images).float()
        self.labels = torch.from_numpy(self.labels).float()

        # In pytorch, conv2D expect input shape to be in this
        # form: (batch_size, channels, height, weight).
        self.images = self.images.permute(0, 3, 1, 2)

        self.preprocessing = preprocessing
        if self.preprocessing:
            self.images = self.preprocessing.normalize(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        X = self.images[index]
        y = self.labels[index]
        return X, y


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
            for i, line in enumerate(lines[:300]):
                row = line.split(' ')
                self.images.append(images_path + "/" + row[0])
                labels = [row[n:n+5] for n in range(1, len(row), 5)]
                self.labels.append(labels)

        print("\tLoading images...")
        self.images = self.load_images(self.images)

        print("\tEncoding labels...")
        self.labels = self.encode_labels(self.labels, self.images)

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
        train_dataset = Dataset(self.trainX, self.trainY, self.preprocessing)
        val_dataset = Dataset(self.valX, self.valY, self.preprocessing)
        test_dataset = Dataset(self.testX, self.testY, self.preprocessing)
        return train_dataset, val_dataset, test_dataset

    def load_images(self, paths):
        # Read images to memory
        # Doing this because I have enough ram space:)
        images = []
        for path in tqdm(self.images):
            img = self.read_image(path)
            img = self.preprocessing.BGR2RGB(img)
            images.append(img)
        return np.asarray(images)

    def encode_labels(self, labels, images):
        S = self.S
        B = self.B
        C = self.C

        ground_truth = np.zeros((len(labels), S, S, B * 5 + C))
        relative_labels = []
        for i, label in enumerate(tqdm(labels)):
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

                target = ([x, y, w, h, confidence] * B) + \
                    encoded_class  # length = B * xywhc + num_classes
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
