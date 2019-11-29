import torch
from torch.utils import data
import numpy as np
import glob
import cv2
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
            for line in lines:
                row = line.split(' ')
                self.images.append(images_path + "/" + row[0])
                labels = [row[n:n+5] for n in range(1, len(row), 5)]
                self.labels.append(labels)






        # cat_images = glob.glob(dirpath + "/Cat/*.jpg")[:]

        # # Filter out bad images
        # print("\tFiltering corrupt images... ")
        # cat_images = self.filter_corrupt_images(cat_images, input_shape)
        # dog_images = self.filter_corrupt_images(dog_images, input_shape)

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

    def filter_corrupt_images(self, paths, input_shape):
        good_imgpaths = []
        for path in tqdm(paths):
            try:
                img = self.read_image(path, input_shape)
                good_imgpaths.append(img)
            except:
                pass
        return good_imgpaths

    def read_image(self, filepath, input_shape):
        img = cv2.imread(filepath)
        img = cv2.resize(img, input_shape)
        return img
