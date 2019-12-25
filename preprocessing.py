import cv2
from torchvision import transforms
import torch
import numpy as np
import math
import random

from albumentations import (
    BboxParams,
    CenterCrop,
    Compose,
    Crop,
    HorizontalFlip,
    PadIfNeeded,
    RandomCrop,
    RandomSizedBBoxSafeCrop,
    RandomSizedCrop,
    Resize,
    Rotate,
    VerticalFlip,
)


class Preprocessing():
    def __init__(self, S, B, C, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 brightness=0, saturation=0, contrast=0, hue=0):
        self.S, self.B, self.C = S, B, C

        self.mean = mean
        self.std = std
        self.inv_mean = [((-1 * mean[i]) / std[i]) for i, m in enumerate(mean)]
        self.inv_std = [(1 / x) for x in std]

        self.normalize_transform = transforms.Normalize(mean=self.mean,
                                                        std=self.std)
        self.unnormalize_transform = transforms.Normalize(mean=self.inv_mean,
                                                          std=self.inv_std)

        self.brightness = brightness
        self.saturation = saturation
        self.contrast = contrast
        self.hue = hue
        self.color_transform = transforms.ColorJitter(
            brightness, saturation, contrast, hue)

        self.ToPILImage = transforms.ToPILImage(mode='RGB')
        self.ToTensor = transforms.ToTensor()

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

    def decode_label(self, label, img_h, img_w, obj_thresh=0.25):
        """
            label: torch tensor (S, S, B*5 + C)
            Return:
                list of boxes with each box represented by
                xmin,ymin,xmax,ymax,class, confidence
        """

        # Convert to numpy
        np_labels = label.cpu().numpy()

        boxes = []  # x1, y1, x2, y2, class_num, confidence
        for i in range(self.S):
            for j in range(self.S):
                bbox1, bbox2 = np_labels[i, j, :5], np_labels[i, j, 5:10]
                if bbox1[4] > bbox2[4]:
                    box = bbox1
                else:
                    box = bbox2
                if box[4] > obj_thresh:
                    # x and y relative to the image instead of grid_cell
                    cell_size = 1./self.S
                    x, y = box[:2]
                    x *= cell_size
                    y *= cell_size
                    cell_xmin = j * cell_size
                    cell_ymin = i * cell_size
                    x = x + cell_xmin
                    y = y + cell_ymin
                    box[:2] = [x, y]

                    # scale x, y, w, h from [0-1] to [0-255]
                    x, y, w, h = box[:4]
                    x, w = x * img_w, w * img_w
                    y, h = y * img_h, h * img_h

                    # from xy,wh to x1y1,x2y2 (top left - botom right)
                    xmin = x - w/2
                    xmax = x + w/2
                    ymin = y - h/2
                    ymax = y + h/2
                    xmin, xmax, ymin, ymax = [
                        int(i) for i in [xmin, xmax, ymin, ymax]]

                    class_number = np.argmax(np_labels[i, j, 10:])
                    confidence = box[4]

                    boxes.append(
                        [xmin, ymin, xmax, ymax, class_number, confidence])
        return boxes

    def post_process_image(self, torch_img):
        # Unormalize
        img = self.unnormalize(torch_img)
        img = self.channel_first_to_channel_last(img)

        # Convert to numpy images
        np_img = img.cpu().numpy().astype(np.uint8)
        np_img = self.RGB2BGR(np_img)

        return np_img

    def encode_class(self, a_class, num_class):
        encoded_class = [0] * num_class
        encoded_class[a_class] = 1
        return encoded_class

    def channel_first_to_channel_last(self, img):
        return img.permute(1, 2, 0)

    def BGR2RGB(self, img):
        """
            img: Numpy array
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def RGB2BGR(self, img):
        """
            img: Numpy array
        """
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def normalize(self, imgs):
        """
            imgs: Torch images range: 0-1, format RGB
        """
        if len(imgs.shape) == 3:
            # one image
            return self.normalize_transform(imgs)
        else:
            # batch of images
            ret = torch.zeros(imgs.shape)
            for i in range(imgs.shape[0]):
                ret[i] = self.normalize_transform(imgs[i])
            return ret

    def unnormalize(self, imgs):
        if len(imgs.shape) == 3:
            # one image
            ret = self.unnormalize_transform(imgs)
        else:
            # batch of images
            ret = torch.zeros(imgs.shape)
            for i in range(imgs.shape[0]):
                ret[i] = self.normalize_transform(imgs[i])

        ret *= 255.
        return ret

    def apply_random_transforms(self, images, labels):
        return self.random_shrink(images, labels)

    def random_color_transform(self, img):
        """
            Numpy image (RGB)
            Randomly change brightness, saturation,
            contrast and hue of a given torch image.
        """
        pil_img = self.ToPILImage(img)
        pil_img = self.color_transform(pil_img)
        return np.array(pil_img)

    def random_shrink(self, img, labels):
        height = img.shape[0]
        width = img.shape[1]
        stripped_size = round(random.random() * 100)
        resize = Resize(height - stripped_size, width - stripped_size,
                        always_apply=False, p=1)
        img, labels = self._transform(img, labels, [resize])
        pad = PadIfNeeded(min_height=height, min_width=width,
                          border_mode=4, value=None,
                          mask_value=None, always_apply=False, p=1.0)
        return self._transform(img, labels, [pad])


    def vertical_flip(self, img, labels):
        return self._transform(img, labels, [VerticalFlip(p=1)])

    def horizontal_flip(self, img, labels):
        return self._transform(img, labels, [HorizontalFlip(p=1)])

    def random_crop(self, img, labels):
        height = img.shape[0]
        width = img.shape[1]
        crop = RandomSizedCrop(300, height, width, always_apply=False, p=1.0)
        # crop = RandomSizedBBoxSafeCrop(height=height, width=height, p=1)
        return self._transform(img, labels, [crop])

    def random_rotate(self, img, labels):
        angle = 20 # random rotate [-angle, angle]
        border_mode = cv2.BORDER_CONSTANT # default 4
        rotation = Rotate(limit=angle, border_mode=4,
                          value=None, mask_value=None,
                          always_apply=False, p=1)
        return self._transform(img, labels, [rotation])

    def _transform(self, img, labels, transformations):
        aug = self._compose(transformations)
        annotations = self._format_annotation(img, labels)
        augmented_ann = aug(**annotations)
        return self._unformat_annotation(augmented_ann)



    def _format_annotation(self, image, label):
        """
        image: Numpy image in format RGB
        labels: List of bounding boxes with format
                xmin, ymin, xmax, ymax, classe number.
        """
        bboxes = []
        classe_ids = []
        for xyxyc in label:
            bboxes.append(xyxyc[:4])
            classe_ids.append(xyxyc[4])
        return {'image': image, 'bboxes': bboxes, 'class_ids': classe_ids}

    def _unformat_annotation(self, annotations):
        image = annotations['image']
        label = []
        bboxes = annotations['bboxes']
        class_ids = annotations['class_ids']
        for bbox, class_id in zip(bboxes, class_ids):
            label.append(list(bbox) + [class_id])
        return image, label

    def _compose(self, aug, min_area=0., min_visibility=0.):
        bbox_params = BboxParams(format='albumentations', # xmin,xmax,ymin,ymax relatives
                                 min_area=min_area,
                                 min_visibility=min_visibility,
                                 label_fields=['class_ids'])
        return Compose(aug, bbox_params=bbox_params)
