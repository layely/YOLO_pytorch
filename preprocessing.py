import cv2
from torchvision import transforms
import torch


class Preprocessing():
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
        self.inv_mean = [((-1 * mean[i]) / std[i]) for i,m in enumerate(mean)]
        self.inv_std = [1 / x for x in std]

        self.normalize_transform = transforms.Normalize(mean=self.mean,
                                              std=self.std)
        self.unnormalize_transform = transforms.Normalize(mean=self.inv_mean,
                                                std=self.inv_std)

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
            imgs: Torch images uint8 type: 0-255, format RGB
        """
        norm = imgs / 255.
        print(norm.shape)

        if len(norm.shape) == 3:
            # one image
            return self.normalize_transform(norm)
        else:
            # batch of images
            ret = torch.zeros(norm.shape)
            for i in range(norm.shape[0]):
                ret[i] = self.normalize_transform(norm[i])
            return ret

    def unnormalize(imgs):
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
