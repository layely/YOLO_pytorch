import cv2
from torchvision import transforms
import torch


class Preprocessing():
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 brightness=0, saturation=0, contrast=0, hue=0):
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
        self.color_transform = transforms.ColorJitter(brightness, saturation, contrast, hue)

        self.ToPILImage = transforms.ToPILImage()
        self.ToTensor = transforms.ToTensor()

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

    def random_color_transform(self, img):
        """
            Randomly change brightness, saturation,
            contrast and hue of a given torch image.
        """
        pil_img = self.ToPILImage(img)
        pil_img = self.color_transform(pil_img)
        return self.ToTensor(pil_img)

