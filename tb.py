import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid

TENSORBOARD_LOG_DIR = "tensorboard_logs"


class Tensorboard():
    """
        A wrapper around Tensorboard.
    """

    def __init__(self, log_dir=TENSORBOARD_LOG_DIR):
        self.writer = SummaryWriter(
            log_dir=log_dir, comment='', purge_step=None)

    def add_scalar(self, name, scalar, epoch):
        self.writer.add_scalar(name, scalar, epoch)

    def add_images(self, name, model, arg_images):
        images = arg_images * 255
        grid = make_grid(images)
        self.writer.add_image(name, grid, 0)
        # if model:
        # self.writer.add_graph(model, images)

    def close(self):
        self.writer.close()
