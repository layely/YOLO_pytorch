import torch

from voc_classes import get_class_name

CHECKPOINT_PATH = "checkpoint.tar"


def save_checkpoint(model, optimizer, epoch, loss):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss
    }
    torch.save(checkpoint, CHECKPOINT_PATH)


def load_checkpoint(model):
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, optimizer_state_dict, loss

def write_result_to_disk(file_name, bboxes, type="gt"):
    """
    Write gt or pred result to disk with respect to
    https://github.com/rafaelpadilla/Object-Detection-Metrics#how-to-use-this-project
    Args:
        file_name: name of the file where to store the result
        bboxes: list of bounding boxes where each bbox consist of
                x1,y1,x2,y2,class,confidence
        type: gt or pred for ground truth or prediction, respectively
    """
    text = ""
    for bbox in bboxes:
        text += get_class_name(bbox[4]) # class name
        if type == "pred":
            text += " " + str(bbox[5])
        # from xyxy to xywh format
        xmin, ymin, xmax, ymax = bbox[:4]
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin
        for i in [x,y,w,h]:
            text += " " + str(i)
        text += "\n"

    with open(file_name, "w") as f:
        f.write(text)