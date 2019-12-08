import torch

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
