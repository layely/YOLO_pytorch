import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class YoloLoss(nn.Module):
    def __init__(self, grid_size, num_bboxes_per_grid, num_classes, lamda_coord=5., lamda_noob=0.5, device='cpu'):
        super(YoloLoss, self).__init__()
        self.S = grid_size
        self.B = num_bboxes_per_grid
        self.C = num_classes
        self.lamda_coord = lamda_coord
        self.lamda_noob = lamda_noob
        self.type = torch.FloatTensor
        self.device = device

    def IOU(self, box1, box2):
        # box: Tensor [x, y, w, h]
        x1, y1, w1, h1 = box1.numpy()
        x2, y2, w2, h2 = box2.numpy()

        x1min = x1 - w1/2
        x1max = x1 + w1/2
        y1min = y1 - h1/2
        y1max = y1 + h1/2

        x2min = x2 - w2/2
        x2max = x2 + w2/2
        y2min = y2 - h2/2
        y2max = y2 + h2/2

        # determine the coordinates of the intersection rectangle
        x_left = max(x1min, x2min)
        y_top = max(y1min, y2min)
        x_right = min(x1max, x2max)
        y_bottom = min(y1max, y2max)

        if x_right < x_left or y_bottom < y_top:
            return torch.Tensor([0])

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        area1 = w1 * h1
        area2 = w2 * h2

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(area1 + area2 - intersection_area)
        return torch.Tensor([iou])

    def forward(self, y, gt):
        """
        pred and target must have a shape like this:
        (batch_size, S, S, B*5 + num_classes)
        """

        pred = y.clone().detach()
        target = gt.clone().detach()

        mask_object = target[:, :, :, 4] > 0
        mask_no_object = target[:, :, :, 4] == 0

        # Cast bool types to float
        mask_object = mask_object.type(self.type)
        mask_no_object = mask_no_object.type(self.type)

        # repeat elements to make masks matches target size
        mask_object = mask_object.unsqueeze(-1).expand_as(target)
        mask_no_object = mask_no_object.unsqueeze(-1).expand_as(target)

        # class loss
        class_loss = (mask_object[:, :, :, 10:] * (target[:,
                                                          :, :, 10:] - pred[:, :, :, 10:]).pow(2)).sum()

        # Confidence loss no object
        # c_indexes = [4, 9], we just consider the first bounding box when there is no object in cell
        conf_loss_no_obj = (
            mask_no_object[:, :, :, 4] * (target[:, :, :, 4] - pred[:, :, :, 4]).pow(2)).sum()

        # Confidence targets must be computed
        confidence_targets = torch.zeros(target.size())
        if self.device == 'gpu':
            confidence_targets = confidence_targets.cuda()

        # Check the bounding the box with the max IOU
        batch_size = pred.shape[0]
        for batch in range(batch_size):
            for grid_i in range(pred.shape[1]):
                for grid_j in range(pred.shape[2]):
                    contain_obj = mask_object[batch, grid_i, grid_j, 4]
                    if (contain_obj.item() == 0.):
                        # Just continue if this is a no object cell.
                        continue
                    pred_box1 = pred[batch, grid_i, grid_j, 0:4]  # xywh
                    pred_box2 = pred[batch, grid_i, grid_j, 5:9]  # xywh
                    target_box = target[batch, grid_i, grid_j, 0:4]
                    iou1 = self.IOU(target_box, pred_box1)
                    iou2 = self.IOU(target_box, pred_box2)

                    # We want the confidence targets to equal the IOU
                    confidence_targets[batch, grid_i, grid_j, 4] = iou1
                    confidence_targets[batch, grid_i, grid_j, 9] = iou2

                    if iou1.item() > iou2.item():
                        mask_object[batch, grid_i, grid_j, 5:10] = 0.
                    else:
                        mask_object[batch, grid_i, grid_j, 0:5] = 0.

        # Confidence obj with object
        conf_loss_obj = (mask_object[:, :, :, [
                         4, 9]] * (confidence_targets[:, :, :, [4, 9]] - pred[:, :, :, [4, 9]]).pow(2)).sum()

        # X,Y loss
        xy_indexes = [0, 1, 5, 6]
        xy_loss = (mask_object[:, :, :, xy_indexes] * (target[:, :,
                                                              :, xy_indexes] - pred[:, :, :, xy_indexes]).pow(2)).sum()

        # W, H loss
        wh_indexes = [2, 3, 7, 8]
        wh_loss = (mask_object[:, :, :, wh_indexes] * (target[:, :,
                                                              :, wh_indexes] - pred[:, :, :, wh_indexes]).pow(2)).sum()

        total_loss = (self.lamda_coord*xy_loss + self.lamda_coord*wh_loss +
                      conf_loss_obj + self.lamda_noob*conf_loss_no_obj + class_loss) / batch_size
        return Variable(total_loss, requires_grad=True)
