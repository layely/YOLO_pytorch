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

    def IOU(self, boxes1, boxes2):
        # boxes: Tensor of shape (N, 4) 4 => [x, y, w, h]
        x1, y1, w1, h1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
        x2, y2, w2, h2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

        x1min = (x1 - w1/2).view(-1, 1)
        x1max = (x1 + w1/2).view(-1, 1)
        y1min = (y1 - h1/2).view(-1, 1)
        y1max = (y1 + h1/2).view(-1, 1)

        x2min = (x2 - w2/2).view(-1, 1)
        x2max = (x2 + w2/2).view(-1, 1)
        y2min = (y2 - h2/2).view(-1, 1)
        y2max = (y2 + h2/2).view(-1, 1)

        # determine the coordinates of the intersection rectangle
        x_left = torch.max(x1min, x2min)
        y_top = torch.max(y1min, y2min)
        x_right = torch.min(x1max, x2max)
        y_bottom = torch.min(y1max, y2max)

        mask = (x_right < x_left) | (y_bottom < y_top)
        mask = mask.type(x_left.dtype)

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = mask * (x_right - x_left) * (y_bottom - y_top)

        area1 = w1.view(-1, 1) * h1.view(-1, 1)
        area2 = w2.view(-1, 1) * h2.view(-1, 1)
        
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        ious = intersection_area / (area1 + area2 - intersection_area)
        return ious

    def mse(self, pred, target):
        return F.mse_loss(pred, target, reduction="sum")

    def forward(self, pred, target):
        """
        pred and target must have a shape like this:
        (batch_size, S, S, B*5 + num_classes)
        """

        device = pred.device

        mask_object = target[:, :, :, 4] > 0
        mask_no_object = target[:, :, :, 4] == 0

        # repeat elements to make masks matches target size
        mask_object = mask_object.unsqueeze(-1).expand_as(target)
        mask_no_object = mask_no_object.unsqueeze(-1).expand_as(target)

        print("mask object shape: ", mask_object.shape)

        target_obj = target[mask_object].view(-1, 30)
        target_noobj = target[mask_no_object].view(-1, 30)
        pred_obj = pred[mask_object].view(-1, 30)
        pred_noobj = pred[mask_no_object].view(-1, 30)

        print("target shape in loss: ", target_obj.shape)

        pred_boxes = pred_obj[:, :10].contiguous().view(-1, 5)
        target_boxes = target_obj[:, :10].contiguous().view(-1, 5)
        ious = self.IOU(pred_boxes, target_boxes)
        ious = ious.view(-1, 2)

        target_boxes_iou = target_obj[:, :5]
        pred_boxes_iou = Variable(torch.zeros(ious.shape[0], 5).to(device))

        for i in range(ious.shape[0]):
            iou1 = ious[i, 0]
            iou2 = ious[i, 1]
            if iou1.item() > iou2.item():
                pred_boxes_iou[i] = pred_obj[i, :5]
            else:
                pred_boxes_iou[i] = pred_obj[i, 5:10]

        xy_loss = self.mse(pred_boxes_iou[:, :2], target_boxes_iou[:, :2])
        wh_loss = self.mse(torch.sqrt(pred_boxes_iou[:, 2:4]), torch.sqrt(target_boxes_iou[:, 2:4]))

        confidence_loss_obj = self.mse(pred_boxes_iou[:, 4], target_boxes_iou[:, 4])
        confidence_loss_noobj = self.mse(pred_noobj[:, [4,9]], target_noobj[:, [4,9]])

        class_loss = self.mse(pred_obj[:, 10:], target_obj[:, 10:])

        # print("Losses: xy: {}, wh: {}, conf_obj: {}, conf_noobj: {}, class: {}".format(xy_loss, wh_loss, confidence_loss_obj, confidence_loss_noobj, class_loss))

        total_loss = self.lamda_coord * xy_loss + self.lamda_coord * wh_loss + confidence_loss_obj + self.lamda_noob * confidence_loss_noobj + class_loss
        batch_size = target.shape[0]

        individual_losses = [xy_loss.item(), wh_loss.item(), confidence_loss_obj.item(), confidence_loss_noobj.item(), class_loss.item()]
        individual_losses = [x/batch_size for x in individual_losses]
        return total_loss / batch_size, individual_losses